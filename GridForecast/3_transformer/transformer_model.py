"""Transformer-based time series forecasting model and trainer with custom losses.

Features:
  * Preprocessing: feature engineering + scaling (train/val split by grid id)
  * Sequence assembly: builds (seq_len = core_len + 2*pad) overlapping windows.
        * Model: Per-timestep input MLP -> sinusoidal positional encoding -> Transformer encoder stack -> (optional CNN aggregation with conv_pad) -> per-timestep output MLP.
  * Custom losses:
        - mae: standard MAE on central (trimmed) window
        - mse: standard MSE on central window
        - mae_maex: mean(|(y_pred - y_true)/maex|)
        - alpha_peak: peak-aware loss combining abs + squared scaled error weighted by alpha
  * Trimming: only central window (loss_trim_left/right) contributes to loss.
  * Trainer: PyTorch training loop + early stopping + Ray Tune compatible step() API.
  * Ray Tune example helper with Optuna TPE + ASHA + TensorBoard logger (TBX).

Assumptions:
  * Raw data stored in an HDF5 file with keys 'X' and 'y'. MultiIndex (grid_id, hour_index)
  * One target column defined in Preprocessor.TARGET_COLS.

Minimal usage (manual):
    from transformer_model import TransformerTrainer, TransformerConfig
    cfg = TransformerConfig(in_features=F, out_features=1, epochs=10)
    trainer = TransformerTrainer({**cfg.__dict__, '_data': {...}})
    for _ in range(cfg.epochs):
        print(trainer.step())

Ray Tune usage:
    from transformer_model import build_tune_search_space, build_tune_trainable
    space = build_tune_search_space(tune)
    trainable = build_tune_trainable()
    Tuner(trainable, param_space=space, ...)

"""
# ---------------------------------------------------------------------------
# Environment safety knobs (must run BEFORE importing torch)
# - PYTORCH_NVML_DISABLE=1 prevents allocator NVML queries that can assert
#   in some container/driver combos (e.g., under Ray workers or enroot).
# - PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync enables the modern
#   async allocator, which pairs well with NVML disabled and reduces
#   fragmentation. These are no-ops on CPU-only runs.
# ---------------------------------------------------------------------------
try:  # keep import-time side effects minimal and safe
     import os as _os  # type: ignore
     _os.environ.setdefault("PYTORCH_NVML_DISABLE", "1")
     _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
except Exception:
     pass

# from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import random
import time
import json
from pathlib import Path

import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from sklearn.preprocessing import StandardScaler
from vmdpy import VMD
import multiprocessing as mp

from evaluation import EvaluationMetrics  # type: ignore
from resource_report import resource_report  # timing CPU resources

# Optional Ray / TensorBoard integrations
_RAY_AVAILABLE = False
_TB_AVAILABLE = False
session = None
SummaryWriter = None
try:  # Ray AIR session for reporting metrics to Tune
    from ray.air import session as _ray_session  # type: ignore
    session = _ray_session
    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    session = None
    _RAY_AVAILABLE = False
try:  # TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover
    SummaryWriter = None
    _TB_AVAILABLE = False


# ============================================================
# Preprocessor
# ============================================================
class Preprocessor:
        """Feature/target preprocessing and optional VMD augmentation for the Transformer.

        Responsibilities:
            - Select base features (`X_BASE_COLS`) from raw `X`.
            - Add engineered features:
                    * zero indicators
                    * log1p transforms
                    * seasonal sin/cos terms from the hour index
            - Optionally compute Variational Mode Decomposition (VMD) modes for selected
                time series columns (configured via `VMD_COLS` / `VMD_K_MODES`).
            - Scale X and y using `StandardScaler`.

        Data assumptions:
            - Input X/y are DataFrames indexed by MultiIndex with levels `batch` and `hour`.
            - If dual-target mode is used, targets are interpreted as (P, Q).
        """

    def __init__(self, data_cfg) -> None: 
                """Create the preprocessor from the provided data configuration."""
        self.X_BASE_COLS        = data_cfg["X_BASE_COLS"]
        self.TARGET_COLS        = data_cfg["TARGET_COLS"]
        self.ZERO_BASE_FEATURES = data_cfg["ZERO_BASE_FEATURES"]
        self.LOG1P_COLS         = data_cfg["LOG1P_COLS"]
        # Accept TS_PERIODS as list[int] or a comma-separated string. Normalize to List[int].
        self.TS_PERIODS         = self._normalize_ts_periods(data_cfg["TS_PERIODS"])  # type: ignore
        self.VMD_APPROACH       = data_cfg["VMD_APPROACH"]
        self.VMD_COLS           = data_cfg["VMD_COLS"]
        self.VMD_K_MODES        = data_cfg["VMD_K_MODES"]

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.fitted = False
        self.zero_names: List[str] = []
        self.period_names: List[str] = []
        self.feature_names_: List[str] = []
        # Store exactly the list (and order) of feature names passed into scaler_X.fit
        # so that later transform() calls cannot drift due to re-generated engineered
        # features (VMD modes, seasonal sin/cos) or list mutation side effects.
        self.scale_feature_names_: List[str] = []

    @staticmethod
    def _normalize_ts_periods(val):
        """Normalize TS_PERIODS into a list of ints.

        Accepts:
          - list[int]: returned as-is
          - list[str]: converted to ints
          - str: comma/space-separated numbers (e.g., "8760,168,24" or "8760 168 24"). Empty -> []
          - None or falsy: []
        """
        if val is None:
            return []
        # If Ray Tune injects strings, handle here
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return []
            # Allow comma and/or whitespace separators
            parts = [p for chunk in s.replace(" ", ",").split(",") for p in [chunk.strip()] if p]
            out: List[int] = []
            for p in parts:
                try:
                    out.append(int(p))
                except Exception:
                    # Ignore tokens that are not integers
                    continue
            return out
        # If it's an iterable of numbers/strings
        try:
            lst = list(val)
        except Exception:
            return []
        out: List[int] = []
        for x in lst:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    @staticmethod
    def _compute_vmd_for_grid(inputgrid: tuple):
        #-----------------------------------------------------------#
        grid_id, _ = inputgrid
        grid_df, vmd_settings = _
        #----------------------- Settings --------------------------#
        alpha= vmd_settings["alpha"]
        tau=   vmd_settings["tau"]
        K=     vmd_settings["K"]
        DC=    vmd_settings["DC"]
        init=  vmd_settings["init"]
        tol=   vmd_settings["tol"]
        #-------------------------- VMD ----------------------------#
        vmd_signal_list = []
        for col in grid_df.columns:
            signal = np.asarray(grid_df[col], dtype=float)
            if np.all(signal==0):
                modes_df = pd.DataFrame(np.zeros((len(signal), K+1), dtype=float), columns=[f'{col}_{i+1}' for i in range(K)]+[f'{col}_res'])
                vmd_signal_list.append(modes_df)
            else:
                u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
                modes_df = pd.DataFrame(u.T, columns=[f'{col}_{i+1}' for i in range(K)])
                modes_df[f'{col}_res'] = signal - modes_df.sum(axis=1)
                vmd_signal_list.append(modes_df)
        out = pd.concat(vmd_signal_list, axis=1)
        out.index = pd.MultiIndex.from_arrays([[grid_id]*len(out), out.index], names=['batch', 'hour'])
        #-----------------------------------------------------------#
        return out

    def run_vmd(self, signals, vmd_approach, type="train"):
        """Compute or load cached VMD decompositions for the provided signals.

        Args:
            signals: DataFrame of shape (batch,hour) x n_signals (MultiIndex).
            vmd_approach: 'read' to load from cache, 'write' to compute and store,
                or any other value to compute without saving.
            type: Cache key suffix, typically 'train' or 'val'.

        Returns:
            DataFrame of VMD mode features aligned to `signals.index`.
        """
        #----------------------- Settings --------------------------#
        vmd_settings = {
            "alpha": 2000,              # moderate bandwidth constraint
            "tau":   0.,                # noise-tolerance (0 for noiseless)
            "K":     self.VMD_K_MODES,  # number of modes
            "DC":    1,                 # do not include the DC part as separate mode
            "init":  1,                 # initialize omegas uniformly
            "tol":   1e-5               # convergence tolerance
        }
        
        ### Obtain savefile name
        vmd_cols = signals.columns
        sorted_cols = sorted(vmd_cols)
        cleaned_cols = [col.replace("_", "") for col in sorted_cols]
        output_name = "_".join(cleaned_cols)
        filename = f"VMD{vmd_settings['K']}_{output_name}.h5"

        if vmd_approach == "read":
            try:
                grid_vmd_signal = pd.read_hdf(f"/dss/dsshome1/05/ge96ton2/GridForecast/3_transformer/data/{filename}", key=f"data_{type}")
                grid_vmd_signal.index = signals.index
                return grid_vmd_signal
            except (FileNotFoundError, KeyError) as e:
                print(f"Error reading VMD file: {e}")
                print("Precomputing new output instead!")
                vmd_approach = "write"

        NUM_VMD_CPUS = 2    # 2 seems to work most efficiently
        VMD_CHUNK_SIZE = 4  # tasks per worker fetch; keep 1 because work units similar size
        
        # Select grids
        grids = signals.index.get_level_values("batch").unique().values
        grid_data_dict = {grid: (signals.xs(grid, level="batch").copy(), vmd_settings) for grid in grids}
        
        # Pass only the necessary data to each worker
        with mp.Pool(processes=NUM_VMD_CPUS) as pool:
            results = list(pool.imap(self._compute_vmd_for_grid, grid_data_dict.items(), chunksize=VMD_CHUNK_SIZE))
        grid_vmd_signal = pd.concat(results, axis=0)
        grid_vmd_signal.index = signals.index

        if vmd_approach == "write":
            grid_vmd_signal.to_hdf(f"/dss/dsshome1/05/ge96ton2/GridForecast/3_transformer/data/{filename}", key=f"data_{type}", mode="a")

        return grid_vmd_signal

    # ------------- internal helpers -------------
    def _universal_X_trafo(self, df_X: pd.DataFrame, type, fit: bool) -> pd.DataFrame:
        """Core feature engineering pipeline.

        When fit=True we record zero_names / period_names. When fit=False we DO NOT
        mutate those lists again (to keep exclusion sets stable) and instead use the
        already learned lists purely to reconstruct / select features.
        """
        df_X = df_X[self.X_BASE_COLS].copy()

        # --- Zero indicators (record only during fit) ---
        zeros = (df_X[self.ZERO_BASE_FEATURES] == 0).astype('int32') if self.ZERO_BASE_FEATURES else pd.DataFrame(index=df_X.index)
        if not zeros.empty:
            zeros.columns = [f"{c}_zero" for c in self.ZERO_BASE_FEATURES]
            if fit:
                self.zero_names = list(zeros.columns)
        # If not fit, ensure columns appear in same order as stored zero_names
        if not fit and self.zero_names:
            # Reorder zeros to stored order (in case DataFrame column order differs)
            zeros = zeros.reindex(columns=self.zero_names)

        # --- Log1p transforms (in-place, deterministic) ---
        for col in self.LOG1P_COLS:
            if col in df_X.columns:
                df_X[col] = np.log1p(df_X[col].astype('float32'))

        # --- Seasonal features ---
        hours = df_X.index.get_level_values(1).to_numpy()
        seasonal_cols: List[pd.Series] = []
        seasonal_names: List[str] = []
        for freq in self.TS_PERIODS:
            sin_name = f"sin_{freq}"
            cos_name = f"cos_{freq}"
            seasonal_cols.append(pd.Series(np.sin(2 * np.pi / freq * hours), index=df_X.index, name=sin_name))
            seasonal_cols.append(pd.Series(np.cos(2 * np.pi / freq * hours), index=df_X.index, name=cos_name))
            if fit:
                self.period_names.extend([sin_name, cos_name])
        if seasonal_cols:
            seasonal_df = pd.concat(seasonal_cols, axis=1)
            # On transform ensure column ordering matches training order
            if not fit and self.period_names:
                seasonal_df = seasonal_df.reindex(columns=[c for c in self.period_names if c in seasonal_df.columns])
            df_X = pd.concat([df_X, seasonal_df], axis=1)

        # --- VMD decomposition (order guaranteed by self.VMD_COLS list) ---
        if self.VMD_K_MODES >= 1 and self.VMD_COLS:
            signals = df_X[self.VMD_COLS]
            grid_vmd_signal = self.run_vmd(signals, self.VMD_APPROACH, type=type)
            # Ensure deterministic ordering: for each original column, modes 1..K + _res
            ordered_vmd_cols: List[str] = []
            for base in self.VMD_COLS:
                for k in range(self.VMD_K_MODES):
                    ordered_vmd_cols.append(f"{base}_{k+1}")
                ordered_vmd_cols.append(f"{base}_res")
            grid_vmd_signal = grid_vmd_signal.reindex(columns=ordered_vmd_cols)
            engineered = pd.concat([df_X.drop(columns=self.VMD_COLS), zeros, grid_vmd_signal], axis=1)
        else:
            engineered = pd.concat([df_X, zeros], axis=1)

        return engineered

    def _fit_or_transform_X(self, df_X: pd.DataFrame, fit: bool, type) -> pd.DataFrame:
        df_X = self._universal_X_trafo(df_X, type=type, fit=fit)

        if fit:
            # Determine which columns are scaled (exclude indicator & seasonal & raw VMD source cols)
            scale_feats = [c for c in df_X.columns if (c not in self.zero_names and c not in self.period_names and c not in self.VMD_COLS)]
            df_X[scale_feats] = self.scaler_X.fit_transform(df_X[scale_feats])
            self.scale_feature_names_ = list(scale_feats)  # exact order used for scaler
            self.feature_names_ = list(df_X.columns)       # full feature order
        else:
            if not self.fitted:
                raise ValueError("Preprocessor not fitted for transform().")
            # Recompute engineered features then enforce scaling on the stored ordered list
            missing_scaled = [c for c in self.scale_feature_names_ if c not in df_X.columns]
            if missing_scaled:
                raise ValueError(f"Missing engineered feature(s) at transform time: {missing_scaled}")
            df_X[self.scale_feature_names_] = self.scaler_X.transform(df_X[self.scale_feature_names_])
            # Reorder / subset to training feature order (drop unexpected extras silently)
            df_X = df_X.reindex(columns=self.feature_names_)
        return df_X

    ### Target aggregation (sum) BEFORE scaling if agg_hours > 1
    def _aggregate_target(self, y_df: pd.DataFrame, agg_h: int) -> pd.DataFrame:
        if agg_h == 1: return y_df
        # Expect MultiIndex (grid,hour). Build aggregated hour bucket.
        lvl_grid = y_df.index.get_level_values(0)
        hours = y_df.index.get_level_values(1).astype(int)
        agg_hour = (hours // agg_h).astype(int) # all hours in the same agg bucket get the same value e.g. [0,0,1,1,2,2,...]
        new_index = pd.MultiIndex.from_arrays([lvl_grid, agg_hour], names=[y_df.index.names[0], 'agg_hour'])
        y_tmp = y_df.copy()
        y_tmp.index = new_index
        # Sum within each (grid, agg_hour) bucket
        y_agg = y_tmp.groupby(level=[0, 1]).sum()
        return y_agg.astype('float32')

    def _fit_or_transform_y(self, df_y: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df_y = df_y[self.TARGET_COLS].copy()
        # Dual-target case: learn scaler on S=sqrt(P^2+Q^2) and apply same rescaling to P and Q
        if len(self.TARGET_COLS) == 2:
            p_col, q_col = self.TARGET_COLS
            P = df_y[p_col].astype('float32').values.reshape(-1, 1)
            Q = df_y[q_col].astype('float32').values.reshape(-1, 1)
            S = np.sqrt(P**2 + Q**2).astype('float32')
            if fit:
                self.scaler_y.fit(S)
            if not fit and not self.fitted:
                raise ValueError("Preprocessor not fitted for transform().")
            mean_S = float(self.scaler_y.mean_[0])
            scale_S = float(self.scaler_y.scale_[0]) + 1e-12
            P_scaled = (P - mean_S) / scale_S
            Q_scaled = (Q - mean_S) / scale_S
            out = np.concatenate([P_scaled, Q_scaled], axis=1)
            return pd.DataFrame(out, index=df_y.index, columns=self.TARGET_COLS)
        # Single-target: standard scaling
        if fit:
            self.scaler_y.fit(df_y.values)
            arr = self.scaler_y.transform(df_y.values)
            return pd.DataFrame(arr, index=df_y.index, columns=self.TARGET_COLS)
        else:
            if not self.fitted:
                raise ValueError("Preprocessor not fitted for transform().")
            arr = self.scaler_y.transform(df_y.values)
            return pd.DataFrame(arr, index=df_y.index, columns=self.TARGET_COLS)

    def fit_transform(self, df_X: pd.DataFrame, df_y: pd.DataFrame, aggregation: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = self._fit_or_transform_X(df_X, fit=True, type="train")
        df_y_agg = self._aggregate_target(df_y, aggregation)
        y = self._fit_or_transform_y(df_y_agg, fit=True)
        self.fitted = True
        return X, y

    def transform(self, df_X: pd.DataFrame, df_y: Optional[pd.DataFrame] = None, aggregation: int = 1, type = "val"):
        X = self._fit_or_transform_X(df_X, fit=False, type=type)
        if df_y is not None:
            df_y_agg = self._aggregate_target(df_y, aggregation)
            y = self._fit_or_transform_y(df_y_agg, fit=False)
            return X, y
        return X 

    def inverse_transform_y(self, df_y_scaled: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor not fitted.")
        if len(self.TARGET_COLS) == 2 and df_y_scaled.shape[1] == 2:
            mean_S = float(self.scaler_y.mean_[0])
            scale_S = float(self.scaler_y.scale_[0])
            vals = df_y_scaled.values.astype('float32')
            out = vals * scale_S + mean_S
            return pd.DataFrame(out, index=df_y_scaled.index, columns=self.TARGET_COLS)
        arr = self.scaler_y.inverse_transform(df_y_scaled.values)
        return pd.DataFrame(arr, index=df_y_scaled.index, columns=self.TARGET_COLS)


# ============================================================
# Config
# ============================================================
@dataclass
class TransformerConfig:
    """Hyperparameter container for `TransformerForecastModel` and `TransformerTrainer`.

    This dataclass is populated from a plain dict (e.g. Ray Tune config). Only
    keys defined here are considered by the trainer.
    """
    # Sequence lengths
    # core_len and pad_hours control how sequences are assembled from raw series: seq_len = core_len + 2 * pad_hours
    core_len: int = 146
    pad_hours: int = 24
    # Aggregate target (sum over agg_hours). 1 means no aggregation. Allowed typical values: 1,2,4,12,24
    agg_hours: int = 1
    in_features: int = 1
    out_features: int = 1

    # Architectures
    input_mlp_layers: int = 1
    input_mlp_hidden: int = 128
    input_dropout: float = 0.0

    d_model: int = 128
    num_heads: int = 1
    ff_dim: int = 128
    num_transformer_layers: int = 1
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0

    output_mlp_layers: int = 1
    output_mlp_hidden: int = 128
    output_dropout: float = 0.0

    activation: str = "gelu"

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "none"  # none | cosine | plateau
    scheduler_epochs: int = 50
    epochs: int = 50
    batch_size: int = 16
    grad_clip: float = 1.0
    loss_type: str = "mae_maex"  # mae | mse | mae_maex | alpha_peak
    patience: int = 3
    # Loss blending weight for multi-target case: lambda/2*(P-term + Q-term) + (1-lambda)*S-term
    multi_target_lambda: float = 0.5

    # Aggregation mode:
    #  - 'sum' (default): original behavior (trim padding then sum blocks of size agg_hours)
    #  - 'conv': learnable depthwise Conv1d aggregation (symmetric context).
    #            Spec:
    #              * Transformer produces sequence of length core_len + 2*pad_hours.
    #              * For conv mode we extract slice: indices [pad_hours - conv_padding : pad_hours + core_len + conv_padding]
    #                giving length core_len + 2*conv_padding (symmetric extra context from both sides of central window).
    #              * kernel_size = agg_hours + 2*conv_padding, stride = agg_hours.
    #              * Output length: ((core_len + 2*conv_pad) - (agg_hours + 2*conv_pad)) / agg_hours + 1 = core_len/agg_hours.
    #              * If agg_hours == 1 and conv_padding > 0, Conv1d still applied (kernel=1+2*conv_pad, stride=1) returning core_len outputs.
    aggregation_mode: str = "sum"  # sum | conv
    # Amount of original sequence padding (<= pad_hours) to include inside each convolution window on both sides.
    # Effective kernel size for conv mode: agg_hours + 2*conv_padding.
    conv_padding: int = 0

    # Reporting and others
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 
    compile_model: bool = True
    bottom_rung_report: Optional[int] = None
    full_metrics_every: int = 5
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    return {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
    }[name]


# ============================================================
# Positional Encoding
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (adds to embeddings)."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============================================================
# Per-timestep MLP (shared weights)
# ============================================================
class PerTimestepMLP(nn.Module):
    """Shared-weight MLP applied independently at each timestep."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int, 
                 dropout: float, activation: str, final_activation: bool = False):
        super().__init__()
        modules: List[nn.Module] = []
        act = get_activation(activation)
        if layers <= 0:
            modules.append(nn.Linear(in_dim, out_dim))
            if final_activation:
                modules.append(act)
        else:
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(act)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            for _ in range(layers - 1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(act)
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
            if hidden_dim != out_dim:
                modules.append(nn.Linear(hidden_dim, out_dim))
                if final_activation:
                    modules.append(act)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        # Use reshape instead of view to support non-contiguous inputs (e.g. after transpose under torch.compile)
        y = self.net(x.reshape(b * s, f))
        return y.reshape(b, s, -1)


# ============================================================
# Core Model
# ============================================================
class TransformerForecastModel(nn.Module):
    """Transformer-based forecasting model operating on (B, T, F) windows."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.aggregation_mode not in ('sum', 'conv'):
            raise ValueError("aggregation_mode must be 'sum' or 'conv'")
        if cfg.conv_padding < 0:
            raise ValueError("conv_padding must be >=0")
        # Auto-cap conv_padding to pad_hours to avoid invalid negative slice starts
        if cfg.conv_padding > cfg.pad_hours:
            try:
                print(f"[TransformerForecastModel] conv_padding ({cfg.conv_padding}) > pad_hours ({cfg.pad_hours}); capping to {cfg.pad_hours}.")
            except Exception:
                pass
            cfg.conv_padding = int(cfg.pad_hours)
        self.input_mlp = PerTimestepMLP(cfg.in_features, 
                                        cfg.d_model, 
                                        cfg.input_mlp_hidden, 
                                        cfg.input_mlp_layers, 
                                        cfg.input_dropout, 
                                        cfg.activation,
                                        final_activation=True)
        self.positional = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.core_len + 2*cfg.pad_hours)
        encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=cfg.d_model,
                                        nhead=cfg.num_heads,
                                        dim_feedforward=cfg.ff_dim,
                                        dropout=max(cfg.attn_dropout, cfg.ffn_dropout),
                                        activation=cfg.activation,
                                        batch_first=True,
                                        norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_transformer_layers)
        # Optional learnable depthwise Conv1d aggregation with symmetric conv_padding
        if cfg.aggregation_mode == 'conv' and (cfg.agg_hours > 1 or cfg.conv_padding > 0):
            kernel = cfg.agg_hours + 2 * cfg.conv_padding
            self.conv_agg = nn.Conv1d(
                in_channels=cfg.d_model,
                out_channels=cfg.d_model,
                kernel_size=kernel,
                stride=cfg.agg_hours,
                groups=cfg.d_model,
                bias=True,
                padding=0
            )
            # Kaiming uniform (linear) initialization
            nn.init.kaiming_uniform_(self.conv_agg.weight, a=0.0, nonlinearity='linear')
            if self.conv_agg.bias is not None:
                nn.init.zeros_(self.conv_agg.bias)
        else:
            self.conv_agg = None  # type: ignore
        self.output_mlp = PerTimestepMLP(cfg.d_model, 
                                         cfg.out_features, 
                                         cfg.output_mlp_hidden, 
                                         cfg.output_mlp_layers, 
                                         cfg.output_dropout, 
                                         cfg.activation,
                                         final_activation=False)
        self._init_parameters()

    def _init_parameters(self):  # simple init
        act = self.cfg.activation.lower()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act in ('relu',):
                    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                elif act in ('leaky_relu',):
                    nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')  # adjust slope if you use a different one
                elif act in ('gelu', 'silu'):
                    # Often Xavier works well for smoother activations
                    nn.init.xavier_uniform_(m.weight)
                elif act in ('tanh',):
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_mlp(x)
        h = self.positional(h)
        h = self.transformer(h)
        if self.cfg.aggregation_mode == 'conv':
            pad = int(self.cfg.pad_hours)
            # Safety clamp in forward as well (in case cfg mutated after init)
            conv_pad = int(min(self.cfg.conv_padding, self.cfg.pad_hours))
            agg = int(self.cfg.agg_hours)
            # Symmetric slice including context on both sides: core_len + 2*conv_pad
            start = pad - conv_pad if conv_pad > 0 else pad
            end = pad + self.cfg.core_len + conv_pad if conv_pad > 0 else pad + self.cfg.core_len
            h_slice = h[:, start:end]  # (B, core_len + 2*conv_pad, d_model) or (B, core_len, d_model)
            if (agg > 1) or (conv_pad > 0):
                # Conv expects (B, d_model, T)
                h_conv_in = h_slice.transpose(1, 2)
                h = self.conv_agg(h_conv_in).transpose(1, 2)  # (B, core_len/agg, d_model)
            else:
                h = h_slice  # identity case
        y = self.output_mlp(h)
        return y


# ============================================================
# Custom Losses
# ============================================================
class MAEMaexLoss(nn.Module):
    """MAE normalized by MAEx: mean(|(pred-target)/maex|)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor, maex: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Numerical safety: avoid division by zero and sanitize NaN/Inf
        eps = 1e-8
        denom = torch.clamp(maex.abs(), min=eps)
        nerr = (pred - target) / denom
        nerr = torch.nan_to_num(nerr, nan=0.0, posinf=0.0, neginf=0.0)
        return nerr.abs().mean()


class AlphaPeakLoss(nn.Module):
    """Peak-aware loss combining absolute and squared normalized error.

    The loss uses a per-sample `alpha` weight (typically related to peaks) to
    upweight peak errors.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor, maex: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Numerical safety: avoid division by zero and sanitize alpha
        eps = 1e-8
        denom = torch.clamp(maex.abs(), min=eps)
        nerr = (pred - target) / denom
        nerr = torch.nan_to_num(nerr, nan=0.0, posinf=0.0, neginf=0.0)
        alpha_safe = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        w = torch.maximum(torch.ones_like(alpha_safe), alpha_safe)
        return (w * nerr.abs() + alpha_safe * (nerr ** 2)).mean()


# ============================================================
# Trainer
# ============================================================
class TransformerTrainer:
        """Train/evaluate a `TransformerForecastModel` from a config dictionary.

        The trainer owns:
            - preprocessing and dataset windowing
            - PyTorch model + optimizer/scheduler
            - metric evaluation via `EvaluationMetrics`
            - optional Ray Tune reporting

        Training is performed one epoch at a time via `step()` (Ray-friendly).
        """
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer from configuration (and optionally auto-load data)."""
        # Merge dict into config dataclass
        cfg_kwargs = {k: v for k, v in config.items() if k in TransformerConfig.__dataclass_fields__}
        self.cfg = TransformerConfig(**cfg_kwargs)  # type: ignore[arg-type]
        self.device = torch.device(self.cfg.device)
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_state = None
        # Track epoch number (1-based) at which best_val_loss was observed
        self.best_epoch: Optional[int] = None
        self.history = []
        self.train_loader = None
        self.val_loader = None
        self.evaluator = EvaluationMetrics()
        self.writer = None
        self._set_seed(self.cfg.seed)

        data_cfg = config.get('_data')
        if data_cfg is not None:
            self._auto_load_data(data_cfg)

        self.model = TransformerForecastModel(self.cfg).to(self.device)
        if self.cfg.compile_model and hasattr(torch, 'compile'):  # PyTorch 2.0+
            try:
                self.model = torch.compile(self.model)  # type: ignore
            except Exception:  # pragma: no cover
                pass
        self._setup_optim()
        # Optional TensorBoard logging
        if _TB_AVAILABLE and SummaryWriter is not None:
            log_dir = None
            if _RAY_AVAILABLE and session is not None:
                try:
                    log_dir = session.get_trial_dir()
                except Exception:
                    log_dir = None
            if log_dir is None:
                ts = time.strftime("%Y%m%d-%H%M%S")
                log_dir = os.path.join("runs", f"transformer_{ts}")
            try:
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.writer = None

        # Track a bottom-rung metric that should be monotonic increasing until the
        # bottom_rung epoch is reached, then stay fixed afterward. This allows
        # search algorithms that expect 'val_loss_bottom' to only reflect the
        # "rung" evaluation while still satisfying strict metric checking.
        self._val_loss_bottom: Optional[float] = None


    # ---------------- Explainability: Integrated Gradients -----------------
    def _get_dataset_sample(self, split: str = 'val', sample_idx: int = 0) -> torch.Tensor:
        """Return a single window tensor (1, T, F) from train/val dataset on device.

        split: 'train' | 'val'
        sample_idx: index within the corresponding TensorDataset
        """
        if split not in ('train', 'val'):
            raise ValueError("split must be 'train' or 'val'")
        dataset = self.train_dataset if split == 'train' else self.val_dataset  # type: ignore[attr-defined]
        if dataset is None:
            raise RuntimeError("Datasets are not initialized.")
        item = dataset[sample_idx]
        # TensorDataset returns tuple (X, y[, maex][, alpha])
        if isinstance(item, (tuple, list)):
            xb = item[0]
        else:
            xb = item
        if xb.dim() == 2:
            xb = xb.unsqueeze(0)
        return xb.to(self.device)

    def _forward_aggregated(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that mirrors the training loss aggregation path.

        Returns a tensor with shape:
          - sum mode: (B, core_len/agg_hours, out_features)
          - conv mode: already aggregated inside forward (or identity if agg==1), shape (B, T_out, out_features)
        """
        pred = self.model(x)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')
        if aggregation_mode == 'sum':
            pad = int(self.cfg.pad_hours)
            agg = int(self.cfg.agg_hours)
            if pad > 0:
                pred = pred[:, pad:-pad]
            if agg > 1:
                Bm, Tm, Fm = pred.shape
                if Tm % agg != 0:
                    raise RuntimeError("After trimming, core length not divisible by agg_hours in explainability path.")
                pred = pred.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
        # conv mode is already in the right shape
        return pred

    @torch.no_grad()
    def _get_feature_names(self) -> List[str]:
        try:
            return list(self.preprocessor.feature_names_)  # type: ignore[attr-defined]
        except Exception:
            return [f"f{i}" for i in range(int(self.cfg.in_features))]

    def integrated_gradients(
        self,
        split: str = 'val',
        sample_idx: int = 0,
        target: str = 'both',
        steps: int = 32,
        baseline: Optional[torch.Tensor] = None,
        include_pad: bool = True,
        output_reduce: str = 'sum',  # 'sum' | 'mean'
        abs_attributions: bool = True,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients (IG) attributions for a given sample window.

        - Attributions are computed w.r.t. the model input window (time x features) in scaled space.
        - The scalar objective is constructed from the aggregated output used during training
          (trim pads, then sum by agg_hours in 'sum' mode; conv path already aggregated),
          selecting channel P (index 0) or Q (index 1), and reducing over time by `output_reduce`.

        Returns a dict with keys:
          - if target in {'P','Q'}: {
                'target': 'P'|'Q',
                'attributions': np.ndarray [T, F],
                'feature_names': List[str],
                'times': np.ndarray [T],
                'include_pad': bool,
                'sample_idx': int,
            }
          - if target == 'both': {'P': {...}, 'Q': {...}}
        """
        if steps <= 0:
            raise ValueError("steps must be >= 1")
        # Prepare input window (1, T, F)
        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state, strict=False)
            except Exception:
                pass

        x = self._get_dataset_sample(split=split, sample_idx=sample_idx).detach()
        x = x.requires_grad_(True)
        B, T, F = x.shape
        if B != 1:
            raise RuntimeError("Expected a single sample (B=1) for IG computation.")

        # Baseline
        if baseline is None:
            b = torch.zeros_like(x)
        else:
            b = baseline.to(self.device)
            if b.shape != x.shape:
                raise ValueError("baseline must have the same shape as the input sample (1, T, F)")

        # Alphas for Riemann approximation
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        delta = (x - b)

        def _ig_for_channel(ch_index: int) -> np.ndarray:
            grads_sum = torch.zeros_like(x)
            # Use enable_grad within loop (model.eval above)
            for a in alphas:
                x_i = (b + a * delta).detach().requires_grad_(True)
                y = self._forward_aggregated(x_i)  # (1, T_out, C)
                if ch_index >= y.shape[-1]:
                    raise ValueError(f"Requested channel index {ch_index} not available in model output with shape {tuple(y.shape)}")
                y_sel = y[..., ch_index]  # (1, T_out)
                if output_reduce == 'sum':
                    y_scalar = y_sel.sum()
                elif output_reduce == 'mean':
                    y_scalar = y_sel.mean()
                else:
                    raise ValueError("output_reduce must be 'sum' or 'mean'")
                (grad_x,) = torch.autograd.grad(y_scalar, x_i, retain_graph=False, create_graph=False)
                # Accumulate
                grads_sum = grads_sum + grad_x
            ig = delta * (grads_sum / float(steps))
            ig = ig.detach()[0]  # (T, F)
            if not include_pad:
                pad = int(self.cfg.pad_hours)
                if pad > 0:
                    ig = ig[pad:-pad]
            if abs_attributions:
                ig = ig.abs()
            ig_np = ig.cpu().numpy()
            if normalize:
                # Scale by max absolute to [0,1] to aid visualization (avoid divide-by-zero)
                mx = np.nanmax(np.abs(ig_np)) if ig_np.size else 0.0
                if mx > 0:
                    ig_np = ig_np / mx
            return ig_np

        feature_names = self._get_feature_names()
        if not include_pad:
            t_len = int(self.cfg.core_len)
        else:
            t_len = int(self.cfg.core_len + 2 * self.cfg.pad_hours)
        times = np.arange(t_len, dtype=int)

        def _pack(name: str, arr: np.ndarray) -> Dict[str, Any]:
            return {
                'target': name,
                'attributions': arr,
                'feature_names': feature_names,
                'times': times,
                'include_pad': include_pad,
                'sample_idx': sample_idx,
            }

        target_lower = str(target).lower()
        if target_lower == 'p':
            return _pack('P', _ig_for_channel(0))
        if target_lower == 'q':
            return _pack('Q', _ig_for_channel(1))
        if target_lower == 'both':
            return {
                'P': _pack('P', _ig_for_channel(0)),
                'Q': _pack('Q', _ig_for_channel(1)),
            }
        raise ValueError("target must be one of {'P','Q','both'}")

    def plot_integrated_gradients(
        self,
        ig_result: Dict[str, Any],
        top_k_features: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = 'coolwarm',
    ):
        """Plot IG heatmaps (x=time, y=features). Accepts a single- or dual-target result from integrated_gradients().

        - If ig_result contains keys 'P' and 'Q', two subplots are drawn side-by-side.
        - top_k_features: if provided, select features with largest overall attribution (time-sum) to plot.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"Matplotlib is required for plotting IG heatmaps: {e}")

        def _plot_single(ax, payload: Dict[str, Any], title_suffix: str = ""):
            A = payload['attributions']  # (T, F)
            fnames: List[str] = payload['feature_names']
            # Select top-k features by total attribution
            if top_k_features is not None and top_k_features > 0 and top_k_features < A.shape[1]:
                scores = np.sum(np.abs(A), axis=0)
                idx = np.argsort(scores)[-top_k_features:][::-1]
                A = A[:, idx]
                ylabels = [fnames[i] for i in idx]
            else:
                ylabels = fnames
            # Transpose to (features, time) for plotting with y=features
            img = ax.imshow(A.T, aspect='auto', interpolation='nearest', cmap=cmap)
            ax.set_xlabel('t')
            ax.set_ylabel('features')
            try:
                ax.set_yticks(range(len(ylabels)))
                ax.set_yticklabels(ylabels, fontsize=8)
            except Exception:
                pass
            ax.set_title(f"IG heatmap {title_suffix}")
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        # Determine single vs both
        if 'P' in ig_result and 'Q' in ig_result:
            if figsize is None:
                figsize = (12, 6)
            fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
            _plot_single(axes[0], ig_result['P'], title_suffix='(P)')
            _plot_single(axes[1], ig_result['Q'], title_suffix='(Q)')
            try:
                plt.show()
            except Exception:
                pass
            return fig, axes
        else:
            if figsize is None:
                figsize = (7, 5)
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            _plot_single(ax, ig_result, title_suffix=f"({ig_result.get('target','')})")
            try:
                plt.show()
            except Exception:
                pass
            return fig, ax


    # ---------------- Seed -----------------
    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    # ---------------- Optim -----------------
    def _setup_optim(self):
        ### optimizer
        if self.cfg.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer {self.cfg.optimizer}")
        ### scheduler
        if self.cfg.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.scheduler_epochs)
        elif self.cfg.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.cfg.patience, factor=0.5)
        else:
            self.scheduler = None

    # ---------------- Data Loading -----------------
    def _auto_load_data(self, data_cfg: Dict[str, Any]):
        ### Settings
        hdf_data_path = data_cfg["hdf_data_path"]
        key_X = data_cfg["key_X"]
        key_y = data_cfg["key_y"]
        train_grids = data_cfg["train_grids"]
        test_ratio = float(data_cfg["test_ratio"])
        random_state = int(data_cfg["random_state"])
        agg = int(self.cfg.agg_hours)

        if agg < 1: raise ValueError("agg_hours must be >=1")
        if self.cfg.core_len % agg != 0: raise ValueError(f"core_len ({self.cfg.core_len}) must be divisible by agg_hours ({agg})")
        # pad_hours can be arbitrary since we trim before aggregation; still ensure non-negative
        if self.cfg.pad_hours < 0: raise ValueError("pad_hours must be >=0")

        ### Data file Readout
        if hdf_data_path is None: raise ValueError("_data.hdf_data_path required")
        df_X = pd.read_hdf(hdf_data_path, key=key_X).astype('float32')
        df_y = pd.read_hdf(hdf_data_path, key=key_y).astype('float32')

        ### Train/Val split by grid id
        unique_grids = df_X.index.get_level_values(0).unique().tolist()
        if len(unique_grids) < 2:
            raise ValueError("Need at least 2 grids for train/val split")
        rng = random.Random(random_state)
        rng.shuffle(unique_grids)
        n_val = max(1, int(len(unique_grids) * test_ratio))
        val_grids = set(unique_grids[:n_val])
        train_grids_all = set(unique_grids[n_val:])

        if train_grids not in (None, 'all'):
            if isinstance(train_grids, int):
                train_grids_all = set(list(train_grids_all)[:train_grids])
            elif isinstance(train_grids, (list, tuple, set)):
                train_grids_all = set(train_grids_all).intersection(set(train_grids))

        train_mask = df_X.index.get_level_values(0).isin(train_grids_all)
        val_mask = df_X.index.get_level_values(0).isin(val_grids)
        X_train_raw, y_train_raw = df_X[train_mask], df_y[train_mask]
        X_val_raw, y_val_raw = df_X[val_mask], df_y[val_mask]


        ### Data Preprocessing (feature engineering + scaling)
        self.preprocessor = Preprocessor(data_cfg)
        # Fit scaler on aggregated targets
        X_train, y_train = self.preprocessor.fit_transform(X_train_raw, y_train_raw, agg)
        X_val, y_val = self.preprocessor.transform(X_val_raw, y_val_raw, agg, type="val")

        print(X_train.columns)
        print(y_train.columns)

        ### Precompute loss function helpers 
        target_cols = list(self.preprocessor.TARGET_COLS)
        single_target = len(target_cols) == 1
        target_col = target_cols[0]
        loss_type = self.cfg.loss_type.lower()
        extras_train: Dict[str, pd.Series] = {}
        extras_val: Dict[str, pd.Series] = {}
        if loss_type in ('mae_maex', 'alpha_peak'):
            if single_target:
                # maex per grid: mean absolute value of target in training set (to normalize grid demand scale)
                maex_train_by_grid = y_train.groupby(level=0)[target_col].apply(lambda s: s.abs().mean())
                maex_val_by_grid = y_val.groupby(level=0)[target_col].apply(lambda s: s.abs().mean())
                maex_train = y_train.index.get_level_values(0).map(maex_train_by_grid).astype('float32')
                maex_val = y_val.index.get_level_values(0).map(maex_val_by_grid).astype('float32')
                extras_train['maex'] = pd.Series(maex_train.values, index=y_train.index)
                extras_val['maex'] = pd.Series(maex_val.values, index=y_val.index)
                if loss_type == 'alpha_peak': # increase loss on peaks
                    stats_train = y_train.groupby(level=0)[target_col].agg(['mean', 'std'])
                    stats_val = y_val.groupby(level=0)[target_col].agg(['mean', 'std'])
                    def _alpha(df_scaled: pd.DataFrame, stats: pd.DataFrame):
                        grid = df_scaled.index.get_level_values(0)
                        mu = grid.map(stats['mean']).values
                        sigma = grid.map(stats['std']).values
                        sigma = np.where(sigma == 0, 1.0, sigma)
                        return (np.abs(df_scaled[target_col].values - mu) / sigma).astype('float32')
                    alpha_train = _alpha(y_train, stats_train)
                    alpha_val = _alpha(y_val, stats_val)
                    alpha_train = np.nan_to_num(alpha_train, nan=0.0, posinf=0.0, neginf=0.0)
                    alpha_val = np.nan_to_num(alpha_val, nan=0.0, posinf=0.0, neginf=0.0)
                    extras_train['alpha'] = pd.Series(alpha_train, index=y_train.index)
                    extras_val['alpha'] = pd.Series(alpha_val, index=y_val.index)
            else:
                # Multi-target: build P,Q,S helpers on scaled space
                p_col, q_col = target_cols[0], target_cols[1]
                S_train = np.sqrt(y_train[p_col].values**2 + y_train[q_col].values**2).astype('float32')
                S_val = np.sqrt(y_val[p_col].values**2 + y_val[q_col].values**2).astype('float32')
                S_train_series = pd.Series(S_train, index=y_train.index, name='S')
                S_val_series = pd.Series(S_val, index=y_val.index, name='S')

                # MAEX per grid
                maex_P_train_g = y_train.groupby(level=0)[p_col].apply(lambda s: s.abs().mean())
                maex_Q_train_g = y_train.groupby(level=0)[q_col].apply(lambda s: s.abs().mean())
                maex_S_train_g = S_train_series.groupby(level=0).apply(lambda s: s.abs().mean())
                maex_P_val_g = y_val.groupby(level=0)[p_col].apply(lambda s: s.abs().mean())
                maex_Q_val_g = y_val.groupby(level=0)[q_col].apply(lambda s: s.abs().mean())
                maex_S_val_g = S_val_series.groupby(level=0).apply(lambda s: s.abs().mean())

                g_train = y_train.index.get_level_values(0)
                g_val = y_val.index.get_level_values(0)
                maex_train_mat = np.stack([
                    g_train.map(maex_P_train_g).values,
                    g_train.map(maex_Q_train_g).values,
                    g_train.map(maex_S_train_g).values
                ], axis=1).astype('float32')
                maex_val_mat = np.stack([
                    g_val.map(maex_P_val_g).values,
                    g_val.map(maex_Q_val_g).values,
                    g_val.map(maex_S_val_g).values
                ], axis=1).astype('float32')
                extras_train['maex'] = pd.DataFrame(maex_train_mat, index=y_train.index, columns=['P','Q','S'])
                extras_val['maex'] = pd.DataFrame(maex_val_mat, index=y_val.index, columns=['P','Q','S'])

                if loss_type == 'alpha_peak':
                    # Compute per-grid mean/std for P,Q,S
                    stats_P_tr = y_train.groupby(level=0)[p_col].agg(['mean', 'std'])
                    stats_Q_tr = y_train.groupby(level=0)[q_col].agg(['mean', 'std'])
                    stats_S_tr = S_train_series.groupby(level=0).agg(['mean', 'std'])
                    stats_P_va = y_val.groupby(level=0)[p_col].agg(['mean', 'std'])
                    stats_Q_va = y_val.groupby(level=0)[q_col].agg(['mean', 'std'])
                    stats_S_va = S_val_series.groupby(level=0).agg(['mean', 'std'])

                    def _alpha_triple(df_scaled: pd.DataFrame, S_series: pd.Series, statsP: pd.DataFrame, statsQ: pd.DataFrame, statsS: pd.DataFrame):
                        grid = df_scaled.index.get_level_values(0)
                        muP = grid.map(statsP['mean']).values; sdP = grid.map(statsP['std']).values
                        muQ = grid.map(statsQ['mean']).values; sdQ = grid.map(statsQ['std']).values
                        muS = grid.map(statsS['mean']).values; sdS = grid.map(statsS['std']).values
                        sdP = np.where(sdP == 0, 1.0, sdP)
                        sdQ = np.where(sdQ == 0, 1.0, sdQ)
                        sdS = np.where(sdS == 0, 1.0, sdS)
                        aP = np.abs(df_scaled[p_col].values - muP) / sdP
                        aQ = np.abs(df_scaled[q_col].values - muQ) / sdQ
                        aS = np.abs(S_series.values - muS) / sdS
                        out = np.stack([aP, aQ, aS], axis=1).astype('float32')
                        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                    alpha_train_mat = _alpha_triple(y_train, S_train_series, stats_P_tr, stats_Q_tr, stats_S_tr)
                    alpha_val_mat = _alpha_triple(y_val, S_val_series, stats_P_va, stats_Q_va, stats_S_va)
                    extras_train['alpha'] = pd.DataFrame(alpha_train_mat, index=y_train.index, columns=['P','Q','S'])
                    extras_val['alpha'] = pd.DataFrame(alpha_val_mat, index=y_val.index, columns=['P','Q','S'])

        ### Transform data to torch tensors and build datasets/loaders
        (X_train_t, y_train_t, train_extras_t) = self._build_sequence_tensors(X_train, y_train, extras_train)
        (X_val_t, y_val_t, val_extras_t) = self._build_sequence_tensors(X_val, y_val, extras_val)

        if loss_type in ('mae_maex', 'alpha_peak'):
            if 'alpha' in train_extras_t:
                self.train_dataset = TensorDataset(X_train_t, y_train_t, train_extras_t['maex'], train_extras_t['alpha'])
                self.val_dataset = TensorDataset(X_val_t, y_val_t, val_extras_t['maex'], val_extras_t['alpha'])
            else:
                self.train_dataset = TensorDataset(X_train_t, y_train_t, train_extras_t['maex'])
                self.val_dataset = TensorDataset(X_val_t, y_val_t, val_extras_t['maex'])
        else:
            self.train_dataset = TensorDataset(X_train_t, y_train_t)
            self.val_dataset = TensorDataset(X_val_t, y_val_t)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=False)
        
        self.cfg.in_features = X_train_t.shape[-1]
        self.cfg.out_features = int(y_train_t.shape[-1])

    # ---------------- Sequence windows -----------------
    def _build_sequence_tensors(self, X: pd.DataFrame, y: pd.DataFrame, extras: Dict[str, pd.Series]):
        # Settings
        core_len = int(self.cfg.core_len)   # in hours
        pad = int(self.cfg.pad_hours)       # in hours
        agg = int(self.cfg.agg_hours)       # in hours
        core_len_agg = core_len // agg

        # Collect per-grid batches as numpy arrays, convert to tensors once
        X_batches: List[np.ndarray] = []
        Y_batches: List[np.ndarray] = []  # aggregated target windows (no pad)
        extras_batches: Dict[str, List[np.ndarray]] = {k: [] for k in extras.keys()}

        for grid, Xg in X.groupby(level=0):
            # Check length of grid divisible by agg_hours
            yg = y.loc[grid]
            n_hours = len(Xg)
            if n_hours % agg != 0: 
                raise ValueError(f"Data length for grid {grid} not divisible by agg_hours {agg}")
            n_agg = n_hours // agg
            if len(yg) != n_agg: 
                raise ValueError(f"Grid {grid} has inconsistent number of aggregated target buckets in target and features")

            ### Build index windows serving as batches (idea: sliding windows of len core_len, stride core_len and with padding pad on each side)
            # Start positions on hour resolution (same logic as before)
            starts_hours = np.arange(0, n_hours - core_len + 1, core_len, dtype=int)

            # Build X indices with padding (circular)
            base_hours = np.arange(-pad, core_len + pad, dtype=int)
            idx_X = (starts_hours[:, None] + base_hours[None, :]) % n_hours

            # For aggregated targets, compute starting aggregated bucket index
            starts_agg = (starts_hours // agg).astype(int)
            base_agg = np.arange(0, core_len_agg, dtype=int)
            idx_y = (starts_agg[:, None] + base_agg[None, :]) % n_agg  # (num_windows, core_len_agg)

            X_arr = Xg.to_numpy().astype(np.float32, copy=False)
            y_arr = yg.to_numpy().astype(np.float32, copy=False)
            X_batches.append(X_arr[idx_X])  # (batch, seq_len_hours, F_in)
            Y_batches.append(y_arr[idx_y])  # (batch, core_len_agg, F_out)

            for k, series in extras.items():
                eg = series.loc[grid]
                e_arr = eg.to_numpy().astype(np.float32, copy=False)
                extras_batches[k].append(e_arr[idx_y])  # aggregated (batch, core_len_agg)

        # Concatenate all grids and convert to tensors
        X_np = np.concatenate(X_batches, axis=0)
        y_np = np.concatenate(Y_batches, axis=0)
        X_t = torch.from_numpy(X_np)
        y_t = torch.from_numpy(y_np)

        extras_out: Dict[str, torch.Tensor] = {}
        for k, parts in extras_batches.items():
            if parts:
                e_np = np.concatenate(parts, axis=0)
                # If extras are 2D (B,T), add feature dim; if 3D (B,T,C), keep as is
                if e_np.ndim == 2:
                    e_np = e_np[..., None]
                extras_out[k] = torch.from_numpy(e_np.astype(np.float32, copy=False))
        return X_t, y_t, extras_out

    # ---------------- Loss -----------------
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, batch_extras: Optional[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        # pred: (B, seq_len_hours, out_features)
        # target: (B, core_len_agg, out_features) if agg_hours > 1 else (B, core_len, out_features)
        pad = int(self.cfg.pad_hours)
        agg = int(self.cfg.agg_hours)
        loss_type = self.cfg.loss_type.lower()
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')

        if aggregation_mode == 'sum':
            # Original behavior: trim pad then sum aggregates
            if pad > 0:
                pred = pred[:, pad:-pad]
            if agg > 1:
                B, T, F = pred.shape
                if T % agg != 0:
                    raise RuntimeError("After trimming, core length not divisible by agg_hours")
                pred = pred.reshape(B, T // agg, agg, F).sum(dim=2)
        else:
            # conv mode already produced aggregated sequence (or same length if agg_hours == 1)
            pass
        if pred.shape != target.shape:
            raise RuntimeError(f"Prediction/target shape mismatch after aggregation path: pred={pred.shape} target={target.shape}")

        ### Compute loss
        F = pred.shape[-1]
        if F == 1:
            if loss_type == 'mae_maex':
                assert batch_extras is not None and len(batch_extras) >= 1
                maex = batch_extras[0]
                return MAEMaexLoss()(pred, target, maex)
            if loss_type == 'alpha_peak':
                assert batch_extras is not None and len(batch_extras) == 2
                maex, alpha = batch_extras[0], batch_extras[1]
                return AlphaPeakLoss()(pred, target, maex, alpha)
        elif F == 2:
            lam = float(getattr(self.cfg, 'multi_target_lambda', 0.5))
            Pp, Qp = pred[..., 0], pred[..., 1]
            Pt, Qt = target[..., 0], target[..., 1]
            Sp = torch.sqrt(torch.clamp(Pp**2 + Qp**2, min=0.0))
            St = torch.sqrt(torch.clamp(Pt**2 + Qt**2, min=0.0))
            assert batch_extras is not None and len(batch_extras) >= 1
            maex = batch_extras[0]
            # maex expected shape (B,T,3) with order [P,Q,S]; fallback if single channel
            if maex.shape[-1] == 1:
                mP = mQ = mS = maex[..., 0]
            else:
                mP, mQ, mS = maex[..., 0], maex[..., 1], maex[..., 2]
            eps = 1e-12
            if loss_type == 'mae_maex':
                termP = ((Pp - Pt) / (mP + eps)).abs().mean()
                termQ = ((Qp - Qt) / (mQ + eps)).abs().mean()
                termS = ((Sp - St) / (mS + eps)).abs().mean()
                return lam * 0.5 * (termP + termQ) + (1.0 - lam) * termS
            if loss_type == 'alpha_peak':
                assert len(batch_extras) == 2
                alpha = batch_extras[1]
                if alpha.shape[-1] == 1:
                    aP = aQ = aS = alpha[..., 0]
                else:
                    aP, aQ, aS = alpha[..., 0], alpha[..., 1], alpha[..., 2]
                one = 1.0
                wP = torch.maximum(torch.ones_like(aP), aP)
                wQ = torch.maximum(torch.ones_like(aQ), aQ)
                wS = torch.maximum(torch.ones_like(aS), aS)
                nerrP = (Pp - Pt) / (mP + eps)
                nerrQ = (Qp - Qt) / (mQ + eps)
                nerrS = (Sp - St) / (mS + eps)
                lossP = (wP * nerrP.abs() + aP * (nerrP ** 2)).mean()
                lossQ = (wQ * nerrQ.abs() + aQ * (nerrQ ** 2)).mean()
                lossS = (wS * nerrS.abs() + aS * (nerrS ** 2)).mean()
                return lam * 0.5 * (lossP + lossQ) + (1.0 - lam) * lossS
        raise ValueError(f"Unknown loss_type {self.cfg.loss_type}")

    # ---------------- Epoch -----------------
    def _run_epoch(self, train: bool, collect_for_metrics: bool = False):
        ### Settings
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        pad = int(self.cfg.pad_hours)
        agg = int(self.cfg.agg_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')

        ### Intermediate storages
        simple_preds: List[np.ndarray] = []
        simple_trues: List[np.ndarray] = []
        total = 0.0
        count = 0

        for batch in loader:
            ### Load data from loader
            if self.cfg.loss_type.lower() in ('mae_maex', 'alpha_peak'):
                if self.cfg.loss_type.lower() == 'alpha_peak':
                    xb, yb, maex, alpha = batch
                    extras = (maex.to(self.device), alpha.to(self.device))
                else:
                    xb, yb, maex = batch
                    extras = (maex.to(self.device),)
            else:
                xb, yb = batch
                extras = None
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            with torch.set_grad_enabled(train):
                ### Forward pass
                pred = self.model(xb)
                loss = self._compute_loss(pred, yb, extras)

                ### If all metrics due, collect full validation series reconstruction data
                if (not train) and collect_for_metrics:
                    t = yb.detach()
                    p = pred.detach()
                    if aggregation_mode == 'sum':
                        if pad > 0:
                            p = p[:, pad:-pad]
                        if agg > 1:
                            Bm, Tm, Fm = p.shape
                            p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                    # conv mode already aggregated inside forward
                    B2, T2, Fp = p.shape
                    Ft = t.shape[-1]
                    simple_preds.append(p.cpu().numpy().reshape(-1, Fp))
                    simple_trues.append(t.cpu().numpy().reshape(-1, Ft))

                ### Backward pass
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

            ### Collect stats
            total += loss.detach().item() * yb.size(0) * yb.size(1)
            count += yb.size(0) * yb.size(1)
        avg = total / max(count, 1)

        ### Scheduler step
        if (not train) and self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg)
        if train and self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        ### If all metrics due, reconstruct full series and inverse transform to original scale
        if (not train) and collect_for_metrics:
            # Concatenate validation data and batch into grids of size 8760 hours
            target_cols = list(self.preprocessor.TARGET_COLS)
            if simple_preds and simple_trues:
                pred_arr = np.concatenate(simple_preds, axis=0)
                true_arr = np.concatenate(simple_trues, axis=0)
                N = pred_arr.shape[0]
                base_period = 8760 // agg if agg > 1 else 8760
                hours = np.arange(N, dtype=int) % base_period
                batches = np.arange(N, dtype=int) // base_period
                index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
                cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
                y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
                y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)
            else:
                # Nothing collected; return empty frames
                y_true_scaled = pd.DataFrame(columns=target_cols)
                y_pred_scaled = pd.DataFrame(columns=target_cols)

            y_true = self.preprocessor.inverse_transform_y(y_true_scaled)
            y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)
            return avg, y_true, y_pred
        return avg

    # ---------------- Public API -----------------
    def train_one_epoch(self) -> Dict[str, float]:
        ### Run one training epoch + one validation epoch
        collect_full = bool(self.cfg.full_metrics_every) and ((self.epoch + 1) % max(1, int(self.cfg.full_metrics_every)) == 0)
        train_loss = self._run_epoch(train=True)
        val_epoch_out = self._run_epoch(train=False, collect_for_metrics=collect_full)

        ### Extract metrics
        if isinstance(val_epoch_out, tuple):
            val_loss, y_true_orig, y_pred_orig = val_epoch_out
        else:
            val_loss = val_epoch_out
            y_true_orig = None
            y_pred_orig = None

        ### Process metrics
        if val_loss < self.best_val_loss:
            # Note: current epoch is (self.epoch + 1) because we increment below
            next_epoch = self.epoch + 1
            self.best_val_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_epoch = next_epoch
        self.epoch += 1
        metrics = {
            'epoch': self.epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        # Report 'val_loss_bottom' as a monotonic (non-increasing) series until
        # hitting the configured bottom_rung epoch (running minimum), then freeze
        # it and keep reporting the fixed value thereafter.
        grace = int(getattr(self.cfg, 'bottom_rung_report', 0) or 0)
        if grace > 0:
            if grace > self.epoch:
                self._val_loss_bottom_fixed = self.best_val_loss
            metrics['val_loss_bottom'] = self._val_loss_bottom_fixed 
        self.history.append(metrics)

        ### TensorBoard logging
        if _TB_AVAILABLE and self.writer:
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
            if 'val_loss_bottom' in metrics:
                self.writer.add_scalar('Loss/val_bottom', metrics['val_loss_bottom'], self.epoch)
            try:
                self.writer.flush()
            except Exception:
                pass
        if collect_full and (y_true_orig is not None) and (y_pred_orig is not None):
            # Handle single or multi-target
            if y_true_orig.shape[1] == 1:
                full = self.evaluator.evaluate(y_true_orig, y_pred_orig, base_agg_hours=int(self.cfg.agg_hours))
                for k, v in full.items():
                    metrics[f"val_{k}"] = float(v)
                    if _TB_AVAILABLE and self.writer:
                        self.writer.add_scalar(f"ValMetrics/{k}", metrics[f"val_{k}"] , self.epoch)
            elif y_true_orig.shape[1] == 2:
                p_col, q_col = list(y_true_orig.columns)
                # Build S
                S_true = pd.DataFrame(
                    np.sqrt(y_true_orig[p_col].values**2 + y_true_orig[q_col].values**2),
                    index=y_true_orig.index, columns=['S']
                )
                S_pred = pd.DataFrame(
                    np.sqrt(y_pred_orig[p_col].values**2 + y_pred_orig[q_col].values**2),
                    index=y_pred_orig.index, columns=['S']
                )
                for name, yt, yp in (
                    ("P", y_true_orig[[p_col]].rename(columns={p_col: 'P'}), y_pred_orig[[p_col]].rename(columns={p_col: 'P'})),
                    ("Q", y_true_orig[[q_col]].rename(columns={q_col: 'Q'}), y_pred_orig[[q_col]].rename(columns={q_col: 'Q'})),
                    ("S", S_true, S_pred),
                ):
                    full = self.evaluator.evaluate(yt, yp, base_agg_hours=int(self.cfg.agg_hours), skip_feedin=(name=="S"))
                    for k, v in full.items():
                        metrics[f"val_{name}_{k}"] = float(v)
                        if _TB_AVAILABLE and self.writer:
                            self.writer.add_scalar(f"ValMetrics/{name}/{k}", metrics[f"val_{name}_{k}"] , self.epoch)
        return metrics

    def step(self) -> Dict[str, float]:  # Ray Tune compatibility
        metrics = self.train_one_epoch()
        # Prefer tune.report when running under Ray Tune to avoid deprecation warning
        try:
            from ray import tune  # type: ignore
            tune.report(**metrics)
        except Exception:
            if _RAY_AVAILABLE and session is not None:
                try:
                    session.report(metrics)
                except Exception:
                    pass
        return metrics

    def fit(self) -> Dict[str, float]:
        start = time.time()
        for _ in range(self.cfg.epochs):
            metrics = self.train_one_epoch()
        return {
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epoch,
            'train_time_sec': time.time() - start,
        }

    def predict(self, X_unscaled, use_best: bool = True, type: str = "test"):
        """Predict on unscaled feature data and return y in original units.

        Expected input format:
          - A pandas DataFrame of features with a MultiIndex (batch, hour) on rows,
            identical to the raw/original feature layout used for training.
            Columns must match (or be a superset of) the training feature columns
            expected by the Preprocessor.

        Behavior:
          - Applies the same preprocessing/scaling to X as during training
          - Builds sliding windows (core_len with pad_hours circular padding) per batch
          - Runs the model forward pass on each window
          - Applies the same aggregation behavior as in training (sum mode trims pad
            then sums over agg_hours; conv mode is already aggregated in forward)
          - Reassembles window predictions back into a continuous time series per batch
          - Inverse-transforms predictions to original target scale and returns a DataFrame

        Returns:
          - pandas DataFrame with MultiIndex (batch, hour) containing predictions in
            original target units. If agg_hours > 1, "hour" refers to aggregated-hour
            buckets (size = agg_hours) and length will be n_hours/agg per batch.
        """
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor not initialized. Ensure the trainer was created with _data and trained/fitted before calling predict().")

        import pandas as pd  # local import to avoid issues if not used elsewhere
        import numpy as np

        # Normalize input to a pandas DataFrame with MultiIndex (batch, hour)
        if isinstance(X_unscaled, pd.DataFrame):
            df_X_raw = X_unscaled.copy()
            if not isinstance(df_X_raw.index, pd.MultiIndex) or df_X_raw.index.nlevels < 2:
                raise ValueError("X_unscaled must have a MultiIndex with at least two levels: (batch, hour)")
            # Use the first two index levels as (batch, hour)
            if df_X_raw.index.nlevels > 2:
                # Collapse any extra levels beyond the first two to keep structure stable
                df_X_raw.index = pd.MultiIndex.from_arrays([
                    df_X_raw.index.get_level_values(0),
                    df_X_raw.index.get_level_values(1)
                ], names=['batch', 'hour'])
            else:
                # Ensure consistent names
                df_X_raw.index = pd.MultiIndex.from_arrays([
                    df_X_raw.index.get_level_values(0),
                    df_X_raw.index.get_level_values(1)
                ], names=['batch', 'hour'])
        else:
            # Try to coerce array-like of shape (B, T, F) or (B, T)
            arr = np.asarray(X_unscaled)
            if arr.ndim == 2:
                arr = arr[:, :, None]  # (B, T) -> (B, T, 1)
            if arr.ndim != 3:
                raise ValueError("X_unscaled must be a pandas DataFrame with MultiIndex (batch,hour) or an array-like of shape (B, T[, F]).")
            B, T, F = arr.shape
            batches = np.repeat(np.arange(B), T)
            hours = np.tile(np.arange(T), B)
            df_X_raw = pd.DataFrame(
                arr.reshape(B * T, F),
                index=pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
            )

        # Model state
        self.model.eval()
        if use_best and self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)

        # Config shortcuts
        core_len = int(self.cfg.core_len)
        pad = int(self.cfg.pad_hours)
        agg = int(self.cfg.agg_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')

        # Validate lengths w.r.t. aggregation
        # (We follow training behavior: require each batch length divisible by agg_hours; any tail < core_len is ignored.)
        def _validate_lengths(df_grouped):
            for bid, Xg in df_grouped:
                n_hours = len(Xg)
                if n_hours % agg != 0:
                    raise ValueError(f"Batch {bid} has length {n_hours} not divisible by agg_hours={agg}")

        # Preprocess (scale/feature-engineer) using fitted preprocessor
        df_X_scaled = self.preprocessor.transform(df_X_raw, df_y=None, aggregation=agg, type=type)  # type: ignore[arg-type]
        grouped = df_X_scaled.groupby(level=0)  # by batch id
        _validate_lengths(grouped)

        preds_scaled_parts = []
        index_parts = []

        with torch.no_grad():
            for bid, Xg in grouped:
                X_arr = Xg.to_numpy(dtype=np.float32, copy=False)
                n_hours = X_arr.shape[0]

                # Build sliding window start positions and indices (wrap-around padding)
                starts_hours = np.arange(0, max(n_hours - core_len + 1, 0), core_len, dtype=int)
                if len(starts_hours) == 0:
                    # If the series is shorter than core_len, create a single window starting at 0
                    starts_hours = np.array([0], dtype=int)
                base_hours = np.arange(-pad, core_len + pad, dtype=int)
                idx_X = (starts_hours[:, None] + base_hours[None, :]) % n_hours  # (num_windows, seq_len_hours)

                X_win = torch.from_numpy(X_arr[idx_X]).to(self.device)  # (W, core_len+2*pad, F)
                pred = self.model(X_win)  # (W, seq_len_out, out_features)

                # Apply aggregation identical to training loss path for "sum" mode
                if aggregation_mode == 'sum':
                    p = pred
                    if pad > 0:
                        p = p[:, pad:-pad]
                    if agg > 1:
                        Bm, Tm, Fm = p.shape
                        if Tm % agg != 0:
                            raise RuntimeError("After trimming, core length not divisible by agg_hours in predict().")
                        p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                else:
                    p = pred  # conv mode already aggregated (or identity if agg==1)

                p_np = p.detach().cpu().numpy()
                # Reassemble windows by simple concatenation (stride = core_len)
                # Total length becomes n_hours/agg if agg>1 else n_hours
                p_cat = p_np.reshape(-1, p_np.shape[-1])  # (W*T_out, F_out)
                n_out = n_hours // agg if agg > 1 else n_hours
                if p_cat.shape[0] != n_out:
                    # If the last partial chunk is dropped due to non-multiple-of-core_len, trim to min length
                    min_len = min(p_cat.shape[0], n_out)
                    p_cat = p_cat[:min_len]
                    n_out = min_len

                # Build index for this batch
                idx = pd.MultiIndex.from_arrays([
                    np.full(n_out, bid),
                    np.arange(n_out, dtype=int)
                ], names=['batch', 'hour'])

                preds_scaled_parts.append(p_cat)
                index_parts.append(idx)

        if not preds_scaled_parts:
            # No predictions produced; return empty DataFrame
            return pd.DataFrame(columns=list(self.preprocessor.TARGET_COLS))  # type: ignore[attr-defined]

        preds_scaled = np.concatenate(preds_scaled_parts, axis=0)
        if len(index_parts) > 1:
            index_full = index_parts[0]
            for idx in index_parts[1:]:
                index_full = index_full.append(idx)
        else:
            index_full = index_parts[0]

        # Determine target columns (single or multi-target)
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]
        if preds_scaled.shape[1] == len(target_cols):
            cols = target_cols
        else:
            # Fallback for single-target
            cols = [target_cols[0]]

        y_pred_scaled_df = pd.DataFrame(preds_scaled, index=index_full, columns=cols)
        # Inverse-transform to original scale
        y_pred_orig = self.preprocessor.inverse_transform_y(y_pred_scaled_df)  # type: ignore[attr-defined]
        return y_pred_orig

    def save(self, path: str, use_best: bool = True):
        state = self.best_state if (use_best and self.best_state is not None) else self.model.state_dict()
        torch.save({'config': self.cfg.__dict__, 'state_dict': state}, path)

    @classmethod
    def load(cls, path: str) -> 'TransformerTrainer':
        payload = torch.load(path, map_location='cpu')
        trainer = cls(payload['config'])
        trainer.model.load_state_dict(payload['state_dict'], strict=False)
        trainer.best_state = payload['state_dict']
        trainer.best_val_loss = float('inf')
        return trainer

    # ---------------- Evaluation helper -----------------
    def evaluate_best_with_plots(self) -> Dict[str, Any]:
        """
        Run inference with the best model on both training and validation data,
        convert predictions back to original scale, and call
        EvaluationMetrics.run_model_evaluation with plotting enabled.

        Returns a dict containing metrics and the involved DataFrames.
        """
        ### Setup
        if self.val_loader is None:
            raise RuntimeError("Data loaders not initialized; provide _data config and call after setup.")
        # Build non-shuffling eval loader for train data
        eval_train_loader = DataLoader(
            self.train_dataset,  # type: ignore[attr-defined]
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False)
        # Ensure eval mode and load best state
        self.model.eval()
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)
        agg = int(self.cfg.agg_hours)
        pad = int(self.cfg.pad_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]


        ### Inference helper
        def _infer_to_df(loader: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Run model on a loader, trim pads, flatten across all sequences, then build pseudo-batches."""
            flat_preds: List[np.ndarray] = []
            flat_trues: List[np.ndarray] = []

            with torch.no_grad():
                ### Load data and predict
                for batch in loader:
                    if self.cfg.loss_type.lower() in ('mae_maex', 'alpha_peak'):
                        if self.cfg.loss_type.lower() == 'alpha_peak':
                            xb, yb, maex, alpha = batch
                        else:
                            xb, yb, maex = batch
                    else:
                        xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)

                    
                    p = pred.detach()
                    t = yb.detach()

                    if aggregation_mode == 'sum':
                        if pad > 0:
                            p = p[:, pad:-pad]
                        if agg > 1:
                            Bm, Tm, Fm = p.shape
                            p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                    B, T, Fp = p.shape
                    Ft = t.shape[-1]
                    flat_preds.append(p.detach().cpu().numpy().reshape(-1, Fp))
                    flat_trues.append(t.detach().cpu().numpy().reshape(-1, Ft))

            pred_arr = np.concatenate(flat_preds, axis=0)
            true_arr = np.concatenate(flat_trues, axis=0)
            N = pred_arr.shape[0]
            base_period = 8760 // agg if agg > 1 else 8760
            hours = np.arange(N, dtype=int) % base_period
            batches = np.arange(N, dtype=int) // base_period
            index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
            cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
            y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
            y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)

            # Inverse-transform to original units
            y_true = self.preprocessor.inverse_transform_y(y_true_scaled)  # type: ignore[attr-defined]
            y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)  # type: ignore[attr-defined]
            return y_true, y_pred

        # Prepare outputs for train and validation
        y_train_true, y_train_pred = _infer_to_df(eval_train_loader)
        y_val_true, y_val_pred = _infer_to_df(self.val_loader)  # type: ignore[arg-type]

        evaluator = EvaluationMetrics()
        # If multi-target, create separate evaluations for P, Q, and S
        if len(target_cols) == 2:
            p_col, q_col = target_cols
            # Build S DataFrames
            train_S_true = pd.DataFrame(np.sqrt(y_train_true[p_col].values**2 + y_train_true[q_col].values**2), index=y_train_true.index, columns=['S']) if not y_train_true.empty else None
            train_S_pred = pd.DataFrame(np.sqrt(y_train_pred[p_col].values**2 + y_train_pred[q_col].values**2), index=y_train_pred.index, columns=['S']) if not y_train_pred.empty else None
            val_S_true = pd.DataFrame(np.sqrt(y_val_true[p_col].values**2 + y_val_true[q_col].values**2), index=y_val_true.index, columns=['S'])
            val_S_pred = pd.DataFrame(np.sqrt(y_val_pred[p_col].values**2 + y_val_pred[q_col].values**2), index=y_val_pred.index, columns=['S'])

            results: Dict[str, Any] = {}
            for name, yt, yp, yt_tr, yp_tr in (
                ("P", y_val_true[[p_col]].rename(columns={p_col: 'P'}), y_val_pred[[p_col]].rename(columns={p_col: 'P'}),
                 y_train_true[[p_col]].rename(columns={p_col: 'P'}) if not y_train_true.empty else None,
                 y_train_pred[[p_col]].rename(columns={p_col: 'P'}) if not y_train_pred.empty else None),
                ("Q", y_val_true[[q_col]].rename(columns={q_col: 'Q'}), y_val_pred[[q_col]].rename(columns={q_col: 'Q'}),
                 y_train_true[[q_col]].rename(columns={q_col: 'Q'}) if not y_train_true.empty else None,
                 y_train_pred[[q_col]].rename(columns={q_col: 'Q'}) if not y_train_pred.empty else None),
                ("S", val_S_true, val_S_pred, train_S_true, train_S_pred),
            ):
                results[name] = evaluator.run_model_evaluation(
                    y_test_true=yt,
                    y_test_pred=yp,
                    y_train_true=yt_tr,
                    y_train_pred=yp_tr,
                    no_plots=False,
                    skip_feedin_metrics=(name=="S"),
                    base_agg_hours=int(self.cfg.agg_hours)
                )

            # Manual: plot apparent power S phase angle and print its metrics
            try:
                # Use validation P and Q to compute angles and plot
                val_P_true = y_val_true[[p_col]].rename(columns={p_col: 'P'})
                val_Q_true = y_val_true[[q_col]].rename(columns={q_col: 'Q'})
                val_P_pred = y_val_pred[[p_col]].rename(columns={p_col: 'P'})
                val_Q_pred = y_val_pred[[q_col]].rename(columns={q_col: 'Q'})

                # Build angle DataFrames (deg) and plot angle-specific figure
                ang_true = evaluator._angles_deg_from_pq(val_P_true, val_Q_true)
                ang_pred = evaluator._angles_deg_from_pq(val_P_pred, val_Q_pred)
                fig = evaluator.plot_phase_angle_halfnormal_quantile_errors(ang_true, ang_pred, "Phase Angle of S")
                # Try to display like EvaluationMetrics.plot_evaluation does
                _display = None
                _plt = None
                try:
                    from IPython.display import display as _display  # type: ignore
                except Exception:
                    _display = None
                try:
                    import matplotlib.pyplot as _plt  # type: ignore
                except Exception:
                    _plt = None
                if _display is not None:
                    try:
                        _display(fig)
                    except Exception:
                        pass
                elif _plt is not None:
                    try:
                        _plt.show()
                    except Exception:
                        pass
                if _plt is not None:
                    try:
                        _plt.close(fig)
                    except Exception:
                        pass

                # Compute angle metrics across aggregation scales and print them
                angle_scores = evaluator.evaluate(
                    val_P_true, val_P_pred,
                    base_agg_hours=int(self.cfg.agg_hours),
                    skip_feedin=True,
                    y_true_reactive=val_Q_true,
                    y_pred_reactive=val_Q_pred,
                )
                angle_metrics = {k: v for k, v in angle_scores.items() if "Phase Angle MAE (deg)" in k}
                print("Phase Angle of S metrics (deg):")
                for k, v in angle_metrics.items():
                    try:
                        print(f"  {k}: {float(v):.4f}")
                    except Exception:
                        print(f"  {k}: {v}")
                # Include in results for convenience
                results["S_angle"] = angle_metrics
            except Exception as e:
                print(f"[TransformerTrainer] Skipped phase-angle plotting/metrics due to error: {e}")
            return results
        # Single-target default
        return evaluator.run_model_evaluation(
            y_test_true=y_val_true,
            y_test_pred=y_val_pred,
            y_train_true=y_train_true if not y_train_true.empty else None,
            y_train_pred=y_train_pred if not y_train_pred.empty else None,
            no_plots=False,
            base_agg_hours=int(self.cfg.agg_hours)
        )

    def evaluate_best_metrics(self) -> Dict[str, Any]:
        """
        Compute validation metrics using the best (lowest val_loss) checkpoint.

        Returns a dict with keys:
          - For 2-target setup: {'P': {...}, 'Q': {...}, 'S': {...}}
          - For 1-target setup: {'target': {...}}

        Each value is a dict of metrics as returned by EvaluationMetrics.evaluate,
        e.g., includes keys like "MA(E/MAEx) (1h)", "Peak Import MAPE (1h)", etc.
        """
        if self.val_loader is None:
            raise RuntimeError("Data loaders not initialized; provide _data config and call after setup.")

        # Ensure eval mode and load best state
        self.model.eval()
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)

        agg = int(self.cfg.agg_hours)
        pad = int(self.cfg.pad_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]

        # Run inference on validation loader (no shuffling)
        def _infer_to_df(loader: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
            flat_preds: List[np.ndarray] = []
            flat_trues: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    if self.cfg.loss_type.lower() in ('mae_maex', 'alpha_peak'):
                        if self.cfg.loss_type.lower() == 'alpha_peak':
                            xb, yb, maex, alpha = batch
                        else:
                            xb, yb, maex = batch
                    else:
                        xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)

                    p = pred.detach()
                    t = yb.detach()
                    if aggregation_mode == 'sum':
                        if pad > 0:
                            p = p[:, pad:-pad]
                        if agg > 1:
                            Bm, Tm, Fm = p.shape
                            p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                    B, T, Fp = p.shape
                    Ft = t.shape[-1]
                    flat_preds.append(p.cpu().numpy().reshape(-1, Fp))
                    flat_trues.append(t.cpu().numpy().reshape(-1, Ft))

            pred_arr = np.concatenate(flat_preds, axis=0)
            true_arr = np.concatenate(flat_trues, axis=0)
            N = pred_arr.shape[0]
            base_period = 8760 // agg if agg > 1 else 8760
            hours = np.arange(N, dtype=int) % base_period
            batches = np.arange(N, dtype=int) // base_period
            index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
            cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
            y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
            y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)

            y_true = self.preprocessor.inverse_transform_y(y_true_scaled)  # type: ignore[attr-defined]
            y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)  # type: ignore[attr-defined]
            return y_true, y_pred

        y_val_true, y_val_pred = _infer_to_df(self.val_loader)  # type: ignore[arg-type]
        evaluator = EvaluationMetrics()

        if len(target_cols) == 2:
            p_col, q_col = target_cols
            # Build S
            S_true = pd.DataFrame(
                np.sqrt(y_val_true[p_col].values**2 + y_val_true[q_col].values**2),
                index=y_val_true.index, columns=['S']
            )
            S_pred = pd.DataFrame(
                np.sqrt(y_val_pred[p_col].values**2 + y_val_pred[q_col].values**2),
                index=y_val_pred.index, columns=['S']
            )
            out: Dict[str, Any] = {}
            for name, yt, yp in (
                ("P", y_val_true[[p_col]].rename(columns={p_col: 'P'}), y_val_pred[[p_col]].rename(columns={p_col: 'P'})),
                ("Q", y_val_true[[q_col]].rename(columns={q_col: 'Q'}), y_val_pred[[q_col]].rename(columns={q_col: 'Q'})),
                ("S", S_true, S_pred),
            ):
                metrics = evaluator.evaluate(yt, yp, base_agg_hours=int(self.cfg.agg_hours), skip_feedin=(name=="S"))
                out[name] = metrics
            return out
        else:
            # Single-target
            metrics = evaluator.evaluate(y_val_true, y_val_pred, base_agg_hours=int(self.cfg.agg_hours))
            return { 'target': metrics }

    def evaluate_best_metrics_mdape(self) -> Dict[str, Any]:
        """Compute validation metrics using MdAPE + percentile spreads instead of MAPE.

        Format mirrors evaluate_best_metrics() but replaces MAPE metric keys with
        MdAPE plus additional keys for + and - percentile spreads (p95 and p5).

        Returns
        -------
        dict
            Same nesting as evaluate_best_metrics().
        """
        if self.val_loader is None:
            raise RuntimeError("Data loaders not initialized; provide _data config and call after setup.")

        self.model.eval()
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)

        agg = int(self.cfg.agg_hours)
        pad = int(self.cfg.pad_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]

        def _infer_to_df(loader: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
            flat_preds: List[np.ndarray] = []
            flat_trues: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    if self.cfg.loss_type.lower() in ('mae_maex', 'alpha_peak'):
                        if self.cfg.loss_type.lower() == 'alpha_peak':
                            xb, yb, maex, alpha = batch
                        else:
                            xb, yb, maex = batch
                    else:
                        xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    p = pred.detach()
                    t = yb.detach()
                    if aggregation_mode == 'sum':
                        if pad > 0:
                            p = p[:, pad:-pad]
                        if agg > 1:
                            Bm, Tm, Fm = p.shape
                            p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                    flat_preds.append(p.cpu().numpy().reshape(-1, p.shape[-1]))
                    flat_trues.append(t.cpu().numpy().reshape(-1, t.shape[-1]))
            pred_arr = np.concatenate(flat_preds, axis=0)
            true_arr = np.concatenate(flat_trues, axis=0)
            N = pred_arr.shape[0]
            base_period = 8760 // agg if agg > 1 else 8760
            hours = np.arange(N, dtype=int) % base_period
            batches = np.arange(N, dtype=int) // base_period
            index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
            cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
            y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
            y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)
            y_true = self.preprocessor.inverse_transform_y(y_true_scaled)  # type: ignore[attr-defined]
            y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)  # type: ignore[attr-defined]
            return y_true, y_pred

        y_val_true, y_val_pred = _infer_to_df(self.val_loader)  # type: ignore[arg-type]
        evaluator = EvaluationMetrics()
        if len(target_cols) == 2:
            p_col, q_col = target_cols
            S_true = pd.DataFrame(
                np.sqrt(y_val_true[p_col].values**2 + y_val_true[q_col].values**2),
                index=y_val_true.index, columns=['S']
            )
            S_pred = pd.DataFrame(
                np.sqrt(y_val_pred[p_col].values**2 + y_val_pred[q_col].values**2),
                index=y_val_pred.index, columns=['S']
            )
            out: Dict[str, Any] = {}
            for name, yt, yp in (
                ("P", y_val_true[[p_col]].rename(columns={p_col: 'P'}), y_val_pred[[p_col]].rename(columns={p_col: 'P'})),
                ("Q", y_val_true[[q_col]].rename(columns={q_col: 'Q'}), y_val_pred[[q_col]].rename(columns={q_col: 'Q'})),
                ("S", S_true, S_pred),
            ):
                metrics = evaluator.evaluate_mdape(yt, yp, base_agg_hours=int(self.cfg.agg_hours), skip_feedin=(name=="S"))
                out[name] = metrics
            return out
        else:
            metrics = evaluator.evaluate_mdape(y_val_true, y_val_pred, base_agg_hours=int(self.cfg.agg_hours))
            return { 'target': metrics }

    # ---------------- Train on Train+Val, then Test Eval (with plots) -----------------
    def train_on_trainval(self, epochs: int, shuffle: bool = True) -> Dict[str, Any]:
        """Train the current model for a number of epochs on the concatenation of
        the training and validation datasets prepared during initialization.

        Notes:
          - Uses the same preprocessing and window assembly already built.
          - Does not run validation during these epochs; it's a pure fit on all available
            (train+val) data to maximize data usage before testing.
          - At the end, sets best_state to the final weights for downstream inference.
        """
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            raise RuntimeError("Datasets not initialized. Create the trainer with _data config first.")

        combined_ds = ConcatDataset([self.train_dataset, self.val_dataset])  # type: ignore[arg-type]
        loader = DataLoader(
            combined_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        # Train loop (no validation)
        losses: List[float] = []
        for ep in range(int(epochs)):
            self.model.train(True)
            total = 0.0
            count = 0
            for batch in loader:
                # Unpack batch depending on loss/extras presence
                if self.cfg.loss_type.lower() in ('mae_maex', 'alpha_peak'):
                    if self.cfg.loss_type.lower() == 'alpha_peak':
                        xb, yb, maex, alpha = batch
                        extras = (maex.to(self.device), alpha.to(self.device))
                    else:
                        xb, yb, maex = batch
                        extras = (maex.to(self.device),)
                else:
                    xb, yb = batch
                    extras = None
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self._compute_loss(pred, yb, extras)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                total += float(loss.detach().item()) * yb.size(0) * yb.size(1)
                count += yb.size(0) * yb.size(1)

            avg = total / max(count, 1)
            losses.append(avg)
            # Optional TB log
            if _TB_AVAILABLE and self.writer:
                self.writer.add_scalar('Loss/train_trainval', avg, ep + 1)
                try: 
                    self.writer.flush()
                except Exception:
                    pass

        # Mark final weights as "best" for downstream inference
        self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return { 'epochs': int(epochs), 'final_loss': float(losses[-1]) if losses else float('nan') }

    def evaluate_on_test_with_plots(self, test_data_cfg: Dict[str, Any], n_bldng_lim = None) -> Dict[str, Any]:
        """Evaluate the (already-trained) model on a held-out test set with plots.

        test_data_cfg should contain:
          - 'hdf_data_path': path to the HDF5 file for test set
          - 'key_X': key for features in HDF5
          - 'key_y': key for targets in HDF5

        Uses the fitted preprocessor to transform test features/targets and assembles
        windows identically to training. Produces plots and returns computed metrics.
        """
        required_keys = ['hdf_data_path', 'key_X', 'key_y']
        for k in required_keys:
            if k not in test_data_cfg:
                raise ValueError(f"Missing required test_data_cfg['{k}']")

        # Load raw test frames
        df_X_test_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_X']).astype('float32')
        df_y_test_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_y']).astype('float32')

        if n_bldng_lim is not None:
            df_X_test_raw = df_X_test_raw[(df_X_test_raw["n_nonres_buildings"] + df_X_test_raw["n_res_buildings"]) > n_bldng_lim]
            df_y_test_raw = df_y_test_raw.loc[df_X_test_raw.index]

        # Transform using fitted preprocessor
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor not available. Initialize trainer with _data and call training first.")
        agg = int(self.cfg.agg_hours)
        if n_bldng_lim is None:
            X_test, y_test = self.preprocessor.transform(df_X_test_raw, df_y_test_raw, agg, type="test")  # type: ignore[arg-type]
        else:
            X_test, y_test = self.preprocessor.transform(df_X_test_raw, df_y_test_raw, agg, type=f"test_{n_bldng_lim}")  # type: ignore[arg-type]

        # Assemble windows/tensors
        X_test_t, y_test_t, _ = self._build_sequence_tensors(X_test, y_test, extras={})
        test_ds = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader( 
            test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        # Ensure eval mode with best weights
        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state, strict=False)
            except Exception:
                pass

        # Inference identical to validation path, then inverse-transform
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]
        pad = int(self.cfg.pad_hours)
        agg = int(self.cfg.agg_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')

        flat_preds: List[np.ndarray] = []
        flat_trues: List[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                p = pred.detach()
                t = yb.detach()
                if aggregation_mode == 'sum':
                    if pad > 0:
                        p = p[:, pad:-pad]
                    if agg > 1:
                        Bm, Tm, Fm = p.shape
                        p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                flat_preds.append(p.cpu().numpy().reshape(-1, p.shape[-1]))
                flat_trues.append(t.cpu().numpy().reshape(-1, t.shape[-1]))

        pred_arr = np.concatenate(flat_preds, axis=0) if flat_preds else np.empty((0, len(target_cols)), dtype=np.float32)
        true_arr = np.concatenate(flat_trues, axis=0) if flat_trues else np.empty((0, len(target_cols)), dtype=np.float32)
        N = pred_arr.shape[0]
        base_period = 8760 // agg if agg > 1 else 8760
        hours = np.arange(N, dtype=int) % base_period
        batches = np.arange(N, dtype=int) // base_period
        index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
        cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
        y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
        y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)

        y_true = self.preprocessor.inverse_transform_y(y_true_scaled)  # type: ignore[attr-defined]
        y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)  # type: ignore[attr-defined]

        evaluator = EvaluationMetrics()
        # Multi- or single-target plots/metrics
        if len(target_cols) == 2:
            p_col, q_col = target_cols
            S_true = pd.DataFrame(
                np.sqrt(y_true[p_col].values**2 + y_true[q_col].values**2),
                index=y_true.index, columns=['S']
            )
            S_pred = pd.DataFrame(
                np.sqrt(y_pred[p_col].values**2 + y_pred[q_col].values**2),
                index=y_pred.index, columns=['S']
            )

            results: Dict[str, Any] = {}
            for name, yt, yp in (
                ("P", y_true[[p_col]].rename(columns={p_col: 'P'}), y_pred[[p_col]].rename(columns={p_col: 'P'})),
                ("Q", y_true[[q_col]].rename(columns={q_col: 'Q'}), y_pred[[q_col]].rename(columns={q_col: 'Q'})),
                ("S", S_true, S_pred),
            ):
                results[name] = evaluator.run_model_evaluation(
                    y_test_true=yt,
                    y_test_pred=yp,
                    y_train_true=None,
                    y_train_pred=None,
                    no_plots=False,
                    skip_feedin_metrics=(name == "S"),
                    base_agg_hours=int(self.cfg.agg_hours)
                )

            # Optional: angle plot/metrics for S
            try:
                val_P_true = y_true[[p_col]].rename(columns={p_col: 'P'})
                val_Q_true = y_true[[q_col]].rename(columns={q_col: 'Q'})
                val_P_pred = y_pred[[p_col]].rename(columns={p_col: 'P'})
                val_Q_pred = y_pred[[q_col]].rename(columns={q_col: 'Q'})
                ang_true = evaluator._angles_deg_from_pq(val_P_true, val_Q_true)
                ang_pred = evaluator._angles_deg_from_pq(val_P_pred, val_Q_pred)
                fig = evaluator.plot_phase_angle_halfnormal_quantile_errors(ang_true, ang_pred, "Phase Angle of S (Test)")
                # Best effort show
                try:
                    import matplotlib.pyplot as _plt  # type: ignore
                    _plt.show()
                    _plt.close(fig)
                except Exception:
                    pass
                angle_scores = evaluator.evaluate(
                    val_P_true, val_P_pred,
                    base_agg_hours=int(self.cfg.agg_hours),
                    skip_feedin=True,
                    y_true_reactive=val_Q_true,
                    y_pred_reactive=val_Q_pred,
                )
                results["S_angle"] = {k: v for k, v in angle_scores.items() if "Phase Angle MAE (deg)" in k}
            except Exception as e:
                print(f"[TransformerTrainer] Skipped test phase-angle plotting/metrics due to error: {e}")
            return results

        # Single-target
        return evaluator.run_model_evaluation(
            y_test_true=y_true,
            y_test_pred=y_pred,
            y_train_true=None,
            y_train_pred=None,
            no_plots=False,
            base_agg_hours=int(self.cfg.agg_hours)
        )

    def evaluate_on_test_mdape(self, test_data_cfg: Dict[str, Any], n_bldng_lim: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the trained model on a held-out test set and return MdAPE-based metrics.

        - Uses EvaluationMetrics.evaluate_mdape, which reports MdAPE with +(p95-MdAPE) and -(MdAPE-p5) spreads.
    - Also includes Median Δt metrics with percentile spreads added in evaluate_mdape.

        Prints a transposed DataFrame-like dict view similar to other evaluation helpers.
        Returns the raw metrics dict structure.
        """
        required_keys = ['hdf_data_path', 'key_X', 'key_y']
        for k in required_keys:
            if k not in test_data_cfg:
                raise ValueError(f"Missing required test_data_cfg['{k}']")

        # Load test raw
        df_X_test_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_X']).astype('float32')
        df_y_test_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_y']).astype('float32')
        if n_bldng_lim is not None:
            df_X_test_raw = df_X_test_raw[(df_X_test_raw["n_nonres_buildings"] + df_X_test_raw["n_res_buildings"]) > n_bldng_lim]
            df_y_test_raw = df_y_test_raw.loc[df_X_test_raw.index]

        # Transform with fitted preprocessor
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor not available. Initialize trainer with _data and train before calling test evaluation.")
        agg = int(self.cfg.agg_hours)
        type_tag = "test" if n_bldng_lim is None else f"test_{n_bldng_lim}"
        X_test, y_test = self.preprocessor.transform(df_X_test_raw, df_y_test_raw, agg, type=type_tag)  # type: ignore[arg-type]

        # Windows and inference (scaled)
        X_test_t, y_test_t, _ = self._build_sequence_tensors(X_test, y_test, extras={})
        test_ds = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False,
                                 num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=False)

        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state, strict=False)
            except Exception:
                pass

        pad = int(self.cfg.pad_hours)
        agg = int(self.cfg.agg_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')
        target_cols = list(self.preprocessor.TARGET_COLS)  # type: ignore[attr-defined]

        flat_preds: List[np.ndarray] = []
        flat_trues: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                p = pred.detach()
                t = yb.detach()
                if aggregation_mode == 'sum':
                    if pad > 0:
                        p = p[:, pad:-pad]
                    if agg > 1:
                        Bm, Tm, Fm = p.shape
                        p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                flat_preds.append(p.cpu().numpy().reshape(-1, p.shape[-1]))
                flat_trues.append(t.cpu().numpy().reshape(-1, t.shape[-1]))

        pred_arr = np.concatenate(flat_preds, axis=0) if flat_preds else np.empty((0, len(target_cols)), dtype=np.float32)
        true_arr = np.concatenate(flat_trues, axis=0) if flat_trues else np.empty((0, len(target_cols)), dtype=np.float32)
        N = pred_arr.shape[0]
        base_period = 8760 // agg if agg > 1 else 8760
        hours = np.arange(N, dtype=int) % base_period
        batches = np.arange(N, dtype=int) // base_period
        index = pd.MultiIndex.from_arrays([batches, hours], names=['batch', 'hour'])
        cols = target_cols if true_arr.shape[1] == len(target_cols) else [target_cols[0]]
        y_true_scaled = pd.DataFrame(true_arr, index=index, columns=cols)
        y_pred_scaled = pd.DataFrame(pred_arr, index=index, columns=cols)

        y_true = self.preprocessor.inverse_transform_y(y_true_scaled)  # type: ignore[attr-defined]
        y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)  # type: ignore[attr-defined]

        evaluator = EvaluationMetrics()
        if len(target_cols) == 2:
            p_col, q_col = target_cols
            S_true = pd.DataFrame(np.sqrt(y_true[p_col].values**2 + y_true[q_col].values**2), index=y_true.index, columns=['S'])
            S_pred = pd.DataFrame(np.sqrt(y_pred[p_col].values**2 + y_pred[q_col].values**2), index=y_pred.index, columns=['S'])
            out: Dict[str, Any] = {}
            for name, yt, yp in (
                ("P", y_true[[p_col]].rename(columns={p_col: 'P'}), y_pred[[p_col]].rename(columns={p_col: 'P'})),
                ("Q", y_true[[q_col]].rename(columns={q_col: 'Q'}), y_pred[[q_col]].rename(columns={q_col: 'Q'})),
                ("S", S_true, S_pred),
            ):
                out[name] = evaluator.evaluate_mdape(yt, yp, base_agg_hours=int(self.cfg.agg_hours), skip_feedin=(name=="S"))
            # Pretty print similar to run_model_evaluation
            try:
                dfP = pd.DataFrame(out.get('P', {}), index=['Test']).T if 'P' in out else None
                dfQ = pd.DataFrame(out.get('Q', {}), index=['Test']).T if 'Q' in out else None
                dfS = pd.DataFrame(out.get('S', {}), index=['Test']).T if 'S' in out else None
                print("MdAPE Metric Results (Test):")
                if dfP is not None:
                    print("P:")
                    print(dfP)
                if dfQ is not None:
                    print("Q:")
                    print(dfQ)
                if dfS is not None:
                    print("S:")
                    print(dfS)
            except Exception:
                pass
            return out
        else:
            metrics = evaluator.evaluate_mdape(y_true, y_pred, base_agg_hours=int(self.cfg.agg_hours))
            try:
                df = pd.DataFrame(metrics, index=['Test']).T
                print("MdAPE Metric Results (Test):")
                print(df)
            except Exception:
                pass
            return {'target': metrics}

    # ---------------------------------------------------------------------
    # Benchmark: Per-grid end-to-end test (preprocess -> predict -> postprocess)
    # ---------------------------------------------------------------------
    def benchmark_test_grids(
        self,
        test_data_cfg: Dict[str, Any],
        logfile: str = "grid_inference_benchmark.log",
        include_children: bool = False,
        flush_every: int = 1,
        device_sync: bool = True,
    ) -> pd.DataFrame:
        """Run an end-to-end timing benchmark for each grid in the provided test
        dataset. For every grid we measure:

        - preprocess_cpu_seconds: feature engineering + scaling (CPU)
        - inference_gpu_seconds: active GPU wall time (synchronous) for forward pass windows
        - postprocess_cpu_seconds: inverse scaling + assembly
        - total_wall_seconds: overall wallclock for the grid (including everything)
        - n_hours_raw: number of original hours
        - n_hours_out: number of prediction hours after aggregation
        - n_windows: number of sliding windows evaluated
        - peak_rss_bytes: peak process RSS at end of full grid run (from resource_report)

        The function appends one JSON line per grid to ``logfile`` (newline delimited JSON)
        and returns a DataFrame of the collected metrics.

        Parameters
        ----------
        test_data_cfg: Dict[str, Any]
            Must contain 'hdf_data_path', 'key_X', 'key_y'. Uses the same raw feature
            layout as training. Targets are only used for dimensional validation (not timing).
        logfile: str
            Path to append JSONL metrics. Created if missing.
        include_children: bool
            Whether to include child process CPU time in resource reports.
        flush_every: int
            Flush file handle after this many grid entries (default 1 -> flush each line).
        device_sync: bool
            If True and CUDA available, calls torch.cuda.synchronize() around measured GPU blocks
            to obtain accurate timings.
        """
        required = ['hdf_data_path', 'key_X', 'key_y']
        for k in required:
            if k not in test_data_cfg:
                raise ValueError(f"benchmark_test_grids missing test_data_cfg['{k}']")
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor not fitted. Train the model before benchmarking.")

        # Load raw (unscaled) test features/targets (targets optional but helpful for shape checks)
        df_X_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_X']).astype('float32')
        df_y_raw = pd.read_hdf(test_data_cfg['hdf_data_path'], key=test_data_cfg['key_y']).astype('float32')

        # Group by grid id (first index level assumed 'batch')
        if not isinstance(df_X_raw.index, pd.MultiIndex) or df_X_raw.index.nlevels < 2:
            raise ValueError("Expected MultiIndex (batch, hour) for test features")
        grid_ids = df_X_raw.index.get_level_values(0).unique().tolist()

        logfile_path = Path(logfile)
        logfile_path.parent.mkdir(parents=True, exist_ok=True)
        # We'll write JSONL; no separate header required but we note start if empty.
        new_file = not logfile_path.exists() or logfile_path.stat().st_size == 0

        agg = int(self.cfg.agg_hours)
        core_len = int(self.cfg.core_len)
        pad = int(self.cfg.pad_hours)
        aggregation_mode = getattr(self.cfg, 'aggregation_mode', 'sum')

        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state, strict=False)
            except Exception:
                pass

        results: List[Dict[str, Any]] = []
        line_count = 0
        # Reuse scalers; do NOT refit.
        with open(logfile_path, 'a', encoding='utf-8') as fh:
            if new_file:
                fh.write(json.dumps({"_meta": "grid_inference_benchmark_v1"}) + "\n")
                fh.flush()
            for gid in grid_ids:
                # Slice grid
                Xg_raw = df_X_raw.xs(gid, level=0)
                n_hours = len(Xg_raw)
                # Optional target just for dimensional sanity (not scaled if not needed)
                yg_raw = df_y_raw.xs(gid, level=0) if gid in df_y_raw.index.get_level_values(0) else None
                start_grid = time.perf_counter()

                with resource_report(include_children=include_children, name=f"grid_{gid}_total") as rr_total:
                    # ---------------- Preprocess ----------------
                    t0 = time.perf_counter()
                    # Reconstruct MultiIndex for transform
                    Xg_raw_reindexed = Xg_raw.copy()
                    Xg_raw_reindexed.index = pd.MultiIndex.from_arrays([[gid]*n_hours, np.arange(n_hours)], names=['batch','hour'])
                    if yg_raw is not None:
                        yg_raw_re = yg_raw.copy()
                        yg_raw_re.index = pd.MultiIndex.from_arrays([[gid]*len(yg_raw), np.arange(len(yg_raw))], names=['batch','hour'])
                    else:
                        yg_raw_re = None
                    Xg_scaled = self.preprocessor.transform(Xg_raw_reindexed, df_y=None, aggregation=agg, type="test")  # type: ignore[arg-type]
                    preprocess_sec = time.perf_counter() - t0

                    # ---------------- Window Assembly ----------------
                    # Build windows exactly like predict() but for single grid to measure GPU time precisely
                    X_arr = Xg_scaled.to_numpy(dtype=np.float32, copy=False)
                    seq_len_hours = core_len + 2*pad
                    starts_hours = np.arange(0, max(len(X_arr) - core_len + 1, 0), core_len, dtype=int)
                    if len(starts_hours) == 0:
                        starts_hours = np.array([0], dtype=int)
                    base_hours = np.arange(-pad, core_len + pad, dtype=int)
                    idx_X = (starts_hours[:, None] + base_hours[None, :]) % len(X_arr)
                    X_win = torch.from_numpy(X_arr[idx_X]).to(self.device)
                    n_windows = X_win.shape[0]

                    # ---------------- Inference (GPU timed) ----------------
                    if device_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_inf0 = time.perf_counter()
                    with torch.no_grad():
                        pred = self.model(X_win)
                        if aggregation_mode == 'sum':
                            p = pred
                            if pad > 0:
                                p = p[:, pad:-pad]
                            if agg > 1:
                                Bm, Tm, Fm = p.shape
                                if Tm % agg != 0:
                                    raise RuntimeError("Core length not divisible by agg_hours during benchmark.")
                                p = p.view(Bm, Tm // agg, agg, Fm).sum(dim=2)
                        else:
                            p = pred
                    if device_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_gpu_sec = time.perf_counter() - t_inf0

                    # ---------------- Postprocess (inverse + assembly) ----------------
                    t_post0 = time.perf_counter()
                    p_np = p.detach().cpu().numpy().reshape(-1, p.shape[-1])
                    n_out = len(X_arr) // agg if agg > 1 else len(X_arr)
                    if p_np.shape[0] != n_out:
                        n_out = min(p_np.shape[0], n_out)
                        p_np = p_np[:n_out]
                    # Build scaled DataFrame for inverse
                    target_cols = list(self.preprocessor.TARGET_COLS)
                    cols = target_cols if p_np.shape[1] == len(target_cols) else [target_cols[0]]
                    idx = pd.MultiIndex.from_arrays([[gid]*n_out, np.arange(n_out)], names=['batch','hour'])
                    y_scaled = pd.DataFrame(p_np, index=idx, columns=cols)
                    y_orig = self.preprocessor.inverse_transform_y(y_scaled)
                    postprocess_sec = time.perf_counter() - t_post0

                total_wall = time.perf_counter() - start_grid
                rr_metrics = rr_total.last_metrics or {}
                entry = {
                    'grid_id': gid,
                    'preprocess_cpu_seconds': preprocess_sec,
                    'inference_gpu_seconds': inference_gpu_sec,
                    'postprocess_cpu_seconds': postprocess_sec,
                    'total_wall_seconds': total_wall,
                    'n_hours_raw': int(len(X_arr)),
                    'n_hours_out': int(n_out),
                    'n_windows': int(n_windows),
                    'peak_rss_bytes': rr_metrics.get('peak_rss_bytes'),
                    'total_cpu_seconds': rr_metrics.get('total_cpu_seconds'),
                    'user_seconds': rr_metrics.get('user_seconds'),
                    'system_seconds': rr_metrics.get('system_seconds'),
                }
                fh.write(json.dumps(entry) + "\n")
                line_count += 1
                if flush_every and (line_count % flush_every == 0):
                    fh.flush()
                results.append(entry)

        # Return tabular view
        return pd.DataFrame(results)

# ============================================================
__all__ = [
    'TransformerConfig',
    'TransformerForecastModel',
    'TransformerTrainer',
    'train_on_trainval_and_eval_test'
]


# ============================================================
# Ray Tune helper API (simple example)
# ============================================================
def build_tune_search_space(tune):
    """Return a simple Ray Tune search space dict for the TransformerTrainer.

    The space mirrors the MLP example: we expose a handful of important knobs
    while keeping data/config defaults suitable for the provided dataset.
    """
    space = {
        # Core searchable training hparams
        'learning_rate': tune.loguniform(1e-4, 5e-2),
        'weight_decay': tune.loguniform(1e-6, 1e-1),
        'batch_size': tune.choice([128, 256, 512, 1024]),
        'optimizer': tune.choice(['adamw', 'sgd']),
        'scheduler': tune.choice(['none', 'cosine', 'plateau']),

        # Model size/regularization
        'd_model': tune.choice([256, 512, 1024]),
        'num_heads': tune.choice([4, 8, 16]),
        'ff_dim': tune.choice([512, 1024, 2048]),
        'num_transformer_layers': tune.choice([1, 2, 3, 4]),
        'input_mlp_hidden': tune.choice([128, 256, 512]),
        'output_mlp_hidden': tune.choice([128, 256, 512]),
        'input_mlp_layers': tune.choice([1, 2, 3]),
        'output_mlp_layers': tune.choice([1, 2, 3]),
        'input_dropout': tune.uniform(0.0, 0.3),
        'output_dropout': tune.uniform(0.0, 0.3),
        'attn_dropout': tune.uniform(0.0, 0.3),
        'ffn_dropout': tune.uniform(0.0, 0.3),

        # Sequence/aggregation knobs
        'core_len': tune.choice([24, 120]),
        'pad_hours': tune.choice([12, 24, 48, 96]),
        'aggregation_mode': "conv",
        'agg_hours': 1 , #tune.choice([1, 2, 4]),
        'conv_padding': tune.choice([0, 6, 12, 24]),

        # Loss
        'loss_type': 'mae_maex',
        'multi_target_lambda': 1, #lambda/2*(P-term + Q-term) + (1-lambda)*S-term

        # Fixed/defaults suitable for this repo
        'activation': 'gelu',
        'grad_clip': 0.0,
        'compile_model': True,
        'num_workers': 14,
        'pin_memory': True,
        'seed': 42,

        # Data pipeline
        '_data': {
            'hdf_data_path': '/dss/dsshome1/05/ge96ton2/GridForecast/0_preprocessing/Data/ts_train.h5',
            'key_X': 'X',
            'key_y': 'y',
            'train_grids': 'all',
            'test_ratio': 0.2,
            'random_state': 42,
            'X_BASE_COLS': [
                #'T', 
                'demand_net_active_pre',
                'mob_avail', "mob_last_avail", 'mobility',
                'heat_water', 'heat_space', 'cop_avg',
                'PV_prod_expected',
                'res_bldng_area_base_sum', 'nonres_bldng_area_base_sum', 'bldng_area_floors_sum',
                'n_cars', 'n_res_buildings', 'n_nonres_buildings', 'n_flats', 'n_occ',
                'n_lines', 'tot_R_grid', 'regiostar7',
                ],
            'TARGET_COLS': ['demand_net_active_post', 'demand_net_reactive_post'],
            'ZERO_BASE_FEATURES': [],
            'LOG1P_COLS': ['n_lines', 'tot_R_grid'],
            # Let Ray Tune choose among several seasonal feature sets.
            # Use strings to avoid nested list identity/sampling quirks, then parse in Preprocessor.
            'TS_PERIODS': tune.choice([
                '8760,168,24,12',
                '8760,168,24',
                '8760,24',
                '8760',
                ''
            ]),
            'VMD_COLS': [#'demand_net_active_pre', 'heat_water', 'heat_space', 'cop_avg', 'PV_prod_expected'],
                'demand_net_active_pre', 'heat_water', 'heat_space', 'cop_avg', 'mob_avail', 'mobility', "mob_last_avail", 'PV_prod_expected'],
            # Explore different numbers of VMD modes
            'VMD_K_MODES': tune.choice([0, 1, 2, 3, 4]),
            'VMD_APPROACH': 'read',
        },
    }
    return space


def build_tune_trainable():
    """Return a Ray Tune-compatible trainable callable for TransformerTrainer.

    The trainable creates a trainer from the config, then reports metrics each
    epoch via the trainer.step() method.
    """
    def _trainable(config: Dict[str, Any]):
        # Ensure bottom_rung_report is present for the TPE+ASHA example if not set
        config = dict(config)
        grace = int(config.get('bottom_rung_report', 0))
        if grace <= 0:
            config['bottom_rung_report'] = 5
        # epochs controls max training iterations for ASHA
        epochs = int(config.get('epochs', config.get('scheduler_epochs', 20)))
        config['epochs'] = epochs

        trainer = TransformerTrainer(config)
        for _ in range(epochs):
            trainer.step()
    return _trainable


# ============================================================
# Convenience runner
# ============================================================
def train_on_trainval_and_eval_test(
    config: Dict[str, Any],
    trainval_epochs: int,
    test_data_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """End-to-end helper:
    1) Initialize trainer (builds train/val datasets from config['_data']).
    2) Optionally warm up with existing epochs in config (epochs>0) via trainer.fit().
    3) Train on train+val for 'trainval_epochs'.
    4) Evaluate on the provided test set with plots.

    Returns the test metrics dict.
    """
    cfg = dict(config)
    # If the caller wants a brief warmup using the standard train/val split epochs
    trainer = TransformerTrainer(cfg)
    # Train on all available (train+val)
    trainer.train_on_trainval(int(trainval_epochs))
    # Test evaluation with plots
    trainer.evaluate_on_test_with_plots(test_data_cfg)

    return trainer