"""Compact MLP trainer aligned with transformer preprocessing (no VMD), dual-target support, MAE/MAEx, Ray Tune.

- Preprocessing: log1p on selected cols, seasonal sin/cos by TS_PERIODS, StandardScaler on X and y.
    * Dual-target scaling: learn scaler on S = sqrt(P^2+Q^2), apply same std(S) to both channels. Assumes first
        target is active power P and second is reactive power Q when two targets are provided.
- Model: simple feed-forward MLP; output dim equals number of targets (1 or 2).
- Loss: MAE/MAEx on scaled space. Any non-'mae_maex' loss in config maps to 'mae_maex' (for compatibility).
- Trainer: Ray Tune-compatible (Trainable). Reports train/val loss; optional bottom rung and extended metrics
  on the original scale via EvaluationMetrics every full_metrics_every epochs.
- Inference: FullModel wrapper with predict().
"""
from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

try:  # Ray
    from ray import tune  # type: ignore
    from ray.tune.trainable import Trainable  # type: ignore
except Exception:  # pragma: no cover
    tune = None  # type: ignore 
    Trainable = object  # type: ignore

from evaluation import EvaluationMetrics


# ========================= Preprocessor (no VMD) =========================
class Preprocessor:
        """Feature/target preprocessing for the MLP baseline.

        Responsibilities:
            - Select base feature columns (`X_BASE_COLS`) from the raw `X` table.
            - Add feature engineering:
                    * zero-indicator columns for selected features
                    * log1p for selected features
                    * seasonal sin/cos terms derived from the `hour` index level
            - Scale X features using `StandardScaler`.
            - Scale y targets.

        Dual-target behavior (P/Q):
            If `TARGET_COLS` has length 2, scaling is learned from apparent power
            $S=\sqrt{P^2+Q^2}$ and then applied to both channels.
        """

    def __init__(self, data_cfg: Dict[str, Any]):
                """Create a Preprocessor from a data configuration dict."""
        self.X_BASE_COLS: List[str] = list(data_cfg["X_BASE_COLS"])  # mandatory
        self.TARGET_COLS: List[str] = list(data_cfg["TARGET_COLS"])  # mandatory (len 1 or 2)
        self.ZERO_BASE_FEATURES: List[str] = list(data_cfg.get("ZERO_BASE_FEATURES", []))
        self.LOG1P_COLS: List[str] = list(data_cfg.get("LOG1P_COLS", []))
        self.TS_PERIODS: List[int] = self._normalize_ts_periods(data_cfg.get("TS_PERIODS", []))

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.fitted = False

        self.zero_names: List[str] = []
        self.period_names: List[str] = []
        self.feature_names_: List[str] = []

    @staticmethod
    def _normalize_ts_periods(val: Any) -> List[int]:
        if val is None:
            return []
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return []
            return [int(x) for x in s.split(',') if str(x).strip()]
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        return []

    def _universal_X_trafo(self, df_X: pd.DataFrame) -> pd.DataFrame:
        df_X = df_X[self.X_BASE_COLS].copy()
        # Zero indicators
        if self.ZERO_BASE_FEATURES:
            zeros = (df_X[self.ZERO_BASE_FEATURES] == 0).astype('int32')
            zeros.columns = [f"{c}_zero" for c in self.ZERO_BASE_FEATURES]
            self.zero_names = list(zeros.columns)
        else:
            zeros = pd.DataFrame(index=df_X.index)
            self.zero_names = []
        # Log1p columns
        for col in self.LOG1P_COLS:
            if col in df_X.columns:
                df_X[col] = np.log1p(np.clip(df_X[col].astype('float32'), a_min=0.0, a_max=None))
        # Seasonal features
        self.period_names = []
        if df_X.index.nlevels >= 2:
            hours = df_X.index.get_level_values(1).to_numpy()
        else:
            hours = np.arange(len(df_X), dtype=int)
        for freq in self.TS_PERIODS:
            if freq and freq > 0:
                df_X[f"sin_{freq}"] = np.sin(2 * np.pi * hours / freq).astype('float32')
                df_X[f"cos_{freq}"] = np.cos(2 * np.pi * hours / freq).astype('float32')
                self.period_names.extend([f"sin_{freq}", f"cos_{freq}"])
        # Concat zeros at the end
        if not zeros.empty:
            df_X = pd.concat([df_X, zeros], axis=1)
        return df_X

    def _fit_or_transform_X(self, df_X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df_X = self._universal_X_trafo(df_X)
        scale_feats = [c for c in df_X.columns if (c not in self.zero_names and c not in self.period_names)]
        if fit:
            arr = df_X[scale_feats].astype('float32').values
            self.scaler_X.fit(arr)
            df_X[scale_feats] = self.scaler_X.transform(arr)
            self.feature_names_ = list(df_X.columns)
        else:
            if not self.feature_names_:
                self.feature_names_ = list(df_X.columns)
            for col in self.feature_names_:
                if col not in df_X.columns:
                    df_X[col] = 0.0
            df_X = df_X[self.feature_names_]
            scale_feats = [c for c in df_X.columns if (c not in self.zero_names and c not in self.period_names)]
            df_X[scale_feats] = self.scaler_X.transform(df_X[scale_feats].astype('float32').values)
        return df_X

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

    def fit_transform(self, df_X: pd.DataFrame, df_y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit scalers on `(df_X, df_y)` and return the transformed copies."""
        X = self._fit_or_transform_X(df_X, fit=True)
        y = self._fit_or_transform_y(df_y, fit=True)
        self.fitted = True
        return X, y

    def transform(self, df_X: pd.DataFrame, df_y: Optional[pd.DataFrame] = None):
        """Transform `df_X` (and optionally `df_y`) using already-fitted scalers."""
        X = self._fit_or_transform_X(df_X, fit=False)
        if df_y is not None:
            y = self._fit_or_transform_y(df_y, fit=False)
            return X, y
        return X

    def inverse_transform_y(self, df_y_scaled: pd.DataFrame) -> pd.DataFrame:
        """Inverse-transform scaled targets back to the original target scale."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted.")
        y = df_y_scaled.copy().astype('float32')
        if len(self.TARGET_COLS) == 2 and y.shape[1] == 2:
            mean_S = float(self.scaler_y.mean_[0]) if hasattr(self.scaler_y, 'mean_') else 0.0
            scale_S = float(self.scaler_y.scale_[0]) if hasattr(self.scaler_y, 'scale_') else 1.0
            y.iloc[:, 0] = y.iloc[:, 0] * scale_S + mean_S
            y.iloc[:, 1] = y.iloc[:, 1] * scale_S + mean_S
            y.columns = self.TARGET_COLS
            return y
        arr = self.scaler_y.inverse_transform(y.values)
        return pd.DataFrame(arr, index=y.index, columns=self.TARGET_COLS[:arr.shape[1]])


# ========================= MLP =========================
class MLP(nn.Module):
    """Simple feed-forward MLP used as baseline forecaster."""

    def __init__(self, in_dim: int, hidden_layers: Sequence[int], out_dim: int,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        act_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        Act = act_map.get(activation.lower(), nn.ReLU)
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev, out_dim)
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x))


# ========================= Loss =========================
class MAEMaexLoss(nn.Module):
    """MAE normalized by MAEx: mean(|(pred-target)/maex|).

    `maex` is expected to be a per-sample (or per-batch) scale term, typically
    the mean absolute exchange for each grid.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, maex: torch.Tensor) -> torch.Tensor:  # type: ignore
        eps = 1e-8
        denom = torch.clamp(maex.abs(), min=eps)
        nerr = (pred - target) / denom
        nerr = torch.nan_to_num(nerr, nan=0.0, posinf=0.0, neginf=0.0)
        return nerr.abs().mean()


# ========================= Trainer (Ray Tune) =========================
class MLPTrainer(Trainable):  # pragma: no cover
    """Ray Tune `Trainable` wrapper for MLP training.

    The trainer:
      - loads `X`/`y` from an HDF5 file defined in `config['_data']`
      - splits grids into train/val by grid id
      - fits preprocessing scalers on train and applies to val
      - trains for one epoch per `step()` call
      - reports `train_loss`, `val_loss` and optional extended metrics
    """

    def setup(self, config: Dict[str, Any]):  # type: ignore[override]
        """Initialize datasets, model, optimizer, and scheduler from config."""
        if tune is None:
            raise RuntimeError("ray.tune not available – use Ray to run MLPTrainer.")
        self.cfg = dict(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = EvaluationMetrics()
        self.history: List[Dict[str, float]] = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.base_agg_hours = 1

        # Map legacy loss names to mae_maex
        if str(self.cfg.get('loss_type', 'mae_maex')).lower() != 'mae_maex':
            self.cfg['loss_type'] = 'mae_maex'

        self._set_seed(int(self.cfg.get('random_state', self.cfg.get('_data', {}).get('random_state', 42))))
        self._load_data(self.cfg.get('_data', {}))
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self.loss_fn = MAEMaexLoss()
        self.loss_type = 'mae_maex'

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # ---------------- Data ----------------
    def _load_data(self, data_cfg: Dict[str, Any]):
        hdf_path = data_cfg.get('hdf_data_path')
        key_X = data_cfg.get('key_X', 'X')
        key_y = data_cfg.get('key_y', 'y')
        train_grids = data_cfg.get('train_grids', 'all')
        test_ratio = float(data_cfg.get('test_ratio', 0.2))
        random_state = int(data_cfg.get('random_state', 42))
        if not hdf_path or not os.path.exists(str(hdf_path)):
            raise FileNotFoundError(f"HDF data file not found: {hdf_path}")

        X_df: pd.DataFrame = pd.read_hdf(str(hdf_path), key_X)
        y_df: pd.DataFrame = pd.read_hdf(str(hdf_path), key_y)

        # Preprocessor spec
        data_cfg_pp = {
            'X_BASE_COLS': data_cfg['X_BASE_COLS'],
            'TARGET_COLS': data_cfg['TARGET_COLS'],
            'ZERO_BASE_FEATURES': data_cfg.get('ZERO_BASE_FEATURES', []),
            'LOG1P_COLS': data_cfg.get('LOG1P_COLS', []),
            'TS_PERIODS': data_cfg.get('TS_PERIODS', []),
        }
        self.preprocessor = Preprocessor(data_cfg_pp)

        # Split by grid id
        rng = np.random.default_rng(seed=random_state)
        grids = np.array(X_df.index.get_level_values(0).unique())
        rng.shuffle(grids)
        n_val = max(1, int(round(test_ratio * len(grids))))
        val_grids = set(grids[:n_val].tolist())
        train_grids_all = set(grids[n_val:].tolist())
        if train_grids not in (None, 'all'):
            try:
                train_grids_all = train_grids_all.intersection(set(map(int, train_grids)))  # type: ignore[arg-type]
            except Exception:
                pass

        train_idx = X_df.index.get_level_values(0).isin(list(train_grids_all))
        val_idx = X_df.index.get_level_values(0).isin(list(val_grids))
        X_train_raw, y_train_raw = X_df.loc[train_idx], y_df.loc[train_idx]
        X_val_raw, y_val_raw = X_df.loc[val_idx], y_df.loc[val_idx]

        X_train, y_train = self.preprocessor.fit_transform(X_train_raw, y_train_raw)
        X_val, y_val = self.preprocessor.transform(X_val_raw, y_val_raw)

        self.y_train_index = y_train.index
        self.y_val_index = y_val.index

        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train[self.preprocessor.TARGET_COLS].values, dtype=torch.float32)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val[self.preprocessor.TARGET_COLS].values, dtype=torch.float32)

        # Per-grid MAEx on scaled y (per target)
        def _maex_per_row(df_scaled: pd.DataFrame) -> np.ndarray:
            grouped = df_scaled.groupby(level=0).apply(lambda g: g.abs().mean())
            batches = df_scaled.index.get_level_values(0)
            return grouped.reindex(batches).values.astype('float32')

        train_maex = _maex_per_row(y_train)
        val_maex = _maex_per_row(y_val)

        self.train_dataset = TensorDataset(X_train_t, y_train_t, torch.tensor(train_maex, dtype=torch.float32))
        self.val_dataset = TensorDataset(X_val_t, y_val_t, torch.tensor(val_maex, dtype=torch.float32))

        self.input_dim = X_train_t.shape[1]
        self.output_dim = y_train_t.shape[1]

        bs = int(self.cfg.get('batch_size', 8192))
        nw = int(self.cfg.get('num_workers', 0))
        pin = bool(self.cfg.get('pin_memory', True))
        self.train_loader = DataLoader(self.train_dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
        self.val_loader = DataLoader(self.val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    # ---------------- Model/optim ----------------
    def _build_model(self):
        # Build hidden layers from simplified parameters
        if 'hidden_layers' in self.cfg:
            raise ValueError("'hidden_layers' is deprecated. Use 'num_layers' and 'layer_size' instead.")
        num_layers = int(self.cfg.get('num_layers', 4))
        layer_size = int(self.cfg.get('layer_size', 256))
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if layer_size < 1:
            raise ValueError(f"layer_size must be >= 1, got {layer_size}")
        hidden_layers = [layer_size] * num_layers
        activation = self.cfg.get('activation', 'relu')
        dropout = float(self.cfg.get('dropout', 0.0))
        self.model = MLP(self.input_dim, hidden_layers, self.output_dim, activation, dropout).to(self.device)
        if bool(self.cfg.get('compile_model', False)) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _build_optimizer(self):
        opt_name = str(self.cfg.get('optimizer', 'adamw')).lower()
        lr = float(self.cfg.get('learning_rate', 1e-3))
        wd = float(self.cfg.get('weight_decay', 0.0))
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _build_scheduler(self):
        sched = str(self.cfg.get('scheduler', 'none')).lower()
        if sched == 'none':
            self.scheduler = None
            return
        scheduler_epochs = int(self.cfg.get('scheduler_epochs', 50))
        steps_per_epoch = max(1, len(self.train_dataset) // int(self.cfg.get('batch_size', 1)))
        if sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_epochs)
        elif sched == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3)
        elif sched == 'onecycle':
            max_lr = float(self.cfg.get('learning_rate', 1e-3))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                                 steps_per_epoch=steps_per_epoch,
                                                                 epochs=scheduler_epochs)
        else:
            self.scheduler = None

    # ---------------- Epoch ----------------
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, maex: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target, maex)  # type: ignore[arg-type]

    def _run_epoch(self, train: bool, collect: bool = False) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
        loader = self.train_loader if train else self.val_loader
        assert loader is not None
        self.model.train(mode=train)
        total_loss = 0.0
        n_batches = 0
        y_list: List[torch.Tensor] = []
        yp_list: List[torch.Tensor] = []
        with torch.set_grad_enabled(train):
            for xb, yb, maex in loader:  # type: ignore
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                maex = maex.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = self._compute_loss(pred, yb, maex)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_clip = float(self.cfg.get('grad_clip', 0.0) or 0.0)
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                total_loss += float(loss.detach().item())
                n_batches += 1
                if not train and collect:
                    y_list.append(yb.detach().cpu())
                    yp_list.append(pred.detach().cpu())
        avg_loss = total_loss / max(1, n_batches)
        if not train and collect and y_list and yp_list:
            y_scaled = torch.cat(y_list, dim=0).numpy()
            yp_scaled = torch.cat(yp_list, dim=0).numpy()
            return avg_loss, {"y_scaled": y_scaled, "yp_scaled": yp_scaled}
        return avg_loss, None

    # ---------------- Ray step ----------------
    def step(self) -> Dict[str, Any]:  # type: ignore[override]
        """Run one training epoch and one validation pass and return metrics."""
        t0 = time.time()
        self.current_epoch += 1
        # Compute/report extensive metrics every N epochs (default aligns with transformer: 5)
        full_every = int(self.cfg.get('full_metrics_every', 5))
        collect = (full_every > 0) and (self.current_epoch % full_every == 0)
        bottom_rung_at = int(self.cfg.get('bottom_rung_report', 0) or 0)
        do_bottom = (bottom_rung_at > 0 and self.current_epoch == bottom_rung_at)

        train_loss, _ = self._run_epoch(train=True, collect=False)
        val_loss, collect_data = self._run_epoch(train=False, collect=collect)

        rec: Dict[str, float] = {
            'epoch': float(self.current_epoch),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(self.optimizer.param_groups[0]['lr'])
        }

        # Extended metrics on original scale
        if collect and collect_data is not None:
            y_scaled = collect_data['y_scaled']
            yp_scaled = collect_data['yp_scaled']
            cols = self.preprocessor.TARGET_COLS[:y_scaled.shape[1]]
            y_df_scaled = pd.DataFrame(y_scaled, index=self.y_val_index, columns=cols)
            yp_df_scaled = pd.DataFrame(yp_scaled, index=self.y_val_index, columns=cols)
            try:
                y_df = self.preprocessor.inverse_transform_y(y_df_scaled)
                yp_df = self.preprocessor.inverse_transform_y(yp_df_scaled)
                y_df.index = y_df_scaled.index
                yp_df.index = yp_df_scaled.index
                # Ensure index level names are consistent with EvaluationMetrics expectations
                if isinstance(y_df.index, pd.MultiIndex) and y_df.index.nlevels >= 2:
                    try:
                        y_df.index = y_df.index.set_names(['batch', 'hour'] + list(y_df.index.names[2:]))
                        yp_df.index = yp_df.index.set_names(['batch', 'hour'] + list(yp_df.index.names[2:]))
                    except Exception:
                        pass
            except Exception:
                y_df, yp_df = y_df_scaled, yp_df_scaled
                # Also try to enforce index names if MultiIndex is present
                if isinstance(y_df.index, pd.MultiIndex) and y_df.index.nlevels >= 2:
                    try:
                        y_df.index = y_df.index.set_names(['batch', 'hour'] + list(y_df.index.names[2:]))
                        yp_df.index = yp_df.index.set_names(['batch', 'hour'] + list(yp_df.index.names[2:]))
                    except Exception:
                        pass

            # Evaluate per-target metrics (P, Q if present) and derived apparent power S when dual-target
            def _short_label(idx: int, name: str) -> str:
                # Prefer canonical P/Q when available, fallback to original column name
                return 'P' if idx == 0 else ('Q' if idx == 1 else name)

            for i, col in enumerate(cols):
                try:
                    # For S (i==2), skip feed-in specific metrics
                    m = self.metrics.evaluate(
                        y_df[[col]],
                        yp_df[[col]],
                        base_agg_hours=self.base_agg_hours,
                        skip_feedin=(i == 2),
                    )
                    label = _short_label(i, col)
                    for k, v in m.items():
                        val = float(v) if np.isfinite(v) else float('nan')
                        # Standard keys (match transformer style with underscores)
                        rec[f"val_{label}_{k}"] = val
                        # Also provide original column name variant
                        rec[f"val_{col}_{k}"] = val
                        # Backward-compat aliases with ':' (can be removed later)
                        rec[f"val_{label}:{k}"] = val
                        rec[f"val_{col}:{k}"] = val
                except Exception:
                    pass

            # Apparent power S = sqrt(P^2 + Q^2)
            if len(cols) >= 2:
                try:
                    P_true = y_df.iloc[:, 0]
                    Q_true = y_df.iloc[:, 1]
                    P_pred = yp_df.iloc[:, 0]
                    Q_pred = yp_df.iloc[:, 1]
                    S_true = np.sqrt(P_true**2 + Q_true**2).to_frame(name='S')
                    S_pred = np.sqrt(P_pred**2 + Q_pred**2).to_frame(name='S')
                    mS = self.metrics.evaluate(
                        S_true,
                        S_pred,
                        base_agg_hours=self.base_agg_hours,
                        skip_feedin=False,
                    )
                    for k, v in mS.items():
                        val = float(v) if np.isfinite(v) else float('nan')
                        rec[f"val_S_{k}"] = val
                        # Backward-compat alias
                        rec[f"val_S:{k}"] = val
                except Exception:
                    pass

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        if val_loss < (self.best_val_loss - 1e-6):
            self.best_val_loss = float(val_loss)
            self.best_state = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

        if do_bottom:
            rec['val_loss_bottom'] = float(val_loss)

        rec['epoch_wall_s'] = float(time.time() - t0)
        self.history.append(rec)
        return rec

    # ---------------- Train on Train+Val, then Test Eval (no Ray) ----------------
    def train_on_trainval(self, epochs: int, shuffle: bool = True) -> Dict[str, Any]:
        """Train the current model on the concatenation of train and val datasets for a
        specified number of epochs (pure fit, no validation). Keeps preprocessing
        and MAEx logic intact by reusing prepared datasets."""
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            raise RuntimeError("Datasets are not initialized; call setup() first or construct through Ray.")
        combo = ConcatDataset([self.train_dataset, self.val_dataset])  # type: ignore[arg-type]
        loader = DataLoader(
            combo,
            batch_size=int(self.cfg.get('batch_size', 8192)),
            shuffle=shuffle,
            num_workers=int(self.cfg.get('num_workers', 14)),
            pin_memory=bool(self.cfg.get('pin_memory', True)),
            drop_last=False,
        )

        self.model.train(True)
        train_losses: List[float] = []
        for ep in range(int(epochs)):
            total = 0.0
            n = 0
            for xb, yb, maex in loader:  # type: ignore
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                maex = maex.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = self._compute_loss(pred, yb, maex)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(self.cfg.get('grad_clip', 0.0) or 0.0)
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                total += float(loss.detach().item())
                n += 1
            avg = total / max(1, n)
            train_losses.append(avg)
        # Capture final weights as best_state for inference convenience
        self.best_state = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}
        return { 'epochs': int(epochs), 'final_loss': float(train_losses[-1]) if train_losses else float('nan') }

    def evaluate_on_test_with_plots(self, test_data_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate this model (using current or best_state) on a held-out test set.
        Expects an HDF5 with the same schema and keys used for training.
        test_data_cfg requires: hdf_data_path, key_X, key_y."""
        req = ['hdf_data_path', 'key_X', 'key_y']
        for k in req:
            if k not in test_data_cfg:
                raise ValueError(f"Missing test_data_cfg['{k}']")
        hdf_path = str(test_data_cfg['hdf_data_path'])
        key_X = str(test_data_cfg['key_X'])
        key_y = str(test_data_cfg['key_y'])
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(f"Test HDF not found: {hdf_path}")

        X_test_raw: pd.DataFrame = pd.read_hdf(hdf_path, key_X)
        y_test_raw: pd.DataFrame = pd.read_hdf(hdf_path, key_y)
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor is missing. Initialize via setup() with training data first.")
        X_test, y_test = self.preprocessor.transform(X_test_raw, y_test_raw)

        # Build tensors; reuse MAEx construction on the fly from scaled y
        X_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_t = torch.tensor(y_test[self.preprocessor.TARGET_COLS].values, dtype=torch.float32)
        # per-grid MAEx for each row
        def _maex_per_row(df_scaled: pd.DataFrame) -> np.ndarray:
            grouped = df_scaled.groupby(level=0).apply(lambda g: g.abs().mean())
            batches = df_scaled.index.get_level_values(0)
            return grouped.reindex(batches).values.astype('float32')
        test_maex = _maex_per_row(y_test)
        test_ds = TensorDataset(X_t, y_t, torch.tensor(test_maex, dtype=torch.float32))
        test_loader = DataLoader(test_ds, batch_size=int(self.cfg.get('batch_size', 8192)), shuffle=False)

        # Use the best_state if available
        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state, strict=False)
            except Exception:
                pass
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb, _ in test_loader:
                out = self.model(xb.to(self.device)).detach().cpu()
                preds.append(out)
                trues.append(yb.detach().cpu())
        y_s = torch.cat(trues).numpy()
        yp_s = torch.cat(preds).numpy()

        cols = self.preprocessor.TARGET_COLS[:y_s.shape[1]]
        y_df_s = pd.DataFrame(y_s, index=y_test.index, columns=cols)
        yp_df_s = pd.DataFrame(yp_s, index=y_test.index, columns=cols)
        y_df = self.preprocessor.inverse_transform_y(y_df_s)
        yp_df = self.preprocessor.inverse_transform_y(yp_df_s)
        y_df.index = y_df_s.index
        yp_df.index = yp_df_s.index

        ev = self.metrics
        results: Dict[str, Any] = {}
        # P
        col0 = cols[0]
        results['P'] = ev.run_model_evaluation(
            y_test_true=y_df[[col0]],
            y_test_pred=yp_df[[col0]],
            y_train_true=None,
            y_train_pred=None,
            no_plots=False,
            base_agg_hours=self.base_agg_hours,
        )
        # Q if present
        if len(cols) >= 2:
            col1 = cols[1]
            results['Q'] = ev.run_model_evaluation(
                y_test_true=y_df[[col1]],
                y_test_pred=yp_df[[col1]],
                y_train_true=None,
                y_train_pred=None,
                no_plots=False,
                base_agg_hours=self.base_agg_hours,
                skip_feedin_metrics=False,
            )
            # S
            S_true = np.sqrt(y_df.iloc[:, 0]**2 + y_df.iloc[:, 1]**2).to_frame(name='S')
            S_pred = np.sqrt(yp_df.iloc[:, 0]**2 + yp_df.iloc[:, 1]**2).to_frame(name='S')
            results['S'] = ev.run_model_evaluation(
                y_test_true=S_true,
                y_test_pred=S_pred,
                y_train_true=None,
                y_train_pred=None,
                no_plots=False,
                base_agg_hours=self.base_agg_hours,
            )
        return results

    def evaluate_on_test_mdape(self, test_data_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate this model on a held-out test set and report MdAPE metrics with percentile spreads.

        Uses EvaluationMetrics.evaluate_mdape which provides:
          - MdAPE (median absolute percentage error) replacing MAPE
          - +(p95-MdAPE) and -(MdAPE-p5) spreads
          - Mean Δt metrics also with percentile spreads if implemented in evaluation
        Prints a compact table per target similar to evaluate_on_test_with_plots.
        """
        req = ['hdf_data_path', 'key_X', 'key_y']
        for k in req:
            if k not in test_data_cfg:
                raise ValueError(f"Missing required test_data_cfg['{k}']")

        hdf_path = str(test_data_cfg['hdf_data_path'])
        key_X = str(test_data_cfg['key_X'])
        key_y = str(test_data_cfg['key_y'])
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(f"Test HDF not found: {hdf_path}")

        # Load and transform
        X_test_raw: pd.DataFrame = pd.read_hdf(hdf_path, key_X)
        y_test_raw: pd.DataFrame = pd.read_hdf(hdf_path, key_y)
        if not hasattr(self, 'preprocessor'):
            raise RuntimeError("Preprocessor is missing. Initialize via setup() with training data first.")
        X_test, y_test = self.preprocessor.transform(X_test_raw, y_test_raw)

        # Tensors and loader
        X_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_t = torch.tensor(y_test[self.preprocessor.TARGET_COLS].values, dtype=torch.float32)
        # Dummy MAEx not required for inference; keep dataset simple
        test_ds = TensorDataset(X_t, y_t)
        test_loader = DataLoader(test_ds, batch_size=int(self.cfg.get('batch_size', 8192)), shuffle=False)

        # Best weights if available
        self.model.eval()
        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state)
            except Exception:
                pass

        # Inference
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yp = self.model(xb)
                preds.append(yp.cpu())
                trues.append(yb.cpu())
        if not preds:
            return {}
        y_s = torch.cat(trues).numpy()
        yp_s = torch.cat(preds).numpy()

        # Back to original units
        cols = self.preprocessor.TARGET_COLS[:y_s.shape[1]]
        y_df_s = pd.DataFrame(y_s, index=y_test.index, columns=cols)
        yp_df_s = pd.DataFrame(yp_s, index=y_test.index, columns=cols)
        y_df = self.preprocessor.inverse_transform_y(y_df_s)
        yp_df = self.preprocessor.inverse_transform_y(yp_df_s)
        y_df.index = y_df_s.index
        yp_df.index = yp_df_s.index

        ev = self.metrics
        results: Dict[str, Any] = {}
        # P
        col0 = cols[0]
        results['P'] = ev.evaluate_mdape(
            y_true=y_df[[col0]],
            y_pred=yp_df[[col0]],
            base_agg_hours=self.base_agg_hours,
            skip_feedin=False,
        )
        # Q and S if present
        if len(cols) >= 2:
            col1 = cols[1]
            results['Q'] = ev.evaluate_mdape(
                y_true=y_df[[col1]],
                y_pred=yp_df[[col1]],
                base_agg_hours=self.base_agg_hours,
                skip_feedin=False,
            )
            S_true = np.sqrt(y_df.iloc[:, 0]**2 + y_df.iloc[:, 1]**2).to_frame(name='S')
            S_pred = np.sqrt(yp_df.iloc[:, 0]**2 + yp_df.iloc[:, 1]**2).to_frame(name='S')
            # For S we skip feed-in metrics (no positive/negative separation)
            results['S'] = ev.evaluate_mdape(
                y_true=S_true,
                y_pred=S_pred,
                base_agg_hours=self.base_agg_hours,
                skip_feedin=True,
            )

        # Pretty print similar to transformer
        try:
            print("MdAPE Metric Results (Test):")
            if 'P' in results:
                print('P:')
                print(pd.DataFrame(results['P'], index=['Test']).T)
            if 'Q' in results:
                print('Q:')
                print(pd.DataFrame(results['Q'], index=['Test']).T)
            if 'S' in results:
                print('S:')
                print(pd.DataFrame(results['S'], index=['Test']).T)
        except Exception:
            pass

        return results

    # ---------------- Checkpointing ----------------
    def save_checkpoint(self, checkpoint_dir: str):  # type: ignore[override]
        path = Path(checkpoint_dir) / 'state.pt'
        payload = {
            'config': self.cfg,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'preprocessor': self.preprocessor,
            'history': self.history,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_state': self.best_state,
            'y_train_index': self.y_train_index,
            'y_val_index': self.y_val_index,
        }
        torch.save(payload, path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: str):  # type: ignore[override]
        cp = Path(checkpoint_path)
        if cp.is_dir():
            cp = cp / 'state.pt'
        payload = torch.load(cp, map_location='cpu', weights_only=False)
        self.cfg = payload['config']
        self.best_val_loss = payload['best_val_loss']
        self.history = payload['history']
        self.current_epoch = payload['current_epoch']
        self.best_state = payload['best_state']
        self.y_train_index = payload['y_train_index']
        self.y_val_index = payload['y_val_index']
        self.preprocessor = payload['preprocessor']
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self.model.load_state_dict(payload['model_state'])
        self.optimizer.load_state_dict(payload['optimizer_state'])
        if self.scheduler and payload.get('scheduler_state') is not None:
            try:
                self.scheduler.load_state_dict(payload['scheduler_state'])
            except Exception:
                pass

    # ---------------- Evaluation helper ----------------
    def evaluate_best_with_plots(self, no_training_data: bool = False) -> Dict[str, Any]:
        self.model.eval()
        bs = int(self.cfg.get('batch_size', 4096))
        val_loader = DataLoader(self.val_dataset, batch_size=bs, shuffle=False)
        train_loader = None if no_training_data else DataLoader(self.train_dataset, batch_size=bs, shuffle=False)

        def _predict(loader):
            preds, ys = [], []
            with torch.no_grad():
                for xb, yb, _ in loader:  # type: ignore
                    out = self.model(xb.to(self.device)).detach().cpu()
                    preds.append(out)
                    ys.append(yb.detach().cpu())
            return torch.cat(ys).numpy(), torch.cat(preds).numpy()

        if train_loader is not None:
            y_train_s, y_trainp_s = _predict(train_loader)
        y_val_s, y_valp_s = _predict(val_loader)

        cols = self.preprocessor.TARGET_COLS[:y_val_s.shape[1]]
        y_val_df_s = pd.DataFrame(y_val_s, index=self.y_val_index, columns=cols)
        y_valp_df_s = pd.DataFrame(y_valp_s, index=self.y_val_index, columns=cols)
        y_val_df = self.preprocessor.inverse_transform_y(y_val_df_s)
        y_valp_df = self.preprocessor.inverse_transform_y(y_valp_df_s)
        y_val_df.index = y_val_df_s.index
        y_valp_df.index = y_valp_df_s.index

        if train_loader is not None:
            y_train_df_s = pd.DataFrame(y_train_s, index=self.y_train_index, columns=cols)
            y_trainp_df_s = pd.DataFrame(y_trainp_s, index=self.y_train_index, columns=cols)
            y_train_df = self.preprocessor.inverse_transform_y(y_train_df_s)
            y_trainp_df = self.preprocessor.inverse_transform_y(y_trainp_df_s)
            y_train_df.index = y_train_df_s.index
            y_trainp_df.index = y_trainp_df_s.index
        else:
            y_train_df = None
            y_trainp_df = None

        ev = self.metrics

        # Always evaluate and plot for first target (typically P)
        col0 = cols[0]
        print("\n=== Evaluation: P (active power) ===")
        res_P = ev.run_model_evaluation(
            y_test_true=y_val_df[[col0]],
            y_test_pred=y_valp_df[[col0]],
            y_train_true=None if y_train_df is None else y_train_df[[col0]],
            y_train_pred=None if y_trainp_df is None else y_trainp_df[[col0]],
            no_plots=False,
            base_agg_hours=self.base_agg_hours,
        )

        results: Dict[str, Any] = {"P": res_P}

        # If Q available, evaluate it as well
        if len(cols) >= 2:
            col1 = cols[1]
            print("\n=== Evaluation: Q (reactive power) ===")
            res_Q = ev.run_model_evaluation(
                y_test_true=y_val_df[[col1]],
                y_test_pred=y_valp_df[[col1]],
                y_train_true=None if y_train_df is None else y_train_df[[col1]],
                y_train_pred=None if y_trainp_df is None else y_trainp_df[[col1]],
                no_plots=False,
                base_agg_hours=self.base_agg_hours,
                skip_feedin_metrics=False,  # Q can be signed; skip feed-in specific metrics for clarity
            )
            results["Q"] = res_Q

            # Derived apparent power S
            S_true = np.sqrt(y_val_df.iloc[:, 0]**2 + y_val_df.iloc[:, 1]**2).to_frame(name='S')
            S_pred = np.sqrt(y_valp_df.iloc[:, 0]**2 + y_valp_df.iloc[:, 1]**2).to_frame(name='S')
            S_train_true = None
            S_train_pred = None
            if y_train_df is not None and y_trainp_df is not None:
                S_train_true = np.sqrt(y_train_df.iloc[:, 0]**2 + y_train_df.iloc[:, 1]**2).to_frame(name='S')
                S_train_pred = np.sqrt(y_trainp_df.iloc[:, 0]**2 + y_trainp_df.iloc[:, 1]**2).to_frame(name='S')
            print("\n=== Evaluation: S (apparent power) ===")
            res_S = ev.run_model_evaluation(
                y_test_true=S_true,
                y_test_pred=S_pred,
                y_train_true=S_train_true,
                y_train_pred=S_train_pred,
                no_plots=False,
                base_agg_hours=self.base_agg_hours,
            )
            results["S"] = res_S

        return results


# ========================= Inference wrapper =========================
class FullModel:
    """Inference wrapper bundling a fitted `Preprocessor` with a trained MLP.

    Use this for out-of-Ray prediction on raw (unscaled) feature tables.
    """

    def __init__(self, preprocessor: Preprocessor, model: nn.Module, device: Optional[str] = None):
        """Create an inference wrapper and move model to the requested device."""
        self.preprocessor = preprocessor
        self.model = model.eval()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, df_X: pd.DataFrame) -> pd.DataFrame:
        """Predict targets for the provided raw feature dataframe."""
        X_t = self.preprocessor.transform(df_X)
        X_t = torch.tensor(X_t.values, dtype=torch.float32, device=self.device)
        out = self.model(X_t).detach().cpu().numpy()
        cols = self.preprocessor.TARGET_COLS[:out.shape[1]]
        df_scaled = pd.DataFrame(out, index=df_X.index, columns=cols)
        df = self.preprocessor.inverse_transform_y(df_scaled)
        df.index = df_X.index
        return df

    @classmethod
    def from_trainer(cls, trainer: MLPTrainer, use_best: bool = True) -> 'FullModel':  # type: ignore[name-defined]
        """Create a `FullModel` from a trainer instance.

        Args:
            trainer: A configured/trained `MLPTrainer`.
            use_best: If True and `trainer.best_state` is available, reconstructs
                a fresh model and loads the best checkpoint weights.
        """
        if use_best and trainer.best_state is not None:
            if 'hidden_layers' in trainer.cfg:
                raise ValueError("'hidden_layers' is deprecated. Use 'num_layers' and 'layer_size' instead.")
            num_layers = int(trainer.cfg.get('num_layers', 4))
            layer_size = int(trainer.cfg.get('layer_size', 256))
            if num_layers < 1 or layer_size < 1:
                raise ValueError("Invalid model dimensions: num_layers and layer_size must be >= 1")
            hidden_layers = [layer_size] * num_layers
            activation = trainer.cfg.get('activation', 'relu')
            dropout = float(trainer.cfg.get('dropout', 0.0))
            model = MLP(trainer.input_dim, hidden_layers, trainer.output_dim, activation, dropout)
            model.load_state_dict(trainer.best_state)
        else:
            model = trainer.model
        return cls(trainer.preprocessor, model, device=str(trainer.device))

def train_on_trainval_and_eval_test(
    config: Dict[str, Any],
    trainval_epochs: int,
    test_data_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """End-to-end runner for the MLP without Ray orchestration.
    Steps:
      1) Build an MLPTrainer-like object and call setup(config) to prepare data/model.
      2) Optionally perform a brief warmup using step() loops if config['epochs'] > 0.
      3) Train on train+val for trainval_epochs.
      4) Evaluate on the given test set with plots.
    Returns the test metrics dict.
    """
    # Instantiate with config so Trainable passes it to setup() during __init__
    trainer = MLPTrainer(config)  # type: ignore[arg-type]
    # Optional warmup on the original train/val split
    warmup = int(config.get('epochs', 0))
    for _ in range(max(0, warmup)):
        trainer.step()
    # Train on concatenated train+val
    trainer.train_on_trainval(int(trainval_epochs))
    trainer.evaluate_on_test_with_plots(test_data_cfg)
    return trainer


__all__ = ['Preprocessor', 'MLP', 'MLPTrainer', 'FullModel', 'train_on_trainval_and_eval_test']
    
# ========================= Utility loss (reuse) =========================
