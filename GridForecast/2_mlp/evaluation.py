"""Evaluation utilities for GridForecast models.

This module provides a collection of metrics used to evaluate forecasts on a
per-grid (batch) basis, including normalized errors (MAE/MAEx), peak metrics,
and phase-angle metrics for the (P, Q) two-target case.

Most metric functions expect inputs as `pandas.DataFrame` objects indexed by a
`MultiIndex` with a `batch` level (grid id) and usually an `hour` level.
"""

import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import torch


import concurrent.futures

from scipy.stats import norm, halfnorm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset  # local import for clarity

import concurrent.futures
import matplotlib.pyplot as plt
from IPython.display import display

class EvaluationMetrics:
    """Metric computation helper used by both MLP and Transformer trainers.

    The class is intentionally mostly stateless: metric methods are implemented
    as `@staticmethod` so they can be called without keeping internal state.
    """

    def __init__(self):
        """Create an EvaluationMetrics instance."""
        pass

    ############################################################################################################
    ############################################ Evaluation metrics ############################################
    ############################################################################################################
    # Losses
    @staticmethod
    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R^2 safely without exceptions:
        - Aligns and filters to finite values
        - Returns NaN if <2 points or variance of y_true == 0
        """
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[m]
        yp = yp[m]
        if yt.size < 2:
            return float("nan")
        denom = np.sum((yt - yt.mean())**2)
        if denom == 0:
            return float("nan")
        ss_res = np.sum((yt - yp)**2)
        return float(1.0 - ss_res/denom)

    @staticmethod
    def _align_angles_for_regression(true_deg: np.ndarray, pred_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Align predicted angles to true by adding multiples of 360 so that the
        pointwise difference lies in [-180, 180]. This reduces wrap-around issues
        before computing R^2 on angles.
        """
        a = np.asarray(true_deg, dtype=float).reshape(-1)
        b = np.asarray(pred_deg, dtype=float).reshape(-1)
        k = np.rint((a - b) / 360.0)  # nearest integer multiples
        b_aligned = b + 360.0 * k
        return a, b_aligned

    @staticmethod
    def _angles_deg_from_pq(p_df: pd.DataFrame, q_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute phase angle of apparent power S = P + jQ, in degrees, for matching P and Q DataFrames.
        Returns a single-column DataFrame with the same index.
        """
        if p_df.shape[1] != 1 or q_df.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames for P and Q.")
        # Align indices
        if not p_df.index.equals(q_df.index):
            q_df = q_df.reindex(p_df.index)
        p = p_df.iloc[:, 0].astype(float).to_numpy()
        q = q_df.iloc[:, 0].astype(float).to_numpy()
        angles = np.degrees(np.arctan2(q, p))  # range (-180, 180]
        return pd.DataFrame(angles, index=p_df.index, columns=["angle_deg"]) 

    @staticmethod
    def _angle_abs_diff_deg(true_deg: pd.Series | np.ndarray, pred_deg: pd.Series | np.ndarray) -> np.ndarray:
        """
        Minimal absolute angular difference in degrees in [0, 180]. Works with arrays/Series.
        Computes wrap-around using modulo 360.
        """
        a = np.asarray(true_deg, dtype=float)
        b = np.asarray(pred_deg, dtype=float)
        d = b - a
        d_wrapped = (d + 180.0) % 360.0 - 180.0
        return np.abs(d_wrapped)

    @staticmethod
    def mean_phase_angle_abs_error_deg(y_true_p: pd.DataFrame,
                                       y_true_q: pd.DataFrame,
                                       y_pred_p: pd.DataFrame,
                                       y_pred_q: pd.DataFrame) -> float:
        """
        Mean absolute error of apparent power angle (deg) averaged over batches.
        1) Compute angles φ_true=atan2(Q_true,P_true), φ_pred=atan2(Q_pred,P_pred) in degrees
        2) Angular abs error uses wrap-around to keep within [0,180]
        3) Mean per batch, then average across batches
        """
        ang_true = EvaluationMetrics._angles_deg_from_pq(y_true_p, y_true_q)
        ang_pred = EvaluationMetrics._angles_deg_from_pq(y_pred_p, y_pred_q)
        # Align
        if not ang_true.index.equals(ang_pred.index):
            ang_pred = ang_pred.reindex(ang_true.index)

        abs_err = EvaluationMetrics._angle_abs_diff_deg(
            ang_true.iloc[:, 0].values, ang_pred.iloc[:, 0].values
        )
        # Group by batch and average
        if not (isinstance(ang_true.index, pd.MultiIndex) and 'batch' in ang_true.index.names):
            # If no batches, return global mean
            return float(np.nanmean(abs_err))
        # attach to Series for groupby
        err_series = pd.Series(abs_err, index=ang_true.index)
        per_batch = err_series.groupby(level='batch').mean()
        return float(per_batch.mean())

    @staticmethod
    def mean_mse_per_mean_absolute_exchange(y_true, y_pred):
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_true.columns[0]
        df = pd.concat([
            y_true.rename(columns={col: "true"}),
            y_pred.rename(columns={col: "pred"})
            ], axis=1)

        # 1. Compute for each batch: (MSE / mean_abs_true)
        rel_mse_per_batch = (
            df.groupby(level="batch").apply(lambda g: ((g["true"] - g["pred"])**2).mean() / g["true"].abs().mean())
        )

        # 2. Final metric: mean over batches
        normalized_mse = rel_mse_per_batch.mean()
        return normalized_mse

    @staticmethod
    def mean_rmse_per_mean_absolute_exchange(y_true, y_pred):
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_true.columns[0]
        df = pd.concat([
            y_true.rename(columns={col: "true"}),
            y_pred.rename(columns={col: "pred"})
            ], axis=1)

        # 1. Compute for each batch: (RMSE / mean_abs_true)
        rel_rmse_per_batch = (
            df.groupby(level="batch").apply(lambda g: ((g["true"] - g["pred"])**2).mean()**0.5 / g["true"].abs().mean())
        )

        # 2. Final metric: mean over batches
        normalized_rmse = rel_rmse_per_batch.mean()
        return normalized_rmse

    @staticmethod
    def mean_mae_per_mean_absolute_exchange(y_true, y_pred):
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_true.columns[0]
        df = pd.concat([
            y_true.rename(columns={col: "true"}),
            y_pred.rename(columns={col: "pred"})
            ], axis=1)

        # 1. Compute for each batch: (MAE / mean_abs_true)
        rel_mae_per_batch = (
            df.groupby(level="batch").apply(lambda g: (g["true"] - g["pred"]).abs().mean() / g["true"].abs().mean())
        )

        # 2. Final metric: mean over batches
        normalized_mae = rel_mae_per_batch.mean()
        return normalized_mae
    
    @staticmethod
    def mean95_mae_per_mean_absolute_exchange(y_true, y_pred):
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_true.columns[0]
        df = pd.concat(
            [
                y_true.rename(columns={col: "true"}),
                y_pred.rename(columns={col: "pred"})
            ],
            axis=1
        )

        # Per-sample absolute errors and absolute true
        abs_err = (df["true"] - df["pred"]).abs()
        abs_true = df["true"].abs()

        # Per-batch mean absolute error and mean absolute true
        mae_per_batch = abs_err.groupby(level="batch").mean()
        mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()

        # Normalized MAE per batch
        rel_mae_per_batch = mae_per_batch / mean_abs_true_per_batch

        if len(rel_mae_per_batch) == 0:
            return float("nan")

        # 95% best (lowest) batches
        q95 = rel_mae_per_batch.quantile(0.95)
        best_95 = rel_mae_per_batch[rel_mae_per_batch <= q95]

        # Fallback: if filtering removed all (edge numeric issues), use all
        if best_95.empty:
            best_95 = rel_mae_per_batch

        return best_95.mean()
    
    @staticmethod
    def median_mae_per_mean_absolute_exchange(y_true, y_pred):
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_true.columns[0]
        df = pd.concat(
            [
                y_true.rename(columns={col: "true"}),
                y_pred.rename(columns={col: "pred"})
            ],
            axis=1
        )

        if df.empty:
            return float("nan")

        abs_err = (df["true"] - df["pred"]).abs()
        abs_true = df["true"].abs()

        mae_per_batch = abs_err.groupby(level="batch").mean()
        mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()

        rel_mae_per_batch = mae_per_batch / mean_abs_true_per_batch

        if len(rel_mae_per_batch) == 0:
            return float("nan")

        return rel_mae_per_batch.median()

    @staticmethod
    def mean_absolute_peak_demand_percentage_error(y_true, y_pred):
        col = y_true.columns[0]
        agg_true = y_true.groupby(level='batch')[col].agg([("max", "max")])
        agg_pred = y_pred.groupby(level='batch')[y_pred.columns[0]].agg([("max", "max")])

        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(agg_true, agg_pred)
        return mape
    
    @staticmethod
    def mean_absolute_peak_feedin_percentage_error(y_true, y_pred):
        col = y_true.columns[0]
        agg_true = y_true.groupby(level='batch')[col].agg([("min", "min")])
        agg_pred = y_pred.groupby(level='batch')[y_pred.columns[0]].agg([("min", "min")])

        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(agg_true, agg_pred)
        return mape
    
    @staticmethod
    def mean_absolute_aggregated_demand_percentage_error(y_true, y_pred):
        col = y_true.columns[0]
        agg_true = y_true.groupby(level='batch')[col].agg([("positive_sum", lambda x: x[x>0].sum())])
        agg_pred = y_pred.groupby(level='batch')[y_pred.columns[0]].agg([("positive_sum", lambda x: x[x>0].sum())])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(agg_true, agg_pred)
        return mape

    @staticmethod
    def mean_absolute_aggregated_feedin_percentage_error(y_true, y_pred):
        col = y_true.columns[0]
        agg_true = y_true.groupby(level='batch')[col].agg([("negative_sum", lambda x: x[x<0].sum())])
        agg_pred = y_pred.groupby(level='batch')[y_pred.columns[0]].agg([("negative_sum", lambda x: x[x<0].sum())])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = mean_absolute_percentage_error(agg_true, agg_pred)
        return mape

    @staticmethod
    def mean_peak_demand_time_dist(y_true, y_pred):
        n_timesteps = len(y_true.index.get_level_values(1).unique())
        true_peak_hours = y_true[y_true.columns[0]].groupby(level='batch').idxmax().map(lambda x: x[1])
        pred_peak_hours = y_pred[y_pred.columns[0]].groupby(level='batch').idxmax().map(lambda x: x[1])
        
        diff = np.abs(true_peak_hours - pred_peak_hours)
        circ_dist = np.minimum(diff, n_timesteps - diff)
        # Calculate the average absolute difference in hours
        return np.mean(circ_dist)
    
    @staticmethod
    def mean_peak_feedin_time_dist(y_true, y_pred, n_timesteps=8760):
        n_timesteps = len(y_true.index.get_level_values(1).unique())
        true_peak_hours = y_true[y_true.columns[0]].groupby(level='batch').idxmin().map(lambda x: x[1])
        pred_peak_hours = y_pred[y_pred.columns[0]].groupby(level='batch').idxmin().map(lambda x: x[1])
        
        diff = np.abs(true_peak_hours - pred_peak_hours)
        circ_dist = np.minimum(diff, n_timesteps - diff)
        # Calculate the average absolute difference in hours
        return np.mean(circ_dist)
        
    @staticmethod
    def reaggregate_n_steps(y_true, y_pred, n_agg):
        """Compute aggregated sums over n_steps consecutive steps."""
        if n_agg == 1:
            return y_true, y_pred

        ### Ensure Formatting
        if y_true.shape[1] != 1 or y_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        if not y_true.index.equals(y_pred.index):
            y_pred = y_pred.reindex(y_true.index)
        if not (isinstance(y_true.index, pd.MultiIndex) and 'batch' in y_true.index.names):
            raise ValueError("Expected MultiIndex with level name 'batch' and an hourly level (e.g. 'hour').")
        if y_true.index.nlevels < 2:
            raise ValueError("Need at least two index levels (batch, hour).")

        ### Reagregate
        # Identify hour level (fallback to position 1 if not explicitly named 'hour')
        if 'hour' in y_true.index.names: hour_vals = y_true.index.get_level_values('hour')
        else: hour_vals = y_true.index.get_level_values(1)
        agg_vals = (hour_vals // n_agg).astype(int)
        batch_vals = y_true.index.get_level_values('batch')
        # Reindex to (batch, day) then aggregate by sum
        new_index = pd.MultiIndex.from_arrays([batch_vals, agg_vals], names=['batch', 'agg'])
        y_true_n_steps = y_true.copy()
        y_pred_n_steps = y_pred.copy()
        y_true_n_steps.index = new_index
        y_pred_n_steps.index = new_index
        y_true_n_steps = y_true_n_steps.groupby(level=['batch', 'agg']).sum()
        y_pred_n_steps = y_pred_n_steps.groupby(level=['batch', 'agg']).sum()
        return y_true_n_steps, y_pred_n_steps


    ### Evaluation
    def evaluate(self,
                 y_true,
                 y_pred,
                 base_agg_hours: int = 1,
                 skip_feedin: bool = False,
                 y_true_reactive: pd.DataFrame | None = None,
                 y_pred_reactive: pd.DataFrame | None = None):
        ### Ensure formatting
        # Ensure DataFrames (the trainer passes DataFrames already)
        if isinstance(y_true, (pd.Series, np.ndarray)):
            y_true = pd.DataFrame(y_true)
        if isinstance(y_pred, (pd.Series, np.ndarray)):
            y_pred = pd.DataFrame(y_pred)
        # Align indices (just in case) and filter any non-finite rows to avoid sklearn ValueError
        if not y_true.index.equals(y_pred.index):
            y_pred = y_pred.reindex(y_true.index)

        has_reactive = (y_true_reactive is not None) and (y_pred_reactive is not None)
        if has_reactive:
            # Ensure DataFrames
            if isinstance(y_true_reactive, (pd.Series, np.ndarray)):
                y_true_reactive = pd.DataFrame(y_true_reactive)
            if isinstance(y_pred_reactive, (pd.Series, np.ndarray)):
                y_pred_reactive = pd.DataFrame(y_pred_reactive)
            if not y_true.index.equals(y_true_reactive.index):
                y_true_reactive = y_true_reactive.reindex(y_true.index)
            if not y_true.index.equals(y_pred_reactive.index):
                y_pred_reactive = y_pred_reactive.reindex(y_true.index)

        ### Reaggregate to coarser levels and compute metrics there
        candidate_aggs = [1,2,4,12,24]
        metrics = {}
        for target_agg in candidate_aggs:
            if target_agg < base_agg_hours: continue       # only larger aggregations
            if target_agg % base_agg_hours != 0: continue  # only multiples of base
            steps = target_agg // base_agg_hours           # Number of base steps to aggregate
            try: # Reaggregate by 'steps' over existing aggregated series (each row already base_agg_hours hours)
                y_true_aggr, y_pred_aggr = self.reaggregate_n_steps(y_true, y_pred, n_agg=steps)
                if target_agg == base_agg_hours:
                    metrics["Agg. Import MAPE"] = self.mean_absolute_aggregated_demand_percentage_error(y_true_aggr, y_pred_aggr)
                    if not skip_feedin:
                        metrics["Agg. Feed-In MAPE"] = self.mean_absolute_aggregated_feedin_percentage_error(y_true_aggr, y_pred_aggr)
                metrics[f"MA(E/MAEx) ({target_agg}h)"] = self.mean_mae_per_mean_absolute_exchange(y_true_aggr, y_pred_aggr)
                metrics[f"Peak Import MAPE ({target_agg}h)"] = self.mean_absolute_peak_demand_percentage_error(y_true_aggr, y_pred_aggr)
                if not skip_feedin:
                    metrics[f"Peak Feed-In MAPE ({target_agg}h)"] = self.mean_absolute_peak_feedin_percentage_error(y_true_aggr, y_pred_aggr)
                metrics[f"Peak Import Mean Δt ({target_agg}h)"] = base_agg_hours*steps*self.mean_peak_demand_time_dist(y_true_aggr, y_pred_aggr)
                if not skip_feedin:
                    metrics[f"Peak Feed-In Mean Δt ({target_agg}h)"] = base_agg_hours*steps*self.mean_peak_feedin_time_dist(y_true_aggr, y_pred_aggr)

                # Phase angle MAE (deg) if reactive power is provided
                if has_reactive:
                    y_true_q_aggr, y_pred_q_aggr = self.reaggregate_n_steps(y_true_reactive, y_pred_reactive, n_agg=steps)
                    metrics[f"Phase Angle MAE (deg) ({target_agg}h)"] = self.mean_phase_angle_abs_error_deg(
                        y_true_aggr, y_true_q_aggr, y_pred_aggr, y_pred_q_aggr
                    )

                    # Phase Angle R^2 across time within each batch, averaged over batches
                    ang_true = self._angles_deg_from_pq(y_true_aggr, y_true_q_aggr)
                    ang_pred = self._angles_deg_from_pq(y_pred_aggr, y_pred_q_aggr)
                    if not ang_true.index.equals(ang_pred.index):
                        ang_pred = ang_pred.reindex(ang_true.index)
                    # compute per-batch R^2 then average (drop NaN batches)
                    if isinstance(ang_true.index, pd.MultiIndex) and 'batch' in ang_true.index.names:
                        r2_vals = []
                        for b, gt in ang_true.groupby(level='batch'):
                            gp = ang_pred.xs(b, level='batch')
                            a_true, a_pred = self._align_angles_for_regression(gt.iloc[:,0].values, gp.iloc[:,0].values)
                            r2_b = self._safe_r2(a_true, a_pred)
                            if np.isfinite(r2_b):
                                r2_vals.append(r2_b)
                        metrics[f"Phase Angle R^2 ({target_agg}h)"] = float(np.mean(r2_vals)) if len(r2_vals)>0 else float("nan")
            except Exception as e:  # pragma: no cover
                print(f"[EvaluationMetrics] Skipped aggregation {target_agg}h due to error: {e}")

            # Additional R^2 metrics (grid-aggregated) per aggregation level
            try:
                # Peak value R^2 across batches
                col = y_true.columns[0]
                agg_true_max = y_true_aggr.groupby(level='batch')[col].max()
                agg_pred_max = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].max()
                metrics[f"Peak Import R^2 ({target_agg}h)"] = self._safe_r2(agg_true_max.values, agg_pred_max.values)
                if not skip_feedin:
                    agg_true_min = y_true_aggr.groupby(level='batch')[col].min()
                    agg_pred_min = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].min()
                    metrics[f"Peak Feed-In R^2 ({target_agg}h)"] = self._safe_r2(agg_true_min.values, agg_pred_min.values)

                # Peak time R^2 across batches (with wrap handling)
                # True times
                true_peak_hours = y_true_aggr[col].groupby(level='batch').idxmax().map(lambda x: x[1])
                pred_peak_hours = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmax().map(lambda x: x[1])
                # Align times and handle wrap by appending overhang relative to true
                t_true_arr = true_peak_hours.values.astype(float)
                t_pred_adj = self.append_overhang(pred_peak_hours.values, true_peak_hours.values, period=len(y_true_aggr.index.get_level_values(1).unique()))
                # Scale to hours at this aggregation granularity
                scale = base_agg_hours*steps
                metrics[f"Peak Import Δt R^2 ({target_agg}h)"] = self._safe_r2(scale*t_true_arr, scale*t_pred_adj.values)

                if not skip_feedin:
                    true_peak_min = y_true_aggr[col].groupby(level='batch').idxmin().map(lambda x: x[1])
                    pred_peak_min = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmin().map(lambda x: x[1])
                    t_true_min = true_peak_min.values.astype(float)
                    t_pred_min_adj = self.append_overhang(pred_peak_min.values, true_peak_min.values, period=len(y_true_aggr.index.get_level_values(1).unique()))
                    metrics[f"Peak Feed-In Δt R^2 ({target_agg}h)"] = self._safe_r2(scale*t_true_min, scale*t_pred_min_adj.values)
            except Exception as e:  # pragma: no cover
                print(f"[EvaluationMetrics] Skipped R^2 add-ons for {target_agg}h due to error: {e}")

            # Base-aggregation-only R^2 for aggregated import/feed-in sums (to mirror MAPE)
            if target_agg == base_agg_hours:
                try:
                    col = y_true.columns[0]
                    # Import (sum of positives)
                    agg_true_imp = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x>0].sum())
                    agg_pred_imp = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].apply(lambda x: x[x>0].sum())
                    metrics["Agg. Import R^2"] = self._safe_r2(agg_true_imp.values, agg_pred_imp.values)
                    if not skip_feedin:
                        # Feed-in (sum of negatives)
                        agg_true_fe = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x<0].sum())
                        agg_pred_fe = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].apply(lambda x: x[x<0].sum())
                        metrics["Agg. Feed-In R^2"] = self._safe_r2(agg_true_fe.values, agg_pred_fe.values)
                except Exception as e:  # pragma: no cover
                    print(f"[EvaluationMetrics] Skipped aggregated import/feed-in R^2 at {target_agg}h due to error: {e}")
        return metrics

    ############################################################################################################
    ######################################## MdAPE Evaluation #################################################
    ############################################################################################################
    def evaluate_mdape(self,
                       y_true,
                       y_pred,
                       base_agg_hours: int = 1,
                       skip_feedin: bool = False,
                       y_true_reactive: pd.DataFrame | None = None,
                       y_pred_reactive: pd.DataFrame | None = None):
        """Same as evaluate() but replaces MAPE metrics with MdAPE (median absolute percentage error)
        and adds asymmetric percentile spreads (+(p95-MdAPE), -(MdAPE-p5)). Other metrics unchanged.

        Keys replaced:
          Agg. Import MAPE -> Agg. Import MdAPE (+/- spread rows)
          Agg. Feed-In MAPE -> Agg. Feed-In MdAPE (+/- ...)
          Peak Import/Feed-In MAPE (Xh) -> Peak Import/Feed-In MdAPE (Xh) (+/- ...)
        """
        # --- Helper to compute median APE + spreads on paired arrays ---
        def _mdape_spreads(true_arr: np.ndarray, pred_arr: np.ndarray, epsilon: float = 1e-8):
            true_arr = np.asarray(true_arr, dtype=float)
            pred_arr = np.asarray(pred_arr, dtype=float)
            if true_arr.size == 0:
                return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            denom = np.maximum(np.abs(true_arr), epsilon)
            ape = np.abs(true_arr - pred_arr) / denom
            p5, p50, p95 = np.percentile(ape, [5, 50, 95])
            return p50, (p95 - p50), (p50 - p5), p5, p95, ape.mean()

        # Ensure DataFrames
        if isinstance(y_true, (pd.Series, np.ndarray)):
            y_true = pd.DataFrame(y_true)
        if isinstance(y_pred, (pd.Series, np.ndarray)):
            y_pred = pd.DataFrame(y_pred)
        if not y_true.index.equals(y_pred.index):
            y_pred = y_pred.reindex(y_true.index)

        has_reactive = (y_true_reactive is not None) and (y_pred_reactive is not None)
        if has_reactive:
            if isinstance(y_true_reactive, (pd.Series, np.ndarray)):
                y_true_reactive = pd.DataFrame(y_true_reactive)
            if isinstance(y_pred_reactive, (pd.Series, np.ndarray)):
                y_pred_reactive = pd.DataFrame(y_pred_reactive)
            if not y_true.index.equals(y_true_reactive.index):
                y_true_reactive = y_true_reactive.reindex(y_true.index)
            if not y_true.index.equals(y_pred_reactive.index):
                y_pred_reactive = y_pred_reactive.reindex(y_true.index)

        candidate_aggs = [1,2,4,12,24]
        metrics: dict[str, float] = {}
        for target_agg in candidate_aggs:
            if target_agg < base_agg_hours: continue
            if target_agg % base_agg_hours != 0: continue
            steps = target_agg // base_agg_hours
            try:
                y_true_aggr, y_pred_aggr = self.reaggregate_n_steps(y_true, y_pred, n_agg=steps)
                col = y_true_aggr.columns[0]
                # Base aggregation aggregated demand/feed-in MdAPE
                if target_agg == base_agg_hours:
                    # Aggregated demand (sum of positives) per batch
                    true_imp = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x>0].sum())
                    pred_imp = y_pred_aggr.groupby(level='batch')[col].apply(lambda x: x[x>0].sum())
                    md, plus, minus, p5, p95, mean_ape = _mdape_spreads(true_imp.values, pred_imp.values)
                    metrics["Agg. Import MdAPE"] = md
                    metrics["Agg. Import MdAPE +(p95-MdAPE)"] = plus
                    metrics["Agg. Import MdAPE -(MdAPE-p5)"] = minus
                    if not skip_feedin:
                        true_fe = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x<0].sum())
                        pred_fe = y_pred_aggr.groupby(level='batch')[col].apply(lambda x: x[x<0].sum())
                        md, plus, minus, p5, p95, mean_ape = _mdape_spreads(true_fe.values, pred_fe.values)
                        metrics["Agg. Feed-In MdAPE"] = md
                        metrics["Agg. Feed-In MdAPE +(p95-MdAPE)"] = plus
                        metrics["Agg. Feed-In MdAPE -(MdAPE-p5)"] = minus

                # Normalized MAE metrics: keep mean and add median + percentile spreads
                metrics[f"MA(E/MAEx) ({target_agg}h)"] = self.mean_mae_per_mean_absolute_exchange(y_true_aggr, y_pred_aggr)
                # Compute per-batch normalized MAE distribution
                col = y_true_aggr.columns[0]
                df_tmp = pd.concat(
                    [y_true_aggr.rename(columns={col: "true"}), y_pred_aggr.rename(columns={y_pred_aggr.columns[0]: "pred"})],
                    axis=1
                )
                abs_err = (df_tmp["true"] - df_tmp["pred"]).abs()
                abs_true = df_tmp["true"].abs()
                mae_per_batch = abs_err.groupby(level="batch").mean()
                mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()
                rel_mae_per_batch = (mae_per_batch / mean_abs_true_per_batch).to_numpy()
                if rel_mae_per_batch.size > 0:
                    p5_mae, p50_mae, p95_mae = np.percentile(rel_mae_per_batch, [5, 50, 95])
                    metrics[f"MA(E/MAEx) Md ({target_agg}h)"] = float(p50_mae)
                    metrics[f"MA(E/MAEx) Md +(p95-Md) ({target_agg}h)"] = float(p95_mae - p50_mae)
                    metrics[f"MA(E/MAEx) Md -(Md-p5) ({target_agg}h)"] = float(p50_mae - p5_mae)

                # Peak Import MdAPE
                true_peak_imp = y_true_aggr.groupby(level='batch')[col].max()
                pred_peak_imp = y_pred_aggr.groupby(level='batch')[col].max()
                md, plus, minus, p5, p95, _ = _mdape_spreads(true_peak_imp.values, pred_peak_imp.values)
                metrics[f"Peak Import MdAPE ({target_agg}h)"] = md
                metrics[f"Peak Import MdAPE +(p95-MdAPE) ({target_agg}h)"] = plus
                metrics[f"Peak Import MdAPE -(MdAPE-p5) ({target_agg}h)"] = minus
                if not skip_feedin:
                    true_peak_fe = y_true_aggr.groupby(level='batch')[col].min()
                    pred_peak_fe = y_pred_aggr.groupby(level='batch')[col].min()
                    md, plus, minus, p5, p95, _ = _mdape_spreads(true_peak_fe.values, pred_peak_fe.values)
                    metrics[f"Peak Feed-In MdAPE ({target_agg}h)"] = md
                    metrics[f"Peak Feed-In MdAPE +(p95-MdAPE) ({target_agg}h)"] = plus
                    metrics[f"Peak Feed-In MdAPE -(MdAPE-p5) ({target_agg}h)"] = minus

                # Peak time median absolute distance + percentile spreads (hours)
                # Import peak time distance distribution per batch
                n_steps = len(y_true_aggr.index.get_level_values(1).unique())
                scale_h = base_agg_hours * steps
                true_peak_hours = y_true_aggr[col].groupby(level='batch').idxmax().map(lambda x: x[1]).astype(float).values
                pred_peak_hours = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmax().map(lambda x: x[1]).astype(float).values
                diff = np.abs(true_peak_hours - pred_peak_hours)
                circ_dist = np.minimum(diff, n_steps - diff) * scale_h
                if circ_dist.size:
                    p5_dt, p50_dt, p95_dt = np.percentile(circ_dist, [5, 50, 95])
                    median_dt_imp = float(p50_dt)
                    plus_dt = float(p95_dt - p50_dt)
                    minus_dt = float(p50_dt - p5_dt)
                else:
                    median_dt_imp = float('nan')
                    plus_dt = minus_dt = float('nan')
                metrics[f"Peak Import Median Δt ({target_agg}h)"] = median_dt_imp
                metrics[f"Peak Import Median Δt +(p95-Md) ({target_agg}h)"] = plus_dt
                metrics[f"Peak Import Median Δt -(Md-p5) ({target_agg}h)"] = minus_dt
                if not skip_feedin:
                    true_peak_hours_fe = y_true_aggr[col].groupby(level='batch').idxmin().map(lambda x: x[1]).astype(float).values
                    pred_peak_hours_fe = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmin().map(lambda x: x[1]).astype(float).values
                    diff_fe = np.abs(true_peak_hours_fe - pred_peak_hours_fe)
                    circ_dist_fe = np.minimum(diff_fe, n_steps - diff_fe) * scale_h
                    if circ_dist_fe.size:
                        p5_dt_fe, p50_dt_fe, p95_dt_fe = np.percentile(circ_dist_fe, [5, 50, 95])
                        median_dt_fe = float(p50_dt_fe)
                        plus_dt_fe = float(p95_dt_fe - p50_dt_fe)
                        minus_dt_fe = float(p50_dt_fe - p5_dt_fe)
                    else:
                        median_dt_fe = float('nan')
                        plus_dt_fe = minus_dt_fe = float('nan')
                    metrics[f"Peak Feed-In Median Δt ({target_agg}h)"] = median_dt_fe
                    metrics[f"Peak Feed-In Median Δt +(p95-Md) ({target_agg}h)"] = plus_dt_fe
                    metrics[f"Peak Feed-In Median Δt -(Md-p5) ({target_agg}h)"] = minus_dt_fe

                # Phase angle metrics (reuse existing) if reactive provided
                if has_reactive:
                    y_true_q_aggr, y_pred_q_aggr = self.reaggregate_n_steps(y_true_reactive, y_pred_reactive, n_agg=steps)  # type: ignore[arg-type]
                    metrics[f"Phase Angle MAE (deg) ({target_agg}h)"] = self.mean_phase_angle_abs_error_deg(
                        y_true_aggr, y_true_q_aggr, y_pred_aggr, y_pred_q_aggr
                    )
                    ang_true = self._angles_deg_from_pq(y_true_aggr, y_true_q_aggr)
                    ang_pred = self._angles_deg_from_pq(y_pred_aggr, y_pred_q_aggr)
                    if not ang_true.index.equals(ang_pred.index):
                        ang_pred = ang_pred.reindex(ang_true.index)
                    if isinstance(ang_true.index, pd.MultiIndex) and 'batch' in ang_true.index.names:
                        r2_vals = []
                        for b, gt in ang_true.groupby(level='batch'):
                            gp = ang_pred.xs(b, level='batch')
                            a_true, a_pred = self._align_angles_for_regression(gt.iloc[:,0].values, gp.iloc[:,0].values)
                            r2_b = self._safe_r2(a_true, a_pred)
                            if np.isfinite(r2_b):
                                r2_vals.append(r2_b)
                        metrics[f"Phase Angle R^2 ({target_agg}h)"] = float(np.mean(r2_vals)) if len(r2_vals)>0 else float("nan")
            except Exception as e:  # pragma: no cover
                print(f"[EvaluationMetrics] MdAPE: Skipped aggregation {target_agg}h due to error: {e}")
                continue

            # R^2 add-ons (unchanged)
            try:
                col = y_true_aggr.columns[0]
                agg_true_max = y_true_aggr.groupby(level='batch')[col].max()
                agg_pred_max = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].max()
                metrics[f"Peak Import R^2 ({target_agg}h)"] = self._safe_r2(agg_true_max.values, agg_pred_max.values)
                if not skip_feedin:
                    agg_true_min = y_true_aggr.groupby(level='batch')[col].min()
                    agg_pred_min = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].min()
                    metrics[f"Peak Feed-In R^2 ({target_agg}h)"] = self._safe_r2(agg_true_min.values, agg_pred_min.values)
                true_peak_hours = y_true_aggr[col].groupby(level='batch').idxmax().map(lambda x: x[1])
                pred_peak_hours = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmax().map(lambda x: x[1])
                t_true_arr = true_peak_hours.values.astype(float)
                t_pred_adj = self.append_overhang(pred_peak_hours.values, true_peak_hours.values, period=len(y_true_aggr.index.get_level_values(1).unique()))
                scale = base_agg_hours*steps
                metrics[f"Peak Import Δt R^2 ({target_agg}h)"] = self._safe_r2(scale*t_true_arr, scale*t_pred_adj.values)
                if not skip_feedin:
                    true_peak_min = y_true_aggr[col].groupby(level='batch').idxmin().map(lambda x: x[1])
                    pred_peak_min = y_pred_aggr[y_pred_aggr.columns[0]].groupby(level='batch').idxmin().map(lambda x: x[1])
                    t_true_min = true_peak_min.values.astype(float)
                    t_pred_min_adj = self.append_overhang(pred_peak_min.values, true_peak_min.values, period=len(y_true_aggr.index.get_level_values(1).unique()))
                    metrics[f"Peak Feed-In Δt R^2 ({target_agg}h)"] = self._safe_r2(scale*t_true_min, scale*t_pred_min_adj.values)
            except Exception as e:  # pragma: no cover
                print(f"[EvaluationMetrics] MdAPE: Skipped R^2 add-ons for {target_agg}h due to error: {e}")

            if target_agg == base_agg_hours:
                try:
                    col = y_true_aggr.columns[0]
                    agg_true_imp = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x>0].sum())
                    agg_pred_imp = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].apply(lambda x: x[x>0].sum())
                    metrics["Agg. Import R^2"] = self._safe_r2(agg_true_imp.values, agg_pred_imp.values)
                    if not skip_feedin:
                        agg_true_fe = y_true_aggr.groupby(level='batch')[col].apply(lambda x: x[x<0].sum())
                        agg_pred_fe = y_pred_aggr.groupby(level='batch')[y_pred_aggr.columns[0]].apply(lambda x: x[x<0].sum())
                        metrics["Agg. Feed-In R^2"] = self._safe_r2(agg_true_fe.values, agg_pred_fe.values)
                except Exception as e:  # pragma: no cover
                    print(f"[EvaluationMetrics] MdAPE: Skipped aggregated import/feed-in R^2 at {target_agg}h due to error: {e}")
        return metrics
    




    ###############################################################################################################
    ###################################### Evaluation Plots #######################################################
    ###############################################################################################################
    @staticmethod
    def plot_true_vs_pred_and_quantile_errors(y_true, y_pred, name, grid_batched=False, n_samples_true_vs_predicted=10000):
        ##### Randomly select subset of points to plot ###
        n_samples = len(y_true)
        if n_samples_true_vs_predicted > n_samples: 
            y_true_plot = y_true
            y_pred_plot = y_pred
        else:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_samples, size=n_samples_true_vs_predicted, replace=False)
            y_true_plot = y_true.iloc[idx]
            y_pred_plot = y_pred.iloc[idx]

        ##### Create a 1×2 subplot
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        ### Left plot: True vs Predicted values
        ax0 = axes[0]
        ax0.scatter(
            y_true_plot,
            y_pred_plot,
            alpha=0.6,
            edgecolor='none'
        )
        ax0.grid(True)
        # 45° reference line
        min_y, max_y = y_true_plot.values.min(), y_true_plot.values.max()
        ax0.plot([min_y, max_y], [min_y, max_y], 'r--', linewidth=1)
        ax0.set_xlabel('True Values')
        ax0.set_ylabel('Predicted Values')
        ax0.set_title('True vs Predicted')
        # ax0.set_aspect('equal', adjustable='box')

        ### Right plot: QQ‐style plot of prediction errors ###
        # Compute prediction errors)
        if grid_batched: y_test_error = (y_pred_plot - y_true_plot)/y_true_plot.abs()
        else: y_test_error = (y_pred_plot - y_true_plot)
        sorted_errors = np.sort(y_test_error.to_numpy().ravel())
        n = len(sorted_errors)
        quantiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = norm.ppf(quantiles)

        ax1 = axes[1]
        ax1.plot(
            theoretical_quantiles,
            sorted_errors,
            marker='o',
            linestyle='',
            color='blue',
            markersize=3,
            alpha=0.6
        )
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Theoretical Quantiles (Std Normal)')
        if grid_batched: ax1.set_ylabel('Prediction Percentage Errors')
        else: ax1.set_ylabel('Prediction Errors')
        ax1.set_title('Quantile Error Plot')
        ax1.grid(True)
        fig.suptitle(name, fontsize=16)

        plt.tight_layout()
        # plt.show()
        return fig
    
    @staticmethod
    def plot_true_vs_pred_and_halfnormal_quantile_errors(y_true, y_pred, name, grid_batched=False, n_samples_true_vs_predicted=10000):
        """
        Half-normal (folded) QQ-style plot:
        - Compute errors (or percentage errors if grid_batched)
        - Take absolute value to 'fold' negative deviations
        - Plot sorted absolute errors vs theoretical half-normal quantiles
        - If any absolute error > 5, switch to log y-scale
        """
        # Subsample
        n_samples = len(y_true)
        if n_samples_true_vs_predicted > n_samples: 
            y_true_plot = y_true
            y_pred_plot = y_pred
        else:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_samples, size=n_samples_true_vs_predicted, replace=False)
            y_true_plot = y_true.iloc[idx]
            y_pred_plot = y_pred.iloc[idx]

        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))

        # Left: True vs Pred
        ax0 = axes[0]
        ax0.scatter(
            y_true_plot,
            y_pred_plot,
            alpha=0.6,
            edgecolor='none'
        )
        ax0.grid(True)
        min_y, max_y = y_true_plot.values.min(), y_true_plot.values.max()
        ax0.plot([min_y, max_y], [min_y, max_y], 'r--', linewidth=1)
        ax0.set_xlabel('True Values')
        ax0.set_ylabel('Predicted Values')
        ax0.set_title('True vs Predicted')

        # Right: Half-normal QQ of absolute errors
        if grid_batched:
            errors = (y_pred_plot - y_true_plot) / y_true_plot.abs()
        else:
            errors = (y_pred_plot - y_true_plot)

        errors_np = np.abs(errors.to_numpy().ravel())
        # Remove inf / nan (can arise from division by zero)
        finite_mask = np.isfinite(errors_np)
        errors_np = errors_np[finite_mask]

        sorted_abs_errors = np.sort(errors_np)
        n = len(sorted_abs_errors)
        if n == 0:
            # Fallback empty plot
            ax1 = axes[1]
            ax1.text(0.5, 0.5, 'No finite errors', ha='center', va='center')
            ax1.set_axis_off()
            fig.suptitle(name + " (No finite errors)", fontsize=16)
            plt.tight_layout()
            return fig

        quantiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = halfnorm.ppf(quantiles)  # Half-normal reference

        ax1 = axes[1]
        ax1.plot(
            theoretical_quantiles,
            sorted_abs_errors,
            marker='o',
            linestyle='',
            color='blue',
            markersize=3,
            alpha=0.6
        )
        ax1.axhline(1, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Theoretical Half-Normal Quantiles')
        if grid_batched:
            ax1.set_ylabel('Absolute Percentage Errors')
        else:
            ax1.set_ylabel('Absolute Errors')
        ax1.set_title('Half-Normal QQ of |Errors|')
        ax1.grid(True)

        # Log scale if large errors
        if sorted_abs_errors.max() > 5:
            ax1.set_yscale('log')

        fig.suptitle(f"{name} (Absolute / Folded Errors)", fontsize=16)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_phase_angle_halfnormal_quantile_errors(y_true_angle: pd.DataFrame,
                                                    y_pred_angle: pd.DataFrame,
                                                    name: str,
                                                    n_samples_true_vs_predicted: int = 10000):
        """
        QQ-style plot for absolute angular errors (deg) with minimal wrap-around difference.
        Left: scatter of true vs pred angles. Right: half-normal QQ of |Δφ| with modulo 360 handling.
        """
        # Subsample
        n_samples = len(y_true_angle)
        if n_samples_true_vs_predicted > n_samples:
            yt = y_true_angle
            yp = y_pred_angle
        else:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_samples, size=n_samples_true_vs_predicted, replace=False)
            yt = y_true_angle.iloc[idx]
            yp = y_pred_angle.iloc[idx]

        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))

        # Left: True vs Pred angles
        ax0 = axes[0]
        ax0.scatter(yt, yp, alpha=0.6, edgecolor='none')
        ax0.grid(True)
        min_y = min(yt.values.min(), yp.values.min())
        max_y = max(yt.values.max(), yp.values.max())
        ax0.plot([min_y, max_y], [min_y, max_y], 'r--', linewidth=1)
        ax0.set_xlabel('True Angle (deg)')
        ax0.set_ylabel('Pred Angle (deg)')
        ax0.set_title('True vs Predicted Phase Angle')

        # Right: Half-normal QQ of absolute angular errors with wrap-around
        true_vals = yt.iloc[:, 0].to_numpy()
        pred_vals = yp.iloc[:, 0].to_numpy()
        abs_ang_err = EvaluationMetrics._angle_abs_diff_deg(true_vals, pred_vals)
        abs_ang_err = abs_ang_err[np.isfinite(abs_ang_err)]
        sorted_abs = np.sort(abs_ang_err)
        n = len(sorted_abs)
        if n == 0:
            ax1 = axes[1]
            ax1.text(0.5, 0.5, 'No finite errors', ha='center', va='center')
            ax1.set_axis_off()
            fig.suptitle(name + " (No finite errors)", fontsize=16)
            plt.tight_layout()
            return fig
        quantiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical = halfnorm.ppf(quantiles)
        ax1 = axes[1]
        ax1.plot(theoretical, sorted_abs, marker='o', linestyle='', color='blue', markersize=3, alpha=0.6)
        ax1.set_xlabel('Theoretical Half-Normal Quantiles')
        ax1.set_ylabel('Absolute Angle Error (deg)')
        ax1.set_title('Half-Normal QQ of |Δφ|')
        ax1.grid(True)
        if sorted_abs.max() > 30:
            ax1.set_yscale('log')
        fig.suptitle(f"{name} (deg)", fontsize=16)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_grid_mape_quantile_errors(y_val, y_val_pred):
        col = y_val.columns[0]
        grouped = (y_val_pred - y_val).abs().groupby(level='batch')
        agg_error = grouped[col].agg([("mean", "mean")])
        grouped = y_val.abs().groupby(level='batch')
        agg_mean_demand = grouped[col].agg([("mean", "mean")])
        mape = agg_error/agg_mean_demand

        # Assuming agg_true is defined and contains the grouped absolute errors
        sorted_errors = np.sort(mape.to_numpy().ravel())
        n = len(sorted_errors)
        quantiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = norm.ppf(quantiles)

        # Create single plot
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            theoretical_quantiles,
            sorted_errors,
            marker='o',
            linestyle='',
            color='blue',
            markersize=3,
            alpha=0.6
        )
        # ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.axhline(1, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Theoretical Quantiles (Std Normal)')
        ax.set_ylabel('MAE/Mean Absolute Exchange (Per Grid)')
        ax.set_title('Quantile Error Plot (Grid Batched)')
        ax.grid(True)

        # If any error > 5, switch to log scale
        if sorted_errors[-1] > 5:
            # Avoid issues if zeros present
            if sorted_errors[0] <= 0:
                ax.set_ylim(bottom=1e-3)
            ax.set_yscale('log')        

        fig.suptitle("MAPE (Grid Batched)", fontsize=16)
        plt.tight_layout()
        # plt.show()
        return fig

    @staticmethod
    def plot_mean_timeseries(y_val, y_val_pred, base_agg_hours=1):
        if y_val.shape[1] != 1 or y_val_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_val.columns[0]

        """
        Normalize each batch (z-score) using ground-truth batch mean & std, apply same
        transformation to predictions, then average across batches to obtain:
          - Mean hourly ground-truth series
          - Mean hourly prediction series
        Plot:
          - Faint gray hourly mean ground-truth curve (background)
          - Daily mean (resampled) curves for ground-truth and prediction (colored)
          - X-axis labeled by months Jan..Dec (using a dummy 2024 calendar)
        """
        # Ensure same index
        if not y_val.index.equals(y_val_pred.index):
            y_val_pred = y_val_pred.reindex(y_val.index)

        # Per-batch mean/std from ground truth
        means = y_val[col].groupby(level='batch').transform('mean')
        stds = y_val[col].groupby(level='batch').transform('std').replace(0, 1)

        # Normalized (z-scored) using ground-truth stats
        norm_true = (y_val[col] - means) / stds
        norm_pred = (y_val_pred[col] - means) / stds

        # Mean across batches for each timestep (level=1 assumed to be hour index 0..8760-1)
        mean_true_hourly = norm_true.groupby(level=1).mean()
        mean_pred_hourly = norm_pred.groupby(level=1).mean()

        time_index = pd.date_range(start='2024-01-01', periods=int(8760/base_agg_hours), freq=f'{int(base_agg_hours)}h')

        df_hourly = pd.DataFrame({
            'true': mean_true_hourly.values,
            'pred': mean_pred_hourly.values
        }, index=time_index)

        # Daily means
        # df_daily = df_hourly.resample('D').mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        # Background hourly true (faint)
        ax.plot(df_hourly.index, df_hourly['true'],
                color='lightgray', alpha=0.4, linewidth=1,
                label='Hourly Mean (GT)')

        # Daily means
        ax.plot(df_hourly.index, df_hourly['true'], color='black', linewidth=2,
                label='Daily Mean (GT)', alpha=0.8)
        ax.plot(df_hourly.index, df_hourly['pred'], color='tab:orange', linewidth=2,
                label='Daily Mean (Pred)', alpha=0.8)

        ax.axhline(0, color='gray', linewidth=0.8)

        # Month formatting
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(time_index[0], time_index[-1])

        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Demand (z-score)')
        ax.set_title('Mean Normalized Timeseries Across Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


    @staticmethod
    def plot_agg_timeseries(y_val, y_val_pred):
        if y_val.shape[1] != 1 or y_val_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_val.columns[0]

        """
        Normalize each batch (z-score) using ground-truth batch mean & std, apply same
        transformation to predictions, then average across batches to obtain:
          - Mean hourly ground-truth series
          - Mean hourly prediction series
        Plot:
          - Faint gray hourly mean ground-truth curve (background)
          - Daily mean (resampled) curves for ground-truth and prediction (colored)
          - X-axis labeled by months Jan..Dec (using a dummy 2024 calendar)
        """
        # Ensure same index
        if not y_val.index.equals(y_val_pred.index):
            y_val_pred = y_val_pred.reindex(y_val.index)

        # Per-batch mean/std from ground truth
        means = y_val[col].groupby(level='batch').transform('mean')
        stds = y_val[col].groupby(level='batch').transform('std').replace(0, 1)

        # Normalized (z-scored) using ground-truth stats
        norm_true = (y_val[col] - means) / stds
        norm_pred = (y_val_pred[col] - means) / stds

        # Mean across batches for each timestep (level=1 assumed to be hour index 0..8760-1)
        mean_true_hourly = norm_true.groupby(level=1).mean()
        mean_pred_hourly = norm_pred.groupby(level=1).mean()

        n_hours = len(mean_true_hourly)
        time_index = pd.date_range(start='2024-01-01', periods=n_hours, freq='h')

        df_hourly = pd.DataFrame({
            'true': mean_true_hourly.values,
            'pred': mean_pred_hourly.values
        }, index=time_index)

        # Daily means
        df_daily = df_hourly.resample('D').mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        # Background hourly true (faint)
        ax.plot(df_hourly.index, df_hourly['true'],
                color='lightgray', alpha=0.4, linewidth=1,
                label='Hourly Mean (GT)')

        # Daily means
        ax.plot(df_daily.index, df_daily['true'], color='black', linewidth=2,
                label='Daily Mean (GT)', alpha=0.8)
        ax.plot(df_daily.index, df_daily['pred'], color='tab:orange', linewidth=2,
                label='Daily Mean (Pred)', alpha=0.8)

        ax.axhline(0, color='gray', linewidth=0.8)

        # Month formatting
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(time_index[0], time_index[-1])

        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Demand (z-score)')
        ax.set_title('Mean Normalized Timeseries Across Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_median_error_batch_timeseries(y_val, y_val_pred):
        if y_val.shape[1] != 1 or y_val_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_val.columns[0]

        """
        Plot normalized hourly + daily mean timeseries for the single batch whose
        (MAE / mean_abs_true) is closest to the median across all batches.
        Normalization: z-score per that batch using ground-truth mean & std.
        """
        # Build combined dataframe
        df = pd.concat(
            [
                y_val.rename(columns={col: "true"}),
                y_val_pred.rename(columns={col: "pred"})
            ],
            axis=1
        )

        if df.empty:
            raise ValueError("Empty input data.")

        # Compute per-batch MAE / mean_abs_true
        abs_err = (df["true"] - df["pred"]).abs()
        abs_true = df["true"].abs()
        mae_per_batch = abs_err.groupby(level="batch").mean()
        mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()
        rel_mae_per_batch = mae_per_batch / mean_abs_true_per_batch
        if len(rel_mae_per_batch) == 0:
            raise ValueError("No batches available for median selection.")

        median_ratio = rel_mae_per_batch.median()
        # Select batch id closest to median ratio
        target_batch = (rel_mae_per_batch - median_ratio).abs().idxmin()

        # Slice that batch
        batch_true = df.xs(target_batch, level='batch')["true"].copy()
        batch_pred = df.xs(target_batch, level='batch')["pred"].copy()

        # Normalize using batch ground truth stats
        mean_bt = batch_true.mean()
        std_bt = batch_true.std()
        if std_bt == 0:
            std_bt = 1.0
        norm_true = (batch_true - mean_bt) / std_bt
        norm_pred = (batch_pred - mean_bt) / std_bt

        n_hours = len(norm_true)
        # Assume full-year 8760; if different length still produce sequential hours
        time_index = pd.date_range(start='2024-01-01', periods=n_hours, freq='h')

        df_hourly = pd.DataFrame(
            {"true": norm_true.values, "pred": norm_pred.values},
            index=time_index
        )
        df_daily = df_hourly.resample('D').mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_hourly.index, df_hourly['true'],
                color='lightgray', alpha=0.4, linewidth=1,
                label='Hourly (GT)')
        ax.plot(df_daily.index, df_daily['true'],
                color='black', linewidth=2, alpha=0.85,
                label='Daily Mean (GT)')
        ax.plot(df_daily.index, df_daily['pred'],
                color='tab:orange', linewidth=2, alpha=0.85,
                label='Daily Mean (Pred)')

        ax.axhline(0, color='gray', linewidth=0.8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(time_index[0], time_index[-1])
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Demand (z-score)')
        ax.set_title(f'Batch {target_batch} (Median Normalized MAE Ratio)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_worst_error_batch_timeseries(y_val, y_val_pred, base_agg_hours=1):
        if y_val.shape[1] != 1 or y_val_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_val.columns[0]
        df = pd.concat(
            [
                y_val.rename(columns={col: "true"}),
                y_val_pred.rename(columns={col: "pred"})
            ],
            axis=1
        )
        if df.empty:
            raise ValueError("Empty input data.")

        abs_err = (df["true"] - df["pred"]).abs()
        abs_true = df["true"].abs()
        mae_per_batch = abs_err.groupby(level="batch").mean()
        mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()
        rel_mae_per_batch = mae_per_batch / mean_abs_true_per_batch
        if len(rel_mae_per_batch) == 0:
            raise ValueError("No batches available for worst selection.")

        worst_batch = rel_mae_per_batch.idxmax()
        worst_ratio = rel_mae_per_batch.loc[worst_batch]

        batch_true = df.xs(worst_batch, level='batch')["true"].copy()
        batch_pred = df.xs(worst_batch, level='batch')["pred"].copy()

        mean_bt = batch_true.mean()
        std_bt = batch_true.std() if batch_true.std() != 0 else 1.0
        norm_true = (batch_true - mean_bt) / std_bt
        norm_pred = (batch_pred - mean_bt) / std_bt

        n_hours = len(norm_true)
        time_index = pd.date_range(start='2024-01-01', periods=int(8760/base_agg_hours), freq=f'{int(base_agg_hours)}h')
        df_hourly = pd.DataFrame({"true": norm_true.values, "pred": norm_pred.values}, index=time_index)
        # df_daily = df_hourly.resample('D').mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(df_hourly.index, df_hourly['true'], color='lightgray', alpha=0.35, linewidth=1, label='Hourly (GT)')
        ax.plot(df_hourly.index, df_hourly['true'], color='black', linewidth=2, alpha=0.85, label='Daily Mean (GT)')
        ax.plot(df_hourly.index, df_hourly['pred'], color='tab:red', linewidth=2, alpha=0.85, label='Daily Mean (Pred)')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(time_index[0], time_index[-1])
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Demand (z-score)')
        ax.set_title(f'Batch {worst_batch} (Worst Normalized MAE Ratio={worst_ratio:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_worst_error_batch_timeseries_agg(self, y_val, y_val_pred, base_agg_hours=1):
        if y_val.shape[1] != 1 or y_val_pred.shape[1] != 1:
            raise ValueError("Expected single-column DataFrames.")
        col = y_val.columns[0]
        df = pd.concat(
            [
                y_val.rename(columns={col: "true"}),
                y_val_pred.rename(columns={col: "pred"})
            ],
            axis=1
        )
        if df.empty:
            raise ValueError("Empty input data.")

        abs_err = (df["true"] - df["pred"]).abs()
        abs_true = df["true"].abs()
        mae_per_batch = abs_err.groupby(level="batch").mean()
        mean_abs_true_per_batch = abs_true.groupby(level="batch").mean()
        rel_mae_per_batch = mae_per_batch / mean_abs_true_per_batch
        if len(rel_mae_per_batch) == 0:
            raise ValueError("No batches available for worst selection.")

        worst_batch = rel_mae_per_batch.idxmax()
        worst_ratio = rel_mae_per_batch.loc[worst_batch]

        batch_true = df.xs(worst_batch, level='batch')["true"].copy()
        batch_pred = df.xs(worst_batch, level='batch')["pred"].copy()
        # Recreate a (single) batch MultiIndex required by reaggregate_n_steps
        batch_true.index = pd.MultiIndex.from_product([['batch'], batch_true.index], names=['batch', 'hour'])
        batch_pred.index = pd.MultiIndex.from_product([['batch'], batch_pred.index], names=['batch', 'hour'])

        agg_target = 24
        plot_orig = False
        if (base_agg_hours >= agg_target) or (agg_target % base_agg_hours != 0):
            plot_orig = True
        else:
            # Aggregate (convert single-column Series -> DataFrame first)
            y_true_agg, y_pred_agg = self.reaggregate_n_steps(
                batch_true.to_frame(), batch_pred.to_frame(),
                n_agg=agg_target // base_agg_hours
            )
            # Squeeze to 1D arrays (avoid shape (N,1))
            y_true_agg_series = y_true_agg.iloc[:, 0]
            y_pred_agg_series = y_pred_agg.iloc[:, 0]
            n_agg_periods = len(y_true_agg_series)
            time_index_agg = pd.date_range(
                start='2024-01-01',
                periods=n_agg_periods,
                freq=f'{agg_target}h'
            )
            df_agg = pd.DataFrame(
                {
                    "true": y_true_agg_series.to_numpy(),
                    "pred": y_pred_agg_series.to_numpy()
                },
                index=time_index_agg
            )

        # Normalize (z-score) using ground-truth batch stats
        mean_bt = batch_true.mean()
        std_bt = batch_true.std()
        if std_bt == 0:
            std_bt = 1.0
        norm_true = (batch_true - mean_bt) / std_bt
        norm_pred = (batch_pred - mean_bt) / std_bt

        # Use actual length instead of assuming full year
        n_hours = len(norm_true)
        time_index = pd.date_range(
            start='2024-01-01',
            periods=n_hours,
            freq=f'{int(base_agg_hours)}h'
        )
        df_hourly = pd.DataFrame(
            {"true": norm_true.values, "pred": norm_pred.values},
            index=time_index
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        if plot_orig:
            ax.plot(df_hourly.index, df_hourly['true'],
                    color='black', linewidth=2, alpha=0.85,
                    label='Daily Mean (GT)')
            ax.plot(df_hourly.index, df_hourly['pred'],
                    color='tab:red', linewidth=2, alpha=0.85,
                    label='Daily Mean (Pred)')
        else:
            ax.plot(df_hourly.index, df_hourly['true'],
                    color='lightgray', alpha=0.35, linewidth=1,
                    label=f'{int(base_agg_hours)}h (GT)')
            ax.plot(df_agg.index, df_agg['true'].values,
                    color='black', linewidth=2, alpha=0.85,
                    label='Daily Mean (GT)')
            ax.plot(df_agg.index, df_agg['pred'].values,
                    color='tab:red', linewidth=2, alpha=0.85,
                    label='Daily Mean (Pred)')

        ax.axhline(0, color='gray', linewidth=0.8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_xlim(time_index[0], time_index[-1])
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Demand (z-score)')
        ax.set_title(f'Batch {worst_batch} (Worst Normalized MAE Ratio={worst_ratio:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_sorted_demand_curve(y_val, y_val_pred):
        col = y_val.columns[0]
        gt_means = y_val[col].groupby(level='batch').apply(lambda x: np.mean(np.abs(x)))

        def sorted_normalized_by_gt_mean(df_series, gt):
            return (
                df_series[col]
                .groupby(level='batch')
                .apply(lambda x: pd.Series(np.sort(x.values)) / gt.loc[x.name])
                .groupby(level=1)
                .mean()
            )

        mean_sorted_val  = sorted_normalized_by_gt_mean(y_val, gt_means)
        mean_sorted_pred = sorted_normalized_by_gt_mean(y_val_pred, gt_means)

        fig = plt.figure(figsize=(6, 4))
        plt.plot(mean_sorted_val,  label='Ground Truth (norm. by ground truth mean)',   linewidth=2)
        plt.plot(mean_sorted_pred, label='Prediction (norm. by ground truth mean)',     linewidth=2)
        plt.xlabel('Sorted Time Steps')
        plt.ylabel('Demand / Mean Absolute Exchange (Per Grid)')
        plt.title('Mean Normalized Sorted Demand Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        return fig

    @staticmethod
    ### Any overhang of a time occurence prediction into the last or next year is instead appended creating an overhang
    def wrap_to_half_period(t_pred, t_true, period=8760):
        delta = (t_pred - t_true + period/2) % period - period/2
        return t_pred + delta
    
    @staticmethod
    def append_overhang(t_pred, t_true, period=8760):
        pred = np.asarray(t_pred, dtype=float)
        true = np.asarray(t_true, dtype=float)
        delta = pred - true
        # compute how many whole periods to shift by:
        #   round(delta / period) 
        # puts delta - n*period into [-period/2,period/2].
        n_periods = np.round(delta / period)
        
        return pd.Series(pred - n_periods * period)
        

    def plot_evaluation(self, y_val, y_val_pred, base_agg_hours, skip_feedin: bool = False):
        """
        Optionally, if reactive power DataFrames were attached to the instances (through caller),
        add phase-angle plots. For backward-compatibility, this function keeps the original
        signature, but internal helper will look for attributes set by run_model_evaluation.
        """
        col = y_val.columns[0]
        plot_jobs = [
            (self.plot_true_vs_pred_and_quantile_errors, (y_val, y_val_pred, "Demand (Timestep)"), {}),
            (self.plot_grid_mape_quantile_errors, (y_val, y_val_pred), {}),
            (self.plot_mean_timeseries, (y_val, y_val_pred, base_agg_hours), {}),
            # (self.plot_median_error_batch_timeseries, (y_val, y_val_pred), {}),
            (self.plot_worst_error_batch_timeseries, (y_val, y_val_pred, base_agg_hours), {}),
            (self.plot_worst_error_batch_timeseries_agg, (y_val, y_val_pred, base_agg_hours), {}),
            (self.plot_sorted_demand_curve, (y_val, y_val_pred), {}),
            (self.plot_true_vs_pred_and_halfnormal_quantile_errors,
                (y_val.groupby(level='batch')[col].agg([("max", "max")]),
                 y_val_pred.groupby(level='batch')[col].agg([("max", "max")]),
                 "Peak Demand (Grid Batched)"),
                {"grid_batched": True}),
        ]
        if not skip_feedin:
            plot_jobs.extend([
                (self.plot_true_vs_pred_and_halfnormal_quantile_errors,
                    (y_val.groupby(level='batch')[col].agg([("min", "min")]),
                     y_val_pred.groupby(level='batch')[col].agg([("min", "min")]),
                     "Peak Feed-In (Grid Batched)"),
                    {"grid_batched": True}),
                (self.plot_true_vs_pred_and_halfnormal_quantile_errors,
                    (y_val.groupby(level='batch')[col].agg([("positive_sum", lambda x: x[x>0].sum())]),
                     y_val_pred.groupby(level='batch')[col].agg([("positive_sum", lambda x: x[x>0].sum())]),
                     "Aggregated Demand (Grid Batched)"),
                    {"grid_batched": True}),
                (self.plot_true_vs_pred_and_halfnormal_quantile_errors,
                    (y_val.groupby(level='batch')[col].agg([("negative_sum", lambda x: x[x<0].sum())]),
                     y_val_pred.groupby(level='batch')[col].agg([("negative_sum", lambda x: x[x<0].sum())]),
                     "Aggregated Feed-In (Grid Batched)"),
                    {"grid_batched": True}),
            ])
        # Peak occurrence plots
        plot_jobs.append(
            (self.plot_true_vs_pred_and_quantile_errors,
                (base_agg_hours*y_val[col].groupby(level='batch').idxmax().map(lambda x: x[1]).reset_index(drop=True),
                 base_agg_hours*self.append_overhang(
                        y_val_pred[col].groupby(level='batch').idxmax().map(lambda x: x[1]).values,
                        y_val[col].groupby(level='batch').idxmax().map(lambda x: x[1]).values
                    ).reset_index(drop=True),
                 "Peak Demand Occurence (Grid Batched)"),
                {})
        )
        if not skip_feedin:
            plot_jobs.append(
                (self.plot_true_vs_pred_and_quantile_errors,
                    (base_agg_hours*y_val[col].groupby(level='batch').idxmin().map(lambda x: x[1]).reset_index(drop=True),
                     base_agg_hours*self.append_overhang(
                            y_val_pred[col].groupby(level='batch').idxmin().map(lambda x: x[1]).values,
                            y_val[col].groupby(level='batch').idxmin().map(lambda x: x[1]).values
                        ).reset_index(drop=True),
                     "Peak Feed-In Occurence (Grid Batched)"),
                    {})
            )

        # Phase angle plots if the caller populated reactive power on the instance
        y_val_q = getattr(self, "_plot_y_val_q", None)
        y_val_pred_q = getattr(self, "_plot_y_val_pred_q", None)
        if (y_val_q is not None) and (y_val_pred_q is not None):
            try:
                ang_true = self._angles_deg_from_pq(y_val, y_val_q)
                ang_pred = self._angles_deg_from_pq(y_val_pred, y_val_pred_q)
                plot_jobs.append(
                    (self.plot_phase_angle_halfnormal_quantile_errors,
                     (ang_true, ang_pred, "Phase Angle of S"),
                     {})
                )
            except Exception as e:  # pragma: no cover
                print(f"[EvaluationMetrics] Skipped phase-angle plot due to error: {e}")

        # Submit jobs in order and collect futures in a list
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(func, *args, **kwargs)
                       for func, args, kwargs in plot_jobs]
            # Wait for all to finish
            concurrent.futures.wait(futures)
            # Retrieve results in order
            figures = [f.result() for f in futures]

        # Display all figures in order
        for fig in figures:
            display(fig)
            plt.close(fig)


    ############################################################################################################################
    ################################################## Full Model Evaluation ###################################################
    ############################################################################################################################
    def run_model_evaluation(self,
                             y_test_true: pd.DataFrame,
                             y_test_pred: pd.DataFrame,
                             y_train_true: pd.DataFrame | None = None,
                             y_train_pred: pd.DataFrame | None = None,
                             y_test_true_reactive: pd.DataFrame | None = None,
                             y_test_pred_reactive: pd.DataFrame | None = None,
                             y_train_true_reactive: pd.DataFrame | None = None,
                             y_train_pred_reactive: pd.DataFrame | None = None,
                             no_plots: bool = False,
                             base_agg_hours: int = 1,
                             skip_feedin_metrics: bool = False):
        """
            Provide already computed prediction DataFrames (original scale) and obtain:
                - Metrics table (train optional, test mandatory)
                - Plots (if batch MultiIndex with 'batch' & not suppressed)

        Parameters
        ----------
        y_test_true : pd.DataFrame
            Ground-truth targets for test/validation set.
        y_test_pred : pd.DataFrame
            Predicted targets for test/validation set (aligned row-wise & indexed).
        y_*_reactive : pd.DataFrame, optional
            Reactive power (Q) counterparts for phase-angle metrics/plots. If provided for both
            true and pred, phase-angle metrics (deg) will be reported per aggregation scale.
        y_train_true, y_train_pred : pd.DataFrame, optional
            Training ground-truth & predictions (if you want training metrics shown).
        no_plots : bool
            If True, suppress plot generation.

        Returns
        -------
        dict with:
            train_metrics (optional), test_metrics,
            y_train_true, y_train_pred, y_test_true, y_test_pred
        """
        ### Check inputs
        if y_test_true.shape[0] != y_test_pred.shape[0]:
            raise ValueError("y_test_true and y_test_pred must have identical number of rows.")
        if list(y_test_true.columns) != list(y_test_pred.columns):
            raise ValueError("y_test_true and y_test_pred must share identical columns.")
        if y_train_true is not None and y_train_pred is not None:
            if y_train_true.shape[0] != y_train_pred.shape[0]:
                raise ValueError("y_train_true and y_train_pred must have identical number of rows.")
            if list(y_train_true.columns) != list(y_train_pred.columns):
                raise ValueError("y_train_true and y_train_pred must share identical columns.")
        elif any(v is not None for v in (y_train_true, y_train_pred)):
            raise ValueError("Provide both y_train_true and y_train_pred or neither.")

        # Validate reactive inputs pairing
        def _pair_check(a, b, label):
            if (a is None) ^ (b is None):
                raise ValueError(f"Provide both true and pred reactive for {label}, or neither.")
        _pair_check(y_test_true_reactive, y_test_pred_reactive, "test")
        _pair_check(y_train_true_reactive, y_train_pred_reactive, "train")

        ### Ensure correct index structure
        def _has_batch(df):
            return isinstance(df.index, pd.MultiIndex) and 'batch' in df.index.names
        # auto-create pseudo-batches (8760 hours) if missing
        def _auto_batch(df: pd.DataFrame, name: str) -> pd.DataFrame:
            if df is None: return df
            if _has_batch(df): return df
            n = len(df)
            batch = np.arange(n, dtype=int) // int(8760 / base_agg_hours)
            hour = np.arange(n, dtype=int) % int(8760 / base_agg_hours)
            df = df.copy()
            df.index = pd.MultiIndex.from_arrays([batch, hour], names=['batch', 'hour'])
            print(f"WARNING: '{name}' lacked batch MultiIndex -> pseudo-batches were created.")
            return df
        y_test_true = _auto_batch(y_test_true, "y_test_true")
        y_test_pred = _auto_batch(y_test_pred, "y_test_pred")
        if y_train_true is not None and y_train_pred is not None:
            y_train_true = _auto_batch(y_train_true, "y_train_true")
            y_train_pred = _auto_batch(y_train_pred, "y_train_pred")

        if y_test_true_reactive is not None:
            y_test_true_reactive = _auto_batch(y_test_true_reactive, "y_test_true_reactive")
            y_test_pred_reactive = _auto_batch(y_test_pred_reactive, "y_test_pred_reactive")
        if y_train_true_reactive is not None:
            y_train_true_reactive = _auto_batch(y_train_true_reactive, "y_train_true_reactive")
            y_train_pred_reactive = _auto_batch(y_train_pred_reactive, "y_train_pred_reactive")

        ### Compute metrics
        test_scores = self.evaluate(
            y_test_true, y_test_pred,
            base_agg_hours=base_agg_hours,
            skip_feedin=skip_feedin_metrics,
            y_true_reactive=y_test_true_reactive,
            y_pred_reactive=y_test_pred_reactive,
        )
        train_scores = (self.evaluate(
            y_train_true, y_train_pred,
            base_agg_hours=base_agg_hours,
            skip_feedin=skip_feedin_metrics,
            y_true_reactive=y_train_true_reactive,
            y_pred_reactive=y_train_pred_reactive,
        ) if y_train_true is not None else None)

        ### Format metrics
        df_test = pd.DataFrame(test_scores, index=["Test"]).T
        if train_scores:
            df_train = pd.DataFrame(train_scores, index=["Train"]).T
            overview = pd.concat([df_test, df_train], axis=1)
        else:
            df_train = None
            overview = df_test
        print("Metric Results (original scale):")
        print(overview)

        ### Plots
        if not no_plots:
            # Attach reactive for plotting if available
            if y_test_true_reactive is not None and y_test_pred_reactive is not None:
                # Stash on instance so plot_evaluation can access without signature change for callers
                self._plot_y_val_q = y_test_true_reactive
                self._plot_y_val_pred_q = y_test_pred_reactive
            else:
                self._plot_y_val_q = None
                self._plot_y_val_pred_q = None

            self.plot_evaluation(y_test_true, y_test_pred, base_agg_hours, skip_feedin=skip_feedin_metrics)
            # except Exception as e:
            #     print(f"Plotting failed: {e}")
        else:
            print("Plot generation suppressed (no_plots=True).")