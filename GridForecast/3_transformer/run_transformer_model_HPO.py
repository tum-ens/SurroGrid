"""Run Ray Tune hyperparameter optimization for the Transformer model.

This script configures and runs a Ray Tune HPO experiment using:
    - Optuna TPE search (for bottom-rung metric only)
    - ASHA scheduler (for early stopping / pruning)

It delegates model construction and training to `transformer_model.py`.

Notes:
    - Installs several packages at runtime via `pip`.
    - Uses hard-coded absolute dataset paths by default; adjust `_data.hdf_data_path`.
    - If VMD caching is enabled, cached files are read/written under
        `3_transformer/data/` and should be mounted into containers.
"""

import math
import subprocess
import sys
import os
import argparse

def install(package):
    """Install a Python package into the current interpreter environment."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
	
def parse_args():
    """Parse command line arguments controlling aggregation and loss."""
    parser = argparse.ArgumentParser(description="Run Transformer HPO with Ray Tune")
    parser.add_argument("--agg-hours", type=int, default=1, help="Aggregation hours (int)")
    parser.add_argument("--loss-type", type=str, default="mae_maex", help="Loss type identifier")
    return parser.parse_args()


def main():
    """Entry point: install deps, configure Ray Tune, and run the sweep."""
    args = parse_args()
    install("tables")
    install("ray")
    install("ipywidgets")
    install("tensorboard")
    install("tensorboardX")
    install("optuna")
    install("vmdpy")
    install("seaborn")

    cwd = os.getcwd()
    print(cwd)
    contents = os.listdir(cwd)
    print("Directory contents:", contents)
    ### pytorch
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    import importlib, math
    import transformer_model
    import ray
    from ray import tune
    from ray.tune import Tuner
    from ray.tune import TuneConfig, RunConfig, CheckpointConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.logger import TBXLoggerCallback
    from ray.tune.search.optuna import OptunaSearch
    from optuna.samplers import TPESampler

    # Compute resources
    STORAGE_ROOT = "/dss/dsshome1/05/ge96ton2/GridForecast/3_transformer/ray_tune/"
    n_cpus_total = 14 #48
    n_gpus_total = 1
    max_concurrent = 1
    n_cpus_task = n_cpus_total/max_concurrent
    n_gpus_task = n_gpus_total/max_concurrent

    #
    num_samples = 153
    max_epochs = 45
    grace_period = 5      # minimum number of epochs (= lowest rung) to be run before ASHA curtails
    n_startup_trials = 23 # free dimensions d+1

    agg_hours = int(args.agg_hours)
    loss_type = args.loss_type  #'mae_maex'
    space = transformer_model.build_tune_search_space(tune)
    space.update({
        'epochs': max_epochs,              # max iterations per trial
        'scheduler_epochs': max_epochs,    # for cosine scheduler, if used
        'bottom_rung_report': grace_period,  # grace period: only this rung feeds TPE
        'num_workers': int(math.floor(n_cpus_task)),
        'agg_hours': agg_hours,
        'loss_type': loss_type,
        # Data pipeline
        '_data': {
            'hdf_data_path': '/dss/dsshome1/05/ge96ton2/GridForecast/0_preprocessing/Data/ts_train_large_grids.h5',
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
    })
    # experiment_name = f"HPO_withoutmob_agg{agg_hours}_{loss_type}"
    experiment_name = f"HPO_smallgrid_agg{agg_hours}_{loss_type}"

    #### Ray tune setup
    ray.shutdown()
    ray.init(num_cpus = n_cpus_total, num_gpus = n_gpus_total)
    ray.available_resources()

    # Resource wrapper per trial
    trainable = transformer_model.build_tune_trainable()
    trainable = tune.with_resources(trainable, {"cpu": n_cpus_task, "gpu": n_gpus_task})

    # --- Optuna TPE sampler: random startup for first `grace_period` finished trials ---
    optuna_sampler = TPESampler(n_startup_trials=n_startup_trials, # dimension of search space d + 1
                                multivariate=True, 
                                seed=42)
    # IMPORTANT: OptunaSearch must track the metric that you will report only at the bottom rung.
    # We choose 'val_loss_bottom' for that.
    optuna_search = OptunaSearch(
        sampler=optuna_sampler,
        metric="val_loss_bottom",   # <--- Optuna will only see this metric
        mode="min",
    )
    # --- ASHA scheduler prunes on 'val_loss' (intermediate), but does NOT see 'val_loss_bottom' ---
    asha_scheduler = ASHAScheduler(
        metric="val_loss",         # ASHA prunes using this metric every epoch
        mode="min",
        grace_period=grace_period, # do not prune before this many iterations (enforces bottom rung)
        max_t=max_epochs,
        reduction_factor=3
    )

    # --- checkpoint / run config (same pattern as you used) ---
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute='val_loss',   # choose whichever metric you prefer for checkpoints
        checkpoint_score_order='min',
        #checkpoint_frequency=5
    )

    run_config = RunConfig(
        name=experiment_name,
        storage_path=STORAGE_ROOT,
        verbose=1,
        callbacks=[TBXLoggerCallback()],
        checkpoint_config=checkpoint_config
    )

    # IMPORTANT: TuneConfig.metric should match the metric the searcher uses so that
    # Tuner considers the same objective when returning best trials.
    tune_config = TuneConfig(
        num_samples=num_samples,
        search_alg=optuna_search,
        scheduler=asha_scheduler,
        max_concurrent_trials=max_concurrent
    )

    tuner = Tuner(
        trainable,
        param_space=space,
        tune_config=tune_config,
        run_config=run_config
    )

    results = tuner.fit()

    best = results.get_best_result(metric='val_loss', mode='min')
    print('\nBest config:', best.config)
    # Also print the best val_loss at bottom rung (if available)
    print('Best val_loss:', best.metrics.get('val_loss'))
    # And for completeness you may inspect the epoch-wise val_loss history for the chosen trial.    
    

if __name__ == '__main__':  # pragma: no cover
	raise SystemExit(main())

