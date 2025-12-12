"""Run Ray Tune hyperparameter optimization for the MLP baseline.

This script is the main entrypoint for running an Optuna(TPE) + ASHA HPO sweep
over the MLP model defined in `mlp_model.py`.

Key behaviors to be aware of:
    - It installs several Python packages at runtime via `pip`.
    - It expects the training dataset to exist as an HDF5 file containing keys
        `X` and `y` (see `0_preprocessing/Data/ts_train.h5`).
    - Results are written into `2_mlp/ray_tune/<experiment_name>/...`.
"""

import subprocess
import sys

### Ray tune    
import math, os
### pytorch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def install(package):
    """Install a Python package into the current interpreter environment.

    Note: This performs an online `pip install`. On shared/HPC systems you may
    prefer a pre-built environment or container image.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
	
def main():
    """Configure the Ray Tune experiment and run the HPO sweep."""
    install("tables")
    install("ray")
    install("ipywidgets")
    install("tensorboard")
    install("tensorboardX")
    install("optuna")
    install("vmdpy")
    install("seaborn")
    install("scikit-learn")  # required by mlp_model.py

    import ray
    from ray import tune
    from ray.tune import Tuner
    from ray.tune import TuneConfig, RunConfig, CheckpointConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.logger import TBXLoggerCallback
    from ray.tune.search.optuna import OptunaSearch
    from optuna.samplers import TPESampler
    from mlp_model import MLPTrainer

    # Data config (copied from 2_mlp/run_mlp_models.py)
    DATA_CFG = {
        'hdf_data_path': '/dss/dsshome1/05/ge96ton2/GridForecast/0_preprocessing/Data/ts_train.h5',
        'key_X': 'X',
        'key_y': 'y',
        'train_grids': 'all',
        'test_ratio': 0.2,
        'random_state': 42,
        'X_BASE_COLS': [
            'T', 'demand_net_active_pre',
            'heat_water', 'heat_space', 'cop_avg',
            'PV_prod_expected',
            'res_bldng_area_base_sum', 'nonres_bldng_area_base_sum', 'bldng_area_floors_sum',
            'n_cars', 'n_res_buildings', 'n_nonres_buildings', 'n_flats', 'n_occ',
            'n_lines', 'tot_R_grid', 'regiostar7',
        ],
        'TARGET_COLS': ['demand_net_active_post', 'demand_net_reactive_post'],
        'ZERO_BASE_FEATURES': [],
        'LOG1P_COLS': ['n_lines', 'tot_R_grid'],
        'TS_PERIODS': '8760,168,24',
    }

    # Fixed defaults
    FIXED_CFG = {
        'loss_type': 'alpha_peak',   # mapped to mae_maex internally
        'full_metrics_every': 5,
        'bottom_rung_report': 5,
        'compile_model': True,
        'num_workers': 14,
        'pin_memory': True,
    }

    assert os.path.exists(DATA_CFG['hdf_data_path']), f"Missing HDF5 file: {DATA_CFG['hdf_data_path']}"

    # Edit these parameters to run a single model
    RUN_CFG = {
        # Core hyperparameters
        'learning_rate': 1e-3,
        'batch_size': 32768,
        'optimizer': 'adamw',               # 'adamw' | 'adam' | 'sgd'
        'dropout': 0.1,
        'weight_decay': 1e-4,
        'activation': 'gelu',               # 'relu' | 'gelu' | 'silu' | 'tanh' | 'leaky_relu'
        # New simplified MLP shape
        'num_layers': 4,                    # number of hidden layers
        'layer_size': 256,                  # neurons per hidden layer
        # Scheduler
        'scheduler': 'none',                # 'none' | 'cosine' | 'plateau' | 'onecycle'
        'scheduler_epochs': 10,
        # Reproducibility
        'random_state': 42,
        # Fixed bits
        **FIXED_CFG,
        '_data': {**DATA_CFG},
    }

    print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    # Compute resources (feel free to adjust)
    n_cpus_total = int(os.environ.get('N_CPUS', '14'))
    n_gpus_total = 1 if torch.cuda.is_available() else 0
    max_concurrent = 1
    n_cpus_task = max(1, n_cpus_total // max(1, max_concurrent))
    n_gpus_task = max(0, n_gpus_total / max(1, max_concurrent))

    # Global settings
    STORAGE_ROOT = "/dss/dsshome1/05/ge96ton2/GridForecast/2_mlp/ray_tune"  # keep with other runs
    experiment_name = "mlp_asha_tpe_TS_mae_maex"
    num_samples = 135
    max_epochs = 45
    grace_period = 5
    n_startup_trials = 10

    # Reuse data config from earlier cell
    DATA_CFG_HPO = {**DATA_CFG}

    # Newer API: search over num_layers/layer_size instead of deprecated hidden_layers
    search_space = {
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([8192, 16384, 32768, 65536, 131072, 262144, 524288]),
        'optimizer': tune.choice(['adamw', 'sgd']),
        'dropout': tune.uniform(0.0, 0.4),
        'weight_decay': tune.loguniform(1e-5, 5e-2),
        'activation': tune.choice(['relu', 'gelu']),
        'num_layers': tune.choice([1, 2, 3, 4, 5]),
        'layer_size': tune.choice([64, 128, 256, 512, 1024]),
        'scheduler': tune.choice(['none', 'cosine', 'plateau']),
        # Fixed settings
        'loss_type': 'mae_maex',  # trainer maps to 'mae_maex'
        'scheduler_epochs': max_epochs,
        'full_metrics_every': 5,
        'bottom_rung_report': grace_period,
        'compile_model': True,
        'num_workers': int(math.floor(n_cpus_task)),
        'pin_memory': True,
        # Data
        '_data': DATA_CFG_HPO,
    }

    print("[INFO] Experiment dir:", STORAGE_ROOT)
    ray.shutdown()
    ray.init(num_cpus=n_cpus_total, num_gpus=n_gpus_total)
    ray.available_resources()

    # Function-trainable wrapper around the class-based trainer so we can use with_resources
    from ray import tune as _tune_internal

    def _train_mlp(config):
        """Ray Tune function-trainable wrapper around `MLPTrainer`.

        This wrapper exists so we can attach per-trial resource requirements
        (CPUs/GPUs) via `tune.with_resources`.

        It additionally reports a custom metric `val_loss_bottom` that is
        monotonic (best-so-far) up to the ASHA grace period and then frozen.
        This is used to feed Optuna/TPE only bottom-rung information.
        """
        trainer = MLPTrainer(config)
        trainer.setup(config)
        epochs = int(config.get('scheduler_epochs', 50))
        # Enforce transformer-like bottom-rung behavior:
        # - val_loss_bottom is monotonically non-increasing for epochs <= bottom_rung
        # - val_loss_bottom is frozen (constant) for epochs > bottom_rung
        bottom_rung = int(config.get('bottom_rung_report', 0) or 0)
        best_val_for_bottom = float('inf')
        for _ in range(epochs):
            # Step trainer: this returns a dict with extended metrics every `full_metrics_every` epochs
            rec = trainer.step()
            epoch_i = int(rec.get('epoch', 0) or 0)
            val_loss_cur = rec.get('val_loss', None)
            # Update best only within the bottom rung window
            if val_loss_cur is not None and (bottom_rung == 0 or epoch_i <= bottom_rung):
                try:
                    best_val_for_bottom = min(best_val_for_bottom, float(val_loss_cur))
                except Exception:
                    pass
            # Compute bottom metric with the required behavior
            if bottom_rung > 0:
                if epoch_i <= bottom_rung:
                    val_loss_bottom = best_val_for_bottom
                else:
                    # Freeze after grace period
                    val_loss_bottom = best_val_for_bottom
            else:
                # No designated grace period -> mirror current val_loss
                val_loss_bottom = float(val_loss_cur) if val_loss_cur is not None else best_val_for_bottom

            # Merge bottom-rung metric into trainer record and report EVERYTHING (incl. extensive metrics)
            rec = dict(rec)
            rec['val_loss_bottom'] = val_loss_bottom
            # Ray 2.x: report expects a single dict argument, not kwargs
            _tune_internal.report(rec)

    # Per-trial resource wrapper (works with function trainables)
    trainable = tune.with_resources(_train_mlp, {"cpu": n_cpus_task, "gpu": n_gpus_task})

    # Optuna TPE sampler and search
    optuna_sampler = TPESampler(n_startup_trials=n_startup_trials, multivariate=True, seed=42)
    optuna_search = OptunaSearch(sampler=optuna_sampler, metric="val_loss_bottom", mode="min")

    # ASHA prunes on the usual intermediate metric
    asha_scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        grace_period=grace_period,
        max_t=max_epochs,
        reduction_factor=3,
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute='val_loss',
        checkpoint_score_order='min',
        #checkpoint_frequency=5,
    )

    run_config = RunConfig(
        name=experiment_name,
        storage_path=STORAGE_ROOT,
        verbose=1,
        callbacks=[TBXLoggerCallback()],
        checkpoint_config=checkpoint_config,
    )

    # Keep Tuner API; resources are provided via with_resources wrapper above
    tune_config = TuneConfig(
        num_samples=num_samples,
        search_alg=optuna_search,
        scheduler=asha_scheduler,
        max_concurrent_trials=max_concurrent,
    )

    tuner = Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    best = results.get_best_result(metric='val_loss', mode='min')
    print('\nBest config:', best.config)
    print('Best val_loss:', best.metrics.get('val_loss'))


if __name__ == '__main__':  # pragma: no cover
	raise SystemExit(main())

