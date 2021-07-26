from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import NonlinearSatellite
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    nominal_params = {
        "A": 2e-6,
        "Cd": 2,
        "rho": 9.1515e-5,
        "m": 500,
        "r": 500.0 + NonlinearSatellite.RE,
    }
    scenarios = [
        nominal_params,
    ]

    # Define the dynamics model
    dynamics_model = NonlinearSatellite(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-1.0, 1.0),  # z
        (-1.0, 1.0),  # xdot
        (-1.0, 1.0),  # ydot
        (-1.0, 1.0),  # zdot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V Contour",
        domain=[(-5.0, 5.0), (-5.0, 5.0)],
        n_grid=20,
        x_axis_index=NonlinearSatellite.X,
        y_axis_index=NonlinearSatellite.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[NonlinearSatellite.X, NonlinearSatellite.Y],
        plot_x_labels=["$x$", "$y$"],
        plot_u_indices=[],
        plot_u_labels=[],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    # TODO!!! Nonlinear satellite needs a CBF, not a CLBF
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/linear_satellite",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)