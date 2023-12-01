from argparse import ArgumentParser
import os
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import LinearPoint
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash
# from lightning.pytorch.callbacks import Callback


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 512
controller_period = 0.01

start_x = torch.tensor(
    [
        [0.5, 0.5, -0.1, -0.1],
    ]
)
simulation_dt = 0.01


# class CustomCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.outputs = None

#     def on_validation_batch_end(self, trainer, pl_module, outputs):
#         self.outputs(outputs)

#     def on_validation_epoch_end(self, trainer, pl_module):
#         pl_module._validation_epoch_end(self.outputs)

def main(args):
    torch.set_default_dtype(torch.float64)
    # Define the scenarios
    nominal_params = {
    }
    scenarios = [
        nominal_params,
    ]
    # # If we want robustness, then we can account for some uncertain accelerations
    # # from the target satellite.
    # for ux in [-0.01, 0.01]:
    #     for uy in [-0.01, 0.01]:
    #         for uz in [-0.01, 0.01]:
    #             scenarios.append(
    #                 {
    #                     "a": 6871,
    #                     "ux_target": ux,
    #                     "uy_target": uy,
    #                     "uz_target": uz,
    #                 }
    #             )

    # Define the dynamics model
    dynamics_model = LinearPoint(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
        use_l1_norm=False,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-1.0, 1.0),  # xdot
        (-1.0, 1.0),  # ydot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=100000,
        max_points=50000,
        val_split=0.1,
        batch_size=batch_size,
        quotas={"goal": 0.4, "safe": 0.2},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.5, 2.5), (-2.5, 2.5)],
        n_grid=25,
        x_axis_index=LinearPoint.X,
        y_axis_index=LinearPoint.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_state_space_experiment = RolloutStateSpaceExperiment(
        "Rollout State Space",
        start_x,
        plot_x_index=LinearPoint.X,
        plot_x_label="$x$",
        plot_y_index=LinearPoint.Y,
        plot_y_label="$y$",
        scenarios=[nominal_params],
        n_sims_per_start=1,
        t_sim=10.0,
    )
    experiment_suite = ExperimentSuite(
        [
            V_contour_experiment,
            rollout_state_space_experiment,
        ]
    )

    # Initialize the controller
    clbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=2,
        cbf_hidden_size=64,
        cbf_lambda=0.1,
        controller_period=controller_period,
        cbf_relaxation_penalty=1e4,
        scale_parameter=10.0,
        primal_learning_rate=1e-3,
        learn_shape_epochs=100,
        use_relu=True,
    )
    # cktp_path = "./logs/linear_point_cbf/relu/commit_86a2dc2/version_0/checkpoints/epoch=0-step=175.ckpt/"
    # if os.path.exists(cktp_path):
    #     clbf_controller = NeuralCBFController.load_from_checkpoint(cktp_path)

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/linear_point_cbf/relu",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer(
        logger=tb_logger, max_epochs=201,
        # accelerator=args  .accelerator,
        # devices=args.devices,
        # precision=args.precision,
        # strategy=args.strategy
    )
    # trainer = pl.Trainer(max_epochs=201)

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--devices", default=1)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--precision", default=32)
    parser.add_argument("--strategy", default="ddp")

    args = parser.parse_args()

    main(args)
