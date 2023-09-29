import typer
import torch
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from models.transformer_baseline import TransformerMotionPredictor
from data_utils.dataset_modules import WaymoRoadEnvGraphDataModule

from data_utils.eval_motion_prediction import run_waymo_eval_per_class


def main(
    batch_size: int = 32,
    lr: float = 0.01,
    save_dir: str = "./",
    train_hours: float = 4.5,
    pre_training_checkpoint: str = "",
    prediction_subsampling_rate: int = 1,
    train_sample_limit: int = -1,
    num_nodes: int = 1,
    num_gpus: int = 4,
    prediction_horizon: int = 50,
    num_trajectory_proposals: int = 6,
    run_prefix: str = "scratch",
    train_path: str = "/p/project/hai_mrt_pc/waymo-open-motion-dataset/motion-cnn/train-300k",
    val_path: str = "/p/project/hai_mrt_pc/waymo-open-motion-dataset/motion-cnn/val",
    training_mode: str = "training",
):
    start_time = datetime.utcnow().replace(microsecond=0).isoformat()
    model_name = "baseline"

    loggers = [
        CSVLogger(
            save_dir=f"{save_dir}",
            version=f"{model_name}-{prediction_horizon}-{start_time}",
        ),
        WandbLogger(
            project="road-barlow-twins",
            save_dir=save_dir,
            name=f"{run_prefix}-{model_name}-{prediction_horizon}-{start_time}",
            offline=True,
        ),
    ]

    model = TransformerMotionPredictor(
        dim_road_env_encoder=192, # 256
        dim_road_env_attn_window=16,
        dim_ego_trajectory_encoder=128,
        num_trajectory_proposals=num_trajectory_proposals,
        prediction_horizon=prediction_horizon,
        learning_rate=lr,
        prediction_subsampling_rate=prediction_subsampling_rate,
        mode=training_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        max_time={"days": 0, "hours": train_hours},
        default_root_dir=save_dir,
        callbacks=[lr_monitor],
        logger=loggers,
        strategy="ddp_find_unused_parameters_true",
    )

    dm = WaymoRoadEnvGraphDataModule(
        batch_size=batch_size,
        num_dataloader_workers=12,
        pin_memory=True,
        train_path=train_path,
        val_path=val_path,
        val_limit=24 * 1000,
        train_limit=train_sample_limit,
    )

    if training_mode == "pre-training-graph-dino":
        model.copy_graph_dino_teacher()

    if pre_training_checkpoint:
        model.load_state_dict(torch.load(pre_training_checkpoint), strict=False)

    trainer.fit(model, datamodule=dm)

    if trainer.is_global_zero:
        torch.save(
            model.state_dict(),
            f"{save_dir}/models/{run_prefix}-{model_name}-{start_time}.pt",
        )

        if not run_prefix.startswith("pre-training"):
            if prediction_horizon > 50:
                prediction_horizons = [30, 50, prediction_horizon]
            else:
                prediction_horizons = [30, 50]

            pred_metrics, pred_metrics_per_class = run_waymo_eval_per_class(
                model=model,
                data=val_path,
                prediction_horizons=prediction_horizons,
                red_model=True,
                prediction_subsampling_rate=prediction_subsampling_rate,
            )
            loggers[1].log_table(
                key="motion_prediction_eval",
                dataframe=pred_metrics
            )
            loggers[1].log_table(
                key="motion_prediction_eval_per_class",
                dataframe=pred_metrics_per_class
            )


if __name__ == "__main__":
    typer.run(main)
