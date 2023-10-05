import copy
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn, Tensor

from .road_env_description import (
    LocalTransformerEncoder,
    EgoTrajectoryEncoder,
    REDFusionBlock,
    LocalTransformerEncoderLayer,
)
from .dual_motion_vit import pytorch_neg_multi_log_likelihood_batch
from .pretram import get_pretram_loss, get_mcl_loss
from .graph_dino import get_graph_dino_loss, update_moving_average, ExponentialMovingAverage
from .traj_mae import mask_road_env_tokens


class TransformerMotionPredictor(pl.LightningModule):
    def __init__(
        self,
        dim_road_env_encoder,
        dim_road_env_attn_window,
        dim_ego_trajectory_encoder,
        num_trajectory_proposals,
        prediction_horizon,
        learning_rate,
        epochs=190,
        prediction_subsampling_rate=1,
        num_fusion_layers=6,
        mode: str = "fine-tuning",
        dim_z: int = 512,
    ) -> None:
        super().__init__()
        self.num_trajectory_proposals = num_trajectory_proposals
        self.prediction_horizon = prediction_horizon
        self.prediction_subsampling_rate = prediction_subsampling_rate
        self.lr = learning_rate
        self.epochs = epochs
        self.mode = mode

        self.road_env_encoder = LocalRoadEnvEncoder(
            dim_model=dim_road_env_encoder,
            dim_attn_window_encoder=dim_road_env_attn_window,
            mode="traj-mae" if mode == "pre-training-traj-mae" else "default",
        )
        self.ego_trajectory_encoder = EgoTrajectoryEncoder(
            dim_model=dim_ego_trajectory_encoder,
            dim_output=dim_road_env_encoder,
        )
        self.fusion_block = REDFusionBlock(
            dim_model=dim_road_env_encoder, num_layers=num_fusion_layers
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm((dim_road_env_encoder,), eps=1e-06, elementwise_affine=True),
            nn.Linear(
                in_features=dim_road_env_encoder,
                out_features=num_trajectory_proposals
                * 2
                * (prediction_horizon // prediction_subsampling_rate)
                + num_trajectory_proposals,
            ),  # Multiple trajectory proposals with (x, y) every (0.1 sec * subsampling rate) and confidences
        )
        if self.mode == "pre-training-pretram":
            self.env_projector = nn.Sequential(
                nn.Linear(
                    in_features=dim_road_env_encoder,
                    out_features=1024
                ),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=dim_z),
            )
            self.traj_projector = nn.Sequential(
                nn.Linear(
                    in_features=dim_road_env_encoder,
                    out_features=1024
                ),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=dim_z),
            )
        elif self.mode == "pre-training-pretram-mcl":
            self.env_projector = nn.Sequential(
                nn.Linear(
                    in_features=dim_road_env_encoder,
                    out_features=1024
                ),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=dim_z),
            )
        
        elif self.mode == "pre-training-traj-mae":
            self.traj_mae_decoder = LocalTransformerEncoder(
                num_layers=3,
                dim_model=dim_road_env_encoder,
                dim_attn_window=dim_road_env_attn_window,
                dim_feedforward=dim_road_env_encoder * 4,
            )

        elif self.mode == "pre-training-graph-dino":
            self.graph_dino_env_teacher = nn.Module() # placeholder
            self.teacher_ema_updater = ExponentialMovingAverage()
            self.teacher_centering_ema_updater = ExponentialMovingAverage(decay=0.9)
            self.register_buffer('teacher_centers', torch.zeros(1, dim_road_env_encoder))
            self.register_buffer('previous_centers',  torch.zeros(1, dim_road_env_encoder))
    
    def masked_mean_aggregation(self, token_set, mask):
        denominator = torch.sum(mask, -1, keepdim=True)
        feats = torch.sum(token_set * mask.unsqueeze(-1), dim=1) / denominator
        
        return feats

    def copy_graph_dino_teacher(self):
        self.graph_dino_env_teacher = copy.deepcopy(self.road_env_encoder)
        for p in self.graph_dino_env_teacher.parameters():
            p.requires_grad = False

    def forward(
        self,
        env_idxs_src_tokens: Tensor,
        env_pos_src_tokens: Tensor,
        env_src_mask: Tensor,
        ego_idxs_semantic_embedding: Tensor,
        ego_pos_src_tokens: Tensor,
        env_idxs_src_tokens_b=None,
        env_pos_src_tokens_b=None,
    ):
        if self.mode == "pre-training-traj-mae":
            road_env_tokens, initial_road_env_tokens = self.road_env_encoder(
                env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
            )
            reconstructed_env_tokens = self.traj_mae_decoder(road_env_tokens, env_src_mask)
            loss = F.mse_loss(initial_road_env_tokens[env_src_mask], reconstructed_env_tokens[env_src_mask])

            return loss
            
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        
        if self.mode == "pre-training-pretram-mcl":
            road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask
            )
            road_env_y_a = self.masked_mean_aggregation(
                road_env_tokens, env_src_mask
            )
            road_env_y_b = self.masked_mean_aggregation(
                road_env_tokens_b, env_src_mask
            )

            road_env_z_a = self.env_projector(road_env_y_a)
            road_env_z_b = self.env_projector(road_env_y_b)
            
            loss = get_mcl_loss(road_env_z_a, road_env_z_b, temperature=0.07)

            return loss
        elif self.mode == "pre-training-graph-dino":
            road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask
            )

            with torch.no_grad():
                teacher_road_env_tokens = self.graph_dino_env_teacher(
                    env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
                )
                teacher_road_env_tokens_b = self.graph_dino_env_teacher(
                    env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask
                )

            teacher_logits_avg = torch.concat(
                (teacher_road_env_tokens, teacher_road_env_tokens_b), dim=0
            ).mean(dim=1).mean(dim=0) # mean(dim=1): tokens to logits, mean(dim=0): mean of logits

            self.previous_centers.copy_(teacher_logits_avg)

            loss_a = get_graph_dino_loss(
                teacher_road_env_tokens.mean(dim=1), road_env_tokens_b.mean(dim=1), self.teacher_centers
            )
            loss_b = get_graph_dino_loss(
                teacher_road_env_tokens_b.mean(dim=1), road_env_tokens.mean(dim=1), self.teacher_centers
            )
            loss = (loss_a + loss_b) / 2

            return loss

        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )

        if self.mode == "pre-training-pretram":
            road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask
            )
            road_env_y_a = self.masked_mean_aggregation(
                road_env_tokens, env_src_mask
            )
            road_env_y_b = self.masked_mean_aggregation(
                road_env_tokens_b, env_src_mask
            )
            traj_y = ego_trajectory_tokens.mean(dim=1)

            road_env_z_a = self.env_projector(road_env_y_a)
            road_env_z_b = self.env_projector(road_env_y_b)
            traj_z = self.traj_projector(traj_y)
            
            loss = get_pretram_loss(road_env_z_a, road_env_z_b, traj_z)

            return loss
        
        fused_tokens = self.fusion_block(
            q=ego_trajectory_tokens,
            k=road_env_tokens,
            v=road_env_tokens,
        )
        motion_embedding = self.motion_head(
            fused_tokens.mean(dim=1)
        )  # Sim. to improved ViT global avg pooling before classification
        confidences_logits, logits = (
            motion_embedding[:, : self.num_trajectory_proposals],
            motion_embedding[:, self.num_trajectory_proposals :],
        )
        logits = logits.view(
            -1,
            self.num_trajectory_proposals,
            (self.prediction_horizon // self.prediction_subsampling_rate),
            2,
        )

        return confidences_logits, logits

    def _shared_step(self, batch, batch_idx):
        is_available = batch["future_ego_trajectory"]["is_available"]
        y = batch["future_ego_trajectory"]["trajectory"]

        env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"]
        env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"]
        env_src_mask = batch["src_attn_mask"]
        ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
            "idx_semantic_embedding"
        ]
        ego_pos_src_tokens = batch["past_ego_trajectory"]["pos_src_tokens"]

        if self.mode.startswith("pre-training"):
            env_idxs_src_tokens_b = batch["sample_b"]["idx_src_tokens"]
            env_pos_src_tokens_b = batch["sample_b"]["pos_src_tokens"]

            loss = self.forward(
                env_idxs_src_tokens,
                env_pos_src_tokens,
                env_src_mask,
                ego_idxs_semantic_embedding,
                ego_pos_src_tokens,
                env_idxs_src_tokens_b,
                env_pos_src_tokens_b
            )
            return loss

        y = y[
            :,
            (
                self.prediction_subsampling_rate - 1
            ) : self.prediction_horizon : self.prediction_subsampling_rate,
            :,
        ]
        is_available = is_available[
            :,
            (
                self.prediction_subsampling_rate - 1
            ) : self.prediction_horizon : self.prediction_subsampling_rate,
        ]

        confidences_logits, logits = self.forward(
            env_idxs_src_tokens,
            env_pos_src_tokens,
            env_src_mask,
            ego_idxs_semantic_embedding,
            ego_pos_src_tokens,
        )

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)

        if self.mode == "pre-training-graph-dino":
            new_teacher_centers = update_moving_average(
                self.teacher_ema_updater,
                self.graph_dino_env_teacher,
                self.road_env_encoder,
                self.teacher_centers,
                self.previous_centers,
                self.teacher_centering_ema_updater,
            )
            self.teacher_centers.copy_(new_teacher_centers)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs,
                    eta_min=self.lr * 1e-2,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


class LocalRoadEnvEncoder(nn.Module):
    def __init__(
        self,
        size_encoder_vocab: int = 11,
        dim_encoder_semantic_embedding: int = 4,
        num_encoder_layers: int = 6,
        dim_model: int = 512,
        dim_heads_encoder: int = 64,
        dim_attn_window_encoder: int = 64,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_dist: float = 50.0,
        mode: str = "default",
    ) -> None:
        super().__init__()
        self.encoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_encoder_vocab,
            embedding_dim=dim_encoder_semantic_embedding,
            padding_idx=-1,  # For [pad] token
        )
        self.to_dim_model = nn.Linear(
            in_features=dim_encoder_semantic_embedding + 2,  # For position as (x, y)
            out_features=dim_model,
        )
        self.max_dist = max_dist
        self.encoder = LocalTransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            dim_heads=dim_heads_encoder,
            dim_attn_window=dim_attn_window_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.mode = mode

    def forward(
        self, idxs_src_tokens: Tensor, pos_src_tokens: Tensor, src_mask: Tensor
    ) -> Tensor:
        pos_src_tokens /= self.max_dist
        src = torch.concat(
            (self.encoder_semantic_embedding(idxs_src_tokens), pos_src_tokens), dim=2
        )  # Concat in feature dim
        src = self.to_dim_model(src)

        if self.mode == "traj-mae":
            masked_idxs_src_tokens = mask_road_env_tokens(torch.clone(idxs_src_tokens))
            src_masked = torch.concat(
                (self.encoder_semantic_embedding(masked_idxs_src_tokens), pos_src_tokens), dim=2
            )  # Concat in feature dim
            src_masked = self.to_dim_model(src_masked)

            return self.encoder(src_masked, src_mask), src

        return self.encoder(src, src_mask)
