import torch
import pytorch_lightning as pl

from torch import nn, Tensor

from .road_env_description import (
    LocalTransformerEncoder,
    EgoTrajectoryEncoder,
    REDFusionBlock,
)
from .dual_motion_vit import pytorch_neg_multi_log_likelihood_batch


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
    ) -> None:
        super().__init__()
        self.num_trajectory_proposals = num_trajectory_proposals
        self.prediction_horizon = prediction_horizon
        self.prediction_subsampling_rate = prediction_subsampling_rate
        self.lr = learning_rate
        self.epochs = epochs

        self.road_env_encoder = LocalRoadEnvEncoder(
            dim_model=dim_road_env_encoder,
            dim_attn_window_encoder=dim_road_env_attn_window,
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
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )
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

    def forward(
        self, idxs_src_tokens: Tensor, pos_src_tokens: Tensor, src_mask: Tensor
    ) -> Tensor:
        pos_src_tokens /= self.max_dist
        src = torch.concat(
            (self.encoder_semantic_embedding(idxs_src_tokens), pos_src_tokens), dim=2
        )  # Concat in feature dim
        src = self.to_dim_model(src)

        return self.encoder(src, src_mask)
