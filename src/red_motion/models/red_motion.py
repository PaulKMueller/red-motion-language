import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn, Tensor

from vit_pytorch.cross_vit import CrossTransformer

from .road_env_description import (
    LocalTransformerEncoder,
    EgoTrajectoryEncoder,
    REDFusionBlock,
)
from .dual_motion_vit import pytorch_neg_multi_log_likelihood_batch
from .transformer_baseline import LocalRoadEnvEncoder
from .road_env_description import ParallelTransformerDecoder
from .raster_barlow_twins import BarlowTwinsLoss


class RedMotionQuery(pl.LightningModule):
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
        size_query_decoder_vocab=22,
        batch_size=96,
        dim_query_decoder=256,
        num_query_decoder_layers=6,
        num_heads_query_decoder=8,
        dim_feedforward_query_decoder=512,
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

        self.range_decoder_embedding = torch.arange(size_query_decoder_vocab).expand(
            batch_size, size_query_decoder_vocab
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=dim_query_decoder - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=10,
        )

        self.query_decoder = ParallelTransformerDecoder(
            num_layers=num_query_decoder_layers,
            dim_model=dim_query_decoder,
            num_heads=num_heads_query_decoder,
            dim_feedforward=dim_feedforward_query_decoder,
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
    ):
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )
        self.range_decoder_embedding = self.range_decoder_embedding.to("cuda")

        query_tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_decoder_embedding),
                self.decoder_pos_embedding(self.range_decoder_embedding),
            ),
            dim=2,
        )
        batch_size = env_idxs_src_tokens.size(dim=0)
        ego_mask = torch.ones(11, device="cuda").expand(batch_size, 11)
        ego_traj_queries = self.query_decoder(
            query_tgt, ego_trajectory_tokens, ego_mask
        )

        fused_tokens = self.fusion_block(
            q=ego_traj_queries,
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


class RedMotionQueryConcat(pl.LightningModule):
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
        size_query_decoder_vocab=22,
        batch_size=96,
        dim_query_decoder=128,
        num_query_decoder_layers=6,
        num_heads_query_decoder=8,
        dim_feedforward_query_decoder=512,
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

        self.range_decoder_embedding = torch.arange(size_query_decoder_vocab).expand(
            batch_size, size_query_decoder_vocab
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=dim_query_decoder - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=10,
        )

        self.query_decoder = ParallelTransformerDecoder(
            num_layers=num_query_decoder_layers,
            dim_model=dim_query_decoder,
            num_heads=num_heads_query_decoder,
            dim_feedforward=dim_feedforward_query_decoder,
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
    ):
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )
        self.range_decoder_embedding = self.range_decoder_embedding.to("cuda")

        query_tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_decoder_embedding),
                self.decoder_pos_embedding(self.range_decoder_embedding),
            ),
            dim=2,
        )
        env_queries = self.query_decoder(query_tgt, road_env_tokens, env_src_mask)

        concat_tokens = torch.cat((ego_trajectory_tokens, env_queries), dim=1)
        fused_tokens = self.fusion_block(
            q=concat_tokens,
            k=concat_tokens,
            v=concat_tokens,
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


class RedMotionQueryConcatSkip(pl.LightningModule):
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
        size_query_decoder_vocab=22,
        batch_size=96,
        dim_query_decoder=128,
        num_query_decoder_layers=6,
        num_heads_query_decoder=8,
        dim_feedforward_query_decoder=512,
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

        self.range_decoder_embedding = torch.arange(size_query_decoder_vocab).expand(
            batch_size, size_query_decoder_vocab
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=dim_query_decoder - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_query_decoder_vocab,
            embedding_dim=10,
        )

        self.query_decoder = ParallelTransformerDecoder(
            num_layers=num_query_decoder_layers,
            dim_model=dim_query_decoder,
            num_heads=num_heads_query_decoder,
            dim_feedforward=dim_feedforward_query_decoder,
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
    ):
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )
        self.range_decoder_embedding = self.range_decoder_embedding.to("cuda")

        query_tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_decoder_embedding),
                self.decoder_pos_embedding(self.range_decoder_embedding),
            ),
            dim=2,
        )
        env_queries = self.query_decoder(query_tgt, road_env_tokens, env_src_mask)

        concat_tokens = torch.cat((ego_trajectory_tokens, env_queries), dim=1)
        fused_tokens = self.fusion_block(
            q=concat_tokens,
            k=concat_tokens,
            v=concat_tokens,
        )

        motion_embedding = self.motion_head(
            fused_tokens.mean(dim=1) + ego_trajectory_tokens.mean(dim=1)
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


class RedMotionCrossFusion(pl.LightningModule):
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
        pred_3_points=False,
        num_fusion_layers=6,
        size_global_env_decoder_vocab=100,
        batch_size=96,
        dim_global_env_decoder=128,
        num_global_env_decoder_layers=6,
        num_heads_global_env_decoder=8,
        reduction_b_z_dim=512,
        reduction_feature_aggregation: str = "mean-var",
        mode: str = "fine-tuning",
    ) -> None:
        super().__init__()
        self.num_trajectory_proposals = num_trajectory_proposals
        self.prediction_horizon = prediction_horizon
        self.prediction_subsampling_rate = prediction_subsampling_rate
        self.lr = learning_rate
        self.epochs = epochs
        self.mode = mode
        self.pred_3_points = pred_3_points

        self.road_env_encoder = LocalRoadEnvEncoder(
            dim_model=dim_road_env_encoder,
            dim_attn_window_encoder=dim_road_env_attn_window,
        )
        self.ego_trajectory_encoder = EgoTrajectoryEncoder(
            dim_model=dim_ego_trajectory_encoder,
            dim_output=dim_road_env_encoder,
        )

        self.range_global_decoder_embedding = torch.arange(
            size_global_env_decoder_vocab
        ).expand(
            batch_size,
            size_global_env_decoder_vocab,
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_global_env_decoder_vocab,
            embedding_dim=dim_global_env_decoder - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_global_env_decoder_vocab,
            embedding_dim=10,
        )

        self.global_env_decoder = ParallelTransformerDecoder(
            num_layers=num_global_env_decoder_layers,
            dim_model=dim_global_env_decoder,
            num_heads=num_heads_global_env_decoder,
            dim_feedforward=dim_global_env_decoder * 4,  # As in regular transformer
        )

        self.fusion_embeddings = nn.Embedding(
            num_embeddings=2,
            embedding_dim=dim_global_env_decoder,
        )
        self.dim_global_env_encoder = dim_global_env_decoder

        self.cross_fusion_block = CrossTransformer(
            sm_dim=dim_road_env_encoder,
            lg_dim=dim_global_env_decoder,
            depth=num_fusion_layers,
            heads=8,
            dim_head=16,
            dropout=0.1,
        )

        self.local_fusion_block = REDFusionBlock(
            dim_model=dim_road_env_encoder, num_layers=num_fusion_layers
        )

        if pred_3_points:
            self.motion_head = nn.Sequential(
                nn.LayerNorm((dim_road_env_encoder,), eps=1e-06, elementwise_affine=True),
                nn.Linear(
                    in_features=dim_road_env_encoder,
                    out_features=num_trajectory_proposals * 3 * 2 + num_trajectory_proposals,
                ),
            )
        else:
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

        self.reduction_feature_aggregation = reduction_feature_aggregation
        self.learn_reduction_feats = nn.Linear(in_features=dim_global_env_decoder, out_features=16)
        if mode == "pre-training-traj-env" or reduction_feature_aggregation == "traj-env":
            self.projection_head = nn.Sequential(
                nn.Linear(
                    in_features=dim_global_env_decoder * 2,
                    out_features=1024
                ),  # Mean, var per token
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=reduction_b_z_dim),
            )
        elif mode == "pre-training-red-mae":
            self.road_env_encoder.mode = "traj-mae"
            self.global_mae_decoder = LocalTransformerEncoder(
                num_layers=3,
                dim_model=dim_road_env_encoder,
                dim_attn_window=dim_road_env_attn_window,
                dim_feedforward=dim_road_env_encoder * 4,
            )
            self.global_recon_seeds = nn.Embedding(
                num_embeddings=1200 - size_global_env_decoder_vocab, # Max num - global tokens
                embedding_dim=dim_global_env_decoder,
            )
            self.range_global_recon_seeds = torch.arange(1200 - size_global_env_decoder_vocab).expand(batch_size, 1200 - size_global_env_decoder_vocab)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(
                    in_features=size_global_env_decoder_vocab * 2 if reduction_feature_aggregation == "mean-var" else size_global_env_decoder_vocab * 16,
                    out_features=4096
                ),  # Mean, var per token
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=reduction_b_z_dim),
            )
        self.reduction_loss= BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=5e-3, z_dim=reduction_b_z_dim
        )

    def forward(
        self,
        env_idxs_src_tokens: Tensor,
        env_pos_src_tokens: Tensor,
        env_idxs_src_tokens_b: Tensor,
        env_pos_src_tokens_b: Tensor,
        env_src_mask: Tensor,
        ego_idxs_semantic_embedding: Tensor,
        ego_pos_src_tokens: Tensor,
    ):
        if self.mode == "pre-training-red-mae":
            road_env_tokens, initial_road_env_tokens = self.road_env_encoder(
                env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
            )

            self.range_global_decoder_embedding = self.range_global_decoder_embedding.to("cuda")

            global_env_tgt = torch.concat(
                (
                    self.decoder_semantic_embedding(self.range_global_decoder_embedding),
                    self.decoder_pos_embedding(self.range_global_decoder_embedding),
                ),
                dim=2,
            )
            global_env_tokens = self.global_env_decoder(
                global_env_tgt, road_env_tokens, env_src_mask
            )
            road_env_tokens_b, initial_road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask,
            )
            global_env_tokens_b = self.global_env_decoder(
                global_env_tgt, road_env_tokens_b, env_src_mask
            )
            self.range_global_recon_seeds = self.range_global_recon_seeds.to("cuda")
            
            recon_global_env_tokens_a = torch.concat(
                (global_env_tokens, self.global_recon_seeds(self.range_global_recon_seeds)), dim=1, # To get to max num of tokens
            )
            recon_global_env_tokens_b = torch.concat(
                (global_env_tokens_b, self.global_recon_seeds(self.range_global_recon_seeds)), dim=1, # To get to max num of tokens
            )
            recon_global_env_tokens_a = self.global_mae_decoder(recon_global_env_tokens_a, env_src_mask)
            recon_global_env_tokens_b = self.global_mae_decoder(recon_global_env_tokens_b, env_src_mask)

            # Reconstruct cross-wise to enforce redundancy reduction
            loss_a = F.mse_loss(recon_global_env_tokens_a[env_src_mask], initial_road_env_tokens_b[env_src_mask])
            loss_b = F.mse_loss(recon_global_env_tokens_b[env_src_mask], initial_road_env_tokens[env_src_mask])

            return loss_a + loss_b


        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )

        self.range_global_decoder_embedding = self.range_global_decoder_embedding.to("cuda")

        global_env_tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_global_decoder_embedding),
                self.decoder_pos_embedding(self.range_global_decoder_embedding),
            ),
            dim=2,
        )
        global_env_tokens = self.global_env_decoder(
            global_env_tgt, road_env_tokens, env_src_mask
        )

        if self.mode == "pre-training":
            road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask,
            )
            global_env_tokens_b = self.global_env_decoder(
                global_env_tgt, road_env_tokens_b, env_src_mask
            )
            
            if self.reduction_feature_aggregation == "mean-var":
                y_a = torch.concat(
                    (global_env_tokens.mean(dim=2), global_env_tokens.var(dim=2)), dim=1
                )
                y_b = torch.concat(
                    (global_env_tokens_b.mean(dim=2), global_env_tokens_b.var(dim=2)), dim=1
                )
            elif self.reduction_feature_aggregation == "learned":
                y_a = self.learn_reduction_feats(global_env_tokens)
                y_a = y_a.flatten(start_dim=1, end_dim=-1)
                y_b = self.learn_reduction_feats(global_env_tokens_b)
                y_b = y_b.flatten(start_dim=1, end_dim=-1)

            z_a = self.projection_head(y_a)
            z_b = self.projection_head(y_b)

            loss = self.reduction_loss(z_a, z_b)

            return loss
        
        # Learn the global env rep in a way that guides the future traj, by matching to past traj --> matches the reduction theme
        elif self.mode == "pre-training-traj-env": # Project trajectory and corresponding env into shared embedding space (sim. PreTraM)
            ego_trajectory_tokens = self.ego_trajectory_encoder(
                ego_idxs_semantic_embedding, ego_pos_src_tokens
            )
            road_env_tokens_b = self.road_env_encoder(
                env_idxs_src_tokens_b, env_pos_src_tokens_b, env_src_mask,
            )
            global_env_tokens_b = self.global_env_decoder(
                global_env_tgt, road_env_tokens_b, env_src_mask
            )
            y_a = torch.concat(
                (ego_trajectory_tokens.mean(dim=1), ego_trajectory_tokens.var(dim=1)), dim=1
            )
            y_b = torch.concat(
                (global_env_tokens_b.mean(dim=1), global_env_tokens_b.var(dim=1)), dim=1
            )

            z_a = self.projection_head(y_a)
            z_b = self.projection_head(y_b)

            loss = self.reduction_loss(z_a, z_b)

            return loss
        
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )

        fused_local_tokens = self.local_fusion_block(
            q=ego_trajectory_tokens,
            k=road_env_tokens,
            v=road_env_tokens,
        )

        # Add fusion tokens
        batch_size = env_idxs_src_tokens.size(dim=0)
        local_fusion_token = (
            self.fusion_embeddings(torch.tensor(0, device="cuda"))
            .to("cuda")
            .expand(batch_size, self.dim_global_env_encoder)
        )
        global_fusion_token = (
            self.fusion_embeddings(torch.tensor(1, device="cuda"))
            .to("cuda")
            .expand(batch_size, self.dim_global_env_encoder)
        )

        fused_local_tokens = torch.cat(
            (local_fusion_token[:, None, :], fused_local_tokens), dim=1
        )
        global_env_tokens = torch.cat(
            (global_fusion_token[:, None, :], global_env_tokens), dim=1
        )

        fused_local_tokens, fused_global_tokens = self.cross_fusion_block(
            fused_local_tokens, global_env_tokens
        )
        fused_tokens = torch.cat(
            (fused_local_tokens, fused_global_tokens[:, 0][:, None, :]), dim=1
        )

        motion_embedding = self.motion_head(
            fused_tokens.mean(dim=1)
        )  # Sim. to improved ViT global avg pooling before classification
        confidences_logits, logits = (
            motion_embedding[:, : self.num_trajectory_proposals],
            motion_embedding[:, self.num_trajectory_proposals :],
        )

        if self.pred_3_points:
            logits = logits.view(-1, self.num_trajectory_proposals, 3, 2)
        else:
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
        env_idxs_src_tokens_b = batch["sample_b"]["idx_src_tokens"]
        env_pos_src_tokens_b = batch["sample_b"]["pos_src_tokens"]
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

        if self.mode.startswith("pre-training"):
            loss = self.forward(
                env_idxs_src_tokens,
                env_pos_src_tokens,
                env_idxs_src_tokens_b,
                env_pos_src_tokens_b,
                env_src_mask,
                ego_idxs_semantic_embedding,
                ego_pos_src_tokens,
            )
            return loss

        confidences_logits, logits = self.forward(
            env_idxs_src_tokens,
            env_pos_src_tokens,
            env_idxs_src_tokens_b,
            env_pos_src_tokens_b,
            env_src_mask,
            ego_idxs_semantic_embedding,
            ego_pos_src_tokens,
        )

        if self.pred_3_points:
            loss = pytorch_neg_multi_log_likelihood_batch(
                y[:, (29, 49, 79)], logits, confidences_logits, is_available[:, (29, 49, 79)]
            )
        else:
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


def red_motion_inference(model, batch, device="cuda", return_all_numpy=False):
    model.mode = "fine-tuning"
    is_available = batch["future_ego_trajectory"]["is_available"].to(device)
    y = batch["future_ego_trajectory"]["trajectory"].to(device)

    env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"].to(device)
    env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"].to(device)
    env_src_mask = batch["src_attn_mask"].to(device)
    ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
        "idx_semantic_embedding"
    ].to(device)
    ego_pos_src_tokens = batch["past_ego_trajectory"]["pos_src_tokens"].to(device)

    confidences_logits, logits = model(
        env_idxs_src_tokens=env_idxs_src_tokens,
        env_pos_src_tokens=env_pos_src_tokens,
        env_src_mask=env_src_mask,
        ego_idxs_semantic_embedding=ego_idxs_semantic_embedding,
        ego_pos_src_tokens=ego_pos_src_tokens,
        env_idxs_src_tokens_b=None,
        env_pos_src_tokens_b=None,
    )
    confidences = torch.softmax(confidences_logits, dim=1)

    if not return_all_numpy:
        return confidences, logits
    else:
        logits_np = logits.squeeze(0).cpu().numpy()
        confidences_np = confidences.squeeze(0).cpu().numpy()
        is_available_np = is_available.squeeze(0).long().cpu().numpy()
        y_np = y.squeeze(0).cpu().numpy()

        return logits_np, confidences_np, is_available_np, y_np