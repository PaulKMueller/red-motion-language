import torch
import torch.nn.functional as F

from torch import Tensor


def get_info_nce_loss(z_i: Tensor, z_j: Tensor, temperature: float) -> Tensor:
    """Contrastive InfoNCE loss function
    adapted from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#SimCLR
    """
    feats = torch.concat((z_i, z_j), dim=0)

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find positive example -> batch_size // 2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll


def get_tmcl_loss(map_embeddings, traj_embeddings, temperature):
    return get_info_nce_loss(map_embeddings, traj_embeddings, temperature)


def get_mcl_loss(map_embeddings_i, map_embeddings_j, temperature):
    return get_info_nce_loss(map_embeddings_i, map_embeddings_j, temperature)


def get_pretram_loss(
    map_embeddings_i: Tensor,
    map_embeddings_j: Tensor,
    traj_embeddings: Tensor,
    tmcl_temperature: float = 0.07,
    mcl_temperature: float = 0.07,
    λ: float = 1.0,
):
    """PreTraM loss, a combination of trajectory-map and map contrastive learning
    Default values from: https://github.com/chenfengxu714/PreTraM/blob/master/cfg/nuscenes/configs/tmcl_mcl_pretrain.yml
    """
    tmcl_loss = get_tmcl_loss(map_embeddings_i, traj_embeddings, tmcl_temperature)
    mcl_loss = get_mcl_loss(map_embeddings_i, map_embeddings_j, mcl_temperature)

    return tmcl_loss + λ * mcl_loss
