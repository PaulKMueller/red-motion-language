import torch

from torch import Tensor


def get_masked_indices(
    masking_ratio: float,
    num_tokens: int,
    batch_size: int,
    device: str,
):
    """Get indices to mask for MAE pre-training
    Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
    """
    num_masked = int(masking_ratio * num_tokens)
    rand_indices = torch.rand(batch_size, num_tokens, device=device).argsort(dim=-1)
    masked_indices = rand_indices[:, :num_masked]

    return masked_indices


def mask_road_env_tokens(
    semantic_indices: Tensor,
    masking_ratio: float = 0.6,
    idx_pad_token: int = 10,
):
    batch_size, num_tokens = semantic_indices.shape
    device = semantic_indices.device
    masked_indices = get_masked_indices(masking_ratio, num_tokens, batch_size, device)
    batch_range = torch.arange(batch_size, device=device)[:, None]
    semantic_indices[batch_range, masked_indices] = idx_pad_token

    return semantic_indices
