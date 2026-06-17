import torch
import torch.nn as nn

# =========================================================
# TOP-K + STRAIGHT-THROUGH ESTIMATOR
# =========================================================

class TopKStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, k):

        """
        z: latent vector (B, D)
        k: number of active dimensions
        """

        # ------------------------------------
        # 1. compute magnitude
        # ------------------------------------
        abs_z = z.abs()

        # ------------------------------------
        # 2. get top-k indices
        # ------------------------------------
        topk_vals, topk_idx = torch.topk(abs_z, k, dim=-1)

        # ------------------------------------
        # 3. build mask
        # ------------------------------------
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk_idx, 1.0)

        # ------------------------------------
        # 4. apply mask
        # ------------------------------------
        z_sparse = z * mask

        # save mask for backward (STE)
        ctx.save_for_backward(mask)

        return z_sparse

    @staticmethod
    def backward(ctx, grad_output):

        """
        Straight-Through Estimator:
        gradient passes as if identity function
        """

        mask, = ctx.saved_tensors

        # gradient only flows through selected elements
        grad_input = grad_output * mask

        # no gradient for k (discrete)
        return grad_input, None


# =========================================================
# SPARSE LATENT MODULE
# =========================================================

class SparseLatent(nn.Module):

    def __init__(self):
        super(SparseLatent, self).__init__()

        self.topk_fn = TopKStraightThrough

    def forward(self, z, k):

        """
        z: latent representation (B, D)
        k: dynamic number of active units
        """

        # safety check
        k = int(k)
        k = max(1, min(k, z.size(-1)))

        z_sparse = self.topk_fn.apply(z, k)

        return z_sparse