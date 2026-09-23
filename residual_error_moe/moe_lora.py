"""
moe_lora.py -- residual-guided MoE-LoRA for the frozen DiT.

Design:
  - The base FeedForward F_l is frozen; adapters are added in parallel.
  - One shared, always-on expert
  - E routed experts, each a fully independent LoRA (its own A and B), so experts
    can occupy different subspaces and isolate conflicting gradients
  - Experts are pre-allocated to E_max slots and gated by active_mask
  - Routing is on the residual direction r/||r||, not the raw activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    """Independent LoRA: B(A(x)) with B init zero"""

    def __init__(self, dim, rank, A=None):
        super().__init__()
        # If A is provided (shared-A ablation), reuse it; else own it.
        self._shared_A = A is not None
        self.A = A if self._shared_A else nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.act = nn.GELU()
        if not self._shared_A:
            nn.init.normal_(self.A.weight, std=1.0 / rank)
        nn.init.zeros_(self.B.weight)  # start as no-op

    def forward(self, x):
        return self.B(self.act(self.A(x)))

    def reset_to_noop(self):
        nn.init.zeros_(self.B.weight)


class ResidualRouter(nn.Module):
    """Small MLP over the routing feature to logits over E_max experts.
    Only active experts participate (masked softmax)."""

    def __init__(self, feat_dim, e_max, hidden=256, use_timestep=False, temb_dim=0):
        super().__init__()
        self.e_max = e_max
        in_dim = feat_dim + (temb_dim if use_timestep else 0)
        self.use_timestep = use_timestep
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, e_max),
        )
        # Small init so early routing is near-uniform (avoids early collapse).
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def set_expert_row(self, idx, centroid):
        """Point a freshly-spawned expert's logit at its triggering cluster
        centroid, so it immediately focus on those tokens instead of starting
        from a random hyperplane"""
        with torch.no_grad():
            w = self.net[-1].weight           # [e_max, hidden]
            # Project centroid through the first layer to hidden space, set the
            # last-layer row to align with it.
            h = self.net[1](self.net[0](centroid.unsqueeze(0))).squeeze(0)
            w[idx] = h / (h.norm() + 1e-6)
            self.net[-1].bias[idx] = 0.0

    def forward(self, feat, active_mask, temb=None):
        if self.use_timestep and temb is not None:
            feat = torch.cat([feat, temb], dim=-1)
        logits = self.net(feat)                            # [..., e_max]
        # mask inactive experts to -inf before softmax
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~active_mask.view(1, -1), neg)
        return logits


class MoELoRALayer(nn.Module):
    """Wraps a frozen base FeedForward with a shared expert + routed experts."""

    def __init__(self, base_ff, dim, rank=64, e_max=8, top_k=1,
                 route_feat_dim=None, share_A=False,
                 use_timestep=False, temb_dim=0):
        super().__init__()
        self.base_ff = base_ff
        for p in self.base_ff.parameters():
            p.requires_grad_(False)

        self.dim = dim
        self.rank = rank
        self.e_max = e_max
        self.top_k = top_k
        self.share_A = share_A

        # shared always-on expert
        self.shared_expert = LoRAExpert(dim, rank)

        # optional shared A for routed experts (ablation)
        shared_A_mod = nn.Linear(dim, rank, bias=False) if share_A else None
        if share_A:
            nn.init.normal_(shared_A_mod.weight, std=1.0 / rank)

        self.experts = nn.ModuleList([
            LoRAExpert(dim, rank, A=shared_A_mod) for _ in range(e_max)
        ])

        # active_mask is a buffer (persists across save/load, not a parameter)
        self.register_buffer("active_mask", torch.zeros(e_max, dtype=torch.bool))

        route_feat_dim = route_feat_dim or dim
        self.router = ResidualRouter(route_feat_dim, e_max,
                                     use_timestep=use_timestep, temb_dim=temb_dim)

        # routing feature for the current batch, set externally each step
        # (residual direction). None means Stage 0 (no routing).
        self._route_feat = None
        self._temb = None
        self._hard_route = None  # [B] static per-image expert ids (GRASP-style)
        self._all_active = False  # Option C: all active experts sum, no routing

        # telemetry, refreshed each forward
        self.last_util = torch.zeros(e_max)
        self.last_gate_mean = torch.zeros(e_max)

    @property
    def active_count(self):
        return int(self.active_mask.sum().item())

    def set_routing_feature(self, feat, temb=None):
        """feat: [B, T, route_feat_dim] routing feature for this batch."""
        self._route_feat = feat
        self._temb = temb
        self._hard_route = None 

    def set_hard_route(self, bucket_per_image):
        """Static routing: bucket_per_image [B] gives a fixed expert
        id for every token of image b. No learned router, deterministic top-1.
        Clears the soft routing feature."""
        self._route_feat = None
        self._hard_route = bucket_per_image  # [B] long, values in [0, e_max)

    def set_all_active(self, flag=True):
        """ routing-free growth: every active expert applies to every
        token and the contributions are summed. Experts are still spawned online via
        activate_expert"""
        self._all_active = flag
        self._route_feat = None
        self._hard_route = None

    def activate_expert(self, idx, centroid=None):
        self.active_mask[idx] = True
        self.experts[idx].reset_to_noop()
        if centroid is not None:
            self.router.set_expert_row(idx, centroid)

    def forward(self, x, *args, **kwargs):
        # x: [B, T, dim]
        # It route on the full tensor, so chunking must be off on the host block.
        if self._route_feat is not None and self.active_count > 0:
            if x.shape[:-1] != self._route_feat.shape[:-1]:
                raise RuntimeError(
                    f"routing-feature shape {tuple(self._route_feat.shape[:-1])} != "
                    f"input {tuple(x.shape[:-1])}. Forward chunking must be disabled "
                    f"on MoE blocks (do not call enable_forward_chunking).")
        out = self.base_ff(x) + self.shared_expert(x)

        # routing-free growth. Every active expert applies to every token and the contributions are summed.
        # No router, inference-valid by design.
        if self._all_active and self.active_count > 0:
            for e in range(self.e_max):
                if not self.active_mask[e]:
                    continue
                contrib = self.experts[e](x)
                out = out + contrib.to(out.dtype)
            with torch.no_grad():
                # each active expert sees all tokens equally
                u = self.active_mask.float()
                self.last_util = (u / (u.sum() + 1e-9)).detach().cpu()
            return out

        # each image uses one fixed expert, no router.
        if self._hard_route is not None and self.active_count > 0:
            bucket = self._hard_route  # [B]
            for e in range(self.e_max):
                if not self.active_mask[e]:
                    continue
                img_mask = (bucket == e)          # [B] which images use expert e
                if not img_mask.any():
                    continue
                xe = x[img_mask]                   # [n_img, T, dim]
                contrib = self.experts[e](xe)
                out[img_mask] = out[img_mask] + contrib.to(out.dtype)
            with torch.no_grad():
                cnt = torch.bincount(bucket, minlength=self.e_max).float()
                self.last_util = (cnt / (cnt.sum() + 1e-9)).detach().cpu()
            return out

        if self.active_count == 0 or self._route_feat is None:
            return out

        k_eff = min(self.top_k, self.active_count)  # cannot exceed active
        logits = self.router(self._route_feat, self.active_mask, self._temb)  # [B,T,e_max]
        probs = F.softmax(logits, dim=-1)

        topv, topi = probs.topk(k_eff, dim=-1)               # [B,T,k_eff]
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)    # renorm over chosen

        # telemetry
        with torch.no_grad():
            flat = topi.reshape(-1)
            util = torch.bincount(flat, minlength=self.e_max).float()
            util /= (util.sum() + 1e-9)
            # keep telemetry on CPU
            self.last_util = util.detach().cpu()
            self.last_gate_mean = probs.mean(dim=(0, 1)).detach().cpu()

        # dispatch: loop experts, apply to tokens that selected them
        for e in range(self.e_max):
            if not self.active_mask[e]:
                continue
            sel = (topi == e)                                # [B,T,k_eff] bool
            if not sel.any():
                continue
            gate = (topv * sel).sum(-1, keepdim=True)        # [B,T,1] gate for e
            mask = gate.squeeze(-1) > 0
            if mask.any():
                contrib = self.experts[e](x[mask])           # [n_sel, dim]
                update = (gate[mask] * contrib).to(out.dtype)
                out[mask] = out[mask] + update
        return out


def inject_moe_lora(dit, layer_indices, dim, rank=64, e_max=8, top_k=1,
                    route_feat_dim=None, share_A=False, use_timestep=False, temb_dim=0):
    """Replace the FeedForward at each given transformer block with a
    MoELoRALayer wrapping the frozen original"""
    injected = {}
    for li in layer_indices:
        block = dit.transformer_blocks[li]
        base_ff = block.ff
        moe = MoELoRALayer(base_ff, dim, rank=rank, e_max=e_max, top_k=top_k,
                           route_feat_dim=route_feat_dim, share_A=share_A,
                           use_timestep=use_timestep, temb_dim=temb_dim)
        block.ff = moe
        injected[li] = moe
    return dit, injected


def trainable_parameters(injected):
    """All adapter + router params across injected layers (base stays frozen)."""
    for moe in injected.values():
        yield from moe.shared_expert.parameters()
        for e in moe.experts:
            yield from e.parameters()
        yield from moe.router.parameters()