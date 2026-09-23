import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, rank=16, lora_alpha=32, expert_volume=4.0):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # Standard LoRA scaler.
        self.scaling = lora_alpha / rank

        # It multiplies the expert output at train and inference time and acts as a constant gradient multiplier on A and
        # B. It was a learnable Parameter originally, but the gradient that shrank B toward zero shrank the volume too, so the expert decayed.
        # Total output multiplier = scaling * volume.
        self.register_buffer("expert_volume", torch.ones(1) * float(expert_volume))

        nn.init.orthogonal_(self.A.weight)
        # nn.init.zeros_(self.B.weight)
        nn.init.normal_(self.B.weight, std=1e-3)

    def forward(self, x):
        raw_lora = self.B(self.A(x))
        return raw_lora * self.scaling * self.expert_volume


class DynamicMoELoRA_MLP(nn.Module):
    """
    Top-k MoE-LoRA over a frozen base FFN.

    Notes on a few non-obvious choices:
      - The GELU variant is read off the frozen base FFN (no hardcoded value)
      - The router is timestep-aware
      - Clustering capture is timestep-free by construction. The block feeds us
            x = norm3(h) * (1 + scale_mlp) + shift_mlp
        with shift/scale pure functions of the timestep, x_content = (x - shift_mlp) / (1 + scale_mlp) = norm3(h)
        and the orchestrator clusters content, not noise level. 
      - The update-magnitude diagnostic is decomposed into its factors (how many
        tokens got an expert, how much gate weight they got, how big the expert output is).
      - Telemetry keeps top-1 (argmax) and top-k accounting separate and reports entropy against the live ceiling ln(n_active+1).
      - skip_mode makes the handling of the Skip weight mass an explicit.
    """

    def __init__(self, base_up_proj, base_down_proj, hidden_dim,
                 max_experts=16, rank=16, entropy_threshold=1.0, top_k=2,
                 expert_volume=4.0, gelu_approximate="none", skip_mode="compete",
                 router_mode="shared"):
        super().__init__()

        self.base_up = base_up_proj
        self.base_down = base_down_proj
        self.base_up.requires_grad_(False)
        self.base_down.requires_grad_(False)

        # Match the frozen base FFN's activation exactly
        assert gelu_approximate in ("none", "tanh"), gelu_approximate
        self.gelu_approximate = gelu_approximate

        assert skip_mode in ("compete", "gate"), skip_mode
        self.skip_mode = skip_mode

        # Router topology: the three ablation arms (see paper for more information).
        #
        #   'skip'     [Skip, E0..EN] routed. The original design.
        #   'no_skip'  [E0..EN] routed. Slot 0 masked out entirely.
        #   'shared'   [E0..EN] routed, plus one always-on shared expert applied
        #              to every token unconditionally (DeepSeekMoE "Shared Expert
        #              Isolation", Dai et al. 2024, arXiv:2401.06066).
        #
        # The frozen base FFN always runs regardless of mode (the MoE is a
        # residual add-on, `return base_out + moe_update`).
        assert router_mode in ("skip", "no_skip", "shared"), router_mode
        self.router_mode = router_mode
        self.use_skip = (router_mode == "skip")

        self.max_experts = max_experts
        self.top_k = top_k
        intermediate_dim = self.base_up.out_features
        self.hidden_dim = hidden_dim
        
        
        # All experts are initialized and masked out, if not in used, to ensure that the structure of the model 
        # stays the same across all epochs
        self.up_experts = nn.ModuleList(
            [LoRAExpert(hidden_dim, intermediate_dim, rank, expert_volume=expert_volume)
             for _ in range(max_experts)])
        self.down_experts = nn.ModuleList(
            [LoRAExpert(intermediate_dim, hidden_dim, rank, expert_volume=expert_volume)
             for _ in range(max_experts)])

        # The shared expert is always constructed, even in 'skip'/'no_skip' mode where it is inert and frozen.
        # This keeps the state_dict shape identical across all three arms, so checkpoints are interchangeable, 
        # an ablation is a one-flag change and a run can resume under a different arm.
        self.shared_up = LoRAExpert(hidden_dim, intermediate_dim, rank,
                                    expert_volume=expert_volume)
        self.shared_down = LoRAExpert(intermediate_dim, hidden_dim, rank,
                                      expert_volume=expert_volume)
        if router_mode != "shared":
            self.shared_up.requires_grad_(False)
            self.shared_down.requires_grad_(False)

        # Slot 0 is the Skip path; slots 1..N are experts. The router keeps width
        # max_experts+1 in every mode; in 'no_skip'/'shared' slot 0 is masked to
        # dtype.min, so it gets exactly zero probability and zero gradient. Same
        # tensor shapes, same slot indices, no branching downstream.
        self.router = nn.Linear(hidden_dim, max_experts + 1)
        nn.init.orthogonal_(self.router.weight)
        nn.init.constant_(self.router.bias, 0.0)
        nn.init.constant_(self.router.bias[0], 1.0)

        self.router_ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        nn.init.zeros_(self.router_ada_lin[1].weight)
        nn.init.zeros_(self.router_ada_lin[1].bias)

        self.register_buffer("active_mask", torch.zeros(max_experts, dtype=torch.bool))
        self.active_mask[:2] = True

        self.register_buffer("ema_entropy_threshold", torch.tensor(1.0))
        self.ema_decay = 0.99

        self.current_gumbel_temp = 1.0
        self.current_active_k = 0
        self.latest_unsure_mask = None
        self.latest_poorly_served_mask = None

        # per-token capture for the current forward pass (in content space)
        self.latest_unsure_tokens = None
        self.latest_unsure_rows = None
        self.latest_unsure_pos = None
        self.latest_probs = None
        self.latest_logits = None
        self.latest_balance_loss = None

        # update-magnitude components
        self.latest_update_ratio = 0.0          # global, back-compatible
        self.latest_update_ratio_routed = 0.0   # over tokens that actually got an expert
        self.latest_shared_ratio = 0.0          # always-on shared expert alone
        self.latest_routed_frac = 0.0
        self.latest_mean_gate_weight = 0.0

        self.force_expert_idx = None
        self.force_token_mask = None

        # Warm-start scratch. When a fresh expert is trained offline on a cluster,
        # its router row is still random-orthogonal, so once it comes back online
        # the router sends it near-uniform traffic until it slowly re-learns. We
        # accumulate, during the offline burst, the mean router-input vector of
        # the cluster tokens and all tokens, then point the new router row
        # along (cluster_mean - global_mean).
        self._ws_collect = False
        self._ws_cluster_sum = None
        self._ws_cluster_count = 0
        self._ws_global_sum = None
        self._ws_global_count = 0

        # Stashed by the injection hooks. Not parameters and not buffers: they are
        # per-forward scratch and must never be checkpointed.
        self._hook_t_emb = None
        self._hook_shift_mlp = None
        self._hook_scale_mlp = None
        self._warned_no_temb = False
        self._warned_no_adaln = False

        self.reset_epoch_telemetry()


    def reset_epoch_telemetry(self):
        # Two separate metrics:
        #   epoch_topk_counts : how many times each slot appeared in the top-k.
        #                       Sums to k * tokens_seen (what the experts actually
        #                       computed on).
        #   epoch_top1_counts : argmax choice. Sums to tokens_seen (the router's
        #                       preference). Kept separate from the top-k count.
        self.epoch_topk_counts = torch.zeros(self.max_experts + 1, dtype=torch.long)
        self.epoch_top1_counts = torch.zeros(self.max_experts + 1, dtype=torch.long)
        self.epoch_tokens_seen = 0
        self.epoch_selections = 0

        self.epoch_gating_entropy = 0.0
        self.telemetry_steps = 0

        # update-magnitude accumulators
        self.epoch_update_ratio_sum = 0.0
        self.epoch_update_ratio_routed_sum = 0.0
        self.epoch_routed_frac_sum = 0.0
        self.epoch_gate_weight_sum = 0.0
        self.epoch_shared_ratio_sum = 0.0
        self.epoch_update_steps = 0

        # specialisation accumulators, in content space
        self.epoch_expert_token_sum = None
        self.epoch_expert_token_count = torch.zeros(self.max_experts, dtype=torch.long)
        # The global content-token mean. What matters is the deviation from shared population mean, 
        # which is the space the clustering pipeline already works in (it mean-centres first).
        self.epoch_global_token_sum = None
        self.epoch_global_token_count = 0

    def calculate_entropy(self, probs, eps=1e-9):
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    def entropy_ceiling(self):
        """ln(n_paths) for the currently active set. n_paths excludes Skip when the arm doesn't use it."""
        return math.log(int(self.active_mask.sum().item()) + (1 if self.use_skip else 0))

    def _gelu(self, t):
        return F.gelu(t, approximate=self.gelu_approximate)

    def _content_features(self, x):
        """
        Invert the block's AdaLN-Zero MLP modulation to recover the timestep-free
        content representation:

            x         = norm3(h) * (1 + scale_mlp) + shift_mlp
            x_content = (x - shift_mlp) / (1 + scale_mlp) = norm3(h)

        shift_mlp/scale_mlp are [B, hidden] and are captured from the block's
        norm1 by a forward hook (they are the exact tensors the block used on this
        forward, so the inverse is exact).
        """
        shift, scale = self._hook_shift_mlp, self._hook_scale_mlp
        if shift is None or scale is None:
            if not self._warned_no_adaln:
                print("[MoE][WARN] AdaLN shift/scale unavailable. "
                      "Clustering will be timestep-contaminated.",
                      flush=True)
                self._warned_no_adaln = True
            return x
        if shift.shape[0] != x.shape[0]:
            # stale hook value from a different-sized forward
            return x

        shift = shift.to(x.dtype).unsqueeze(1)          # [B,1,hidden]
        denom = 1.0 + scale.to(x.dtype).unsqueeze(1)    # [B,1,hidden]
        # AdaLN-Zero starts at scale=0 so denom starts at exactly 1, but nothing
        # constrains it later. guard the division without changing the sign.
        eps = 1e-3
        sgn = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        denom = torch.where(denom.abs() < eps, sgn * eps, denom)
        return (x - shift) / denom

    def _switch_balance_loss(self, probs, topk_idx_full):
        # Adaptation of the switch Transfomer load balancing auxiliary loss for the 
        # skip slot and top-routing (Fedus)
        active = self.active_mask
        n_active = int(active.sum().item())
        if n_active == 0:
            return probs.new_tensor(0.0)

        expert_probs = probs[..., 1:]
        # probs[...,1:] sums to (1 - p_skip), not to 1. The Switch loss
        # N*sum(f_i*P_i) assumes P is a distribution; with sum(P) free it can be
        # reduced by shrinking every P_i, i.e. by pushing p_skip -> 1, which is
        # the only router gradient present at init (B=0 => moe_update==0 =>
        # d(MSE)/d(router)==0). Renormalising so P sums to 1 makes the loss
        # invariant to p_skip, so it can only be reduced by balancing experts.
        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        P = expert_probs.reshape(-1, expert_probs.size(-1)).mean(dim=0)

        sel = topk_idx_full.reshape(-1)
        sel = sel[sel > 0] - 1
        if sel.numel() == 0:
            return probs.new_tensor(0.0)
        
        f = torch.bincount(sel, minlength=self.max_experts).float()
        f = f / f.sum().clamp(min=1.0)
        f = f.to(P.device)

        mask = active.to(P.device).float()
        loss = (f * P * mask).sum() * n_active
        return loss


    def forward(self, x, t_emb=None):
        # base model
        base_hidden = self.base_up(x)
        base_hidden = self._gelu(base_hidden)
        base_out = self.base_down(base_hidden)

        # shared expert if needed
        if self.router_mode == "shared":
            shared_hidden = self._gelu(base_hidden + self.shared_up(x))
            shared_delta = self.shared_down(shared_hidden).to(base_out.dtype)
        else:
            shared_delta = None

        # routing
        if t_emb is None:
            t_emb = self._hook_t_emb
        if t_emb is not None and t_emb.shape[0] != x.shape[0]:
            t_emb = None      # stale; never silently modulate with the wrong batch

        if t_emb is not None:
            # timestep-aware router
            ada_params = self.router_ada_lin(t_emb.to(x.dtype)).unsqueeze(1)
            shift, scale = ada_params.chunk(2, dim=-1)
            router_input = x * (1 + scale) + shift
        else:
            if not self._warned_no_temb:
                print("[MoE][WARN] no t_emb available routing is only implicitly timestep-aware via x. "
                      "Check inject_moe_into_dit's hooks.", flush=True)
                self._warned_no_temb = True
            router_input = x

        clean_logits = self.router(router_input)
        self.latest_logits = clean_logits

        # Warm-start collection: during the offline spawn burst only
        if (self.training and self._ws_collect
                and self.force_token_mask is not None):
            with torch.no_grad():
                ri = router_input.detach().float()               # [B,T,H]
                ri_flat = ri.reshape(-1, ri.size(-1))
                if self._ws_global_sum is None:
                    self._ws_global_sum = torch.zeros(ri.size(-1), device=ri.device)
                    self._ws_cluster_sum = torch.zeros(ri.size(-1), device=ri.device)
                self._ws_global_sum += ri_flat.sum(0)
                self._ws_global_count += int(ri_flat.size(0))
                fm = self.force_token_mask
                if fm.any():
                    self._ws_cluster_sum += ri[fm].reshape(-1, ri.size(-1)).sum(0)
                    self._ws_cluster_count += int(fm.sum().item())

        if self.training:
            gumbel_noise = -torch.empty_like(clean_logits).exponential_().log()
            logits = clean_logits + (gumbel_noise * self.current_gumbel_temp)
        else:
            logits = clean_logits

        # slot 0 (Skip) is live only in the 'skip' arm.
        skip_slot = torch.full((1,), self.use_skip, dtype=torch.bool, device=x.device)
        router_mask = torch.cat([skip_slot, self.active_mask])
        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~router_mask, min_value)
        probs = F.softmax(logits, dim=-1)
        self.latest_probs = probs

        __, router_choices = torch.max(probs, dim=-1)

        # unsure tokens and buffer if entropy high
        if self.force_expert_idx is not None and self.force_token_mask is not None:
            unsure_mask = torch.zeros_like(router_choices, dtype=torch.bool)
        else:
            min_value_clean = torch.finfo(clean_logits.dtype).min
            clean_masked_logits = clean_logits.masked_fill(~router_mask, min_value_clean)
            clean_probs = F.softmax(clean_masked_logits, dim=-1)
            entropy = self.calculate_entropy(clean_probs)

            flat_entropy = entropy.view(-1)
            batch_threshold = torch.quantile(flat_entropy.float(), 0.95)
            if self.training:
                self.ema_entropy_threshold.mul_(self.ema_decay).add_(
                    batch_threshold.detach() * (1 - self.ema_decay))
            unsure_mask = entropy > self.ema_entropy_threshold

            # expert-performance filter 
            if self.latest_poorly_served_mask is not None:
                psm = self.latest_poorly_served_mask
                if psm.shape == unsure_mask.shape:
                    unsure_mask = unsure_mask & psm.to(unsure_mask.device)
                else:
                    self.latest_poorly_served_mask = None

            if self.training:
                self.latest_unsure_mask = unsure_mask.detach().cpu()
                with torch.no_grad():
                    # capture in content space, not raw x
                    x_content = self._content_features(x)
                    nz = unsure_mask.nonzero(as_tuple=False)
                    if nz.numel() > 0:
                        rows, pos = nz[:, 0], nz[:, 1]
                        self.latest_unsure_tokens = x_content[rows, pos].detach().float().cpu()
                        self.latest_unsure_rows = rows.detach().cpu()
                        self.latest_unsure_pos = pos.detach().cpu()
                    else:
                        self.latest_unsure_tokens = None
                        self.latest_unsure_rows = None
                        self.latest_unsure_pos = None

                    # top-1 account, kept separate from the top-k one
                    self.epoch_top1_counts += torch.bincount(
                        router_choices.reshape(-1), minlength=self.max_experts + 1).cpu()
                    self.epoch_tokens_seen += int(router_choices.numel())

                    mean_probs = clean_probs.mean(dim=(0, 1)) + 1e-10
                    self.epoch_gating_entropy += -torch.sum(mean_probs * torch.log(mean_probs)).item()
                    self.telemetry_steps += 1

        # expert evaluation
        moe_update = torch.zeros_like(base_out)

        # offline spawn: hard single-expert routing
        if self.force_expert_idx is not None and self.force_token_mask is not None:
            i = int(self.force_expert_idx)
            token_mask = self.force_token_mask
            if token_mask.any():
                selected = x[token_mask]
                up_lora_out = self.up_experts[i](selected)
                hidden_state = self._gelu(base_hidden[token_mask] + up_lora_out)
                down_lora_out = self.down_experts[i](hidden_state)
                moe_update[token_mask] = down_lora_out.to(moe_update.dtype)
            self.latest_balance_loss = base_out.new_tensor(0.0)
            if self.training:
                with torch.no_grad():
                    self._record_diag1(base_out, moe_update, token_mask,
                                       gate_weight_sum=token_mask.float().sum(),
                                       shared_delta=shared_delta)
            # The shared expert still applies during an offline spawn: the new routed expert learns the
            #  residual that remains after the shared correction, not the shared part. 
            out = base_out + moe_update
            return out + shared_delta if shared_delta is not None else out

        active_count = int(self.active_mask.sum().item())
        if active_count == 0:
            self.latest_balance_loss = base_out.new_tensor(0.0)
            return base_out + shared_delta if shared_delta is not None else base_out

        # k must stay strictly below the number of available paths, else every
        # path is always selected, routing degenerates into a dense sum and there
        # is no specialisation reason.
        n_paths = active_count + (1 if self.use_skip else 0)
        dynamic_k = max(1, min(self.top_k, n_paths - 1))
        topk_vals, topk_idx_full = torch.topk(probs, dynamic_k, dim=-1)
        if self.training:
            self.current_active_k = dynamic_k

        # adjust probabilities if skip mode available
        if self.skip_mode == "gate":
            expert_sel = (topk_idx_full > 0).float()
            masked_vals = topk_vals * expert_sel
            topk_weights = masked_vals / (masked_vals.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training:
            with torch.no_grad():
                self.epoch_topk_counts += torch.bincount(
                    topk_idx_full.reshape(-1), minlength=self.max_experts + 1).cpu()
                self.epoch_selections += int(topk_idx_full.numel())

        # content-space features for the specialisation/centroid statistics only
        x_content = self._content_features(x) if self.training else None
        if self.training:
            with torch.no_grad():
                xc_flat = x_content.reshape(-1, x_content.size(-1)).float()
                if self.epoch_global_token_sum is None:
                    self.epoch_global_token_sum = torch.zeros(
                        xc_flat.size(-1), device=xc_flat.device, dtype=torch.float32)
                self.epoch_global_token_sum += xc_flat.sum(dim=0)
                self.epoch_global_token_count += int(xc_flat.size(0))

        gate_weight_total = torch.zeros(base_out.shape[:2], device=base_out.device,
                                        dtype=torch.float32)

        for slot in range(1, self.max_experts + 1):
            i = slot - 1
            if not self.active_mask[i]:
                continue
            sel = (topk_idx_full == slot)
            if not sel.any():
                continue
            w = (topk_weights * sel.float()).sum(dim=-1)
            token_mask = w > 0
            if not token_mask.any():
                continue

            selected = x[token_mask]                       # experts see RAW x
            up_lora_out = self.up_experts[i](selected)
            hidden_state = self._gelu(base_hidden[token_mask] + up_lora_out)
            down_lora_out = self.down_experts[i](hidden_state)
            contrib = down_lora_out * w[token_mask].unsqueeze(-1)
            moe_update[token_mask] += contrib.to(moe_update.dtype)

            if self.training:
                gate_weight_total += w.float()
                with torch.no_grad():
                    sel_content = x_content[token_mask]
                    if self.epoch_expert_token_sum is None:
                        self.epoch_expert_token_sum = torch.zeros(
                            self.max_experts, sel_content.size(-1),
                            device=sel_content.device, dtype=torch.float32)
                    self.epoch_expert_token_sum[i] += sel_content.float().sum(dim=0)
                    self.epoch_expert_token_count[i] += sel_content.size(0)

        if self.training:
            self.latest_balance_loss = self._switch_balance_loss(probs, topk_idx_full)
            with torch.no_grad():
                routed = gate_weight_total > 0
                self._record_diag1(base_out, moe_update, routed,
                                   gate_weight_sum=gate_weight_total.sum(),
                                   shared_delta=shared_delta)
        else:
            self.latest_balance_loss = base_out.new_tensor(0.0)

        out = base_out + moe_update
        return out + shared_delta if shared_delta is not None else out

    # diagnostic
    @torch.no_grad()
    def _record_diag1(self, base_out, moe_update, routed_mask, gate_weight_sum,
                      shared_delta=None):
        """
        Record the expert update magnitude ||moe_update|| / ||base_out|| (how big the experts' contribution
        is relative to the frozen base FFN), broken into: the fraction of tokens routed to an expert, 
        the mean gate weight they received and the raw expert output magnitude.
        """
        n_tok = int(routed_mask.numel())
        n_routed = int(routed_mask.sum().item())

        ratio = (moe_update.norm() / (base_out.norm() + 1e-6)).item()

        if n_routed > 0:
            ratio_routed = (moe_update[routed_mask].norm()
                            / (base_out[routed_mask].norm() + 1e-6)).item()
            mean_w = float(gate_weight_sum) / n_routed
        else:
            ratio_routed, mean_w = 0.0, 0.0

        routed_frac = n_routed / max(1, n_tok)

        # The shared expert's contribution is reported separately. It is always-on, 
        # so folding it into update_ratio would hide whether the routed experts are doing anything:
        #  the shared expert alone could make the number look healthy while every routed expert stayed cosmetic.
        shared_ratio = 0.0
        if shared_delta is not None:
            shared_ratio = (shared_delta.norm() / (base_out.norm() + 1e-6)).item()
        self.latest_shared_ratio = shared_ratio
        self.epoch_shared_ratio_sum += shared_ratio

        self.latest_update_ratio = ratio
        self.latest_update_ratio_routed = ratio_routed
        self.latest_routed_frac = routed_frac
        self.latest_mean_gate_weight = mean_w

        self.epoch_update_ratio_sum += ratio
        self.epoch_update_ratio_routed_sum += ratio_routed
        self.epoch_routed_frac_sum += routed_frac
        self.epoch_gate_weight_sum += mean_w
        self.epoch_update_steps += 1

    def specialization_report(self):
        n = max(1, self.epoch_update_steps)
        active = [int(a) for a in torch.where(self.active_mask)[0].tolist()]
        per_expert = [(a, int(self.epoch_expert_token_count[a].item())) for a in active]
        used = [a for a in active if self.epoch_expert_token_count[a].item() > 0]

        offdiag = float('nan')
        spec_strength = float('nan')
        if (self.epoch_expert_token_sum is not None
                and self.epoch_global_token_sum is not None
                and self.epoch_global_token_count > 0 and len(used) >= 1):
            counts = self.epoch_expert_token_count.to(self.epoch_expert_token_sum.device)
            counts = counts.clamp(min=1).unsqueeze(-1).float()
            means = self.epoch_expert_token_sum / counts
            gmean = self.epoch_global_token_sum / self.epoch_global_token_count

            # Subtract the global token mean before comparing. Without this the
            # off-diagonal cosine reads near 1.0 even for perfectly specialised
            # experts, because every expert mean is mu + (small deviation) and
            # ||mu|| dwarfs the deviation. The clustering pipeline mean-centres;
            # this measurement does too.
            dev = means[used] - gmean.unsqueeze(0)

            # Scale-free: how far this expert's tokens sit from the population,
            # relative to the population itself. Near 0 means the router is not
            # discriminating, whatever the token counts say.
            spec_strength = (dev.norm(dim=-1).mean() / (gmean.norm() + 1e-6)).item()

            if len(used) >= 2:
                m = F.normalize(dev, dim=-1)
                sim = m @ m.t()
                U = sim.size(0)
                offdiag = ((sim.sum() - sim.diagonal().sum()) / (U * (U - 1))).item()

        return {
            "update_ratio": self.epoch_update_ratio_sum / n,
            "update_ratio_routed": self.epoch_update_ratio_routed_sum / n,
            "routed_frac": self.epoch_routed_frac_sum / n,
            "mean_gate_weight": self.epoch_gate_weight_sum / n,
            "shared_ratio": self.epoch_shared_ratio_sum / n,
            "router_mode": self.router_mode,
            "offdiag_cos": offdiag,
            "spec_strength": spec_strength,
            "n_used": len(used),
            "per_expert": per_expert,
        }

    def config_summary(self):
        """
        Everything that changes the model's semantics but not its state_dict shapes.
        Persist this in the checkpoint (`moe_config`) and have eval read it back,
        otherwise a 'shared' checkpoint loads and runs as 'skip' with no error raised.
        """
        return {
            "router_mode": self.router_mode,
            "skip_mode": self.skip_mode,
            "top_k": self.top_k,
            "max_experts": self.max_experts,
            "expert_volume": float(self.up_experts[0].expert_volume.item()),
            "gelu_approximate": self.gelu_approximate,
            "arch_rev": "router-modes-v1",
        }

    def get_expert_input_centroid(self, expert_idx, min_tokens=200):
        """
        The direction in which this expert's routed tokens deviate from the global
        token population, in content space:

            centroid_i = mean(tokens routed to i) - mean(all tokens)

        Returning the deviation (rather than the raw mean) keeps seeds in the same
        mean-centred space as the cluster centroids and keeps them non-parallel.
        """
        if self.epoch_expert_token_sum is None or self.epoch_global_token_sum is None:
            return None
        if self.epoch_global_token_count == 0:
            return None
        count = int(self.epoch_expert_token_count[expert_idx].item())
        if count < min_tokens:
            return None
        mean_i = self.epoch_expert_token_sum[expert_idx] / count
        gmean = self.epoch_global_token_sum / self.epoch_global_token_count
        return (mean_i - gmean).detach().float().cpu()

    # warm-start: give a freshly spawned expert a router row that already points
    # at its cluster, so it is used from the first online step instead of starving
    # until the router happens to re-discover it.
    def ws_begin(self):
        """Enable router-input mean collection for the next offline burst."""
        self._ws_collect = True
        self._ws_cluster_sum = None
        self._ws_cluster_count = 0
        self._ws_global_sum = None
        self._ws_global_count = 0

    def ws_reset(self):
        self._ws_collect = False
        self._ws_cluster_sum = None
        self._ws_cluster_count = 0
        self._ws_global_sum = None
        self._ws_global_count = 0

    @torch.no_grad()
    def warm_start_router_for_expert(self, expert_idx, strength=1.0):
        """
        Point the router row for `expert_idx`'s slot along the cluster's deviation
        from the global mean, in the router's own input space (means gathered by
        ws_begin() during the offline burst). Returns True if applied.

        `strength` scales the new row relative to the mean existing row norm. 1.0
        makes it directly competitive with trained experts
        """
        slot = expert_idx + 1
        if (self._ws_cluster_count < 1 or self._ws_global_count < 1
                or self._ws_cluster_sum is None):
            print("   [warm-start] no router-input stats collected; skipped.", flush=True)
            return False
        cmean = self._ws_cluster_sum / max(1, self._ws_cluster_count)
        gmean = self._ws_global_sum / max(1, self._ws_global_count)
        dev = cmean - gmean
        n = dev.norm()
        if n < 1e-6:
            print("   [warm-start] cluster mean ~= global mean; skipped.", flush=True)
            self.ws_reset()
            return False
        dirv = dev / n
        expert_rows = self.router.weight[1:1 + self.max_experts]
        row_norm = expert_rows.norm(dim=1).mean().clamp(min=1e-6)
        self.router.weight[slot].copy_(
            (dirv * row_norm * float(strength)).to(self.router.weight.dtype))
        self.router.bias[slot].copy_(
            self.router.bias[1:1 + self.max_experts].mean())
        cos_to_others = F.cosine_similarity(
            dirv.unsqueeze(0), F.normalize(expert_rows.float(), dim=1), dim=1)
        print(f"   [warm-start] router row {slot} aligned to cluster deviation "
              f"(|dev|={n:.3f}, max cos to existing rows={cos_to_others.max().item():+.3f}).",
              flush=True)
        self.ws_reset()
        return True


def inject_moe_into_dit(model, layer_idx=6, max_experts=16, rank=16,
                        entropy_threshold=1.0, top_k=2, expert_volume=4.0,
                        skip_mode="compete", router_mode=None):
    """
    router_mode has no default on purpose: it must be passed explicitly.

    The arms deliberately share identical state_dict shapes (so checkpoints stay
    interchangeable), which means a flag that is parsed but never forwarded to
    this call produces no error and cannot be recovered from the checkpoint.
    """
    if router_mode is None:
        raise TypeError(
            "inject_moe_into_dit(): router_mode must be passed EXPLICITLY, e.g.\n"
            "    inject_moe_into_dit(..., router_mode=args.router_mode)\n"
            "Valid: 'skip' | 'no_skip' | 'shared'. There is no default: the arms "
            "share state_dict shapes, so a silent default cannot be detected later "
            "from the checkpoint."
        )
    target_block = model.transformer_blocks[layer_idx]
    base_gelu = target_block.ff.net[0]
    base_up_proj = base_gelu.proj
    base_down_proj = target_block.ff.net[2]
    hidden_dim = base_up_proj.in_features

    # Read the activation variant off the base module instead of hardcoding.
    # diffusers' GELU stores it as `.approximate` ("tanh" for "gelu-approximate").
    gelu_approximate = getattr(base_gelu, "approximate", "none")

    moe_module = DynamicMoELoRA_MLP(
        base_up_proj=base_up_proj, base_down_proj=base_down_proj,
        hidden_dim=hidden_dim, max_experts=max_experts, rank=rank,
        entropy_threshold=entropy_threshold, top_k=top_k,
        expert_volume=expert_volume, gelu_approximate=gelu_approximate,
        skip_mode=skip_mode, router_mode=router_mode,
    )

    class MoE_FFN_Wrapper(torch.nn.Module):
        def __init__(self, moe_layer):
            super().__init__()
            self.moe = moe_layer

        def forward(self, hidden_states, *args, **kwargs):
            # diffusers calls self.ff(norm_hidden_states) with no t_emb, which is
            # why the old kwargs.get('t_emb', zeros) fallback always fired and left
            # router_ada_lin inert. t_emb now arrives via the hooks below; passing
            # None lets the module resolve it and warn if it is genuinely missing.
            return self.moe(hidden_states, kwargs.get('t_emb', None))

    # Hooks: deliver t_emb and the AdaLN MLP modulation to the MoE.
    # Both fire inside the same block forward, strictly before `ff` is called
    # (the block computes norm1 -> attn1 -> attn2 -> norm3 -> modulate -> ff), so
    # the stashed values always belong to the current forward.
    norm1 = target_block.norm1

    if getattr(norm1, "emb", None) is None:
        raise RuntimeError(
            f"Block {layer_idx}.norm1 has no `.emb`; expected AdaLayerNormZero with "
            f"num_embeds_ada_norm set. Timestep-aware routing cannot be wired.")

    def _emb_hook(module, args, output):
        # CombinedTimestepLabelEmbeddings -> [B, hidden_dim]: the combined
        # timestep+class embedding AdaLN-Zero conditions on. 
        moe_module._hook_t_emb = output.detach()

    def _norm1_hook(module, args, output):
        # AdaLayerNormZero returns (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        if isinstance(output, (tuple, list)) and len(output) == 5:
            moe_module._hook_shift_mlp = output[2].detach()
            moe_module._hook_scale_mlp = output[3].detach()

    norm1.emb.register_forward_hook(_emb_hook)
    norm1.register_forward_hook(_norm1_hook)

    target_block.ff = MoE_FFN_Wrapper(moe_module)

    cfg = moe_module.config_summary()
    cfg["layer_index"] = layer_idx
    bar = "#" * 78
    print(bar, flush=True)
    print(f"#  MoE ARM: {router_mode.upper():<12} "
          f"Skip slot: {'LIVE' if moe_module.use_skip else 'MASKED OFF':<11} "
          f"Shared expert: {'ACTIVE' if router_mode == 'shared' else 'inert/frozen'}",
          flush=True)
    print(f"#  layer={layer_idx}  top_k={top_k}  expert_volume={expert_volume}  "
          f"rank={rank}  max_experts={max_experts}  skip_mode={skip_mode}  "
          f"gelu='{gelu_approximate}'", flush=True)
    print(f"#  t_emb + AdaLN hooks registered: timestep-aware router, "
          f"timestep-free clustering.", flush=True)
    print(f"#  >> CHECK THIS LINE MATCHES YOUR JOB SCRIPT BEFORE TRUSTING THE RUN <<",
          flush=True)
    print(bar, flush=True)
    return model