"""
End-of-epoch telemetry for the MoE-LoRA layer, plus the two EMA-shadow helpers
the training loop needs around a spawn.
"""
import math
import torch


def print_telemetry_report(moe_module, epoch, width=88):
    """Print the per-epoch MoE telemetry report for `moe_module`."""
    d = moe_module.specialization_report()
    n_active = int(moe_module.active_mask.sum().item())
    k = moe_module.current_active_k
    toks = moe_module.epoch_tokens_seen
    avg_entropy = moe_module.epoch_gating_entropy / max(1, moe_module.telemetry_steps)
    ceiling = moe_module.entropy_ceiling()

    bar = "=" * width
    print("\n" + bar, flush=True)
    print(f" END OF EPOCH {epoch + 1} MOE TELEMETRY  [arm: {d['router_mode']}]", flush=True)
    print(bar, flush=True)

    # Gating entropy as a fraction of its live ceiling, which is ln(n_paths)
    # for the currently active set (experts, plus the Skip path when the arm
    # uses it). Reporting against the live ceiling rather than a fixed one
    # keeps the percentage meaningful as experts spawn.
    pct = 100 * avg_entropy / max(ceiling, 1e-9)
    verdict = ("balanced" if pct > 90 else
               "skewed" if pct > 60 else "COLLAPSING -> one path dominates")
    print(f"Routing balance   : entropy {avg_entropy:.3f} / {ceiling:.3f} "
          f"(= ln {n_active + (1 if moe_module.use_skip else 0)})  "
          f"= {pct:.1f}% of max  -> {verdict}", flush=True)
    print(f"Tokens seen       : {toks:,}   dynamic k = {k}   active experts = {n_active}",
          flush=True)
    print("-" * width, flush=True)

    # top-1 counts the router's argmax choice and sums to tokens_seen.
    # top-k counts membership in the selected top-k and sums to k*tokens_seen
    print(f"{'Slot':<16} | {'top-1 (argmax)':>21} | {'top-k selected':>21}", flush=True)
    print(f"{'':<16} | {'count':>12} {'%tok':>8} | {'count':>12} {'%tok':>8}", flush=True)
    print("-" * width, flush=True)

    def _row(name, slot):
        t1 = int(moe_module.epoch_top1_counts[slot].item())
        tk = int(moe_module.epoch_topk_counts[slot].item())
        p1 = 100.0 * t1 / toks if toks else 0.0
        pk = 100.0 * tk / toks if toks else 0.0
        print(f"{name:<16} | {t1:>12,} {p1:>7.2f}% | {tk:>12,} {pk:>7.2f}%", flush=True)

    if moe_module.use_skip:
        _row("MoE Skip", 0)
    for idx in range(moe_module.max_experts):
        live = bool(moe_module.active_mask[idx].item())
        if not live and moe_module.epoch_topk_counts[idx + 1].item() == 0:
            continue
        _row(f"Expert {idx}" + ("" if live else " (OFF)"), idx + 1)
    print("-" * width, flush=True)

    # The expert output size relative to the base FFN output, split into its
    # three factors: the global ratio, the ratio measured only over routed
    # tokens, the fraction of tokens that got an expert and the mean gate
    # weight those tokens received. In 'shared' mode the always-on shared
    # expert is reported separately from the routed ones.
    print(f"[DIAG 1] expert magnitude   ||update|| / ||base_out||", flush=True)
    print(f"         routed, global     : {d['update_ratio']:.4f}"
          f"   <- comparable to the old single number", flush=True)
    print(f"         routed, on routed  : {d['update_ratio_routed']:.4f}"
          f"   <- TRUE routed-expert strength", flush=True)
    print(f"         routed token frac  : {d['routed_frac']:.3f}", flush=True)
    print(f"         mean gate weight   : {d['mean_gate_weight']:.4f}"
          f"   <- <1 means Skip is eating update mass", flush=True)
    if d["router_mode"] == "shared":
        print(f"         SHARED expert      : {d['shared_ratio']:.4f}"
              f"   <- always-on; if this is ~0 the shared expert is dead,", flush=True)
        print(f"                              "
              f"      if it dwarfs 'routed' it is doing all the work", flush=True)

    # Mean pairwise (off-diagonal) cosine between expert-mean deviations from
    # the global token mean. Because those deviations sum to zero by
    # construction, the value has a structural floor of -1/(n-1) rather than 0,
    # so both the floor and 0 are printed as reference points alongside it.
    n_used = max(d["n_used"], 1)
    floor = -1.0 / (n_used - 1) if n_used > 1 else float("nan")
    oc = d["offdiag_cos"]
    if n_used >= 2 and oc == oc and floor == floor:
        frac = 100.0 * oc / floor if floor != 0 else float("nan")
        note = ("  [n=2: forced to exactly -1.0000, uninformative]" if n_used == 2 else "")
        print(f"[DIAG 2] specialisation     off-diagonal cosine  : {oc:+.4f}{note}", flush=True)
        print(f"         reference points   : {floor:+.4f} = fully specialised "
              f"(floor -1/(n-1)) | 0.0000 = routing is random", flush=True)
        if n_used > 2:
            print(f"         -> you are {frac:.0f}% of the way from RANDOM to "
                  f"FULLY SPECIALISED", flush=True)
    else:
        print(f"[DIAG 2] specialisation     off-diagonal cosine  : n/a "
              f"(need >=2 experts with tokens)", flush=True)
    print(f"         spec_strength      : {d['spec_strength']:.4f}"
          f"   <- ||expert_mean - global_mean|| / ||global_mean||", flush=True)
    print(f"                              "
          f"      ~0 => router is NOT discriminating, whatever the counts say",
          flush=True)
    print(f"         experts with tokens: {d['n_used']}/{n_active}", flush=True)
    print(bar + "\n", flush=True)


def sync_ema_shadow_for_router(ema_model, model, moe_module):
    """
    Copy the current trained router weight+bias into EMA's shadow of them.

    warm_start_router_for_expert() edits one router row in place when an expert
    spawns and validation runs under EMA weights, so without this the EMA router
    keeps the old row until the EMA catches up. The router tensor is tiny and its
    structure just changed, so a hard copy is preferable to letting the EMA smooth
    it from the stale value.

    Returns the number of shadow tensors copied (router weight and bias, so 2).
    """
    tgt = {id(p) for p in moe_module.router.parameters()}
    n = 0
    for sp, p in zip(ema_model.shadow_params, model.parameters()):
        if id(p) in tgt:
            sp.data.copy_(p.data.to(sp.dtype))
            n += 1
    return n


def sync_ema_shadow_for_expert(ema_model, model, moe_module, expert_idx):
    """
    Copy a freshly spawned expert's trained weights into EMA's shadow of them.

    EMAModel.shadow_params is positionally aligned with the parameter list passed
    to its constructor (model.parameters(), taken after injection). A new expert
    is trained in one offline burst, but its shadow still holds the pre-spawn
    value and validation runs under EMA weights, so until the EMA catches up the
    expert is effectively absent from every validation pass.

    Returns the number of shadow tensors copied (up.A, up.B, down.A, down.B, so 4).
    """
    tgt = {id(p) for p in
           list(moe_module.up_experts[expert_idx].parameters()) +
           list(moe_module.down_experts[expert_idx].parameters())}
    n = 0
    for sp, p in zip(ema_model.shadow_params, model.parameters()):
        if id(p) in tgt:
            sp.data.copy_(p.data.to(sp.dtype))
            n += 1
    return n


def echo_run_manifest(args, moe_module):
    """
    Print the resolved config once at startup, side by side with the values that
    actually live inside the module and raise if they disagree. Call right after
    inject_moe_into_dit().

    The arms share state_dict shapes, so a CLI flag that is parsed but never
    forwarded produces no error and no shape mismatch. Printing argparse's view
    next to the module's view makes that kind of mistake visible at a glance.
    """
    cfg = moe_module.config_summary()
    print("\n" + "#" * 78, flush=True)
    print("#  RUN MANIFEST -- argparse said  vs  what the model actually IS", flush=True)
    print("#" + "-" * 77, flush=True)
    for key in ("router_mode", "skip_mode", "top_k", "expert_volume", "max_experts"):
        want = getattr(args, key, "<not a CLI arg>")
        got = cfg.get(key, "<not in module>")
        ok = "OK " if str(want) == str(got) else "!! MISMATCH -- flag is NOT reaching the model"
        print(f"#  {key:<16} cli={str(want):<12} model={str(got):<12} {ok}", flush=True)
    print(f"#  arch_rev         {cfg['arch_rev']}", flush=True)
    print("#" * 78 + "\n", flush=True)

    for key in ("router_mode", "skip_mode", "top_k"):
        want = getattr(args, key, None)
        if want is not None and str(want) != str(cfg.get(key)):
            raise RuntimeError(
                f"CONFIG MISMATCH: --{key}={want} but the model has {key}={cfg.get(key)}. "
                f"The flag is parsed and then discarded -- forward it into "
                f"inject_moe_into_dit(). Refusing to start: this is precisely how "
                f"three 'different' arms turned out to be the same arm."
            )