import os
import copy
import random
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

_PERSISTENT_CLIENT_RISK_BY_RUN = {}
_PERSISTENT_PAIR_RISK_BY_RUN = {}


def compute_catastrophic_hard_drop_guard(risk, persistent_risk, positive_fraction, drop_k, args):
    """
    Narrow veto for only the most weakly-localized hard-drop rounds.

    It does not disable soft penalties or memory updates. The veto fires only when
    probe evidence is very broad and the current/persistent client rankings lack
    enough separation or agreement to justify removing a full cohort.
    """
    risk = np.asarray(risk, dtype=np.float64)
    persistent_risk = np.asarray(persistent_risk, dtype=np.float64)
    n = min(len(risk), len(persistent_risk))
    k = max(0, min(int(drop_k), n - 1 if n > 1 else 0))
    if k <= 0 or n <= k:
        return {
            "veto": False, "reason": "insufficient_clients",
            "positive_fraction": float(positive_fraction),
            "topk_overlap": 1.0, "rank_agreement": 1.0,
            "boundary_margin": 0.0, "boundary_margin_ratio": 0.0,
            "weak_signal_count": 0,
        }

    current_order = np.argsort(risk[:n])[::-1]
    persistent_order = np.argsort(persistent_risk[:n])[::-1]
    current_top = set(int(x) for x in current_order[:k])
    persistent_top = set(int(x) for x in persistent_order[:k])
    topk_overlap = len(current_top & persistent_top) / float(k)

    # Pearson correlation over rank positions. This is deliberately lightweight
    # and avoids adding scipy as another ceremonial dependency.
    current_rank = np.empty(n, dtype=np.float64)
    persistent_rank = np.empty(n, dtype=np.float64)
    current_rank[current_order] = np.arange(n, dtype=np.float64)
    persistent_rank[persistent_order] = np.arange(n, dtype=np.float64)
    if np.std(current_rank) <= 1e-12 or np.std(persistent_rank) <= 1e-12:
        rank_agreement = 1.0
    else:
        rank_agreement = float(np.corrcoef(current_rank, persistent_rank)[0, 1])
        if not np.isfinite(rank_agreement):
            rank_agreement = 0.0

    sorted_persistent = persistent_risk[persistent_order]
    boundary_margin = float(sorted_persistent[k - 1] - sorted_persistent[k])
    top_scale = max(float(sorted_persistent[0]), 1e-12)
    boundary_margin_ratio = boundary_margin / top_scale

    broad_threshold = float(getattr(args, "probe_catastrophic_broad_fraction", 0.18))
    saturation_threshold = float(getattr(args, "probe_catastrophic_saturation_fraction", 0.25))
    min_overlap = float(getattr(args, "probe_catastrophic_min_topk_overlap", 0.50))
    min_rank_agreement = float(getattr(args, "probe_catastrophic_min_rank_agreement", 0.25))
    min_margin_ratio = float(getattr(args, "probe_catastrophic_min_margin_ratio", 0.01))
    required_weak = int(getattr(args, "probe_catastrophic_required_weak_signals", 2))

    weak_signals = [
        topk_overlap < min_overlap,
        rank_agreement < min_rank_agreement,
        boundary_margin_ratio < min_margin_ratio,
    ]
    weak_signal_count = int(sum(bool(x) for x in weak_signals))

    # Saturated evidence alone is treated as a global-compromise warning. Below
    # saturation, require broad evidence plus multiple localization weaknesses.
    veto = bool(
        float(positive_fraction) >= saturation_threshold
        or (float(positive_fraction) >= broad_threshold and weak_signal_count >= required_weak)
    )
    reason = "catastrophic_diffuse_localization" if veto else "localization_usable"
    return {
        "veto": veto,
        "reason": reason,
        "positive_fraction": float(positive_fraction),
        "topk_overlap": float(topk_overlap),
        "rank_agreement": float(rank_agreement),
        "boundary_margin": float(boundary_margin),
        "boundary_margin_ratio": float(boundary_margin_ratio),
        "weak_signal_count": weak_signal_count,
    }


def compute_boundary_rescue_order(final_persistent_risk, persistent_individual_risk,
                                  individual_risk, drop_k, hard_drop_guard,
                                  positive_fraction, args):
    """
    Rerank only the clients near the hard-drop boundary when localization is
    already coherent. This is designed to rescue a quiet attacker displaced by
    a benign client whose pair pressure is unusually large.

    The catastrophic guard remains authoritative. This helper never runs on
    diffuse or weakly agreed rankings, so it should not revive the old 0/6 edge
    cases merely to make the 6/6 column prettier.
    """
    final_risk = np.asarray(final_persistent_risk, dtype=np.float64)
    persistent_individual = np.asarray(persistent_individual_risk, dtype=np.float64)
    current_individual = np.asarray(individual_risk, dtype=np.float64)
    n = min(len(final_risk), len(persistent_individual), len(current_individual))
    k = max(0, min(int(drop_k), n - 1 if n > 1 else 0))
    original_order = np.argsort(final_risk[:n])[::-1]

    enabled = bool(getattr(args, "probe_boundary_rescue", True))
    max_positive_fraction = float(getattr(args, "probe_boundary_rescue_max_positive_fraction", 0.18))
    min_overlap = float(getattr(args, "probe_boundary_rescue_min_topk_overlap", 0.80))
    min_agreement = float(getattr(args, "probe_boundary_rescue_min_rank_agreement", 0.80))
    pool_extra = int(getattr(args, "probe_boundary_rescue_pool_extra", 3))
    w_final = float(getattr(args, "probe_boundary_rescue_final_weight", 0.40))
    w_persistent_ind = float(getattr(args, "probe_boundary_rescue_persistent_individual_weight", 0.40))
    w_current_ind = float(getattr(args, "probe_boundary_rescue_current_individual_weight", 0.20))

    usable = bool(
        enabled and k > 0 and n > k
        and not bool(hard_drop_guard.get("veto", False))
        and float(positive_fraction) < max_positive_fraction
        and float(hard_drop_guard.get("topk_overlap", 0.0)) >= min_overlap
        and float(hard_drop_guard.get("rank_agreement", 0.0)) >= min_agreement
    )
    if not usable:
        return original_order, {
            "applied": False, "reason": "boundary_rescue_inactive",
            "changed_count": 0, "pool_size": 0,
        }

    pool_size = min(n, k + max(1, pool_extra))
    # Form the pool from both the pair-heavy final ranking and the individual
    # memory ranking. A missed colluder only needs to be near one boundary to be
    # reconsidered; unrelated low-risk clients remain outside the pool.
    final_pool = list(original_order[:pool_size])
    ind_order = np.argsort(persistent_individual[:n])[::-1]
    pool = []
    for idx in final_pool + list(ind_order[:pool_size]):
        idx = int(idx)
        if idx not in pool:
            pool.append(idx)
    pool = pool[:min(n, pool_size + max(1, pool_extra))]

    def percentile_scores(values):
        order = np.argsort(values[:n])
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / max(float(n - 1), 1.0)

    final_pct = percentile_scores(final_risk)
    persistent_ind_pct = percentile_scores(persistent_individual)
    current_ind_pct = percentile_scores(current_individual)
    denom = max(w_final + w_persistent_ind + w_current_ind, 1e-12)
    rescue_score = (
        w_final * final_pct
        + w_persistent_ind * persistent_ind_pct
        + w_current_ind * current_ind_pct
    ) / denom

    reranked_pool = sorted(pool, key=lambda idx: (rescue_score[idx], final_risk[idx]), reverse=True)
    selected = reranked_pool[:k]
    remainder = [int(idx) for idx in original_order if int(idx) not in set(selected)]
    rescued_order = np.array(selected + remainder, dtype=np.int64)
    original_top = set(int(x) for x in original_order[:k])
    rescued_top = set(int(x) for x in selected)
    changed_count = len(original_top.symmetric_difference(rescued_top)) // 2
    return rescued_order, {
        "applied": True,
        "reason": "coherent_boundary_rerank",
        "changed_count": int(changed_count),
        "pool_size": int(len(pool)),
        "original_top": [int(x) for x in original_order[:k]],
        "rescued_top": [int(x) for x in selected],
    }


# RING-Aware Probe-Guided Group Detection scaffold.
# Collusion-aware variant: subgroup probing -> pair/cohort risk -> stronger selective suppression.
# Drop this file into: models/ProbeGroupDefense.py
# Then add the import/branch shown in the accompanying patched main file.


def ProbeGroupDefense(
    w_list,
    w_updates,
    global_model,
    dataset_test,
    args,
    per_run,
    first_call,
    w_length,
    users_idx=None,
    idx_attacker=None,
    debug=True,
):
    """
    Probe-guided random group defense scaffold.

    This is intentionally a first runnable prototype, not a finished paper defense.

    Per round:
      1. Build the full aggregate model from all submitted updates.
      2. Sample random client groups.
      3. Build a temporary aggregate model for each group.
      4. Compare full vs group outputs on clean and probe-triggered inputs.
      5. Score groups by excess probe sensitivity plus update-direction abnormality.
      6. Convert suspicious group appearances into soft client downweights.
      7. Return a weighted aggregate global model.

    This version keeps a persistent cross-round risk table keyed by real client IDs
    when the runner passes users_idx. Current-round group evidence is accumulated
    with exponential decay, then used for soft downweighting.
    """
    n_clients = len(w_updates)
    if n_clients == 0:
        return global_model.state_dict(), {
            "name": "ProbeGroup",
            "empty_round": True,
        }

    device = getattr(args, "device", next(global_model.parameters()).device)
    central_state = global_model.state_dict()

    base_weights = normalize_weights(w_length)

    # Full aggregate reference model.
    full_update = aggregate_updates_by_indices(
        w_updates=w_updates,
        indices=list(range(n_clients)),
        global_state=central_state,
        weights=base_weights,
    )
    full_state = add_update_to_state(central_state, full_update)
    full_model = copy.deepcopy(global_model).to(device)
    full_model.load_state_dict(full_state)
    full_model.eval()

    # Use a few fixed probe batches instead of one shuffled batch. This reduces
    # accidental false positives from one weird batch while keeping the prototype cheap.
    clean_batches = get_probe_batches(dataset_test, args, device)
    trigger_batches = [apply_probe_trigger(x.clone(), args) for x, _ in clean_batches]

    target_class = int(getattr(args, "probe_target", 1))
    target_class = min(max(target_class, 0), int(getattr(args, "num_classes", 10)) - 1)

    with torch.no_grad():
        full_stats = probe_model_stats(full_model, clean_batches, trigger_batches, target_class)
        full_clean_prob = full_stats["clean_target"]
        full_trigger_prob = full_stats["trigger_target"]

    full_vec = parameters_dict_to_vector_flt(full_update).to(device)

    # Random group probe settings. These are hard defaults so options.py does not need edits.
    # This variant is deliberately heavier than the first scaffold. With 30 selected
    # clients, 48 groups gives weak coverage; 180-320 groups is a more realistic
    # stress-test budget for pair/cohort evidence.
    default_group_size = 5 if n_clients >= 12 else max(2, min(4, n_clients // 2))
    group_size = int(getattr(args, "probe_group_size", 0) or default_group_size)
    group_size = max(2, min(group_size, n_clients))
    default_num_groups = min(320, max(96, n_clients * 8))
    num_groups = int(getattr(args, "probe_num_groups", 0) or default_num_groups)

    group_seed = deterministic_group_seed(args=args, users_idx=users_idx, n_clients=n_clients)
    groups = sample_random_groups(n_clients=n_clients, group_size=group_size, num_groups=num_groups, seed=group_seed)

    group_records = []
    appearances = np.zeros(n_clients, dtype=np.float64)
    flags = np.zeros(n_clients, dtype=np.float64)

    for group in groups:
        for idx in group:
            appearances[idx] += 1.0

        group_weights = normalize_weights([w_length[i] for i in group])
        group_update = aggregate_updates_by_indices(
            w_updates=w_updates,
            indices=group,
            global_state=central_state,
            weights=group_weights,
        )
        group_state = add_update_to_state(central_state, group_update)

        group_model = copy.deepcopy(global_model).to(device)
        group_model.load_state_dict(group_state)
        group_model.eval()

        with torch.no_grad():
            group_stats = probe_model_stats(group_model, clean_batches, trigger_batches, target_class)
            group_clean_prob = group_stats["clean_target"]
            group_trigger_prob = group_stats["trigger_target"]

        clean_gap = abs(group_clean_prob - full_clean_prob)
        trigger_gap = abs(group_trigger_prob - full_trigger_prob)
        excess_trigger_gap = trigger_gap - clean_gap

        full_shift = full_trigger_prob - full_clean_prob
        group_shift = group_trigger_prob - group_clean_prob
        shift_gap = abs(group_shift - full_shift)

        clean_vec_gap = l1_vector_distance(group_stats["clean_vec"], full_stats["clean_vec"])
        trigger_vec_gap = l1_vector_distance(group_stats["trigger_vec"], full_stats["trigger_vec"])
        excess_vec_gap = trigger_vec_gap - clean_vec_gap

        full_vec_shift = l1_vector_distance(full_stats["trigger_vec"], full_stats["clean_vec"])
        group_vec_shift = l1_vector_distance(group_stats["trigger_vec"], group_stats["clean_vec"])
        shift_vec_gap = abs(group_vec_shift - full_vec_shift)

        group_vec = parameters_dict_to_vector_flt(group_update).to(device)
        direction_gap = cosine_direction_gap(group_vec, full_vec)

        # Keep the score oriented toward trigger-specific excess behavior.
        # The older vector-heavy score often over-penalized benign non-IID clients.
        # For this collusion-aware variant, vector and direction terms are supporting
        # evidence, not the main event.
        score_mode = str(getattr(args, "probe_score_mode", "excess_only")).lower()
        if score_mode == "target_only":
            # Legacy-ish mode: still primarily target-class excess, with small shift support.
            score = float(max(0.0, excess_trigger_gap) + 0.10 * shift_gap)
        elif score_mode == "original":
            score = float(
                excess_trigger_gap
                + 0.50 * shift_gap
                + 0.75 * excess_vec_gap
                + 0.25 * shift_vec_gap
                + 0.05 * direction_gap
            )
        elif score_mode == "collusion":
            # Older collusion score. Kept for ablation, but no longer the default because
            # it could mark groups suspicious even when trigger excess was negative.
            score = float(
                max(0.0, excess_trigger_gap)
                + 0.20 * shift_gap
                + 0.10 * max(0.0, excess_vec_gap)
                + 0.05 * shift_vec_gap
                + 0.02 * direction_gap
            )
        else:
            # New default: only trigger-specific excess is treated as evidence.
            # Weird clean-distribution behavior can still be logged, but it should
            # not feed persistent risk unless trigger_gap beats clean_gap.
            score = float(max(0.0, excess_trigger_gap))

        real_group = local_to_real_group(
            group,
            users_idx,
        )

        known_attacker_count = (
            count_known_attackers_in_group(
                group=group,
                users_idx=users_idx,
                idx_attacker=idx_attacker,
                n_clients=n_clients,
                args=args,
            )
        )

        known_attacker_fraction = (
            float(known_attacker_count)
            / float(len(group))
            if len(group) > 0
            else 0.0
        )

        group_records.append({
            "group": list(group),
            "real_group": real_group,
            "contains_known_attacker": bool(known_attacker_count > 0),
            "known_attacker_count": int(known_attacker_count),
            "known_attacker_fraction": float(known_attacker_fraction),
            "score": score,
            "clean_gap": float(clean_gap),
            "trigger_gap": float(trigger_gap),
            "excess_trigger_gap": float(excess_trigger_gap),
            "shift_gap": float(shift_gap),
            "direction_gap": float(direction_gap),
            "clean_vec_gap": float(clean_vec_gap),
            "trigger_vec_gap": float(trigger_vec_gap),
            "excess_vec_gap": float(excess_vec_gap),
            "shift_vec_gap": float(shift_vec_gap),
            "group_clean_prob": float(group_clean_prob),
            "group_trigger_prob": float(group_trigger_prob),
        })

    candidate_suspicious_groups = select_suspicious_groups(group_records, args=args)
    candidate_suspicious_group_sets = {tuple(record["group"]) for record in candidate_suspicious_groups}
    score_summary = summarize_score_distribution(group_records, candidate_suspicious_group_sets)

    attack_evidence = compute_attack_evidence_gate(
        group_records=group_records,
        candidate_suspicious_groups=candidate_suspicious_groups,
        args=args,
        idx_attacker=idx_attacker,
    )

    # Candidate suspicious groups are useful for diagnostics, but only gated
    # attack-evidence-positive rounds are allowed to update memory or trigger
    # hard dropping. This prevents clean warmup rounds from teaching the defense
    # that benign non-IID cohorts are attackers.
    if attack_evidence["allow_memory_update"]:
        suspicious_groups = candidate_suspicious_groups
        suspicious_group_sets = candidate_suspicious_group_sets
    else:
        suspicious_groups = []
        suspicious_group_sets = set()

    # Suspicious groups now cast weighted votes. Stronger groups contribute more,
    # and, crucially, they create pair/cohort evidence instead of only individual
    # suspicion. This is the drastic RING-aware part.
    strengths = compute_group_strengths(group_records, suspicious_group_sets)
    for record in suspicious_groups:
        strength = strengths.get(tuple(record["group"]), 1.0)
        for idx in record["group"]:
            flags[idx] += strength

    raw_individual_risk = flags / np.maximum(appearances, 1.0)
    min_appearances = float(getattr(args, "probe_min_appearances", max(8.0, 0.20 * len(groups))))
    confidence = np.minimum(1.0, appearances / max(min_appearances, 1.0))
    individual_risk = raw_individual_risk * confidence

    pair_current, pair_pressure = build_pair_risk_evidence(
        suspicious_groups=suspicious_groups,
        strengths=strengths,
        total_clients=n_clients,
        users_idx=users_idx,
        args=args,
    )

    # Current-round risk is now mostly pair/cohort pressure. If attackers are
    # colluding, repeated suspicious partners should matter more than solo flags.
    pair_weight = float(getattr(args, "probe_pair_weight", 0.70))
    risk = (1.0 - pair_weight) * individual_risk + pair_weight * pair_pressure
    risk_cap = float(getattr(args, "probe_risk_round_cap", 0.60))
    risk = np.minimum(risk, risk_cap)

    persistent_risk_table, persistent_individual_risk = update_persistent_risk_table(
        current_risk=risk,
        users_idx=users_idx,
        args=args,
        per_run=per_run,
        first_call=first_call,
        total_clients=n_clients,
    )

    persistent_pair_table, persistent_pair_pressure = update_persistent_pair_risk_table(
        pair_current=pair_current,
        users_idx=users_idx,
        args=args,
        per_run=per_run,
        first_call=first_call,
        total_clients=n_clients,
    )

    final_persistent_risk = combine_individual_and_pair_risk(
        persistent_individual_risk=persistent_individual_risk,
        persistent_pair_pressure=persistent_pair_pressure,
        args=args,
    )

    telemetry_summary = build_client_telemetry_summary(
        risk=risk,
        flags=flags,
        appearances=appearances,
        users_idx=users_idx,
        idx_attacker=idx_attacker,
        total_clients=n_clients,
        args=args,
        persistent_risk=final_persistent_risk,
    )

    tau = float(getattr(args, "probe_tau", 4.0))
    risk_grace = float(getattr(args, "probe_risk_grace", 0.02))
    max_drop = float(getattr(args, "probe_max_weight_drop", 0.98))
    effective_risk = np.maximum(0.0, final_persistent_risk - risk_grace)
    multiplier = np.exp(-tau * effective_risk)
    multiplier = np.maximum(1.0 - max_drop, multiplier)
    soft_weights = np.array(base_weights, dtype=np.float64) * multiplier

    # Diagnostic hard rejection: if the detector ranks attackers highly but ASR
    # survives, soft exponential downweighting is too polite. Drop the top-risk
    # clients entirely by default, using num_attacker as the test budget. This is
    # intentionally drastic and should be ablated against soft-only behavior.
    hard_dropped_indices = []
    hard_drop_enabled = bool(getattr(args, "probe_hard_drop", True))

    # The normal evidence gate determines which ranking strategy is used,
    # but no longer determines whether a hard drop occurs.
    evidence_allows_hard_drop = bool(
        attack_evidence.get("allow_hard_drop", False)
    )

    default_drop_k = int(getattr(args, "num_attacker", 0))
    drop_k = int(getattr(args, "probe_hard_drop_k", default_drop_k))
    drop_k = max(
        0,
        min(drop_k, n_clients - 1 if n_clients > 1 else 0)
    )

    positive_fraction = (
        float(attack_evidence.get("positive_group_count", 0) or 0)
        / float(len(group_records))
        if len(group_records) > 0
        else 0.0
    )

    hard_drop_guard = compute_catastrophic_hard_drop_guard(
        risk=risk,
        persistent_risk=final_persistent_risk,
        positive_fraction=positive_fraction,
        drop_k=drop_k,
        args=args,
    )

    catastrophic_guard_enabled = bool(
        getattr(args, "probe_catastrophic_guard", False)
    )
    catastrophic_veto = bool(
        catastrophic_guard_enabled
        and hard_drop_guard.get("veto", False)
    )

    # "Normal" hard-drop mode is used only when all existing conditions pass.
    normal_hard_drop_conditions_met = bool(
        hard_drop_enabled
        and evidence_allows_hard_drop
        and not catastrophic_veto
    )

    # Hard dropping itself now happens whenever it is enabled and drop_k > 0.
    hard_drop_action_allowed = bool(
        hard_drop_enabled
        and drop_k > 0
        and len(soft_weights) > 0
    )

    attack_evidence["hard_drop_action_allowed"] = hard_drop_action_allowed
    attack_evidence["hard_drop_normal_conditions_met"] = (
        normal_hard_drop_conditions_met
    )
    attack_evidence["catastrophic_hard_drop_veto"] = catastrophic_veto
    attack_evidence["hard_drop_guard_reason"] = hard_drop_guard.get(
        "reason", ""
    )
    attack_evidence["hard_drop_topk_overlap"] = hard_drop_guard.get(
        "topk_overlap", 1.0
    )
    attack_evidence["hard_drop_rank_agreement"] = hard_drop_guard.get(
        "rank_agreement", 1.0
    )
    attack_evidence["hard_drop_boundary_margin"] = hard_drop_guard.get(
        "boundary_margin", 0.0
    )
    attack_evidence["hard_drop_boundary_margin_ratio"] = (
        hard_drop_guard.get("boundary_margin_ratio", 0.0)
    )

    boundary_rescue_info = {
        "applied": False,
        "reason": "hard_drop_disabled",
        "changed_count": 0,
        "pool_size": 0,
    }

    hard_drop_selection_mode = "disabled"

    if hard_drop_action_allowed:
        if normal_hard_drop_conditions_met:
            # Existing behavior when the evidence gate and guard pass:
            # use the full persistent score with optional boundary rescue.
            candidate_order, boundary_rescue_info = (
                compute_boundary_rescue_order(
                    final_persistent_risk=final_persistent_risk,
                    persistent_individual_risk=persistent_individual_risk,
                    individual_risk=individual_risk,
                    drop_k=drop_k,
                    hard_drop_guard=hard_drop_guard,
                    positive_fraction=positive_fraction,
                    args=args,
                )
            )
            hard_drop_selection_mode = "normal_evidence_ranking"

            # Preserve the existing minimum-risk condition in normal mode.
            min_drop_risk = float(
                getattr(args, "probe_hard_drop_min_risk", 0.0)
            )

            selected_indices = [
                int(idx)
                for idx in candidate_order
                if float(final_persistent_risk[int(idx)]) >= min_drop_risk
            ][:drop_k]

            # Guarantee exactly drop_k removals even if the configured
            # minimum-risk threshold removes too many candidates.
            if len(selected_indices) < drop_k:
                for idx in candidate_order:
                    idx = int(idx)
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                    if len(selected_indices) >= drop_k:
                        break

        else:
            # Fallback behavior:
            # ignore current-round risk, group evidence, boundary rescue,
            # and the minimum-risk threshold. Rank solely by the accumulated
            # persistent score.
            #
            # persistent_individual_risk is the cross-round score produced by
            # update_persistent_risk_table().
            candidate_order = np.argsort(
                persistent_individual_risk
            )[::-1]

            selected_indices = [
                int(idx)
                for idx in candidate_order[:drop_k]
            ]

            hard_drop_selection_mode = "persistent_score_only"
            boundary_rescue_info = {
                "applied": False,
                "reason": "fallback_persistent_score_only",
                "changed_count": 0,
                "pool_size": 0,
            }

        for idx in selected_indices:
            soft_weights[idx] = 0.0
            hard_dropped_indices.append(idx)

    attack_evidence["hard_drop_selection_mode"] = (hard_drop_selection_mode)
    attack_evidence["boundary_rescue_applied"] = bool(boundary_rescue_info.get("applied", False))
    attack_evidence["boundary_rescue_reason"] = boundary_rescue_info.get("reason", "")
    attack_evidence["boundary_rescue_changed_count"] = int(boundary_rescue_info.get("changed_count", 0) or 0)
    attack_evidence["boundary_rescue_pool_size"] = int(boundary_rescue_info.get("pool_size", 0) or 0)

    #FedAvg weights are restored during no-evidence rounds, but mandatory hard drops are retained from persistent score.
    if (
        not bool(attack_evidence.get("allow_penalty", False))
        and bool(getattr(args, "probe_no_evidence_use_fedavg", True))
    ):
    # Restore normal FedAvg weights during no-evidence rounds, but retain
    # the mandatory hard drops selected from persistent score.
        soft_weights = np.array(base_weights, dtype=np.float64)

    for idx in hard_dropped_indices:
        soft_weights[int(idx)] = 0.0

    if soft_weights.sum() <= 0:
        soft_weights = np.array(base_weights, dtype=np.float64)
    soft_weights = (soft_weights / soft_weights.sum()).tolist()

    use_median_clip = bool(getattr(args, "probe_use_median_clip", True))
    aggregation_updates = median_clip_updates(w_updates) if use_median_clip else w_updates
    defended_update = aggregate_updates_by_indices(
        w_updates=aggregation_updates,
        indices=list(range(n_clients)),
        global_state=central_state,
        weights=soft_weights,
    )

    # Drastic default: mostly trust the defended update. Keep a small blend knob
    # available so we can back off if clean accuracy collapses.
    blend_alpha = float(getattr(args, "probe_blend_alpha", 1.0))
    blend_alpha = max(0.0, min(1.0, blend_alpha))
    final_update = blend_updates(defended_update, full_update, alpha=blend_alpha)
    w_avg = add_update_to_state(central_state, final_update)

    probe_summary = build_probe_round_summary_metrics(
        args=args,
        per_run=per_run,
        round_idx=int(getattr(args, "current_round", 0)),
        group_records=group_records,
        candidate_suspicious_groups=candidate_suspicious_groups,
        suspicious_groups=suspicious_groups,
        attack_evidence=attack_evidence,
        telemetry_summary=telemetry_summary,
        hard_dropped_indices=hard_dropped_indices,
        users_idx=users_idx,
        idx_attacker=idx_attacker,
        final_persistent_risk=final_persistent_risk
    )

    client_explanations = summarize_client_group_explanations(
        group_records=group_records,
        suspicious_group_sets=suspicious_group_sets,
        strengths=strengths,
        total_clients=n_clients,
    )

    client_details = []

    for row in telemetry_summary.get("rows", []):
        local_idx = int(row["local_idx"])
        explanation = client_explanations.get(local_idx, {})

        client_item = dict(row)

        client_item.update({
            "individual_risk": float(individual_risk[local_idx]),
            "persistent_individual_risk": float(persistent_individual_risk[local_idx]),
            "final_persistent_risk": float(final_persistent_risk[local_idx]),
            "pair_pressure": float(pair_pressure[local_idx]),
            "persistent_pair_pressure": float(persistent_pair_pressure[local_idx]),
            "base_weight": float(base_weights[local_idx]),
            "final_weight": float(soft_weights[local_idx]),
            "hard_dropped": bool( local_idx in hard_dropped_indices),
        })

        for key, value in explanation.items():
            client_item[key] = float(value)

        client_details.append(client_item)

    group_details = []

    for record in group_records:
        group_key = tuple(record["group"])

        item = dict(record)

        item["candidate_suspicious"] = bool(
            group_key in candidate_suspicious_group_sets
        )

        item["memory_suspicious"] = bool(
            group_key in suspicious_group_sets
        )

        item["strength"] = float(
            strengths.get(group_key, 0.0)
        )

        group_details.append(item)

    probe_telemetry = {
        "name": "ProbeGroup",

        "summary": probe_summary,

        "probe_reference": {
            "target_class": int(target_class),
            "full_clean_prob": float(full_clean_prob),
            "full_trigger_prob": float(full_trigger_prob),
        },

        "sampling": {
            "group_seed": int(group_seed),
            "group_size": int(group_size),
            "requested_num_groups": int(num_groups),
            "actual_num_groups": int(len(groups)),
        },

        "attack_evidence": attack_evidence,
        "score_summary": score_summary,

        "hard_drop_guard": hard_drop_guard,
        "boundary_rescue": boundary_rescue_info,

        "hard_dropped_local_indices": [
            int(x) for x in hard_dropped_indices
        ],

        "users_idx_real_client_order": (
            [int(x) for x in users_idx]
            if users_idx is not None
            else None
        ),

        "attacker_real_client_ids": (
            [int(x) for x in idx_attacker]
            if idx_attacker is not None
            else []
        ),

        "current_risk": risk.tolist(),
        "individual_risk": individual_risk.tolist(),

        "persistent_individual_risk_current_clients":
            persistent_individual_risk.tolist(),

        "final_persistent_risk_current_clients":
            final_persistent_risk.tolist(),

        "persistent_client_risk_table": {
            str(k): float(v)
            for k, v in persistent_risk_table.items()
        },

        "pair_current": {
            str(k): float(v)
            for k, v in pair_current.items()
        },

        "pair_pressure": pair_pressure.tolist(),

        "persistent_pair_pressure":
            persistent_pair_pressure.tolist(),

        "persistent_pair_risk_table": {
            str(k): float(v)
            for k, v in persistent_pair_table.items()
        },

        "base_weights": [
            float(x) for x in base_weights
        ],

        "final_weights": [
            float(x) for x in soft_weights
        ],

        "clients": client_details,
        "groups": group_details,
    }
    print(
        "ProbeGroup hard dropped clients:",
        hard_dropped_indices
    )

    return w_avg, probe_telemetry



def get_client_key(local_idx, users_idx):
    if users_idx is not None and local_idx < len(users_idx):
        return str(int(users_idx[local_idx]))
    return "local_{}".format(int(local_idx))



def update_persistent_risk_table(current_risk, users_idx, args, per_run, first_call, total_clients):
    run_key = int(per_run)

    if first_call or run_key not in _PERSISTENT_CLIENT_RISK_BY_RUN:
        _PERSISTENT_CLIENT_RISK_BY_RUN[run_key] = {}
    

    table = _PERSISTENT_CLIENT_RISK_BY_RUN[run_key]

    decay = float(getattr(args, "probe_risk_decay", 0.85))
    max_risk = float(getattr(args, "probe_max_risk", 2.0))

    # Decay every known client's previous risk once per round. This lets suspicion
    # fade if a client stops appearing in suspicious groups.
    for key in list(table.keys()):
        table[key] = max(0.0, min(max_risk, float(table[key]) * decay))

        if table[key] < 1e-8:
            table.pop(key, None)

    persistent_risk = np.zeros(total_clients, dtype=np.float64)

    for local_idx in range(total_clients):
        key = get_client_key(local_idx, users_idx)
        old_value = float(table.get(key, 0.0))
        new_value = max(0.0, min(max_risk, old_value + float(current_risk[local_idx])))
        table[key] = new_value
        persistent_risk[local_idx] = new_value

    return dict(table), persistent_risk


def get_pair_key(i, j, users_idx):
    if users_idx is not None and i < len(users_idx) and j < len(users_idx):
        a, b = int(users_idx[i]), int(users_idx[j])
    else:
        a, b = int(i), int(j)
    if a > b:
        a, b = b, a
    return "{}:{}".format(a, b)


def update_persistent_pair_risk_table(pair_current, users_idx, args, per_run, first_call, total_clients):
    run_key = int(per_run)

    if first_call or run_key not in _PERSISTENT_PAIR_RISK_BY_RUN:
        _PERSISTENT_PAIR_RISK_BY_RUN[run_key] = {}

    table = _PERSISTENT_PAIR_RISK_BY_RUN[run_key]

    decay = float(getattr(args, "probe_pair_decay", 0.92))
    max_pair_risk = float(getattr(args, "probe_pair_max_risk", 3.0))

    for key in list(table.keys()):
        table[key] = max(0.0, min(max_pair_risk, float(table[key]) * decay))
        if table[key] < 1e-8:
            table.pop(key, None)

    for key, value in pair_current.items():
        old_value = float(table.get(key, 0.0))
        table[key] = max(0.0, min(max_pair_risk, old_value + float(value)))

    persistent_pair_pressure = pair_table_to_client_pressure(
        pair_table=table,
        users_idx=users_idx,
        total_clients=total_clients,
        args=args,
    )
    return dict(table), persistent_pair_pressure


def compute_group_strengths(group_records, suspicious_group_sets):
    if len(group_records) == 0:
        return {}
    scores = np.array([float(r["score"]) for r in group_records], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)) + 1e-12)
    strengths = {}
    for record in group_records:
        group_key = tuple(record["group"])
        if group_key not in suspicious_group_sets:
            continue
        z = max(0.0, (float(record["score"]) - median) / mad)
        # 0.25 floor: a selected suspicious group still counts. Cap prevents one
        # explosive score from dominating the whole round.
        strengths[group_key] = float(max(0.25, min(2.0, z / 3.0)))
    return strengths


def build_pair_risk_evidence(suspicious_groups, strengths, total_clients, users_idx, args):
    pair_values = {}
    pair_counts = {}
    client_pair_matrix = np.zeros((total_clients, total_clients), dtype=np.float64)

    for record in suspicious_groups:
        group = [int(x) for x in record["group"]]
        strength = float(strengths.get(tuple(group), 1.0))
        if len(group) < 2:
            continue
        # Normalize by group size so larger groups do not create unfairly massive
        # pair evidence just because they contain more pairs.
        normalized_strength = strength / max(1.0, float(len(group) - 1))
        for pos, i in enumerate(group):
            for j in group[pos + 1:]:
                key = get_pair_key(i, j, users_idx)
                pair_values[key] = pair_values.get(key, 0.0) + normalized_strength
                pair_counts[key] = pair_counts.get(key, 0) + 1
                client_pair_matrix[i, j] += normalized_strength
                client_pair_matrix[j, i] += normalized_strength

    round_cap = float(getattr(args, "probe_pair_round_cap", 0.80))
    pair_current = {k: min(round_cap, float(v)) for k, v in pair_values.items()}
    pair_pressure = pair_matrix_to_client_pressure(client_pair_matrix, args=args)
    pair_pressure = np.minimum(pair_pressure, round_cap)
    return pair_current, pair_pressure


def pair_matrix_to_client_pressure(matrix, args):
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    top_k = int(getattr(args, "probe_pair_topk", 3))
    top_k = max(1, min(top_k, matrix.shape[0] - 1 if matrix.shape[0] > 1 else 1))
    pressure = np.zeros(matrix.shape[0], dtype=np.float64)
    for i in range(matrix.shape[0]):
        row = np.sort(matrix[i])[::-1]
        vals = row[:top_k]
        pressure[i] = float(np.mean(vals)) if len(vals) else 0.0
    return pressure


def pair_table_to_client_pressure(pair_table, users_idx, total_clients, args):
    matrix = np.zeros((total_clients, total_clients), dtype=np.float64)
    real_to_local = {}
    if users_idx is not None:
        for local_idx, real_id in enumerate(users_idx):
            real_to_local[int(real_id)] = int(local_idx)

    for key, value in pair_table.items():
        try:
            a_str, b_str = str(key).split(":", 1)
            a, b = int(a_str), int(b_str)
        except Exception:
            continue
        if users_idx is not None:
            if a not in real_to_local or b not in real_to_local:
                continue
            i, j = real_to_local[a], real_to_local[b]
        else:
            i, j = a, b
            if i < 0 or j < 0 or i >= total_clients or j >= total_clients:
                continue
        matrix[i, j] = max(matrix[i, j], float(value))
        matrix[j, i] = max(matrix[j, i], float(value))
    return pair_matrix_to_client_pressure(matrix, args=args)


def combine_individual_and_pair_risk(persistent_individual_risk, persistent_pair_pressure, args):
    pair_mix = float(getattr(args, "probe_persistent_pair_mix", 0.80))
    mixed = (1.0 - pair_mix) * persistent_individual_risk + pair_mix * persistent_pair_pressure
    # Use max as a guardrail: if pair evidence is very high, do not let a low
    # individual score wash it away.
    combined = np.maximum(mixed, persistent_pair_pressure)
    max_risk = float(getattr(args, "probe_max_risk", 3.0))
    return np.minimum(combined, max_risk)


def blend_updates(defended_update, reference_update, alpha):
    blended = {}
    for key, value in defended_update.items():
        if key in reference_update and torch.is_floating_point(value):
            blended[key] = alpha * value + (1.0 - alpha) * reference_update[key].to(value.device)
        else:
            blended[key] = value.clone()
    return blended


def local_to_real_group(group, users_idx):
    if users_idx is None:
        return None
    real_group = []
    for idx in group:
        if idx < len(users_idx):
            real_group.append(int(users_idx[idx]))
        else:
            real_group.append(None)
    return real_group


def group_contains_known_attacker(group, users_idx, idx_attacker, n_clients, args):
    # Prefer true global attacker IDs when the runner passes them.
    if users_idx is not None and idx_attacker is not None:
        attacker_set = set(int(x) for x in idx_attacker)
        real_group = local_to_real_group(group, users_idx)
        return any(x in attacker_set for x in real_group if x is not None)

    # Fallback: in this runner, collusion attackers are appended to the end of w_locals_combine.
    attacker_count = int(getattr(args, "num_attacker", 0))
    attacker_start = n_clients - attacker_count
    return any(idx >= attacker_start for idx in group) if attacker_count > 0 else False

def count_known_attackers_in_group(
    group,
    users_idx,
    idx_attacker,
    n_clients,
    args,
):
    """
    Return the number of known attackers inside a temporary ProbeGroup.

    Prefer real/global client IDs when users_idx and idx_attacker are available.
    Fall back to the runner's local ordering assumption otherwise.
    """

    # Preferred path: use real client IDs.
    if users_idx is not None and idx_attacker is not None:
        attacker_set = set(
            int(x) for x in idx_attacker
        )

        real_group = local_to_real_group(
            group,
            users_idx,
        )

        return int(
            sum(
                1
                for client_id in real_group
                if client_id is not None
                and int(client_id) in attacker_set
            )
        )

    # Fallback for older callers where real IDs were not passed.
    attacker_count = int(
        getattr(args, "num_attacker", 0)
    )

    if attacker_count <= 0:
        return 0

    attacker_start = n_clients - attacker_count

    return int(
        sum(
            1
            for local_idx in group
            if int(local_idx) >= attacker_start
        )
    )

def build_client_telemetry_summary(risk, flags, appearances, users_idx, idx_attacker, total_clients, args, persistent_risk=None):
    rows = []
    if persistent_risk is None:
        persistent_risk = risk
    attacker_count = int(getattr(args, "num_attacker", 0))
    fallback_attacker_start = total_clients - attacker_count
    attacker_set = set(int(x) for x in idx_attacker) if idx_attacker is not None else set()

    for local_idx in range(total_clients):
        real_id = int(users_idx[local_idx]) if users_idx is not None and local_idx < len(users_idx) else None
        if users_idx is not None and idx_attacker is not None:
            is_known_attacker = real_id in attacker_set
        else:
            is_known_attacker = attacker_count > 0 and local_idx >= fallback_attacker_start
        rows.append({
            "local_idx": int(local_idx),
            "real_client_id": real_id,
            "is_known_attacker": bool(is_known_attacker),
            "risk": float(risk[local_idx]),
            "persistent_risk": float(persistent_risk[local_idx]),
            "flags": int(flags[local_idx]),
            "appearances": int(appearances[local_idx]),
        })

    attacker_rows = [r for r in rows if r["is_known_attacker"]]
    benign_rows = [r for r in rows if not r["is_known_attacker"]]

    def mean_value(items, key):
        if len(items) == 0:
            return 0.0
        return float(np.mean([x[key] for x in items]))

    return {
        "rows": rows,
        "attacker_rows": attacker_rows,
        "benign_rows": benign_rows,
        "attacker_mean_risk": mean_value(attacker_rows, "risk"),
        "benign_mean_risk": mean_value(benign_rows, "risk"),
        "attacker_mean_persistent_risk": mean_value(attacker_rows, "persistent_risk"),
        "benign_mean_persistent_risk": mean_value(benign_rows, "persistent_risk"),
        "attacker_mean_flags": mean_value(attacker_rows, "flags"),
        "benign_mean_flags": mean_value(benign_rows, "flags"),
    }


def summarize_score_distribution(group_records, suspicious_group_sets):
    if len(group_records) == 0:
        return {}
    scores = np.array([float(r["score"]) for r in group_records], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)) + 1e-12)
    threshold = float(median + 0.75 * mad)

    attacker_scores = [float(r["score"]) for r in group_records if bool(r.get("contains_known_attacker", False))]
    benign_scores = [float(r["score"]) for r in group_records if not bool(r.get("contains_known_attacker", False))]

    def mean_or_zero(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    return {
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "suspicious_count": int(len(suspicious_group_sets)),
        "total_count": int(len(group_records)),
        "attacker_group_mean_score": mean_or_zero(attacker_scores),
        "benign_group_mean_score": mean_or_zero(benign_scores),
        "attacker_group_count": int(len(attacker_scores)),
        "benign_group_count": int(len(benign_scores)),
    }


def print_probe_score_summary(score_summary):
    if not score_summary:
        return
    print(
        "ProbeGroup score summary:",
        {
            "median": round(float(score_summary["median"]), 6),
            "mad": round(float(score_summary["mad"]), 6),
            "threshold": round(float(score_summary["threshold"]), 6),
            "attacker_group_mean": round(float(score_summary["attacker_group_mean_score"]), 6),
            "benign_group_mean": round(float(score_summary["benign_group_mean_score"]), 6),
        }
    )


def print_probe_summary(telemetry_summary, group_records, suspicious_group_sets):
    attacker_rows = telemetry_summary["attacker_rows"]
    benign_rows = telemetry_summary["benign_rows"]
    print("ProbeGroup attacker mean risk:", round(float(telemetry_summary["attacker_mean_risk"]), 3))
    print("ProbeGroup benign mean risk:", round(float(telemetry_summary["benign_mean_risk"]), 3))
    print("ProbeGroup attacker persistent mean risk:", round(float(telemetry_summary["attacker_mean_persistent_risk"]), 3))
    print("ProbeGroup benign persistent mean risk:", round(float(telemetry_summary["benign_mean_persistent_risk"]), 3))
    if attacker_rows:
        print("ProbeGroup attacker local risks:", [
            (r["local_idx"], r["real_client_id"], round(float(r["risk"]), 3), round(float(r["persistent_risk"]), 3))
            for r in attacker_rows
        ])

    top_clients = sorted(
        telemetry_summary["rows"],
        key=lambda r: (r["persistent_risk"], r["risk"], r["flags"], r["appearances"]),
        reverse=True
    )[:10]

    print("ProbeGroup top suspicious clients:", [
        {
            "local": r["local_idx"],
            "real": r["real_client_id"],
            "attacker": r["is_known_attacker"],
            "risk": round(float(r["risk"]), 3),
            "persistent_risk": round(float(r["persistent_risk"]), 3),
            "flags": r["flags"],
            "appearances": r["appearances"],
        }
        for r in top_clients
    ])

    scored = sorted(group_records, key=lambda r: r["score"], reverse=True)[:3]
    compact = []
    for record in scored:
        compact.append({
            "group": record["group"],
            "real_group": record.get("real_group"),
            "suspicious": tuple(record["group"]) in suspicious_group_sets,
            "score": round(float(record["score"]), 4),
            "excess": round(float(record["excess_trigger_gap"]), 4),
            "contains_known_attacker": bool(record.get("contains_known_attacker", False)),
        })
    print("ProbeGroup top groups:", compact)


def get_probe_batches(dataset_test, args, device):
    batch_size = int(getattr(args, "probe_batch_size", 64))
    num_batches = int(getattr(args, "probe_num_batches", 2))
    num_batches = max(1, num_batches)
    loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    batches = []
    for batch_idx, (x, y) in enumerate(loader):
        batches.append((x.to(device), y.to(device)))
        if batch_idx + 1 >= num_batches:
            break
    return batches


def apply_probe_trigger(x, args):
    """
    Apply a simple patch-style probe. For normalized MNIST/CIFAR tensors, use a
    normalized white-pixel value. If input is not image-like, return unchanged.
    """
    if x.dim() != 4:
        return x

    dataset = str(getattr(args, "dataset", "")).lower()
    _, channels, height, width = x.shape
    patch = int(getattr(args, "probe_patch_size", 8))
    patch = max(1, min(patch, height, width))

    row_start = 1 if height >= patch + 1 else 0
    col_start = max(width - patch - 1, 0)
    row_end = min(row_start + patch, height)
    col_end = min(col_start + patch, width)

    if dataset in ("mnist", "fashion-mnist"):
        # White pixel after Normalize((0.1307,), (0.3081,)) for MNIST.
        val = (1.0 - 0.1307) / 0.3081 if dataset == "mnist" else 1.0
        x[:, :, row_start:row_end, col_start:col_end] = val
    elif dataset == "cifar":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, channels, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1, channels, 1, 1)
        val = (torch.ones_like(mean) - mean) / std
        x[:, :, row_start:row_end, col_start:col_end] = val
    elif dataset == "cifar100":
        mean = torch.tensor([0.5071, 0.4867, 0.4408], device=x.device).view(1, channels, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761], device=x.device).view(1, channels, 1, 1)
        val = (torch.ones_like(mean) - mean) / std
        x[:, :, row_start:row_end, col_start:col_end] = val
    else:
        x[:, :, row_start:row_end, col_start:col_end] = x.max().detach()

    return x


def target_probability_mean(model, x, target_class):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
        probs = torch.sigmoid(out.view(-1))
        if target_class == 0:
            probs = 1.0 - probs
        return float(probs.mean().detach().cpu().item())

    probs = F.softmax(out, dim=1)
    return float(probs[:, target_class].mean().detach().cpu().item())


def model_probability_mean_vector(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
        p1 = torch.sigmoid(out.view(-1, 1))
        probs = torch.cat([1.0 - p1, p1], dim=1)
    else:
        probs = F.softmax(out, dim=1)
    return probs.mean(dim=0).detach().cpu()


def probe_model_stats(model, clean_batches, trigger_batches, target_class):
    clean_target_vals = []
    trigger_target_vals = []
    clean_vecs = []
    trigger_vecs = []

    for (clean_x, _), trigger_x in zip(clean_batches, trigger_batches):
        clean_vec = model_probability_mean_vector(model, clean_x)
        trigger_vec = model_probability_mean_vector(model, trigger_x)
        clean_vecs.append(clean_vec)
        trigger_vecs.append(trigger_vec)

        tc = min(max(int(target_class), 0), clean_vec.numel() - 1)
        clean_target_vals.append(float(clean_vec[tc].item()))
        trigger_target_vals.append(float(trigger_vec[tc].item()))

    clean_vec_mean = torch.stack(clean_vecs, dim=0).mean(dim=0)
    trigger_vec_mean = torch.stack(trigger_vecs, dim=0).mean(dim=0)

    return {
        "clean_target": float(np.mean(clean_target_vals)),
        "trigger_target": float(np.mean(trigger_target_vals)),
        "clean_vec": clean_vec_mean,
        "trigger_vec": trigger_vec_mean,
    }


def l1_vector_distance(a, b):
    return float(torch.sum(torch.abs(a - b)).item())




def deterministic_group_seed(args, users_idx, n_clients):
    base = int(getattr(args, "run_seed", 1))
    round_idx = int(getattr(args, "current_round", 0))

    if users_idx is None:
        user_part = n_clients * 9973
    else:
        user_part = 0

        for pos, uid in enumerate(users_idx):
            user_part += (pos + 1) * int(uid) * 131
    return int(base * 1000003 + round_idx * 9176 + user_part + n_clients * 17)


def sample_random_groups(n_clients, group_size, num_groups, seed=None):
    # Deduplicate groups so the probe budget is not wasted on repeated groups.
    # If num_groups exceeds the number of possible unique groups, return all unique groups.
    rng = random.Random(seed)
    max_possible = combination_count(n_clients, group_size)
    target = min(int(num_groups), int(max_possible))
    groups = set()
    attempts = 0
    max_attempts = max(100, target * 20)
    while len(groups) < target and attempts < max_attempts:
        groups.add(tuple(sorted(rng.sample(range(n_clients), group_size))))
        attempts += 1
    if len(groups) < target:
        # Deterministic fallback enumeration if random sampling gets unlucky.
        import itertools
        for group in itertools.combinations(range(n_clients), group_size):
            groups.add(tuple(group))
            if len(groups) >= target:
                break
    return sorted(groups)


def combination_count(n, r):
    try:
        import math
        return math.comb(n, r)
    except Exception:
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        out = 1
        for i in range(1, r + 1):
            out = out * (n - r + i) // i
        return out


def compute_attack_evidence_gate(group_records, candidate_suspicious_groups, args=None, idx_attacker=None):
    """
    Decide whether this round has enough attack-specific evidence to update
    persistent memory or hard-drop clients.

    Two gates are supported:
      1. Debug/controlled gate: if idx_attacker is passed and empty, treat the
         round as clean warmup and do not update memory or hard-drop.
      2. Unsupervised gate: require enough positive trigger-excess groups and a
         max excess that is large relative to the round's excess distribution.

    Ground-truth gating is only for controlled experiments; turn it off with
    probe_use_gt_attack_gate=False when evaluating a deployable variant.
    """
    if group_records is None or len(group_records) == 0:
        return {
            "gt_attack_active": None,
            "gt_gate_used": False,
            "unsupervised_attack_evidence": False,
            "allow_memory_update": False,
            "allow_hard_drop": False,
            "allow_penalty": False,
            "reason": "no_group_records",
            "positive_group_count": 0,
            "candidate_suspicious_count": 0,
            "max_excess": 0.0,
            "median_excess": 0.0,
            "mad_excess": 0.0,
            "excess_threshold": 0.0,
            "min_required_groups": 0,
        }

    use_gt_gate = bool(getattr(args, "probe_use_gt_attack_gate", False)) if args is not None else False
    gt_attack_active = None
    if idx_attacker is not None:
        try:
            gt_attack_active = len(idx_attacker) > 0
        except Exception:
            gt_attack_active = False

    excess_values = np.array([float(r.get("excess_trigger_gap", 0.0)) for r in group_records], dtype=np.float64)
    median_excess = float(np.median(excess_values))
    mad_excess = float(np.median(np.abs(excess_values - median_excess)) + 1e-12)
    max_excess = float(np.max(excess_values))

    min_abs = float(getattr(args, "probe_attack_gate_min_excess", 0.01)) if args is not None else 0.01
    z = float(getattr(args, "probe_attack_gate_z", 2.0)) if args is not None else 2.0
    min_groups = int(getattr(args, "probe_attack_gate_min_groups", 3)) if args is not None else 3
    min_fraction = float(getattr(args, "probe_attack_gate_min_fraction", 0.01)) if args is not None else 0.01
    min_groups = max(min_groups, int(np.ceil(min_fraction * len(group_records))))
    threshold = max(min_abs, median_excess + z * mad_excess)

    candidate_positive_count = sum(
        1 for r in candidate_suspicious_groups
        if float(r.get("excess_trigger_gap", 0.0)) >= threshold
    )
    all_positive_count = int(np.sum(excess_values >= threshold))
    positive_group_count = max(candidate_positive_count, all_positive_count)
    # 2. Always compute unsupervised evidence
    unsupervised = bool(max_excess >= threshold and positive_group_count >= min_groups)

    reason = "unsupervised_gate_pass" if unsupervised else "weak_positive_excess_evidence"
    allow = unsupervised

    # 3. Optional GT gate only for debugging, never default
    if use_gt_gate and gt_attack_active is False:
        allow = False
        reason = "gt_clean_round_gate"
    elif use_gt_gate and gt_attack_active is True:
        requires_evidence = bool(getattr(args, "probe_gt_gate_requires_evidence", True)) if args is not None else True
        allow = unsupervised if requires_evidence else True
        reason = "gt_attack_round_and_evidence" if allow else "gt_attack_round_but_weak_probe_evidence"

    allow_memory = bool(allow and bool(getattr(args, "probe_gate_memory_updates", True))) if args is not None else bool(allow)
    allow_hard = bool(allow and bool(getattr(args, "probe_gate_hard_drop", True))) if args is not None else bool(allow)
    allow_penalty = bool(allow and bool(getattr(args, "probe_gate_soft_penalty", True))) if args is not None else bool(allow)

    return {
        "gt_attack_active": gt_attack_active,
        "gt_gate_used": bool(use_gt_gate and gt_attack_active is not None),
        "unsupervised_attack_evidence": bool(unsupervised),
        "allow_memory_update": bool(allow_memory),
        "allow_hard_drop": bool(allow_hard),
        "allow_penalty": bool(allow_penalty),
        "reason": reason,
        "positive_group_count": int(positive_group_count),
        "candidate_suspicious_count": int(len(candidate_suspicious_groups)),
        "max_excess": float(max_excess),
        "median_excess": float(median_excess),
        "mad_excess": float(mad_excess),
        "excess_threshold": float(threshold),
        "min_required_groups": int(min_groups),
    }


def compact_attack_evidence(info):
    keys = [
        "reason",
        "gt_attack_active",
        "unsupervised_attack_evidence",
        "allow_memory_update",
        "allow_hard_drop",
        "allow_penalty",
        "positive_group_count",
        "min_required_groups",
        "max_excess",
        "excess_threshold",
    ]
    out = {}
    for key in keys:
        value = info.get(key) if isinstance(info, dict) else None
        if isinstance(value, float):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def select_suspicious_groups(group_records, args=None):
    if len(group_records) == 0:
        return []

    # New guardrail: risk memory only receives positive trigger-specific excess.
    # A group can be clean-distribution-weird, shift-weird, or direction-weird,
    # but if trigger_gap <= clean_gap it is not attack evidence for this method.
    require_positive = bool(getattr(args, "probe_require_positive_excess", True)) if args is not None else True
    excess_floor = float(getattr(args, "probe_excess_floor", 0.0)) if args is not None else 0.0
    excess_floor_z = float(getattr(args, "probe_excess_floor_z", 0.0)) if args is not None else 0.0

    excess_values = np.array([float(r.get("excess_trigger_gap", 0.0)) for r in group_records], dtype=np.float64)
    excess_median = float(np.median(excess_values))
    excess_mad = float(np.median(np.abs(excess_values - excess_median)) + 1e-12)
    adaptive_excess_floor = max(excess_floor, excess_median + excess_floor_z * excess_mad)

    candidates = []
    for r in group_records:
        excess = float(r.get("excess_trigger_gap", 0.0))
        score = float(r.get("score", 0.0))
        if require_positive and excess <= adaptive_excess_floor:
            continue
        if score <= 0.0:
            continue
        candidates.append(r)

    if len(candidates) == 0:
        return []

    scores = np.array([float(r["score"]) for r in candidates], dtype=np.float64)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) + 1e-12

    fraction = float(getattr(args, "probe_suspicious_fraction", 0.25)) if args is not None else 0.25
    fraction = max(0.02, min(0.60, fraction))
    k = max(1, int(np.ceil(fraction * len(candidates))))
    k = min(k, len(candidates))

    threshold_z = float(getattr(args, "probe_threshold_z", 0.10)) if args is not None else 0.10
    threshold = median + threshold_z * mad
    top_idx = np.argsort(scores)[-k:]
    suspicious = [candidates[int(i)] for i in top_idx if scores[int(i)] >= threshold]

    # Important: do not backfill from negative-excess groups. If too few groups
    # survive the floor, accept fewer suspicious groups instead of contaminating
    # persistent risk with clean-only weirdness.
    min_groups = int(getattr(args, "probe_min_suspicious_groups", 1)) if args is not None else 1
    if len(suspicious) < min_groups and len(candidates) > 0:
        top_idx = np.argsort(scores)[-min(min_groups, len(candidates)):]
        suspicious = [candidates[int(i)] for i in top_idx]

    return suspicious


def median_clip_updates(w_updates):
    if len(w_updates) == 0:
        return []

    norms = []
    for update in w_updates:
        vec = parameters_dict_to_vector_flt(update)
        norms.append(torch.norm(vec, p=2).item())

    clip_norm = float(np.median(norms)) + 1e-12
    clipped = []

    for update, norm_val in zip(w_updates, norms):
        if norm_val <= clip_norm:
            clipped.append(copy.deepcopy(update))
            continue

        gamma = clip_norm / (norm_val + 1e-12)
        scaled = {}
        for key, value in update.items():
            if torch.is_floating_point(value):
                scaled[key] = value.clone() * gamma
            else:
                scaled[key] = value.clone()
        clipped.append(scaled)

    return clipped


def aggregate_updates_by_indices(w_updates, indices, global_state, weights):
    aggregated = {k: torch.zeros_like(v) for k, v in global_state.items()}

    for idx, weight in zip(indices, weights):
        update = w_updates[idx]
        for key, value in update.items():
            if key not in aggregated:
                continue
            if torch.is_floating_point(aggregated[key]):
                aggregated[key] += value.to(aggregated[key].device) * float(weight)

    return aggregated


def add_update_to_state(global_state, update):
    new_state = {}
    for key, value in global_state.items():
        if key in update and torch.is_floating_point(value):
            new_state[key] = value + update[key].to(value.device)
        else:
            new_state[key] = value.clone()
    return new_state


def normalize_weights(weights):
    weights = [float(w) for w in weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights) for _ in weights]
    return [w / total for w in weights]


def cosine_direction_gap(a, b):
    denom = (torch.norm(a, p=2) * torch.norm(b, p=2)) + 1e-12
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    cos = torch.dot(a, b) / denom
    return float((1.0 - cos).detach().cpu().item())


def parameters_dict_to_vector_flt(update_dict):
    vec = []
    for key, param in update_dict.items():
        last = key.split(".")[-1]
        if last in ("num_batches_tracked", "running_mean", "running_var"):
            continue
        if not torch.is_floating_point(param):
            continue
        vec.append(param.reshape(-1))
    if len(vec) == 0:
        return torch.zeros(1)
    return torch.cat(vec)




def mean_array_at_indices(values, indices):
    if values is None or len(indices) == 0:
        return 0.0
    vals = []
    for idx in indices:
        if 0 <= int(idx) < len(values):
            vals.append(float(values[int(idx)]))
    return float(np.mean(vals)) if vals else 0.0


def mean_list_at_indices(values, indices):
    if values is None or len(indices) == 0:
        return 0.0
    vals = []
    for idx in indices:
        if 0 <= int(idx) < len(values):
            vals.append(float(values[int(idx)]))
    return float(np.mean(vals)) if vals else 0.0


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def summarize_client_group_explanations(group_records, suspicious_group_sets, strengths, total_clients):
    if strengths is None:
        strengths = {}
    metrics = {}
    for local_idx in range(total_clients):
        metrics[local_idx] = {
            "all_group_count": 0.0,
            "suspicious_group_count": 0.0,
            "suspicious_strength_sum": 0.0,
            "all_score_sum": 0.0,
            "suspicious_score_sum": 0.0,
            "suspicious_clean_gap_sum": 0.0,
            "suspicious_trigger_gap_sum": 0.0,
            "suspicious_excess_sum": 0.0,
            "suspicious_shift_gap_sum": 0.0,
            "suspicious_direction_gap_sum": 0.0,
            "suspicious_clean_vec_gap_sum": 0.0,
            "suspicious_trigger_vec_gap_sum": 0.0,
            "attacker_group_count": 0.0,
            "benign_only_suspicious_group_count": 0.0,
        }

    for record in group_records:
        group = [int(x) for x in record.get("group", [])]
        group_key = tuple(group)
        is_suspicious = group_key in suspicious_group_sets
        contains_attacker = bool(record.get("contains_known_attacker", False))
        strength = safe_float(strengths.get(group_key, 0.0), 0.0) if is_suspicious else 0.0
        for idx in group:
            if idx not in metrics:
                continue
            m = metrics[idx]
            m["all_group_count"] += 1.0
            m["all_score_sum"] += safe_float(record.get("score"))
            if not is_suspicious:
                continue
            m["suspicious_group_count"] += 1.0
            m["suspicious_strength_sum"] += strength
            m["suspicious_score_sum"] += safe_float(record.get("score"))
            m["suspicious_clean_gap_sum"] += safe_float(record.get("clean_gap"))
            m["suspicious_trigger_gap_sum"] += safe_float(record.get("trigger_gap"))
            m["suspicious_excess_sum"] += safe_float(record.get("excess_trigger_gap"))
            m["suspicious_shift_gap_sum"] += safe_float(record.get("shift_gap"))
            m["suspicious_direction_gap_sum"] += safe_float(record.get("direction_gap"))
            m["suspicious_clean_vec_gap_sum"] += safe_float(record.get("clean_vec_gap"))
            m["suspicious_trigger_vec_gap_sum"] += safe_float(record.get("trigger_vec_gap"))
            if contains_attacker:
                m["attacker_group_count"] += 1.0
            else:
                m["benign_only_suspicious_group_count"] += 1.0

    for idx, m in metrics.items():
        all_count = max(m["all_group_count"], 1.0)
        sus_count = max(m["suspicious_group_count"], 1.0)
        m["avg_all_score"] = m["all_score_sum"] / all_count
        m["avg_suspicious_score"] = m["suspicious_score_sum"] / sus_count
        m["avg_suspicious_clean_gap"] = m["suspicious_clean_gap_sum"] / sus_count
        m["avg_suspicious_trigger_gap"] = m["suspicious_trigger_gap_sum"] / sus_count
        m["avg_suspicious_excess"] = m["suspicious_excess_sum"] / sus_count
        m["avg_suspicious_shift_gap"] = m["suspicious_shift_gap_sum"] / sus_count
        m["avg_suspicious_direction_gap"] = m["suspicious_direction_gap_sum"] / sus_count
        m["avg_suspicious_clean_vec_gap"] = m["suspicious_clean_vec_gap_sum"] / sus_count
        m["avg_suspicious_trigger_vec_gap"] = m["suspicious_trigger_vec_gap_sum"] / sus_count
    return metrics

# -----------------------------------------------------------------------------
# Per-round CSV telemetry shared with the collusion attack runner.
# ProbeGroupDefense records detector-side values first; the runner later adds
# model accuracy/loss values after evaluation and commits one complete CSV row.
# -----------------------------------------------------------------------------


def _csv_scalar(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool):
        return int(value)
    return value



def build_probe_round_summary_metrics(args, per_run, round_idx, group_records,
                                      candidate_suspicious_groups, suspicious_groups,
                                      attack_evidence, telemetry_summary,
                                      hard_dropped_indices, users_idx, idx_attacker,
                                      final_persistent_risk):
    evidence = attack_evidence or {}
    telemetry = telemetry_summary or {}
    attacker_set = set(int(x) for x in (idx_attacker or []))
    malicious_real_ids = sorted(int(x) for x in (idx_attacker or []))

    allow_memory = bool(evidence.get("allow_memory_update", False))
    allow_hard = bool(evidence.get("hard_drop_action_allowed", evidence.get("allow_hard_drop", False)))
    allow_penalty = bool(evidence.get("allow_penalty", False))
    confidence_tier = 3 if allow_hard else 2 if allow_memory else 1 if allow_penalty else 0

    positive_count = int(evidence.get("positive_group_count", 0) or 0)
    total_groups = int(len(group_records or []))
    max_excess = float(evidence.get("max_excess", 0.0) or 0.0)
    threshold = float(evidence.get("excess_threshold", 0.0) or 0.0)

    attacker_mean = float(telemetry.get("attacker_mean_risk", 0.0) or 0.0)
    benign_mean = float(telemetry.get("benign_mean_risk", 0.0) or 0.0)
    attacker_persistent = float(telemetry.get("attacker_mean_persistent_risk", 0.0) or 0.0)
    benign_persistent = float(telemetry.get("benign_mean_persistent_risk", 0.0) or 0.0)

    dropped_local = [int(x) for x in (hard_dropped_indices or [])]
    dropped_real = []
    for local_idx in dropped_local:
        if users_idx is not None and local_idx < len(users_idx):
            dropped_real.append(int(users_idx[local_idx]))
        else:
            dropped_real.append(local_idx)
    true_positive_drops = sum(1 for real_id in dropped_real if real_id in attacker_set)
    precision = (true_positive_drops / float(len(dropped_real))) if dropped_real else ""
    recall = (true_positive_drops / float(len(attacker_set))) if attacker_set else ""

    sorted_clients = sorted(
        telemetry.get("rows", []),
        key=lambda r: (float(r.get("persistent_risk", 0.0)), float(r.get("risk", 0.0))),
        reverse=True,
    )
    top_client = sorted_clients[0] if sorted_clients else {}
    risk_values = sorted([float(r.get("persistent_risk", 0.0)) for r in telemetry.get("rows", [])], reverse=True)
    hard_drop_margin = (risk_values[0] - risk_values[1]) if len(risk_values) >= 2 else ""

    top_group = max(group_records or [], key=lambda r: float(r.get("score", 0.0)), default={})

    return  {
        "round": int(round_idx),
        "gate_reason": evidence.get("reason", ""),
        "confidence_tier": confidence_tier,
        "positive_fraction": (positive_count / float(total_groups)) if total_groups else 0.0,
        "unsup_gate_pass": bool(evidence.get("unsupervised_attack_evidence", False)),
        "allow_memory": allow_memory,
        "allow_hard_drop": allow_hard,
        "allow_penalty": allow_penalty,
        "candidate_groups": int(len(candidate_suspicious_groups or [])),
        "memory_groups": int(len(suspicious_groups or [])),
        "positive_group_count": positive_count,
        "min_required_groups": int(evidence.get("min_required_groups", 0) or 0),
        "max_excess": max_excess,
        "excess_threshold": threshold,
        "excess_margin": max_excess - threshold,
        "attacker_mean_risk": attacker_mean,
        "benign_mean_risk": benign_mean,
        "risk_gap": attacker_mean - benign_mean,
        "attacker_persistent_mean": attacker_persistent,
        "benign_persistent_mean": benign_persistent,
        "persistent_gap": attacker_persistent - benign_persistent,
        "malicious_real_ids": malicious_real_ids,
        "hard_drop_local": dropped_local,
        "hard_drop_real": dropped_real,
        "hard_drop_count": len(dropped_local),
        "hard_drop_precision": precision,
        "hard_drop_recall": recall,
        "hard_drop_margin": hard_drop_margin,
        "top_group_score": float(top_group.get("score", 0.0) or 0.0),
        "top_group_excess": float(top_group.get("excess_trigger_gap", 0.0) or 0.0),
        "top_group_has_attacker": bool(top_group.get("contains_known_attacker", False)),
        "top_client_real": top_client.get("real_client_id", ""),
        "top_client_is_attacker": bool(top_client.get("is_known_attacker", False)) if top_client else "",
        "hard_drop_action_allowed": bool(evidence.get("hard_drop_action_allowed", allow_hard)),
        "catastrophic_hard_drop_veto": bool(evidence.get("catastrophic_hard_drop_veto", False)),
        "hard_drop_guard_reason": evidence.get("hard_drop_guard_reason", ""),
        "hard_drop_topk_overlap": float(evidence.get("hard_drop_topk_overlap", 1.0) or 0.0),
        "hard_drop_rank_agreement": float(evidence.get("hard_drop_rank_agreement", 1.0) or 0.0),
        "hard_drop_boundary_margin_ratio": float(evidence.get("hard_drop_boundary_margin_ratio", 0.0) or 0.0),
        "boundary_rescue_applied": bool(evidence.get("boundary_rescue_applied", False)),
        "boundary_rescue_reason": evidence.get("boundary_rescue_reason", ""),
        "boundary_rescue_changed_count": int(evidence.get("boundary_rescue_changed_count", 0) or 0),
        "boundary_rescue_pool_size": int(evidence.get("boundary_rescue_pool_size", 0) or 0),
    }