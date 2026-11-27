"""Measure the CPv2 pilot on untouched decisions from new training rollouts."""

import numpy as np
import torch

from civ5rl.core.model import batch_inputs
from civ5rl.core.variants import variant_spec
from civ5rl.engine.ppo.core import encode_episode, episode_returns
from civ5rl.engine.ppo.storage import load_rollout
from civ5rl.experiments.imitation.helpers import isolated_rng
from civ5rl.experiments.ppo.critic import cpv2_contact_mask, critic_statistics
from civ5rl.provenance.common import file_sha256, read_json, write_json
from civ5rl.provenance.jsonl import read_jsonl


def prepare_carrier(config, output):
    """Generate training-world cases for collection tuning without sampling plays."""
    from copy import deepcopy

    from civ5rl.protocols.ppo_target_scale import TargetFactory
    from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
    from civ5rl.provenance.common import ROOT, file_sha256

    cfg = deepcopy(config)
    experiment = cfg["experiment"]
    path = ROOT / experiment["evaluation_manifest"]
    if file_sha256(path) != experiment["evaluation_manifest_sha256"]:
        raise ValueError("Tuning source manifest differs")
    plan = read_json(path)
    verify_manifest_digest(plan, error_msg="Tuning source manifest digest differs")
    experiment["training_namespace"] += "_tuning"
    factory = TargetFactory(cfg, plan["base"])
    start = experiment["start_episode"]
    episodes = [{"spec": factory.make(start + offset), "chosen": []} for offset in range(288)]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(episodes, output)
    return {"cases": len(episodes), "path": str(output)}


def _logits_without_attention(policy, attention_input, mask):
    if policy.variant_spec.action_context_pre_gelu:
        attention_input = policy.action[1](attention_input)
    return policy.action[2](attention_input).squeeze(-1).masked_fill(~mask, -torch.inf)


@torch.no_grad()
def _attention_metrics(policy, episodes, out, device):
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    activations = []

    def capture(_module, args, _output):
        activations.append(args[0])

    with torch.autocast(torch.device(device).type, enabled=False):
        hook = policy.action_attention.register_forward_hook(capture)
        try:
            for episode in episodes:
                hidden = None
                contact = episode["spec"]["opponent"] == "cpv2" and episode["spec"].get("placement") == "contact"
                encoded = encode_episode(episode, policy.variant)
                for start in range(0, len(encoded), 32):
                    if policy.spatial_cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    part = encoded[start : start + 32]
                    batch = batch_inputs(part, device, non_blocking=True)
                    cells, context = policy.encode_spatial(batch)
                    memory, hidden = policy.sequence_memory(context, [len(part)], hidden)
                    logits, features = policy.heads(batch, cells, memory, return_value_features=True)
                    mask = batch["action_mask"]
                    count = mask.sum(dim=1, keepdim=True)
                    valid = mask & (count >= 2)
                    with torch.autocast(torch.device(device).type, enabled=False):
                        no_attention = _logits_without_attention(policy, activations.pop(), mask)
                        delta = (logits - no_attention).masked_fill(~mask, 0)
                        centered = (delta - delta.sum(1, keepdim=True) / count.clamp_min(1)).masked_fill(~valid, 0)
                        switches = ((logits.argmax(1) != no_attention.argmax(1)) & (count[:, 0] >= 2)).sum()
                        totals[0] += centered.square().sum(dtype=torch.float64)
                        totals[1] += valid.sum()
                        totals[2] += switches
                        totals[3] += (count[:, 0] >= 2).sum()
                        if contact:
                            prediction = policy.value_from_features(features)
                            base_value = policy.value(memory.float()).squeeze(-1)
                            totals[4] += (prediction - base_value).square().sum(dtype=torch.float64)
                            totals[5] += len(part)
        finally:
            hook.remove()
    square, actions, switches, free, value_square, contact_count = totals.tolist()
    initial = torch.load(out / "checkpoints/initial.pt", map_location="cpu", weights_only=False)["model"]
    names = [name for name in initial if name.startswith(tuple(f"action_attention.{key}_proj." for key in "qkv"))]
    current = policy.state_dict()
    denominator = sum(float(initial[name].double().square().sum()) for name in names)
    numerator = sum(float((current[name].detach().cpu().double() - initial[name]).square().sum()) for name in names)
    return {"attention_rms": (square / actions) ** 0.5 if actions else None,
            "argmax_switch_rate": switches / free if free else None,
            "qkv_relative_change": (numerator / denominator) ** 0.5,
            "value_context_rms": (value_square / contact_count) ** 0.5 if contact_count else None,
            "legal_actions": int(actions), "free_positions": int(free), "value_positions": int(contact_count)}


def record_pre_update(policy, episodes, config, out, device, rollout):
    path = out / "diagnostic-rollouts.json"
    rows = read_json(path) if path.exists() else {}
    key = str(rollout)
    if key in rows:
        return
    returns = np.concatenate(episode_returns(episodes, gamma=config["ppo"]["gamma"]))
    predictions = np.concatenate([ep["old_value"] for ep in episodes])
    contact = cpv2_contact_mask(episodes)
    rows[key] = {"pre": critic_statistics(predictions, returns),
                 "cpv2_contact_pre": critic_statistics(predictions[contact], returns[contact]) if contact.any() else None,
                 "cpv2_contact_positions": int(contact.sum())}
    if rollout == 3 and not policy.variant_spec.action_context_pre_gelu:
        rows[key].update(_attention_metrics(policy, episodes, out, device))
    write_json(path, rows)


def record_post_r3(session, monitor=None):
    """Bind the attention measurement to the committed R3 boundary, also on resume."""
    policy, state, out = session.policy, session.state, session.out
    if not (policy.variant_spec.action_context_pre_gelu and state["rollouts"] == 3 and state["phase"] == "collect"):
        return
    path = out / "diagnostic-rollouts.json"
    rows = read_json(path) if path.exists() else {}
    if "3_post" in rows:
        return
    updates = read_jsonl(out / "updates.jsonl")
    if [row["rollout"] for row in updates] != [1, 2, 3] or updates[-1]["ppo"]["PPO_skipped"]:
        return
    checkpoint = out / state["latest"]
    rollout = out / "rollouts/latest.pt"
    metadata = read_json(rollout.with_suffix(".json"))
    if (checkpoint.name != "update-0003.pt" or metadata["update"] != 3
            or metadata["sha256"] != file_sha256(rollout)):
        raise ValueError("Post-R3 diagnostic inputs differ from the committed rollout boundary")
    if monitor is not None:
        monitor.mark("diagnostic", rollout=3)
    training = policy.training
    try:
        with isolated_rng(session.shuffle):
            metrics = _attention_metrics(policy.eval(), load_rollout(rollout), out, session.device)
    finally:
        policy.train(training)
    rows["3_post"] = {
        "checkpoint": state["latest"], "checkpoint_sha256": file_sha256(checkpoint),
        "rollout": "rollouts/latest.pt", "rollout_sha256": metadata["sha256"],
        "accepted_steps": state["optimizer_steps"], **metrics,
    }
    write_json(path, rows)


def pilot_report(out):
    rows = read_json(out / "diagnostic-rollouts.json")
    first = read_json(out / "critic-preparation.json")
    updates = read_jsonl(out / "updates.jsonl")
    if [row["rollout"] for row in updates] != [1, 2, 3] or {"2", "3"} - rows.keys():
        raise ValueError("Pilot requires three committed rollouts")
    config = read_json(out / "run.json")["config"]
    post_gelu_pilot = not variant_spec(config["model"]["variant"]).action_context_pre_gelu
    r3 = rows["3"]
    r3_attention = r3 if post_gelu_pilot else rows.get("3_post", {})
    ev1 = (first.get("cpv2_contact_after") or {}).get("explained_variance")
    ev2, ev3 = ((rows[index].get("cpv2_contact_pre") or {}).get("explained_variance") for index in ("2", "3"))
    rms, switches, qkv, value_rms = (r3_attention.get(key) for key in (
        "attention_rms", "argmax_switch_rate", "qkv_relative_change", "value_context_rms"
    ))
    ppo = [row["ppo"] for row in updates[1:]]
    steps = [item["optimizer_steps"] for item in ppo]
    if not post_gelu_pilot and r3_attention and (
        r3_attention["accepted_steps"] != sum(row["ppo"]["optimizer_steps"] for row in updates)
        or r3_attention["checkpoint"] != "checkpoints/update-0003.pt"
        or r3_attention["checkpoint_sha256"] != file_sha256(out / r3_attention["checkpoint"])
        or r3_attention["rollout_sha256"] != read_json(out / "rollouts/latest.json")["sha256"]
    ):
        raise ValueError("Post-R3 metrics do not match the committed ledger and rollout")
    kl_cutoffs = any(item["optimizer_steps"] < 30 and item["stop_reason"] in {"stop_kl", "rejected_full_kl"} for item in ppo)
    overheat = (rms is not None and rms > 0.25) or kl_cutoffs
    underheat = (None not in (rms, switches, qkv)
                 and (rms < 0.015 or switches < 0.002 or qkv < 3e-4))
    attention_ok = not (overheat or underheat) and None not in (rms, switches, qkv) and min(steps) >= 36
    critic_ok = (None not in (ev1, ev2, ev3, value_rms) and ev3 >= 0.25 and ev1 - ev2 <= 0.30
                 and ev3 >= ev2 - 0.05 and value_rms >= 0.01)
    critic_change = {}
    optimizer_change = {}
    if not critic_ok and None not in (ev1, ev2, ev3):
        if (ev2 >= 0.20 and ev3 < 0.15 and rms is not None and rms > 0.25 and kl_cutoffs):
            pass
        elif ev1 > 0.65 and ev2 < 0.15:
            critic_change = {"epochs": 2, "lr": 5e-5}
        elif ev1 < 0.35 and ev2 < 0.20:
            critic_change = {"epochs": 6}
        elif ev2 >= 0.20 and ev3 < 0.15:
            optimizer_change["value"] = 1.5e-5
    context_lr = None
    if overheat:
        context_lr = 2.5e-5
    elif underheat:
        context_lr = 7.5e-5 if critic_change and ev2 <= 0 and 0.008 <= rms < 0.015 else 1e-4
    if context_lr is not None:
        optimizer_change.update(action_context=context_lr, value_actions=context_lr)
    expected_epochs = config["critic"]["epochs"]
    full_fit = first["epochs_completed"] == expected_epochs
    return {"passed": bool(attention_ok and critic_ok and full_fit), "attention_passed": attention_ok,
            "critic_passed": critic_ok, "accepted_ppo_steps": steps, "R3_pre": r3,
            "critic_epochs_completed": first["epochs_completed"], "critic_fit_complete": full_fit,
            "cpv2_contact_EV": {"post_R1": ev1, "pre_R2": ev2, "pre_R3": ev3},
            **({"retry_overrides": {"critic": critic_change, "optimizer": {"group_lrs": optimizer_change}}}
               if post_gelu_pilot else {"R3_post": r3_attention or None})}
