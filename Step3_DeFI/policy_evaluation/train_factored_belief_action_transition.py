from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
    from policy_evaluation.train_joint_execution_update_mlp import build_meta
except ModuleNotFoundError:
    from train_chunk_effect_phi_mlp import split_by_sequence
    from train_joint_execution_update_mlp import build_meta


class BeliefTransitionMLP(nn.Module):
    """T_theta: h_t,o_t,a_t,o_{t+1} -> h_{t+1}."""

    def __init__(self, state_dim: int, summary_dim: int, action_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        in_dim = state_dim * 2 + summary_dim + action_dim + task_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

    def forward(self, h_t: torch.Tensor, state_t: torch.Tensor, action_t: torch.Tensor, state_tp1: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, state_t, action_t, state_tp1, task_vec, meta], dim=-1)
        return self.net(x)


class ActionAdaptationMLP(nn.Module):
    """D_theta: A_remain,h_t,h_{t+1} -> delta_A and learned mode."""

    def __init__(self, summary_dim: int, action_dim: int, chunk_len: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.task_dim = int(task_dim)
        in_dim = summary_dim * 2 + chunk_len * action_dim + task_dim + 3
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, chunk_len * action_dim)
        self.mode_head = nn.Linear(hidden_dim, 3)  # keep, patch, replan
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_remain: torch.Tensor, h_t: torch.Tensor, h_tp1: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([action_remain.reshape(action_remain.shape[0], -1), h_t, h_tp1, task_vec, meta], dim=-1)
        y = self.trunk(x)
        return {
            "action_delta": self.delta_head(y).reshape(action_remain.shape[0], self.chunk_len, self.action_dim),
            "mode_logits": self.mode_head(y),
            "confidence_logit": self.confidence_head(y).squeeze(-1),
        }


def load_memory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    required = [
        "state_start",
        "state_end",
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "action_delta_target",
        "success",
        "task_vec",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
    ]
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")
    return out


def make_bank(memory: dict[str, np.ndarray], keep_delta_norm: float, replan_delta_norm: float) -> dict[str, np.ndarray]:
    action_delta = np.asarray(memory["action_delta_target"], dtype=np.float32)
    delta_norm = np.linalg.norm(action_delta.reshape(action_delta.shape[0], -1), axis=1)
    if "mode_label" in memory:
        # Balanced memories can provide explicit keep/patch/replan labels.
        mode = np.asarray(memory["mode_label"], dtype=np.int64).reshape(-1)
    else:
        mode = np.ones((action_delta.shape[0],), dtype=np.int64)
        mode[delta_norm <= keep_delta_norm] = 0
        if replan_delta_norm > 0:
            mode[delta_norm >= replan_delta_norm] = 2
    return {
        "state_start": np.asarray(memory["state_start"], dtype=np.float32),
        "state_end": np.asarray(memory["state_end"], dtype=np.float32),
        "h_t": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "h_tp1": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "belief_delta": np.asarray(memory["target_summary_exec"], dtype=np.float32) - np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "action_delta": action_delta,
        "executed_action": np.asarray(memory.get("executed_action", np.zeros((action_delta.shape[0], action_delta.shape[-1]), dtype=np.float32)), dtype=np.float32),
        "success": np.asarray(memory["success"], dtype=np.float32).reshape(-1),
        "task_vec": np.asarray(memory["task_vec"], dtype=np.float32),
        "meta": build_meta(memory),
        "mode": mode,
        "sequence_index": np.asarray(memory["sequence_index"], dtype=np.int32).reshape(-1),
    }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in bank.items():
        if value.dtype.kind in "f":
            out[key] = torch.from_numpy(value[ids]).to(device)
        elif key == "mode":
            out[key] = torch.from_numpy(value[ids]).long().to(device)
    return out


def run_models(t_model: BeliefTransitionMLP, d_model: ActionAdaptationMLP, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_delta_h = t_model(batch["h_t"], batch["state_start"], batch["executed_action"], batch["state_end"], batch["task_vec"], batch["meta"])
    pred_h_tp1 = batch["h_t"] + pred_delta_h
    d_out = d_model(batch["action_chunk"], batch["h_t"], pred_h_tp1, batch["task_vec"], batch["meta"])
    return pred_delta_h, d_out


def eval_models(t_model: BeliefTransitionMLP, d_model: ActionAdaptationMLP, bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    t_model.eval()
    d_model.eval()
    vals: dict[str, list[torch.Tensor]] = {"loss": [], "belief_l1": [], "action_l1": [], "mode_acc": [], "confidence_l1": []}
    with torch.no_grad():
        for start in range(0, ids.size, batch_size):
            batch = gather(bank, ids[start : start + batch_size], device)
            pred_delta_h, d_out = run_models(t_model, d_model, batch)
            belief_loss = F.smooth_l1_loss(pred_delta_h, batch["belief_delta"])
            action_loss = F.smooth_l1_loss(d_out["action_delta"], batch["action_delta"])
            mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
            conf_target = (batch["mode"] != 2).float()
            confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
            loss = belief_loss + 0.5 * action_loss + 0.1 * mode_loss + 0.05 * confidence_loss
            vals["loss"].append(loss.cpu())
            vals["belief_l1"].append(F.l1_loss(pred_delta_h, batch["belief_delta"]).cpu())
            vals["action_l1"].append(F.l1_loss(d_out["action_delta"], batch["action_delta"]).cpu())
            vals["mode_acc"].append((d_out["mode_logits"].argmax(dim=-1) == batch["mode"]).float().mean().cpu())
            vals["confidence_l1"].append(F.l1_loss(d_out["confidence_logit"].sigmoid(), conf_target).cpu())
    return {key: float(torch.stack(value).mean().item()) for key, value in vals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train factored T_theta belief dynamics and D_theta action adaptation.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--keep-delta-norm", type=float, default=0.02)
    parser.add_argument("--replan-delta-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = load_memory(args.memory_npz)
    bank = make_bank(memory, args.keep_delta_norm, args.replan_delta_norm)
    for key, value in bank.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite values in {key}")

    train_ids, val_ids = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    if val_ids.size == 0:
        rng = np.random.default_rng(args.seed)
        ids = np.arange(bank["sequence_index"].shape[0])
        rng.shuffle(ids)
        n_val = max(1, int(ids.size * args.val_ratio))
        val_ids, train_ids = ids[:n_val], ids[n_val:]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    t_model = BeliefTransitionMLP(
        state_dim=bank["state_start"].shape[-1],
        summary_dim=bank["h_t"].shape[-1],
        action_dim=bank["executed_action"].shape[-1],
        task_dim=bank["task_vec"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    d_model = ActionAdaptationMLP(
        summary_dim=bank["h_t"].shape[-1],
        action_dim=bank["action_chunk"].shape[-1],
        chunk_len=bank["action_chunk"].shape[-2],
        task_dim=bank["task_vec"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(list(t_model.parameters()) + list(d_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    best_state = None
    log_path = args.output_dir / "train_log.jsonl"

    with log_path.open("w") as log_f:
        for step in range(1, args.steps + 1):
            ids = rng.choice(train_ids, size=args.batch_size, replace=train_ids.size < args.batch_size)
            batch = gather(bank, ids, device)
            t_model.train()
            d_model.train()
            pred_delta_h, d_out = run_models(t_model, d_model, batch)
            belief_loss = F.smooth_l1_loss(pred_delta_h, batch["belief_delta"])
            action_loss = F.smooth_l1_loss(d_out["action_delta"], batch["action_delta"])
            mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
            conf_target = (batch["mode"] != 2).float()
            confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
            loss = belief_loss + 0.5 * action_loss + 0.1 * mode_loss + 0.05 * confidence_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % 200 == 0 or step == args.steps:
                metrics = eval_models(t_model, d_model, bank, val_ids, device, args.eval_batch_size)
                metrics.update({"step": int(step), "train_loss": float(loss.item())})
                log_f.write(json.dumps(metrics) + "\n")
                log_f.flush()
                if best is None or metrics["loss"] < best["loss"]:
                    best = dict(metrics)
                    best_state = {
                        "t_model": {k: v.detach().cpu() for k, v in t_model.state_dict().items()},
                        "d_model": {k: v.detach().cpu() for k, v in d_model.state_dict().items()},
                    }
                print(json.dumps(metrics), flush=True)

    if best_state is not None:
        t_model.load_state_dict(best_state["t_model"])
        d_model.load_state_dict(best_state["d_model"])
    ckpt_path = args.output_dir / "factored_belief_action_transition.pt"
    torch.save(
        {
            "runtime_type": "factored_belief_action_transition",
            "t_model_state": t_model.state_dict(),
            "d_model_state": d_model.state_dict(),
            "state_dim": int(bank["state_start"].shape[-1]),
            "summary_dim": int(bank["h_t"].shape[-1]),
            "action_dim": int(bank["action_chunk"].shape[-1]),
            "chunk_len": int(bank["action_chunk"].shape[-2]),
            "task_dim": int(bank["task_vec"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "keep_delta_norm": float(args.keep_delta_norm),
            "replan_delta_norm": float(args.replan_delta_norm),
            "best": best,
        },
        ckpt_path,
    )
    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(train_ids.size),
        "num_val": int(val_ids.size),
        "mode_counts": {str(k): int(v) for k, v in zip(*np.unique(bank["mode"], return_counts=True))},
        "steps": int(args.steps),
        "best": best,
        "output_ckpt": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
