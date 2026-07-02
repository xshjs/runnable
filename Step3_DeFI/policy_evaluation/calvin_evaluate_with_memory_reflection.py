from collections import Counter
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from policy_evaluation.calvin_evaluate import (
    count_success,
    evaluate_policy as original_evaluate_policy,
    get_log_dir,
    get_video_tag,
    load_eval_sequences,
    print_and_save,
    rollout as original_rollout,
)
from policy_evaluation.evaluate_defi_memory_language_conditioned import (
    build_augmented_instruction,
    build_postexec_reflection_prompt,
    infer_mismatch_type,
    mean_pool_feature,
    load_key_memory_rows,
    make_reflection_row,
    retrieve_adapter_memory_topk,
    retrieve_key_memory,
    summarize_important_key_memory,
    summarize_key_memory,
    summarize_recent_memory,
)
from policy_evaluation.defi_memory_models import load_memory_arrays
from policy_evaluation.memory_state_guided_adapter import MemoryStateGuidedAdapter
from policy_evaluation.memory_weight_calibrator import MemoryWeightCalibrator, pad_weight_inputs
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_models.rollout.rollout_video import RolloutVideo
from policy_models.utils.utils import get_all_checkpoints


logger = logging.getLogger(__name__)


RULE_MEMORY = {
    "default": "Prefer conservative future calibration and preserve verified progress.",
}


def snapshot_env_raw(env):
    return env.env.get_obs()


def reset_env_raw(env, raw):
    env.reset(robot_obs=raw["robot_obs"], scene_obs=raw["scene_obs"])


def make_state_memory_row(memory_id, seq_idx, sub_idx, task, success, steps, source):
    return {
        "memory_id": memory_id,
        "memory_type": source,
        "sequence_index": int(seq_idx),
        "subtask_index": int(sub_idx),
        "task": task,
        "success": bool(success),
        "steps": int(steps),
        "state_semantics": {
            "completed_actions": [task] if success else [],
            "object_state_summary": "state_verified" if success else "state_unverified",
            "relation_summary": "relation_verified" if success else "relation_unverified",
        },
    }


def update_runtime_state_memory(recent_memory, key_state_memory, row, recent_k, key_max):
    recent_memory.append(row)
    del recent_memory[:-recent_k]
    if row.get("success"):
        key_state_memory.append(dict(row, memory_type="key_state"))
        if key_max > 0:
            del key_state_memory[:-key_max]


def rollout_with_lang_text(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    lang_text,
    override_future_feature=None,
):
    old_mode = getattr(model, "future_feature_mode", None)
    old_override = getattr(model, "override_future_feature", None)
    try:
        if override_future_feature is not None:
            model.future_feature_mode = "override"
            model.override_future_feature = override_future_feature
        obs = env.get_obs()
        base_annotation = val_annotations[subtask][0]
        goal = lang_embeddings.get_lang_goal(base_annotation)
        goal["lang_text"] = lang_text

        model.reset()
        start_info = env.get_info()
        for step in range(cfg.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                return True, int(step + 1)
        return False, int(cfg.ep_len)
    finally:
        if old_mode is not None:
            model.future_feature_mode = old_mode
        if old_override is not None:
            model.override_future_feature = old_override
        elif hasattr(model, "override_future_feature"):
            model.override_future_feature = None


def evaluate_sequence_memory(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    lang_embeddings,
    val_annotations,
    cfg,
    record,
    rollout_video,
    seq_idx,
    qwen_getter,
    adapter_getter,
    key_memory_rows,
    runtime_key_state_memory,
    logs,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(seq_idx), caption=caption)

    success_counter = 0
    recent_memory = []

    for sub_idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()

        step_start_raw = snapshot_env_raw(env)
        base_lang_text = val_annotations[subtask][0]
        success = original_rollout(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            record,
            rollout_video,
        )
        if success:
            row = make_state_memory_row(
                f"R_seq{seq_idx}_sub{sub_idx}",
                seq_idx,
                sub_idx,
                subtask,
                True,
                -1,
                "recent_state",
            )
            update_runtime_state_memory(recent_memory, runtime_key_state_memory, row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
            logs.append(
                {
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": True,
                    "used_reflection_retry": False,
                    "lang_text": base_lang_text,
                }
            )
            success_counter += 1
            if record:
                rollout_video.draw_outcome(True)
            continue

        reset_env_raw(env, step_start_raw)
        mismatch_type = infer_mismatch_type(subtask, False, int(cfg.ep_len), cfg.ep_len)
        next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
        runtime_key_rows = retrieve_key_memory(runtime_key_state_memory, subtask, cfg.memory_key_topk)
        historical_key_rows = retrieve_key_memory(key_memory_rows, subtask, max(0, cfg.memory_key_topk - len(runtime_key_rows)))
        key_rows = (runtime_key_rows + historical_key_rows)[: cfg.memory_key_topk]
        recent_summary = summarize_recent_memory(recent_memory, cfg.memory_recent_k)
        key_summary = summarize_key_memory(key_rows)
        important_summary = summarize_important_key_memory(key_rows)
        rule_text = RULE_MEMORY.get(subtask, RULE_MEMORY["default"])
        prompt = build_postexec_reflection_prompt(
            subtask,
            next_task,
            False,
            int(cfg.ep_len),
            mismatch_type,
            recent_summary,
            key_summary,
            important_summary,
            rule_text,
        )
        from policy_evaluation.oracle_hypothesis_rollout import parse_memory_reflection_output

        qwen = qwen_getter()
        decision = parse_memory_reflection_output(qwen.generate_reflection(prompt, max_new_tokens=192))
        reflection_row = make_reflection_row(
            seq_idx,
            sub_idx,
            subtask,
            False,
            int(cfg.ep_len),
            base_lang_text,
            decision,
            mismatch_type,
            recent_summary,
            key_summary,
            important_summary,
            rule_text,
            baseline_success=False,
            baseline_steps=int(cfg.ep_len),
        )
        retry_lang_text = build_augmented_instruction(base_lang_text, reflection_row, important_summary)
        adapter_info = {"future_adapter_enabled": bool(cfg.future_adapter_ckpt), "future_adapter_used": False}
        corrected_future = None
        adapter_bundle = adapter_getter()
        if adapter_bundle is not None and bool(adapter_bundle.get("token_level", False)):
            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

            decision_obs = env.get_obs()
            original_future = defi_future_feature(model, decision_obs, retry_lang_text).detach().to(model.device)
            original_future_np = original_future.detach().cpu().numpy().astype(np.float32)
            memory_features, memory_sims, memory_ids, memory_mask = retrieve_adapter_memory_topk(
                subtask,
                original_future_np,
                adapter_bundle["memory_arrays"],
                int(adapter_bundle["topk"]),
            )
            if memory_ids:
                qwen_weight_map = {memory_id: 1.0 / float(len(memory_ids)) for memory_id in memory_ids}
                similarity_map = {memory_id: float(memory_sims[idx]) for idx, memory_id in enumerate(memory_ids)}
                qwen_weights, sims, mask = pad_weight_inputs(
                    qwen_weight_map,
                    similarity_map,
                    memory_ids,
                    int(adapter_bundle["topk"]),
                )
                future_pooled = mean_pool_feature(original_future_np)
                device = next(model.parameters()).device
                with torch.no_grad():
                    _, memory_state, cal_aux = adapter_bundle["calibrator"](
                        torch.from_numpy(future_pooled).unsqueeze(0).to(device),
                        torch.from_numpy(memory_features).unsqueeze(0).to(device),
                        qwen_weights.unsqueeze(0).to(device),
                        sims.unsqueeze(0).to(device),
                        mask.unsqueeze(0).to(device),
                    )
                    corrected, _, adapter_aux = adapter_bundle["adapter"](
                        original_future.unsqueeze(0).to(device),
                        memory_state,
                    )
                corrected_future = corrected.squeeze(0).detach().to(model.device)
                adapter_info.update(
                    {
                        "future_adapter_used": True,
                        "adapter_memory_ids": list(memory_ids),
                        "adapter_memory_scores": [float(x) for x in memory_sims[: len(memory_ids)].tolist()],
                        "adapter_weight_mean": float(cal_aux["aggregated_memory"].detach().mean().item()),
                        "adapter_gate_mean": float(adapter_aux["gate"].detach().mean().item()),
                        "adapter_delta_norm": float(torch.norm((corrected_future - original_future).reshape(-1), p=2).item()),
                    }
                )
            else:
                adapter_info["future_adapter_skip_reason"] = "no_memory_candidates"
        elif adapter_bundle is not None:
            adapter_info["future_adapter_skip_reason"] = "adapter_checkpoint_not_token_level"
        retry_success, retry_steps = rollout_with_lang_text(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            retry_lang_text,
            corrected_future,
        )
        reflection_row["success"] = bool(retry_success)
        reflection_row["steps"] = int(retry_steps)
        reflection_row["used_reflection_retry"] = True
        reflection_row["language_augmented"] = True
        reflection_row["lang_text"] = retry_lang_text
        reflection_row.update(adapter_info)
        logs.append(reflection_row)

        state_row = make_state_memory_row(
            f"R_seq{seq_idx}_repair{sub_idx}",
            seq_idx,
            sub_idx,
            subtask,
            bool(retry_success),
            int(retry_steps),
            "recent_state",
        )
        update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)

        if record:
            rollout_video.draw_outcome(retry_success)
        if retry_success:
            success_counter += 1
            continue
        return success_counter

    return success_counter


def evaluate_policy_memory(model, env, lang_embeddings, cfg, num_procs, procs_id, checkpoint, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    rollout_video = RolloutVideo(
        logger=logger,
        empty_cache=False,
        log_to_file=True,
        save_dir=save_dir,
        resolution_scale=1,
    )

    eval_sequences = get_sequences(cfg.num_sequences)
    num_seq_per_procs = cfg.num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs * procs_id : num_seq_per_procs * (procs_id + 1)]

    key_memory_rows = load_key_memory_rows(Path(cfg.memory_key_path)) if cfg.memory_key_path else []
    runtime_key_state_memory = []
    logs = []
    qwen = None
    adapter_bundle = None

    def get_qwen():
        nonlocal qwen
        if qwen is None:
            from policy_evaluation.oracle_hypothesis_rollout import QwenReflectionEncoder

            device = next(model.parameters()).device
            qwen = QwenReflectionEncoder(
                Path(cfg.qwen_model_path),
                device,
                lora_path=Path(cfg.qwen_lora_path) if cfg.qwen_lora_path else None,
                python_bin=Path(cfg.qwen_python_bin) if cfg.qwen_python_bin else None,
            )
        return qwen

    def get_adapter():
        nonlocal adapter_bundle
        if not cfg.future_adapter_ckpt:
            return None
        if adapter_bundle is None:
            if not cfg.memory_npz_path or not cfg.calibrator_ckpt:
                raise ValueError("--future_adapter_ckpt requires --memory_npz_path and --calibrator_ckpt")
            memory_arrays = load_memory_arrays(Path(cfg.memory_npz_path))
            device = next(model.parameters()).device
            calibrator_ckpt = torch.load(cfg.calibrator_ckpt, map_location="cpu")
            calibrator = MemoryWeightCalibrator(
                future_dim=int(calibrator_ckpt["future_dim"]),
                memory_dim=int(calibrator_ckpt["memory_dim"]),
                max_memories=int(calibrator_ckpt["max_memories"]),
                hidden_dim=int(calibrator_ckpt["hidden_dim"]),
            )
            calibrator.load_state_dict(calibrator_ckpt["model_state"])
            calibrator = calibrator.to(device).eval()

            adapter_ckpt = torch.load(cfg.future_adapter_ckpt, map_location="cpu")
            adapter = MemoryStateGuidedAdapter(
                future_dim=int(adapter_ckpt["future_dim"]),
                memory_state_dim=int(adapter_ckpt["memory_state_dim"]),
                hidden_dim=int(adapter_ckpt["hidden_dim"]),
            )
            adapter.load_state_dict(adapter_ckpt["model_state"])
            adapter = adapter.to(device).eval()
            adapter_bundle = {
                "memory_arrays": memory_arrays,
                "calibrator": calibrator,
                "adapter": adapter,
                "token_level": bool(adapter_ckpt.get("token_level", False)),
                "future_token_shape": tuple(int(x) for x in adapter_ckpt.get("future_token_shape", ())),
                "topk": int(calibrator_ckpt["max_memories"]),
            }
            print(
                "[INFO] Loaded future adapter: "
                f"token_level={adapter_bundle['token_level']}, "
                f"future_token_shape={adapter_bundle['future_token_shape']}"
            )
        return adapter_bundle

    results = []
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = False
        result = evaluate_sequence_memory(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            lang_embeddings,
            val_annotations,
            cfg,
            record,
            rollout_video,
            i,
            get_qwen,
            get_adapter,
            key_memory_rows,
            runtime_key_state_memory,
            logs,
        )
        results.append(result)
        success_rates = count_success(results)
        average_rate = sum(success_rates) / len(success_rates) * 5
        description = " ".join([f"{idx + 1}/5 : {v * 100:.1f}% |" for idx, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.1f} |"
        eval_sequences.set_description(description)

    results_dict = {checkpoint: results}
    print_and_save(results_dict, cfg, log_dir=save_dir)
    if save_dir is not None:
        with (Path(save_dir) / "memory_rollout_rows.jsonl").open("w") as handle:
            for row in logs:
                handle.write(json.dumps(row) + "\n")
    return results


def main(cfg):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device

    log_wandb = cfg.log_wandb
    seed_everything(0, workers=True)
    checkpoints = get_all_checkpoints(Path(cfg.train_folder))
    lang_embeddings = None
    env = None
    results = {}

    print("train_folder", cfg.train_folder)
    print("\n" + "=" * 80)
    print(f"Found {len(checkpoints)} checkpoint(s) for evaluation:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  [{i}] {str(ckpt)}")
    print("=" * 80 + "\n")

    for checkpoint in checkpoints:
        print(device)
        print(f"\n{'*' * 80}")
        print(f"Processing checkpoint: {str(checkpoint)}")
        print(f"{'*' * 80}")
        device_id = device.index if device.type == "cuda" and device.index is not None else 0
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device_id,
            cfg=cfg,
        )

        print(f"Loading model from {str(checkpoint)}")
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        print(f"✓ Successfully loaded checkpoint from: {str(checkpoint)}")

        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["model"], strict=False)
        model.freeze()

        print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model = acc.prepare(model, device_placement=[True])
        model.process_device()
        model.eval()
        if log_wandb:
            log_dir = get_log_dir(cfg.train_folder)
            os.makedirs(log_dir / "wandb", exist_ok=True)
            evaluator = original_evaluate_policy if cfg.disable_memory_reflection else evaluate_policy_memory
            results[checkpoint] = evaluator(
                model,
                env,
                lang_embeddings,
                cfg,
                acc.num_processes,
                acc.process_index,
                checkpoint,
                save_dir=Path(log_dir),
            )
            avg_reward = torch.tensor(results[checkpoint]).float().mean().to(device)
            acc.wait_for_everyone()
            avg_reward = acc.gather_for_metrics(avg_reward).mean()
            if acc.is_main_process:
                print("average success rate ", avg_reward)

        model.to("cpu")
        model.process_device()
        del model
        del env
        del lang_embeddings
        lang_embeddings = None
        env = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--t5_model_path", type=str, default="")
    parser.add_argument("--language_goal_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    parser.add_argument("--eval_sequences_path", type=str, default="")
    parser.add_argument("--disable_memory_reflection", action="store_true")
    parser.add_argument("--memory_key_path", type=str, default="")
    parser.add_argument("--memory_recent_k", type=int, default=4)
    parser.add_argument("--memory_key_topk", type=int, default=3)
    parser.add_argument("--memory_runtime_key_max", type=int, default=5000)
    parser.add_argument("--qwen_model_path", type=str, default="")
    parser.add_argument("--qwen_lora_path", type=str, default="")
    parser.add_argument("--qwen_python_bin", type=str, default="")
    parser.add_argument("--memory_npz_path", type=str, default="")
    parser.add_argument("--calibrator_ckpt", type=str, default="")
    parser.add_argument("--future_adapter_ckpt", type=str, default="")

    args = parser.parse_args()

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
        print(f"[INFO] Overriding cfg.model.t5_model_path with args.t5_model_path: {cfg.model.t5_model_path}")
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
        print(f"[INFO] Overriding cfg.model.language_goal_path with args.language_goal_path: {cfg.model.language_goal_path}")
    cfg.root_data_dir = args.calvin_abc_dir
    if args.eval_sequences_path:
        with open_dict(cfg):
            cfg.eval_sequences = load_eval_sequences(args.eval_sequences_path)
            cfg.num_sequences = len(cfg.eval_sequences)
            cfg.disable_memory_reflection = bool(args.disable_memory_reflection)
            cfg.memory_key_path = args.memory_key_path
            cfg.memory_recent_k = int(args.memory_recent_k)
            cfg.memory_key_topk = int(args.memory_key_topk)
            cfg.memory_runtime_key_max = int(args.memory_runtime_key_max)
            cfg.qwen_model_path = args.qwen_model_path
            cfg.qwen_lora_path = args.qwen_lora_path
            cfg.qwen_python_bin = args.qwen_python_bin
            cfg.memory_npz_path = args.memory_npz_path
            cfg.calibrator_ckpt = args.calibrator_ckpt
            cfg.future_adapter_ckpt = args.future_adapter_ckpt
    main(cfg)
