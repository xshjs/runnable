# RLBench Suffix Memory Repro Notes

This document records the provenance, label construction, alignment assumptions,
and audit examples for the RLBench suffix transition memory used by the
factored belief/action transition experiments.

## Source Memories

There are two main artifact types.

Raw rollout rows:

```text
suffix_transition_rows.jsonl
suffix_transition/*.npz
```

Example source:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition_rows.jsonl
```

Compact training memory:

```text
bridge_suffix_td_memory_*.npz
```

Example source:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz
```

Recorded local commits at the time of inspection:

```text
BridgeVLA: 0690b6ba71459a76815381e96797d0396c51a6d9
DeFi:      dea4b92fb7b24c482e070d02464c9610b7c49ed3
```

## Keep And Delta Label Source

The labels are generated from model rollout, not directly from demonstration
oracle actions.

The memory construction script is available in this repo at:

```text
Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py
```

The original local BridgeVLA working copy path was:

```text
/mnt/workspace/manipulation/BridgeVLA/finetune/RLBench/build_bridge_suffix_transition_memory.py
```

Rules:

```text
success row -> keep, delta = 0
failure row -> find a successful row from the same task, matched by nearest step, then create a patch delta
```

So keep rows come from successful rollouts. Non-zero delta rows come from a
failure rollout paired with a successful rollout proxy.

## Actual Scripts And Commands

Collection scripts in the local BridgeVLA checkout:

```text
BridgeVLA/finetune/RLBench/eval.py
BridgeVLA/finetune/RLBench/eval.sh
```

This repo also contains a wrapper with the exact environment and parameters used
for the current BridgeVLA suffix TD evaluation:

```text
Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh
```

Memory construction script:

```text
Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py
```

Example memory build command:

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/workspace/manipulation/DeFi/.venv/bin/python \
  Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py \
  --rows-jsonl /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition_rows.jsonl \
  --output-npz /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz \
  --summary-json /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.summary.json \
  --max-per-task-mode 80 \
  --delta-clip 0.2
```

## Action Correction Label

For a failure row, the action correction target is:

```text
target_action = matched_success.action_remain_before
action_delta  = target_action - failure.action_remain_before
```

Equivalently:

```text
Delta A* = A_remain_success_matched - A_remain_failure
```

For a success row:

```text
Delta A* = 0
mode_label = keep
```

For a failure row:

```text
Delta A* is non-zero
mode_label = patch
```

## Meaning Of `action_chunk`

In the compact memory:

```text
action_chunk = action_remain_before
```

It is not the full episode action sequence. It is the short-horizon action
window remaining after the current executed action:

```text
A_remain = [a_{t+1}, ..., a_{t+K}]
```

For current RLBench artifacts the shape is:

```text
[10, 9]
```

That means 10 steps, each with a 9-dimensional BridgeVLA/RLBench action.

## Target Action Source And Assumption

The target action comes from a successful rollout row from the same task:

```text
matched_success.action_remain_before
```

It is a proxy supervision target because it satisfies:

```text
same task
nearby rollout step
successful future outcome
```

The assumption is that this successful suffix is closer to the desired local
correction than the failed suffix. This is not a strict oracle label and should
be described as success-nearest rollout proxy supervision.

## Success / Failure Filtering

The script filters by success state:

```text
success=True  -> keep
success=False -> patch, matched to a successful row
```

If no successful rows exist, the builder fails with:

```text
No successful suffix rows found
```

## Temporal Alignment

The current alignment is weak but explicit:

```text
same task + nearest step
```

Implementation:

```python
min(candidates, key=lambda r: abs(r["step"] - failure_step))
```

This is not a strict state-level nearest-neighbor match. It is task-level and
step-level proxy alignment.

## Time Correspondence Example

For a failed sample at rollout step `t=2`:

```text
o_t              = state_before
a_t              = executed_action
o_{t+1}          = state_after
h_t              = h_future_before + h_action_before
h_{t+1}          = h_future_after + h_action_after
A_remain^t       = action_remain_before
```

If this row failed, a successful row from the same task and nearest step is used:

```text
A_remain_correct = success_row.action_remain_before
Delta A* = A_remain_correct - A_remain_failure
A_remain_after = A_remain_failure + Delta A*
```

## Padding And Last Chunk Step

The builder pads or crops the matched target action to the source suffix shape:

```python
_pad_or_crop(target_data["action_remain_before"], action_remain.shape)
```

Short targets are zero-padded. Long targets are cropped.

Current compact fields:

```text
remaining_steps = action_remain.shape[0]
full_chunk_len  = action_remain.shape[0]
```

So many samples report a fixed value of 10. In the current memory, this is best
interpreted as the window length, not guaranteed true valid remaining control
steps.

## Remaining Length / Mask Caveat

The raw feature files contain masks:

```text
action_remain_before_mask
action_remain_after_mask
next_action_window_mask
```

The current compact memory builder does not preserve these masks or use them to
compute `remaining_steps`. A stricter version should:

```text
remaining_steps = sum(action_remain_before_mask)
compute loss only on valid mask positions
force padded delta positions to 0
```

This is an important caveat when auditing samples near the end of an episode or
near the end of an action window.

## Non-Zero Delta Pair Example

The following inspected memory contains non-zero delta rows:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz
```

Memory summary:

```text
num_rows: 2368
mode_counts: keep=1403, patch=965
delta_norm percentiles: min=0, p50=0, p90=1.3963, p99=1.6190, max=1.7132
```

Example:

```text
idx=2
task=stack_blocks
mode=patch
success=0
cut_idx=2
delta_norm=1.3593
source=/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition/stack_blocks_ep0019_att00_step002.npz
```

Executed action:

```text
[0.4594, -0.1769, 0.9700, -0.9537, 0.3006, -0.0000, 0.0000, 0.0000]
```

First two rows of `A_remain_before`:

```text
[[ 0.2641,  0.0452, 0.9031, -0.0001, 1.0000, -0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.4014, -0.3197, 0.8729, -0.8661, 0.5000, -0.0000, 0.0000, 1.0000, 0.0000]]
```

First two rows of matched target action:

```text
[[ 0.2474, -0.0140, 0.9041,  0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.2463, -0.1197, 0.8540, -0.6661, 0.7000, -0.0000, 0.0001, 1.0000, 0.2000]]
```

First two rows of `Delta A`:

```text
[[-0.0167, -0.0592,  0.0011, 0.0001, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000],
 [-0.1551,  0.2000, -0.0190, 0.2000, 0.2000, 0.0000,  0.0000, 0.0000, 0.2000]]
```

## Audit Command

Use this command to inspect shape, mode counts, and finite values:

```bash
cd /mnt/workspace/manipulation/BridgeVLA

/mnt/workspace/manipulation/DeFi/.venv/bin/python - <<'PY'
import numpy as np

p = "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz"
with np.load(p, allow_pickle=True) as d:
    for k in [
        "state_start",
        "executed_action",
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "action_delta_target",
        "mode_label",
    ]:
        finite = np.isfinite(d[k]).all() if d[k].dtype.kind in "fiu" else "n/a"
        print(k, d[k].shape, d[k].dtype, "finite=", finite)
    print("mode counts:", dict(zip(*np.unique(d["mode_label"], return_counts=True))))
    delta = d["action_delta_target"]
    norms = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    print("delta percentiles:", np.percentile(norms, [0, 50, 90, 99, 100]))
PY
```

## Current Limitation

The current non-zero correction labels are useful for debugging and controlled
ablation, but they are proxy labels:

```text
failure suffix -> matched successful suffix
```

They are not guaranteed to be the unique optimal local action correction. For a
stronger version, collect paired interventions where the same initial state is
run once with baseline and once with successful suffix patch, then use the
successful retry suffix as the target.
