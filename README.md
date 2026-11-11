# Practical DCU training optimization notes

This note collects a small set of training changes measured on a single DCU BW card. It focuses on changes that retained the same examples, objectives, and optimizer update, and records where a result came from a workload replay rather than a live training run.

The figures are configuration-specific measurements, not guarantees for other models, sequence lengths, DTK releases, or cards. No model weights, datasets, checkpoints, account details, hostnames, job IDs, or local paths are included.

## Results at a glance

| Change | Result | Evidence boundary |
| --- | --- | --- |
| Turn off activation recomputation for the tested training step | 1.61× steady updates/s; 1.58× for a 100-update segment including recovery, save, and evaluation | Same parent state and update contents; peak allocated memory rose by about 4.5 GiB. A short fixed development set showed no change. |
| Buffer intermediate tensors in a recurrent Gated DeltaNet (GDN) chunk implementation | 1.23× over the preceding optimized configuration; 1.59× combined over the measured baseline in a replay | Full-size architecture and real sequence shapes, but random weights and synthetic token/pixel values. Not a live-training quality or throughput claim. |
| Wider length grouping within an optimizer update | 1.16× hot-window throughput in one replay | First traversal was slower than the warmed baseline. Re-measure when shapes change often. |

## 1. Disable activation recomputation only when memory allows

Activation recomputation saves memory by calculating some forward activations again during backward. In one matched run, disabling it increased steady update throughput by 1.61×. It used about 4.5 GiB more allocated memory. A separate 100-update segment, including restore, checkpoint save, and a fixed development evaluation, ran at 1.58× the baseline rate.

A safe adoption sequence is:

1. Restore the same model, optimizer, random-number-generator, and data-cursor state for each comparison.
2. Run short parity checks on loss, gradients, clipping, optimizer updates, and optimizer moments.
3. Measure peak allocated and reserved memory at the longest relevant sequence shape.
4. Compare paired windows with the same samples and warmup, then run a short segment that includes saving and evaluation.
5. Keep recomputation enabled if the memory headroom is not sufficient for the target workload.

## 2. Reduce recurrent GDN chunk copy overhead

The tested Python/autograd chunk path repeatedly sliced per-position tensors and copied intermediate results into preallocated outputs. The buffered variant unpacked each input sequence once, kept intermediate chunk values in local lists, and stacked each output once. It preserved recurrence order, equations, masks, scaling, dtypes, and output state.

The isolated forward/backward microbenchmark at a 3,072-position chunk decreased from 201 ms to 105 ms (about 1.91×). In a separate replay using the real sequence shapes, this change added 1.23× over an already optimized setting; the combined configuration measured 1.59× over its replay baseline. Maximum observed loss and gradient-norm relative differences were about 0.05% and 0.11% respectively in the replay. This is promising evidence for a targeted kernel or implementation change, but it still needs a frozen-state check on the intended model and data before training use.

Keep numerically sensitive recurrent state and accumulation in FP32 unless a separate parity study supports changing them. Lower precision in more temporary tensors delivered less than 2% local speed gain in the tested case and caused materially larger input-gradient differences, so that candidate was rejected.

## 3. Tune length grouping without changing the training objective

Grouping similarly sized examples can reduce padding and improve batch utilization. In one replay, widening the allowed length ratio inside each existing optimizer update improved warmed throughput by 1.16×, while the first traversal was slower than the warmed baseline. This suggests the setting may help stable, repeated shapes and may lose when compilation or shape setup happens frequently.

Do not pool examples across optimizer updates just to fill batches: that changes update frequency, optimizer history, and the weighting of loss terms. Track valid target positions/s, padded inputs, update time, first-pass time, and peak memory together. Fewer shapes or fewer padding tokens alone do not prove a faster run.

## Changes that did not pass

- Larger microbatches failed the pre-set numerical parity limits; they were not promoted.
- AdamW foreach failed the same update-parity limits and was slower in the short probe.
- A newer fused GDN implementation could not complete forward/backward compilation on the tested software stack. No speed claim was made.
- A wider hidden width aligned to 64 did not improve warmed throughput in the measured configuration.

These results are useful as examples of keeping failed candidates visible instead of loosening acceptance limits after seeing the output.

## Benchmark protocol worth reusing

- Start every candidate from the same complete model and optimizer state, data order, cursor, and random state.
- Lock tolerances using repeated baseline runs before evaluating a candidate.
- Check loss, representative gradients, clipping, parameter updates, and optimizer moments, not only forward outputs.
- Pair baseline and candidate windows in alternating order; report warmup, first traversal, steady throughput, and peak memory separately.
- Use the actual sequence-length and modality distribution. If the workload replay uses synthetic values or random weights, label the result as replay evidence.
- Measure a short segment with recovery, checkpoint writing, and evaluation in addition to steady-state updates.
- Report untested hardware, sequence lengths, and compiler/runtime combinations as untested.

## Community material we reviewed

- The [光合社区 training Cookbook](https://developer.sourcefind.cn/) and the course *DCU Heterogeneous Programming and Optimization — Debugging and Optimization* are useful starting points for platform-specific examples and performance tooling.
- The community's [Axolotl Qwen3-4B PyTorch recipe](https://developer.sourcefind.cn/codes/modelzoo/axolotl-Qwen3-4B_pytorch) shows separate Distillation, SFT, and RL stages and points to FSDP/QLoRA configurations. Its environment instructions are tied to an older software stack; reuse the training-stage structure only after checking current compatibility. The recipe warns that running generation and RL training together can exceed memory at larger batch sizes.
- A [community BW1000 fine-tuning question](https://forum.sourcefind.cn/) reports a loss plateau but does not provide a verified fix. It is a reminder to inspect the dataset, learning-rate schedule, resume state, and evaluation setup before attributing a plateau to hardware throughput.
- Recent community posts on multi-card long-context serving report useful end-to-end benchmark practice, but those are inference results and their kernels or speedups should not be transferred to training without measurement.

## Scope

Measurements were taken on a single DCU BW card with a platform-provided PyTorch/HIP stack. Software versions and performance vary by platform image. The replay figures preserve the tested model shape and sequence geometry but do not use production weights or private training examples. This note contains no instructions for accessing a particular compute account or private storage.
