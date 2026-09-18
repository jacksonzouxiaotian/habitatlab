# DEGNav current architecture figure caption

## Paper caption (English)

**Current DEGNav strict feasibility architecture.** Solid arrows denote paths
that affect the frozen strict controller, whereas dashed arrows denote audit,
calibration, or legacy-only paths. Current depth/geometry and execution signals
feed structural and pose-conditioned margin estimates; their engineering
uncertainties form interval bounds for Commit, Explore, and structural Reject,
while contact, stuckness, or a failed prior commitment invokes Recover. A
morphology-aware cross-episode memory retrieves only validated positive or
negative geometry evidence and applies a bounded logit correction, which can
downgrade Commit to Explore or trigger Reject under repeated uncontested
negative support. The recursive Gaussian margin posterior contributes an
information-gain signal that bounds active sensing but does not replace the
selector-facing engineering uncertainty. The outcome-updated discrete
posterior over minimum navigable width and its upper quantile remain a
calibration/legacy branch; in the reported strict evaluation,
`W_req_cons` is instead the yaw-aware OBB projection from the shared robot
morphology.

## 中文说明

实线表示当前正式 strict evaluation 中真正影响决策或控制的路径；虚线表示
审计、校准或 legacy-only 路径。Geometry-guided memory 通过有界 logit 修正
影响 Commit/Explore，并在重复、无冲突的强负几何证据下触发 Reject。递归
Gaussian margin posterior 的方差仅用于 information gain 和有限主动扫描退出，
不会替换 selector 使用的工程型 uncertainty。Outcome-based posterior 的
`Q0.75` 分位数目前没有接入 strict `W_req_cons`；当前 `W_req_cons` 来自统一
RobotMorphology 的 yaw-aware OBB 投影。

## Reproduction

```bash
python examples/narrow_passage_rl/plot_current_degnav_architecture.py
```
