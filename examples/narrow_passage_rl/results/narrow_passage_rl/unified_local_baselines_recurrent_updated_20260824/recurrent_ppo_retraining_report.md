# Recurrent PPO 续训与诊断报告

## 结论

更新后的 `Recurrent PPO (Direct-init + PPO fine-tune)` 在完全配对的 Procedural-v2 held-out 协议上达到 **85.0±0.9%** 成功率，超过 50% 门槛。严格成功率同为 85.0±0.9%，碰撞率为 15.0±0.9%，超时率为 0%。

| 方法 | 成功率 | 碰撞率 | 超时/卡住率 | 平均步数 |
|:---|---:|---:|---:|---:|
| Geometry rule | 23.3±1.9% | 70.3±2.2% | 6.3±0.5% | 60.0±2.6 |
| Direct-control PPO | 84.3±0.5% | 15.7±0.5% | 0.0±0.0% | 68.5±0.1 |
| Recurrent PPO（原始 from-scratch） | 0.1±0.1% | 20.3±0.3% | 79.7±0.4% | 323.8±1.1 |
| Recurrent PPO（Direct-init + PPO fine-tune） | **85.0±0.9%** | **15.0±0.9%** | **0.0±0.0%** | **68.3±0.2** |
| DEGNAV + geometry-guided memory | 83.1±0.2% | 0.0±0.0% | 16.9±0.2% | 188.5±1.7 |

## 原始 Recurrent PPO 为什么不如 Direct PPO

1. 当前 19 维观测已经近似 Markov，Direct PPO 可以直接学习 `observation -> velocity`。默认 Recurrent PPO 则在 actor 和 critic 各放置一个 256 维 LSTM，策略参数为 608,709，约为 Direct PPO 11,077 个参数的 55 倍，但没有获得相应的部分可观测性收益。
2. 原始 Recurrent PPO 的 0.5M–3M 训练回报一直为负，最终训练 episode 长度上升到约 205；正式 held-out 评测中 79.7% episode 超时/卡住。这说明它收敛到了低推进动作的局部最优，而不是只发生了最终 checkpoint 退化。
3. 单纯缩小为 64 维 actor LSTM、前馈 critic 并提高到 10 epochs 仍未解决信用分配问题：0.5M 的 100-episode 门槛评测为 0% 成功、24% 碰撞、76% 超时。因此不能把网络缩小后的训练回报波动当成有效成功。

## 达到门槛的训练过程

- 架构：64 维单层 actor LSTM，前馈 critic，共 39,877 个参数。
- 初始化：用 Direct PPO 在 300 个与正式评测隔离的训练 episode（seed 1701）上产生动作标签；教师在这组数据上的成功率为 83.0%。
- 蒸馏：长度 32 的序列监督，动作均值 MSE 从 1.848 降至约 0.002。
- 续训：在 fair reward 下继续 106,496 次 RecurrentPPO on-policy 交互，学习率 `1e-5`、3 epochs、8 个子进程环境；累计计数为 606,496。
- 门槛评测：seed 42 的 100 个 held-out episode 为 90% 成功；随后完整评测 seed 42/43/44、每个 500 episode，最终为 85.0±0.9%。

## 解释边界

更新模型使用了 Direct PPO 的行为初始化，因此它不是与 Direct PPO 完全独立的 from-scratch baseline。它证明的是“Recurrent PPO 模块在正确初始化和保守微调后可以稳定工作并超过 50%”，不能单独用来证明“循环记忆本身优于 Direct PPO”。表格保留原始 from-scratch Recurrent PPO 行，避免隐藏这一差异。

所有五种方法使用相同的 1,500 个有序场景；总计 7,500 条 episode 记录已通过场景种子、通道类型、可通行标签、宽度和唯一 episode ID 的逐项配对检查。
