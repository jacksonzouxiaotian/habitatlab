# VLN 三方法复现状态

任务来源：`/home/xiaotian/下载/VLN_Reproduction_Prompts_ZH.md`。开始日期：2026-09-16。

本轮独立于 EAGOR ObjectNav。仅使用官方 RGB + 指令策略；不接入 A*、GT Direction、Oracle Stop。尚无三方法的本地 SR/SPL，不用既有 NaVILA 4-bit 结果冒充。

## 目录与约束

- 工具和报告：`/home/xiaotian/navigation/habitat-lab/vln_reproduction/`
- 官方源码、模型、隔离环境、原始日志：`/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction/`
- 共享只读数据：`/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/`
- 当前硬件：RTX A4000 16 GiB，桌面占用约 1.3 GiB；系统内存 62 GiB。系统盘剩余约 15 GiB，数据盘约 1.4 TiB。
- 不更改现有 EAGOR、NaVILA、ROS 环境；不购买计算资源；不运行真实机器人非零运动。

## 当前进展（验收层相互独立）

| 方法 | A 官方权重真实推理 | B 正式仿真 | C 导航训练 | D Lite3 |
|---|---|---|---|---|
| NaVid | 进行中：隔离安装、获取权重 | 未开始 | 未开始 | 未开始；非零运动未授权 |
| Uni-NaVid | 未开始：已取得源码 | 未开始，须分别列 VLN/ObjectNav/EQA/Following | 未开始 | 未开始；非零运动未授权 |
| AwareVLN | 未开始：已取得源码 | 未开始 | 未开始 | 未开始；非零运动未授权 |

NaVid 当前公开权重基于后续 Uni-NaVid 训练方案；原 RSS 2024 与当前公开版本分开记录。Uni-NaVid 官方仅公开训练子集；AwareVLN 必须使用 `awarevln/` 最终权重，不使用同仓库初始化 `navila-llama3-8b-8f/` 替代。

## 执行顺序

NaVid → Uni-NaVid → AwareVLN；真实推理通过后依次进行固定 1/10/100/full 的调试/正式评估。随后训练数据与训练 smoke、跨方法原生/受控比较，以及不驱动实机的迁移测试。下载成功、接口单测成功不等于 A/B/C/D 通过。

本文件将在实测阶段更新，所有失败日志保留，不把未运行写为零分。
