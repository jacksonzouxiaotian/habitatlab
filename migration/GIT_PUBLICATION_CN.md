# 代码上传范围与验收（2026-09-18）

目标：`git@github.com:jacksonzouxiaotian/habitatlab.git`。
研究分支：`narrow-passage-rl-memory`。不改 main，不 force push，不重写已有历史。

本次整理当前工作区中的研究代码、配置、测试、中文说明及必要的小型文本报告：窄通道/DEGNav、PointNav、EAGOR、VLN 与 ROS2 源码。保留本地已有修改，不重构导航算法。外部 VLN 源码以固定 commit + 差异补丁保留，并提供默认只打印计划的恢复脚本。

不新增上传场景、模型、环境、原始逐步日志、视频、EAGOR 的原始 episode manifest；它们仍按迁移手册从旧机复制。未纳入提交的结果和图表保持原样留在工作区。旧 Git 历史中原本已有的结果文件不在本轮清理范围。

已完成的本地验证：

- EAGOR 76 项测试通过。
- VLN 契约/指标/动作解析测试 25 项通过；不代表模型推理已复现。
- NaVILA、NaVid 两个补丁对 manifest 指定的原始 commit 校验可应用，使用临时 Git index，不改外部工作区。
- 新机辅助脚本通过 Bash 语法和 plan 模式验证；没有在新机实际解包。
- 暂存内容扫描常见 token、私钥和带凭据 URL，未发现匹配；没有新增大于 2 MiB 的文件。这是启发式检查，不是完整安全审计。
- `git diff --check` 有已有 Markdown 行尾换行空格、文件末尾空行及补丁上下文空格提示；没有为了消除提示改写原始算法或破坏补丁。

新机获取代码：

```bash
git clone --branch narrow-passage-rl-memory \
  git@github.com:jacksonzouxiaotian/habitatlab.git
cd habitatlab
```

接着阅读 `migration/README_MIGRATION_CN.md` 和 `vln_reproduction/README_REPRODUCE_CN.md`；只有 Git clone 不能运行依赖外盘任务、模型和原生环境的完整实验。正式结果使用原任务集合和单独新输出目录。
