---
name: btc-offline-replay
description: 用有序的请求日志复现 BTC prefill KVCache 命中率。用于百炼日志转换、切分、元数据回放、预热与检查点续跑，以及 FULL 和联合命中率的同区间对比。
---

先确认日志的 base tokens per block、hash 是否已经包含前缀、缓存实例归属和请求顺序。用 scripts/prepare.py 将一份有序输入转换为可校验的分片，多种配置复用同一份数据。

让使用者提供待评估 BTC 的 metadata replay 可执行文件、对应源码版本和配置。脚本通过进程接口调用后端，候选选择、CP 映射、联合匹配、insert、水位和 force_drop 由被测后端执行。若只有源码，先核对目标版本的后端接口；版本接入约定见 [使用说明](references/usage.md)。

用 scripts/replay.py 设置预热和正式请求数。分阶段测试时保持最终 --measure 固定，通过 --stop 设置本次累计终点。续跑读取上一阶段验证通过的检查点。

用 scripts/analyze.py 对照多组结果，第一个目录作为达标率基准。报告统计区间、输入 token 总数、FULL 与联合命中率，并标明配置与后端版本。逐请求文件仅包含本次执行新增的记录。

[使用说明、命令及输入输出契约](references/usage.md)
