---
name: btc-offline-replay
description: 用有序的请求日志复现 BTC prefill KVCache 命中率。用于百炼日志转换、切分、元数据回放、预热与检查点续跑，以及 FULL 和联合命中率的同区间对比。
---

先确认日志的 base tokens per block、hash 是否已经包含前缀、缓存实例归属和请求顺序。用 scripts/prepare.py 将一份有序输入转换为可校验的分片，多种配置复用同一份数据。

## 百炼日志样本

JSONL 每行是一条请求。下面是一条合成的 prefill 请求，假设数据源每 256 个 token 记录一个原始块 hash。

```json
{"request_id":"example-001","block_tokens":256,"timestamp":1793000000.125,"input_length":513,"input_block_hash_ids":["0000000000000065","00000000000000ca"]}
```

- `request_id` 是请求标识字符串。`timestamp` 是请求到达的 Unix 秒数，支持小数；样本对应 `1793000000125000000` 纳秒。
- `input_length` 是输入 token 总数。513 个 token 包含两个完整块和 1 个尾部 token，因此这条记录提供两个完整块的 hash。尾部 token 计入命中率分母。
- `input_block_hash_ids` 按输入顺序排列，每个元素是 16 位十六进制字符串。转换器优先读取日志声明，或从相同块 key 的不同前驱识别原始 hash。没有充分证据时需指定模式。本例转换参数为 `--format bailian --base-tokens 256 --hash-mode raw`。若数据源已记录前缀 hash，选择 `--hash-mode prefix`。两种 hash 的文本格式相同，需从数据源确认语义。

## 回放与比较

让使用者提供待评估 BTC 的 metadata replay 可执行文件、对应源码版本和配置。脚本通过进程接口调用后端，候选选择、CP 映射、联合匹配、insert、水位和 force_drop 由被测后端执行。若只有源码，先核对目标版本的后端接口；版本接入约定见 [使用说明](references/usage.md)。

用 scripts/replay.py 设置预热和正式请求数。分阶段测试时保持最终 --measure 固定，通过 --stop 设置本次累计终点。续跑读取上一阶段验证通过的检查点。

用 scripts/backend.py locate 检查指定检出的 BTC 规则；选定已接入元数据回放的可执行目标后，build 子命令用 Bazel cquery 记录依赖并定向编译。

用 scripts/analyze.py --html report.html 对照多组结果，第一个目录作为达标率基准。报告统计区间、输入 token 总数、FULL 与联合命中率，并标明配置与后端版本。逐请求文件仅包含本次执行新增的记录。

[使用说明、命令及输入输出契约](references/usage.md)

默认每个转换文件保存 1,000,000 条请求；最后一个文件保留余数。预处理输出显示粒度、hash 类型与判定来源。
