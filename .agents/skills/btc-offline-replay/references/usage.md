# BTC prefill 离线命中率评估

从检出仓库根目录运行以下命令。需要 Python 3.8+；读取 .zst 需要 zstd 命令。Codex 可以直接发现仓库中的 .agents/skills/btc-offline-replay；也可把整个目录复制到自己的技能目录使用。

## 1. 将请求转换成有序的前缀 key

```sh
S=.agents/skills/btc-offline-replay/scripts
python3 "$S/prepare.py" part01.jsonl.zst part02.jsonl.zst \
  --format bailian --hash-mode raw --base-tokens 256 --domain shared --out trace
```

百炼输入每行含 request_id（字符串）、timestamp（Unix 秒）、input_length（token 数）、input_block_hash_ids（每项 16 位十六进制）。raw 使用已有回放工具的 64 位前缀组合算法；若日志已记录前缀 hash，选 prefix。数据源使用其他原始 hash 算法时，先按数据源规则转成前缀 key。每行只保留完整块，命中率分母保留整个输入长度。多个文件按参数顺序读取，同时间戳保留文件内外原顺序，遇到时间倒序会报出行号。具有不同实例或命名空间的流分别准备。

其他来源可输出 normalized JSONL。下面是单行示例。使用 --format normalized，省略 --hash-mode。

```json
{"request_id":"r1","timestamp_ns":1000000000,"input_len":513,"block_tokens":256,"key_form":"prefix","keys":[101,202],"instance_id":"shared"}
```

输出采用 BTCPFX01。默认每百万条一个原生 .btcbin，以实际记录数定位。支持 header-driven v2 的后端直接读取；旧版 256-token 后端按需生成临时 10 万条兼容视图。manifest.json 记录每片请求数、输入 token 总数和 SHA-256。块序列保持全局基础粒度。业务侧已做 CP 抽样的输入需要与后端另行对齐，重复抽样会改变结果。

## 2. 使用待测版本的原生后端

后端是调用者提供的 CPU 元数据回放程序。已有的 evaluation_replay 可直接通过下列接口接入；新源码版本可提供相同接口的适配程序。仅检出上游 main 或 DSv4.1 不会自动得到此可执行文件。准备二进制时随它分发所需动态库、构建说明及对应源码提交。确认回放保留 BTC 的真实缓存决策，在元数据层处理分配和迁移，避免分配 KV payload 或启动 GPU 计算。

```sh
python3 "$S/replay.py" --backend /path/to/evaluation_replay --revision "$BTC_COMMIT" \
  --config config.json --data trace --warmup 1000000 --measure 10000000 \
  --stop 1000000 --out run-1m
python3 "$S/replay.py" --backend /path/to/evaluation_replay --revision "$BTC_COMMIT" \
  --config config.json --data trace --warmup 1000000 --measure 10000000 \
  --resume run-1m --out run-10m
```

配置 JSON 原样交给后端。容量单位、拓扑及默认值以该二进制为准。例如实验用 DSv4.1 后端读取 tiers.device、tiers.host、tiers.disk 的 full 和 swa 单 rank 逻辑块数。该后端固定 256 基础粒度和 CP=4；工具本身支持其他基础粒度。切换算法、粒度、配置、二进制或输入后应从预热重新开始。

可执行程序的参数契约为 CONFIG CHUNK_DIR NEW_OUTPUT --requests 总终点 --warmup 预热数 --stop-after 本阶段总终点 --identity IDENTITY_JSON，续跑另加 --resume CHECKPOINT_DIR。退出码 0 表示正常完成。后端负责串行处理、完成本请求的缓存状态更新后再开始下一条，并保存 checkpoint/、summary.json、match-lengths.u64。检查点应含 BTC 树、各层驻留和队列顺序、策略状态、请求游标及累计指标，并由后端检查配置和身份。没有实现恢复的后端仅做一次性回放。

预热请求正常更新缓存。summary.json 的 total.requests、input_tokens、full_tokens.sum、joint_tokens.sum 仅累计正式请求，单位分别为请求、输入 token、FULL 命中 token、联合命中 token。match-lengths.u64 每个本阶段新增请求写两个小端 uint64（FULL、联合），续跑后的 summary 仍为累计值。后端可附带 timeline、evictions 和阶段计时；各版本缺失的指标应标记为未采集。

## 3. 对比与复核

```sh
python3 "$S/analyze.py" baseline/run-1m candidate/run-1m
```

FULL 命中率 = FULL 命中 token 总数 / 全部正式输入 token 总数；联合命中率使用联合命中 token 总数作分子。达标率取本组联合命中率除以首组联合命中率，输出均为 0–1 比值，零基准输出 null。离线结果反映给定请求流和后端策略，线上并发、超时、传输延迟与 decode 另行评估。淘汰计数保留后端标签，按层和触发类型解读，区分调度与实际释放。

通过校验的阶段带有 SUCCESS 标记。保留数据清单、配置、二进制身份和完整输出目录，复现时使用相同输入顺序。准备工具发现输入损坏即退出；回放失败时保留诊断输出，重试使用新目录。

接入兼容 DSv4.1 测试配置的原生后端后，可运行 `python3 "$S/smoke.py" /path/to/evaluation_replay`。该用例使用合成请求，比较一次性执行与断点续跑的逐请求结果。

需要定位两种策略的分歧时，加上 `--cases 10`。工具对齐本阶段逐请求 FULL / 联合长度，给出变长、变短的请求数和损失最大的原始请求序号；逐 key 淘汰原因需要对应后端的事件追踪。

## 日志字段与自动判定

从日志字段 block_tokens 读取粒度；缺字段时，百炼输入采样长度与 hash 数求唯一粒度。--base-tokens 只能核对，不能覆盖或替代缺失证据。

百炼输入省略 --hash-mode 时先读取 key_form。没有声明时，检查首个文件最多一万条请求、保存最多 65,536 个不同块的前驱：若同一 key 出现在不同前驱之后（包括起点），在同一命名空间、相同算法且无碰撞的假设下可判定为 raw。无法区分时要求提供来源声明。手工指定与记录声明冲突会报错。混合来源应分开转换；这项抽样不能认证整个数据集的生产方式。

## 定位版本与构建

在完整的待测 Git 检出运行下列命令。locate 展示 HEAD 和 block_tree_cache 路径下 BUILD 中显式命名的规则。宏生成或改名、移动的包可以通过 Bazel query 定位，再显式传目标。编译命令接受一个目标，实际依赖闭包由 Bazel 按相同配置解析。示例标签仅适用于含该离线适配器的检出。

```sh
python3 "$S/backend.py" locate --repo "$RTP_REPO"
python3 "$S/backend.py" build --repo "$RTP_REPO" \
  --target //rtp_llm/cpp/cache/block_tree_cache/benchmark:evaluation_replay \
  --option=--config=cuda13 > build-receipt.json
```

输出包含提交、工作区状态、目标、构建选项、依赖和产物。缺少回放目标时需要接入该版本的适配器。元数据适配器应显式注入 transfer completion 和逻辑块池，保留引用、迁移、回滚及淘汰状态机。工具无法从任意源码自动推断哪些分配或 copy 可以删掉。每个新适配器须核验实际 payload 分配/搬移计数及失败路径。

## 网页报告

```sh
python3 "$S/analyze.py" baseline/run-1m candidate/run-1m --cases 10 --html report.html
```

HTML 可离线打开，包含预热与正式区间、各组 FULL/联合命中率、达标率和回收计数器。--cases 会增加本阶段变长、变短和损失最大的请求。

默认每个转换文件保存 1,000,000 条请求；最后一个文件保留余数。预处理输出显示粒度、hash 类型与判定来源。
