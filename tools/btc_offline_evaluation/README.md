# BTC 离线命中率评估工具旧版归档

本目录冻结 2026-11-06 千万请求实验使用的 CPU 元数据回放实现，并在 2026-11-09 整理归档。基线为 `feature/dsv41-dev-main-202161013` 的 `a3fdca664089a44769b1d8c220ea0141fd55731f`。

## 内容

- `legacy/source/` 保存本地离线适配源码的完整文件副本，逐文件摘要见 `archive-manifest.json`。
- `legacy/source.patch` 包含这些文件相对基线的修改和新增，可在基线 worktree 根目录应用。
- `scripts/` 保存参数化输入打包、范围识别和历史验收脚本。
- `configs/` 保存全保留 1:1 配额，以及 QPS=1000 的 last_n 分池数值。
- `legacy/operations/` 保存历史构建、转换和部署脚本，机器及账号替换为占位符。运行前检查路径、输入清单、镜像和授权请求范围。
- `legacy/input-support/` 保存旧理论对照的输入辅助头文件。旧日志前缀转换另需 LiteHit 的 `request_preprocess.h/.cc` 及对应仓库依赖。
- `docs/legacy-design.md` 描述原生 BTC 调用、线程、hash 和 last_n 语义。
- `docs/known-limitations.md` 记录复现、恢复、构建依赖和入口兼容性限制。
- `docs/next-design.md` 保存后续讨论，仅表示待实现设计。
- `skills/btc-offline-evaluation/SKILL.md` 提供给后续 agent 使用的仓库内操作说明。

## 验证归档

```bash
python3 tools/btc_offline_evaluation/scripts/check_archive.py
```

## 在基线应用旧版源码

先在独立 worktree 检出上述基线，从归档提交拷贝本目录，然后执行：

```bash
git apply --check tools/btc_offline_evaluation/legacy/source.patch
git apply tools/btc_offline_evaluation/legacy/source.patch
```

历史千万请求入口为：

```bash
bazelisk build --config=cuda13 --jobs=16 \
  //rtp_llm/cpp/cache/block_tree_cache/benchmark:resident_metadata_replay
```

编译依赖仍包含 Torch/CUDA 和传输引擎。历史运行环境为 Linux x86_64、Bazel 6.4.0、CUDA 13 构建配置。运行时使用无 GPU 设备的 runc 容器、元数据资源池和模拟传输完成后端。

`resident_metadata_replay` 接收 `PARAMS_JSON INPUT_DIRECTORY OUTPUT_DIRECTORY`。输出目录须预先写入 `authorized-through`，内容为允许处理的请求条数。输入目录包含 `000000.btcbin` 开始的连续分块。原始入口固定 2026-10-27 UTC 日界、65,740,206 条整日请求及每块 100,000 条，详见格式说明。

## 已完成的历史结果

同一批前 10,000,000 条请求，分母为 311,133,819,819 输入 tokens。下表来源于 2026-11-06 会话内读取的完成记录，本次归档未重新运行实验，原始远端完成文件和二进制未重新取回。

| SWA 策略 | Device/Host | Disk | 联合 token 命中率 | 回放耗时 |
|---|---|---|---:|---:|
| 全保留 | LRU | FIFO | 84.372070% | 6268.433 s |
| 全保留 | LRU | LRU | 84.368124% | 6133.417 s |
| last_n=1 | LRU | FIFO | 2.989034% | 2687.074 s |
| last_n=5 | LRU | FIFO | 83.368085% | 4434.633 s |
| last_n=10 | LRU | FIFO | 83.893603% | 4783.286 s |
| last_n=15 | LRU | FIFO | 84.146598% | 4961.845 s |
| last_n=20 | LRU | FIFO | 84.259777% | 5147.647 s |

命中率为原生 BTC 在 CPU 元数据环境下的请求串行回放结果。未命中请求省略模型 prefill 计算，耗时仅描述离线评估器。历史运行各组请求失败和传输失败计数均为零。
