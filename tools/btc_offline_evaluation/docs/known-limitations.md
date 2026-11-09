# 归档限制与证据边界

1. legacy/source 是 2026-11-06 本地源码快照，archive-manifest.json 可验证逐文件内容。本地原 worktree 的三方合并 index 曾显示未合并，本归档保存的是已移除冲突标记的工作文件内容。源文件逐字节保留，未为归档做运行逻辑重构。
2. 历史 native fixture 和 JSONL 验收来自更早的 08cf512139 基线。迁移到 a3fdca664 后已运行的入口是 resident_metadata_replay；旧 MetadataReplayMain.cc、MetadataReplayTest.cc、verify_round.py 中部分 linear_step 及 tail 默认来源描述沿用旧接口。它们作为历史代码保存，本次未重新构建验证。
3. 整个项目的编译目标仍有模型算子、Torch/CUDA 和物理传输依赖。metadata_only 是运行路径隔离，编译依赖瘦身待完成。
4. 请求之间串行，单请求内异步工作顺序可能变化。同配置逐请求确定性仍需验证。std::set 的最终平局比较包含进程地址。
5. held 仅为进程内保活。progress.json、held-N.json 和指标摘要不能恢复缓存树、block ID 分配器及 LRU/FIFO 状态。完整检查点待实现。
6. 顶层旧 implementation-status.json 所述更早基线状态不适用于此次快照，因此以本目录 manifest 和文档为归档入口。
7. legacy/operations 中替换了主机和账号标识，部署路径、镜像、依赖仓库以及请求授权范围需要按目标环境核对。continue_segments.py 描述热续跑，cold-segments-215 描述独立冷段，不能混用。
8. raw logs、完整 trace、原始请求 ID/hash、凭据和模型文件不随本归档发布。Dsv41Profile.h 是既有生成拓扑描述，并包含来源摘要。
9. 不同 last_n 组同时改变 per-request 尾槽和 FULL/SWA 容量配比。结果比较必须包含两项设置。旧理论合池 LRU 与 BTC 分层水位结果分别标注，避免把不同缓存历史的差值当成单项因果贡献。
