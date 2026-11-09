# 已有实现与测试语义

## 请求流程

提前解析和排序请求日志，构造基础粒度完整前缀 hash 并写入 BTCPFX01 分块。回放主线程从分块读取原始 input_length 和完整基础前缀 key，构造 length-only CompleteTokenIds 与 BatchKVCacheResource，依次调用原生 CoordinatorCacheManager 的 malloc、insertIntoCache、free。加载就绪后和请求结束前分别清空本轮后台工作。

实际使用 dsv4.1 模型缓存拓扑和 CPSlotMapper。metadata_only 池只维护逻辑容量、block ID、引用和节点状态。BlockTransferDispatcher 通过 submitter 注入 MetadataCompletion，验证逻辑源目标资源后提交完成回调。NoPayloadOps 对落入物理 execBatchCopy 的情况直接报错。

原生 BTC 执行联合匹配、插入、层间迁移、引用保护、水位淘汰和分配压力 force_drop。失败按原生状态流转处理，指标不另写淘汰模拟器。

## Hash 与尾槽

历史数据基础粒度为 256 tokens。旧原始日志经过 LiteHit 前缀转换后落盘，输入保留全部基础 key。CP=4 时原生 CPSlotMapper 取第 4、8、12 等 key，对应每 1024 tokens 的完整 CP 端点。输入已经是前缀 hash 时应跳过再次前缀转换。基础尾块与 CP 不完整尾段遵循原生映射及长度限制。

last_n 实验通过 CacheConfig.reuse_cache_tail_blocks 设置 1、5、10、15、20。参数影响 SWACacheManager 的 prefill 尾槽分配，未满尾槽也占一个名额。已命中的复用状态和加载所需位置由原生逻辑保留。发布到 BTC 树的端点仍来自完整 canonical keys。因此冷请求 last_n=1 可能只分配无法关联完整端点的尾槽。

## 容量与淘汰

configs 中块数均为单 rank 可用块数，预留零号槽另计。全保留两组各层 FULL:SWA 块数比为 1:1，Device 各 35565，Host 各 166544，Disk 各 3016127。

last_n 各组 Device SWA=1000*n，固定字节预算的剩余部分分配给 FULL，再将 Device 块数比应用到 Host 和 Disk。1000 为容量系数，回放按请求串行执行。Device 水位为 82%/90%，Host 90%/94%，Disk 92%/97%。

LRU 和 FIFO 共用 EvictionHeap，内部是 std::set 加 node 到 iterator 的哈希索引。每个 group set 和 tier 独立维护候选集合。LRU 按 last_access_seq、admission_seq 排序，FIFO 按 admission_seq 排序。迁移更新 admission_seq，保留 last_access_seq。FULL 候选具有分层叶节点约束，选择时还检查引用及冻结等条件。

## 线程与结果

主线程串行发起请求，MetadataBTC 是一个 BTC 业务工作线程，MetadataCompletion 是一个模拟完成工作线程。BlockTreeScan 执行只读一致性扫描。业务完成队列具有优先级，单请求内仍存在异步交错。

progress.json 保存累计计数与下一条输入位置，hit-tokens.u32 保存逐请求复用 token 数，held-N.json 记录处理 N 条后的保活边界。命中率为 hit_tokens/input_tokens，输入长度包括未命中的尾部。计时中的 match_nested 已包含于 malloc 阶段。

## 预处理格式

文件头按 little-endian 写入 magic[8]=BTCPFX01、uint32 base_tokens、uint64 requests、uint64 input_tokens、uint32 domain_bytes 和 UTF-8 domain。

每条记录为 uint64 timestamp_ns、uint32 input_len、uint32 request_id_bytes、request_id 字节，以及 floor(input_len/base_tokens) 个 int64 前缀 key。原始顺序由预处理稳定排序固定，manifest 记录输入和输出摘要。

scripts/prepare_replay_input.py 从规范化 prefix JSONL 生成一个 requests.btcbin。resident 千万请求入口还要求按 100000 条拆分命名分块，legacy/operations/day24-10m/input 保存了原分块转换实现，两种入口须按各自格式约束使用。
