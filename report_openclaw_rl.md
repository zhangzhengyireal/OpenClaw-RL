# OpenClaw-RL 代码仓库调研报告

> 报告日期：2026-04-01  
> 分析对象：zhangzhengyireal/OpenClaw-RL

---

## 目录

1. [项目概览](#1-项目概览)
2. [train_async.py 运行逻辑（流程图）](#2-train_asyncpy-运行逻辑)
   - 2.1 顶层训练循环
   - 2.2 `create_placement_groups` 详解
   - 2.3 `create_rollout_manager` 详解
   - 2.4 `create_training_models` 详解
   - 2.5 `rollout_manager.generate` 详解
3. [OpenClawOPDAPIServer 工作流程（具体示例）](#3-openclawopd-api-server-工作流程)
   - 3.1 系统角色与组件
   - 3.2 端到端数据流（以 combine-001 为例）
4. [关键设计亮点总结](#4-关键设计亮点总结)

---

## 1. 项目概览

OpenClaw-RL 是一个基于 **Reinforcement Learning from Human Feedback (RLHF)** 和 **On-Policy Distillation (OPD)** 的大模型训练框架。其核心组件包括：

| 组件 | 路径 | 职责 |
|------|------|------|
| **Slime 训练引擎** | `slime/` | 基于 Ray + Megatron-LM 的分布式训练核心 |
| **OpenClaw-OPD** | `openclaw-opd/` | OPD 代理服务器，负责采集并构建带教师信号的训练样本 |
| **OpenClaw-Combine** | `openclaw-combine/` | 数据馈送脚本，将种子对话数据注入 OPD 服务器 |

启动入口为 `openclaw-opd/run_qwen3_4b_openclaw_opd.sh`，它：
1. 分配 8 块 GPU（Actor×4、Rollout×2、PRM×2）
2. 启动 Ray 集群
3. 通过 `ray job submit` 调用 `slime/train_async.py`

---

## 2. train_async.py 运行逻辑

### 2.1 顶层训练循环

```
train_async.py  main()
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  parse_args()                                                        │
│  ┌─ 读取所有 CLI 参数（MODEL_ARGS, ROLLOUT_ARGS, OPD_ARGS 等）        │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  configure_logger() + create_placement_groups(args)                  │
│  ┌─ 在 Ray 集群上创建 GPU Placement Group                             │
│  ├─ 分配 Actor GPUs (GPU 0-3)                                         │
│  ├─ 分配 Rollout GPUs (GPU 4-5)                                       │
│  └─ 分配 PRM GPUs (GPU 6-7)                                           │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  init_tracking(args)                                                 │
│  ┌─ 初始化 WandB 追踪（如 USE_WANDB=1）                               │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  create_rollout_manager(args, pgs["rollout"], pgs["prm"])            │
│  ┌─ 创建 Ray Actor: RolloutManager                                    │
│  ├─ 内部启动 SGLang 推理引擎（rollout GPUs）                           │
│  ├─ 内部启动 PRM 评估引擎（PRM GPUs）                                  │
│  └─ 返回 rollout_manager, num_rollout_per_epoch                       │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  create_training_models(args, pgs, rollout_manager)                  │
│  ┌─ 创建 actor_model (RayTrainGroup，Megatron-LM 分布式 Actor)        │
│  ├─ 可选：创建 critic_model                                           │
│  └─ actor_model.async_init() → 加载 HF 检查点，初始化模型            │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  actor_model.update_weights()                                        │
│  ┌─ 将 Megatron Actor 的最新权重同步到 SGLang 推理引擎                 │
│  └─ 确保推理与训练权重对齐（on-policy）                               │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
╔══════════════════════════════════════════════════════════════════════╗
║                   异步训练主循环（Async Train Loop）                 ║
║                                                                      ║
║  rollout_data_next_future = rollout_manager.generate.remote(0)       ║
║                                                                      ║
║  for rollout_id in range(start, num_rollout):                        ║
║    │                                                                 ║
║    ├─[1] ray.get(rollout_data_next_future)  ← 等待当前 rollout 完成  ║
║    │    rollout_data_curr_ref ← 训练样本数据引用                     ║
║    │                                                                 ║
║    ├─[2] rollout_manager.generate.remote(rollout_id+1)              ║
║    │    ← 提前启动下一次 rollout（流水线并行）                        ║
║    │                                                                 ║
║    ├─[3] actor_model.async_train(rollout_id, rollout_data_curr_ref) ║
║    │    ← 用当前 rollout 数据训练 Actor（OPD 损失 + KL 损失）        ║
║    │                                                                 ║
║    ├─[4] 周期性触发（每 save_interval 步）:                          ║
║    │    actor_model.save_model(rollout_id)  ← 保存检查点             ║
║    │                                                                 ║
║    ├─[5] 周期性触发（每 update_weights_interval 步）:                ║
║    │    actor_model.update_weights()  ← 权重热更新到 SGLang          ║
║    │    ← 更新前先同步 rollout，防止权重更新中途推理                  ║
║    │                                                                 ║
║    └─[6] 周期性触发（每 eval_interval 步）:                          ║
║         rollout_manager.eval.remote(rollout_id)  ← 离线评估         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        │
        ▼
    rollout_manager.dispose()  ← 清理资源，结束训练
```

**关键异步设计（流水线并行）：**  
训练步 N 进行 Actor 反向传播的同时，步 N+1 的 rollout 数据已经在 SGLang 上生成。这样 GPU 利用率显著提高，避免 CPU-GPU 空闲等待。

---

### 2.2 `create_placement_groups` 详解

```
create_placement_groups(args)
        │
        ▼
  计算总 GPU 数 = actor_gpus(4) + rollout_gpus(2) + prm_gpus(2) = 8
        │
        ▼
  _create_placement_group(num_gpus=8)
  ├─ 创建 8 个 bundle（每个 bundle: 1 GPU + 1 CPU），strategy="PACK"
  ├─ 启动 8 个 InfoActor（临时 Ray Actor）获取每块 GPU 的 IP + GPU-ID
  ├─ 按照 (节点IP, GPU-ID) 排序，确保 bundle 顺序与物理拓扑一致
  └─ 返回 pg（placement group对象）及有序 bundle index 列表
        │
        ▼
  切片分配：
  ├─ actor_pg   = bundles[0:4]   → Actor 训练使用 GPU 0-3
  ├─ rollout_pg = bundles[4:6]   → SGLang 推理使用 GPU 4-5
  └─ prm_pg     = bundles[6:8]   → PRM/教师模型使用 GPU 6-7
```

---

### 2.3 `create_rollout_manager` 详解

```
create_rollout_manager(args, rollout_pg, prm_pg)
        │
        ▼
  RolloutManager.options(num_cpus=1, num_gpus=0).remote(args, pg, prm_pg)
  │
  └── RolloutManager.__init__(args, pg, prm_pg)
        │
        ├─ _start_router(args)  ← 启动 SGLang Router（负载均衡器）
        │   ├─ 监听 sglang_router_ip:sglang_router_port
        │   └─ 将请求分发给各 SGLangEngine 实例
        │
        ├─ _start_router(args, prm_*)  ← 启动 PRM Router
        │
        ├─ 加载自定义函数：
        │   ├─ generate_rollout = openclaw_opd_rollout.generate_rollout_openclaw_opd
        │   └─ eval_generate_rollout（eval 时使用）
        │
        ├─ init_rollout_engines(args, pg, engines)
        │   ├─ num_engines = rollout_num_gpus(2) / num_gpus_per_engine(2) = 1 个引擎
        │   └─ 每个 SGLangEngine 在分配的 GPU 上加载模型（Qwen3-4B）
        │
        └─ init_prm_engines(args, prm_pg, prm_engines)
            ├─ prm_num_engines = prm_num_gpus(2) / prm_num_gpus_per_engine(2) = 1 个 PRM 引擎
            └─ 加载 PRM/教师模型（Qwen3-4B-Thinking）
```

---

### 2.4 `create_training_models` 详解

```
create_training_models(args, pgs, rollout_manager)
        │
        ▼
  allocate_train_group(actor, pg=pgs["actor"], num_gpus_per_node=4)
  └── RayTrainGroup(args, num_nodes=1, num_gpus_per_node=4, pg=actor_pg)
        ├─ 创建 4 个 Ray Actor（每个占用 0.4 个 GPU slot）
        └─ 每个 Actor 运行 Megatron-LM 分布式训练进程
              │
              ▼
        actor_model.async_init(args, role="actor", with_ref=True)
        ├─ 在各 GPU 上初始化 Megatron Transformer（TP=4）
        ├─ 从 HF checkpoint 加载权重（megatron-to-hf bridge 模式）
        ├─ 初始化 Ref 模型（用于 KL 散度计算）
        └─ 返回 start_rollout_id（从检查点恢复的步数）
              │
              ▼
        actor_model.set_rollout_manager(rollout_manager)
        ← 让 Actor 知道如何触发权重更新
```

---

### 2.5 `rollout_manager.generate` 详解

```
rollout_manager.generate(rollout_id)
        │
        ▼
  health_monitoring_resume()  ← 恢复健康监控线程
        │
        ▼
  _get_rollout_data(rollout_id)
  │
  └── call_rollout_fn(generate_rollout_openclaw_opd, args, rollout_id, data_source)
        │
        └── generate_rollout_openclaw_opd(args, rollout_id, data_buffer)
              │
              ├─ get_global_worker(args)
              │   └── AsyncRolloutWorker（单例）
              │         ├─ output_queue: Queue（接收 OPD 样本）
              │         └─ OpenClawOPDAPIServer（FastAPI 服务，监听 :30000）
              │
              ├─ worker.resume_submission()  ← 打开提交开关
              │
              ├─ _drain_output_queue(args, worker)
              │   │
              │   └─ 循环等待，直到 output_queue 中有 rollout_batch_size(16) 个样本
              │       ├─ poll output_queue.get_nowait()
              │       ├─ 过滤 ABORTED 样本
              │       └─ 每 30s 打印等待进度日志
              │
              ├─ worker.pause_submission()  ← 关闭提交开关
              │
              └─ 返回 RolloutFnTrainOutput(samples=completed_samples)
        │
        ▼
  _save_debug_rollout_data()  ← 可选：保存调试数据
        │
        ▼
  _log_rollout_data()  ← 记录 rollout 统计指标到 WandB
        │
        ▼
  _convert_samples_to_train_data(data)
  ├─ _drop_constant_reward_groups()  ← 删除奖励全相同的组（对 GRPO 无意义）
  ├─ _post_process_rewards()  ← 奖励归一化（组内减均值/除标准差）
  └─ 构建训练数据 dict（包含 tokens, advantages, loss_mask, teacher_log_probs 等）
        │
        ▼
  _split_train_data_by_dp(data, dp_size)
  └── 按 data-parallel size 分片，每个 DP rank 得到对应数据
        │
        ▼
  返回 train_data（分片列表，每片对应一个 DP rank）
```

---

## 3. OpenClawOPD API Server 工作流程

### 3.1 系统角色与组件

```
┌─────────────────────────────────────────────────────────────────────┐
│                   OpenClaw-Combine / OPD 数据管道                   │
│                                                                     │
│  ┌─────────────┐         ┌───────────────────────────────────────┐ │
│  │ feed_data.py│         │      OpenClawOPDAPIServer (:30000)    │ │
│  │ (数据馈送)  │ HTTP ── │  ┌──────────────┐  ┌──────────────┐  │ │
│  │             │ ──────▶ │  │ Turn 缓存    │  │  样本输出队列 │  │ │
│  └─────────────┘         │  │ _pending_    │  │  output_queue│  │ │
│                          │  │ turn_data    │  └──────┬───────┘  │ │
│                          │  └──────────────┘         │          │ │
│                          └───────────────────────────│──────────┘ │
│                                    │                 │            │
│                    ┌───────────────┘                 │            │
│                    ▼                                 ▼            │
│           ┌────────────────┐              ┌──────────────────┐   │
│           │ SGLang (:34000)│              │ RolloutManager   │   │
│           │ (策略模型推理)  │              │ generate()       │   │
│           │ Qwen3-4B       │              │ ← drain_queue()  │   │
│           └────────────────┘              └──────────────────┘   │
│                    │                                              │
│           ┌────────┴────────┐                                    │
│           │ PRM (:34001)    │                                    │
│           │ (教师/评估模型)  │                                    │
│           │ Qwen3-4B-Think  │                                    │
│           └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 端到端数据流（以 combine-001 为例）

**种子数据（来自 `combine_seed_data.jsonl`）：**
```json
{
  "session_id": "combine-001",
  "question": "请帮我写一个Python函数来判断一个数是否为质数，要求效率尽量高。",
  "challenge": "你的代码没有处理n=0和n=1的边界情况，而且对于大数效率很低，没有用到平方根优化。正确的做法应该先处理这两个特殊情况，然后只遍历到sqrt(n)。",
  "expected_signal": "combined"
}
```

---

#### 阶段一：Turn 1 — 模型初次回答

```
feed_data.py
    │
    ├─ session_id = "combine-001-a1b2c3"（原ID + uuid后缀）
    │
    └─ POST /v1/chat/completions
       Headers: X-Session-Id: combine-001-a1b2c3
                X-Turn-Type: main
                (无 X-Session-Done)
       Body:
         messages = [
           {"role": "user", "content": "请帮我写一个Python函数来判断一个数是否为质数..."}
         ]
                │
                ▼
        OpenClawOPDAPIServer._handle_request()
                │
                ├─ turn_type = "main", session_done = False
                ├─ prev_turn_num = 0（首次，无前一轮）
                │   → 不触发 OPD 评估任务
                │
                ├─ 转发请求至 SGLang: POST sglang_router:34000/v1/chat/completions
                │   → 模型生成回答 R1（Python 质数函数）
                │   → 返回 choice, logprobs
                │
                ├─ tokenize(prompt) → prompt_ids
                │  tokenize(R1)     → response_ids, response_logprobs
                │
                ├─ _turn_counts[session_id] = 1
                ├─ _pending_turn_data[session_id][1] = {
                │     prompt_ids, response_ids, response_logprobs,
                │     prompt_text, response_text, messages, tools
                │   }
                │
                ├─ _buffer_record(session_id, 1, ...)
                │   → 暂存到 _pending_records（等下一轮 next_state）
                │
                ├─ _maybe_submit_ready_samples()
                │   → turn 1 无 OPD task，且非 force_drop，跳过
                │
                └─ 返回 R1 给 feed_data.py
```

---

#### 阶段二：Turn 2 — 发送 challenge，触发 OPD 评估

```
feed_data.py
    │
    └─ POST /v1/chat/completions
       Headers: X-Session-Id: combine-001-a1b2c3
                X-Turn-Type: main
                X-Session-Done: true
       Body:
         messages = [
           {"role": "user",      "content": "请帮我写一个Python函数..."},
           {"role": "assistant", "content": R1},
           {"role": "user",      "content": "你的代码没有处理n=0和n=1..."}   ← challenge
         ]
                │
                ▼
        OpenClawOPDAPIServer._handle_request()
                │
                ├─ prev_turn_num = 1（已有第1轮）
                │
                ├─ _flush_pending_record(session_id, next_state=messages[-1])
                │   → 将 Turn 1 记录（含 next_state=challenge）写入 record.jsonl
                │
                ├─ _fire_opd_task(session_id, 1, turn_data_t1, next_state=challenge_msg)
                │   └── 创建异步任务 _opd_evaluate(...)  ← 后台并发执行（不阻塞）
                │
                ├─ 转发请求至 SGLang → 模型对 [Q, R1, challenge] 生成 R2
                │
                ├─ tokenize → 存 Turn 2 数据到 _pending_turn_data[session_id][2]
                │
                ├─ _maybe_submit_ready_samples()
                │   → OPD task for turn 1 可能未完成 → 等待回调
                │
                └─ session_done=True:
                    → _maybe_submit_ready_samples(force_drop_without_next_state=True)
                    → _turn_counts.pop(session_id)  ← 清理会话状态
```

---

#### 阶段三：后台 OPD 评估（并发执行）

```
asyncio.Task: _opd_evaluate(session_id, turn_num=1, turn_data_t1, next_state=challenge)
        │
        ├─ next_state_text = "你的代码没有处理n=0和n=1的边界情况..."
        ├─ next_state_role = "user"
        │
        ├─ _build_hint_judge_messages(R1, challenge_text, role="user")
        │   └── 构建系统提示：
        │       "You are a process reward model used for hindsight hint extraction..."
        │       用户请求："Given R1 and the challenge, should the challenge be a hint?"
        │
        ├─ prm_tokenizer.apply_chat_template(judge_msgs) → judge_prompt（原始文本）
        │
        ├─ asyncio.gather(*[_query_judge_once(judge_prompt, i) for i in range(1)])
        │   └── POST prm_router:34001/generate
        │         payload = {text: judge_prompt, sampling_params: {temp=0.6, max_new_tokens=8192}}
        │         → PRM 模型回答，输出例：
        │           "...分析：challenge 明确指出了代码缺陷... \boxed{1}
        │            [HINT_START]优化质数判断需处理n≤1边界情况并只遍历到sqrt(n)[HINT_END]"
        │         → _parse_judge_result() → score=1, hint="优化质数判断需处理..."
        │
        ├─ _select_best_hint(votes)
        │   → votes=[{"score":1, "hint":"优化质数判断...", "raw":...}]
        │   → 选 score=1 且 hint 长度最长的 → selected = votes[0]
        │
        ├─ hint = "优化质数判断需处理n≤1边界情况并只遍历到sqrt(n)"
        │
        ├─ _append_hint_to_messages(turn_data_t1.messages, hint)
        │   → enhanced_messages = [
        │       {"role":"user", "content": "请帮我写一个Python函数...\n\n[user's hint / instruction]\n优化质数判断需处理n≤1边界情况并只遍历到sqrt(n)"}
        │     ]
        │
        ├─ tokenizer.apply_chat_template(enhanced_messages) → enhanced_prompt_text
        │
        ├─ enhanced_full_text = enhanced_prompt_text + R1
        │  enhanced_ids = tokenizer(enhanced_full_text)["input_ids"]
        │
        ├─ _compute_teacher_log_probs(enhanced_ids, response_len=len(response_ids))
        │   └── POST prm_router:34001/generate
        │         payload = {input_ids: enhanced_ids, max_new_tokens: 0, return_logprob: True}
        │         → 获取每个 token 在"增强后提示"下的 log-prob
        │         → teacher_log_probs = [logp_1, logp_2, ..., logp_T]
        │
        └─ 返回 {
               "accepted": True,
               "teacher_log_probs": [logp_1, ..., logp_T],
               "hint": "优化质数判断需处理n≤1边界情况并只遍历到sqrt(n)",
               "votes": [...],
               "eval_score": None  (eval_mode=0 时不计算)
           }
```

---

#### 阶段四：提交训练样本

```
OPD task 完成后触发回调:
_maybe_submit_ready_samples(session_id)
        │
        ├─ task.done() == True
        ├─ opd_result.accepted == True
        │
        └─ _submit_turn_sample(turn_data_t1, session_id, opd_result)
                │
                ├─ sample = Sample()
                ├─ sample.prompt         = prompt_text（原始 Turn 1 提示）
                ├─ sample.response       = R1（模型原始回答）
                ├─ sample.tokens         = prompt_ids + response_ids
                ├─ sample.response_length= len(response_ids)
                ├─ sample.loss_mask      = [1, 1, ..., 1]（对所有 response token 计算损失）
                ├─ sample.rollout_log_probs = response_logprobs（模型原始 logprob）
                ├─ sample.teacher_log_probs = tensor([logp_1,...,logp_T])（增强提示下的教师 logprob）
                ├─ sample.reward         = {"score": 1.0}（OPD 样本奖励固定为 1.0）
                ├─ sample.index          = 全局自增 ID
                ├─ sample.group_index    = 全局自增 group ID
                ├─ sample.status         = COMPLETED
                │
                └─ output_queue.put((group_index, [sample]))
                        │
                        ▼
        _drain_output_queue() 中轮询到该样本
        → 加入 data 列表
        → 当 len(data) == rollout_batch_size (16) 时返回
                        │
                        ▼
        rollout_manager.generate() 返回分片后的训练数据
                        │
                        ▼
        actor_model.async_train()
        ├─ OPD 损失（on_policy_distillation）：
        │   L_OPD = KL(π_teacher(·|enhanced_prompt) || π_actor(·|original_prompt))
        │   ≈ Σ teacher_log_prob - student_log_prob（token 级别）
        │
        ├─ KL 损失（kl_loss_coef=0.0，本配置中关闭）
        │
        └─ 反向传播 + Adam 优化
```

---

#### 完整数据流总结图

```
combine_seed_data.jsonl
    "combine-001: 质数函数 + challenge"
            │
            │  (feed_data.py 读取并发起请求)
            ▼
┌─────────────────────────────────────────────────────────┐
│                  OpenClawOPD Proxy (:30000)              │
│                                                         │
│  Turn 1 POST ──▶  SGLang(:34000) ──▶ R1(质数函数回答)  │
│                        │                                │
│              存 turn_data[1]                            │
│                        │                                │
│  Turn 2 POST ──▶  SGLang(:34000) ──▶ R2                │
│  (Q + R1 + challenge)  │                               │
│                        │                                │
│         fire_opd_task(turn=1, next_state=challenge)     │
│              │                                          │
│              ▼ (异步)                                   │
│         PRM(:34001) ──▶ hint_judge                      │
│              │  score=1, hint="处理边界+sqrt优化"       │
│              │                                          │
│         PRM(:34001) ──▶ teacher_logprobs                │
│              │  (R1 在 enhanced_prompt 下的 logprob)    │
│              │                                          │
│         Sample 构建 & output_queue.put()               │
└─────────────────────────────────────────────────────────┘
            │
            ▼
  output_queue (rollout_batch_size=16 个样本)
            │
            ▼
  generate_rollout_openclaw_opd() 返回
            │
            ▼
  RolloutManager.generate() 返回训练数据
            │
            ▼
  actor_model.async_train()
  → OPD Loss: 让模型学习"如果知道 hint，R1 应该怎么写"
  → 模型从失败经验中学习改进
```

---

## 4. 关键设计亮点总结

### 4.1 异步流水线训练
- **设计**：`train_async.py` 在 rollout_id=N 训练的同时，提前启动 rollout_id=N+1 的数据生成
- **效果**：消除了生成-训练的串行等待，显著提升 GPU 利用率

### 4.2 On-Policy Distillation (OPD) 的创新点
- **传统 SFT/KD**：使用预先构造的 (问题, 标准答案) 对进行蒸馏
- **OPD 的创新**：
  1. 模型先用原始提示生成 R1（on-policy 样本）
  2. challenge（next_state）作为事后信号，由 PRM 判断 R1 是否有改进空间
  3. 若有，PRM 提取 hint，构建增强提示，计算教师模型在增强提示下对 R1 的 logprob
  4. 用这个教师 logprob 训练模型，让模型"假装已知 hint 时的行为"更接近现实
- **效果**：模型从自身生成的不完美样本中学习，而不依赖人工标注答案

### 4.3 三信号融合（Combined OPD+RL）
```
一条对话数据可同时产生三种训练信号：
┌─────────────────────────────────────────────────────────┐
│  hint accepted AND eval ±1  →  Combined OPD+RL sample  │
│  hint accepted only          →  OPD-only sample         │
│  eval ±1 only                →  RL-only sample          │
└─────────────────────────────────────────────────────────┘
```

### 4.4 分离式 GPU 分配
```
训练 GPUs (0-3, TP=4, Megatron-LM)
    ↕  update_weights
推理 GPUs (4-5, TP=2, SGLang)   ← 负责 on-policy rollout
    ↕  judge & teacher logprob
PRM  GPUs (6-7, TP=2, SGLang)   ← 负责 OPD 评估和教师信号
```
三组 GPU 完全隔离，分别优化各自任务，通过 Ray 远程调用协作。

### 4.5 健壮性设计
- **权重更新锁**：`update_weights` 前强制等待当前 rollout 完成，防止推理和训练权重不一致
- **故障恢复**：`RolloutHealthMonitor` 检测 SGLang 引擎崩溃，自动重启并同步权重
- **提交开关**：`submission_enabled` Event 控制 OPD 样本入队，在权重更新期间暂停数据采集
- **奖励归一化**：对 GRPO 组内奖励做均值/方差归一化，删除奖励恒定组，提高训练稳定性
