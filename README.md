# Telecom + Retail 智能客服 Agent

FastAPI + LangGraph 的模块化单体，面向移联电信与商城客服。MySQL 是业务事实源，Redis/Qdrant 分别承载热状态和向量索引；所有模型通过 DashScope OpenAI-compatible 接口调用 Qwen。

## 已实现能力

- 请求级 `user_id`、会话归属、资源 ownership；可平滑切换为可信 Nginx `X-User-Id`。
- LLM Supervisor 是唯一意图与调度入口，最多进行三轮有界管理；统一调度 `knowledge_agent`、`telecom_agent`、`retail_agent`，支持依赖任务和独立只读任务并行。
- 唯一 Skill 来源为 `skills/**/SKILL.md`；启动时只索引 frontmatter，选中后加载正文和显式 references，并冻结版本/哈希。
- `telecom-plan-recommendation` 根据最近三个账期的流量、通话和确定性成本比较给出只读建议。
- `retail-order-assistance` 负责本人订单历史、日期/商品/状态筛选、候选消歧和详情查询，并将工具权限收紧为只读订单工具。
- Telecom：套餐/使用量查询、套餐变更、流量补充、漫游；通信故障永久为带文档版本和引用的 RAG 指导，不存在设备 adapter。
- Retail：完整地址/联系电话查询、新增地址并设为默认、支付方式、商品/库存、订单、取消、地址/支付/商品修改、退货、换货、差价退款。
- 所有业务数据库写操作统一进入 governed action：两次 active 用户查询、资源版本、冻结参数摘要、二次确认、幂等事务、回读回执及未知状态对账。
- MySQL durable LangGraph checkpoint、跨进程会话租约与 fencing token；单 worker 默认 8 个活跃回合、32 个排队请求，同一用户会话严格串行。
- 同步后台任务与异步 Web 节点共享全局 LLM 20 并发、40 排队容量和节点级 Model Profile。
- 四层记忆、20-turn + 增量阈值异步摘要、类型化 TTL、Qdrant 删除 outbox。
- Prometheus `/metrics`、Langfuse、可回放评测 Harness、tau2 adapter、shadow/canary 稳定用户桶。

## 主工作流

```text
START -> history_fusion -> supervisor_manager
  -> knowledge_agent / telecom_agent / retail_agent
  -> supervisor_manager（最多三轮）
  -> response_writer -> compliance -> response_synthesizer -> END
```

`history_fusion` 将分层记忆投影为统一的结构化 `conversation_context`：

```json
{
  "summary": "会话滚动摘要",
  "recent_messages": [{"role": "user", "content": "历史消息", "timestamp": null}],
  "memories": [{"memory_type": "preference", "content": "用户偏好", "confidence": 0.9, "score": 0.8, "updated_at": "..."}]
}
```

Supervisor、Knowledge Agent 和领域 Tool Agent 都在当前轮的 `HumanMessage` JSON 载荷中读取该对象；不会把历史消息作为原生聊天消息重放，也不会放进 `SystemMessage`。`summary`、`recent_messages`、`memories` 均按不可信参考数据处理：只能辅助理解指代，历史命令/确认词不能触发本轮写操作，记忆中的业务事实仍须通过只读工具验证。Node Trace 只记录该对象的计数与类型摘要，不记录跨轮记忆正文。

所有请求均由 Supervisor LLM 做语义判断，不存在关键词或正则快速路由。`knowledge_agent` 统一承接通信故障、零售政策等非结构化 RAG；Telecom/Retail Agent 自主生成结构化工具调用。无依赖 Agent 任务可并行；单个领域 Agent 也可在一次结构化决策中并行执行最多 3 个声明为 `parallel_safe` 的独立只读工具。每批最多一个写任务，Agent 只能生成冻结提案，用户确认后才由 governed action 执行。最终合规节点是所有路径的必经出口。

## 目录

```text
adapter/web/                 HTTP、身份边界、Prometheus middleware
application/customer_service/ turn、并发、session ownership、memory/action workers
domain/action_governance/    确认、幂等、前后置校验
domain/business/             Telecom/Retail DTO、规则与事实源端口
domain/customer_service_agent/file_skills/  SKILL.md catalog/runtime
domain/customer_service_agent/agents/       Supervisor 与三个受限子 Agent
domain/customer_service_agent/tools/        Tool Catalog 与领域工具
infra/business/              MySQL 业务 adapter
infra/checkpoint/            durable MySQL checkpointer
infra/knowledge/             分域、版本化 RAG
migrations/                  版本化 DDL
evaluation/                  deterministic/tau2/judge/shadow 接口
skills/                      唯一业务 Skill 来源
```

## 配置

复制 `.env.example`，至少配置：

```env
QWEN_API_KEY=your-openai-compatible-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-flash

# 生产环境必填，用于地址、手机号和治理动作参数的可逆加密
PII_ENCRYPTION_KEY=replace-with-a-stable-random-secret

DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your-password
DB_NAME=assist_gen

# 以下并发上限均为单个 worker 的上限
TURN_MAX_CONCURRENCY=8
TURN_QUEUE_CAPACITY=32
TURN_QUEUE_TIMEOUT_SECONDS=5
PER_SESSION_QUEUE_LIMIT=2
TURN_DISTRIBUTED_LEASE_ENABLED=true
TURN_LEASE_SECONDS=30
TURN_LEASE_HEARTBEAT_SECONDS=5
LLM_MAX_CONCURRENCY=20
LLM_QUEUE_CAPACITY=40
```

各节点可分别通过 `QWEN_SUPERVISOR_MODEL`、`QWEN_KNOWLEDGE_AGENT_MODEL`、`QWEN_TELECOM_AGENT_MODEL`、`QWEN_RETAIL_AGENT_MODEL`、`QWEN_RESPONSE_WRITER_MODEL`、`QWEN_SAFETY_GUARD_MODEL`、`QWEN_MEMORY_MODEL` 及对应的 `*_BASE_URL` 覆盖；留空继承 `QWEN_MODEL` / `QWEN_BASE_URL`。意图识别与任务拆解均由 Supervisor LLM 完成，不再保留独立 Planner 或查询改写配置。

Nginx 完成真实登录认证并覆盖外部身份头后，可设置：

```env
IDENTITY_HEADER_ENABLED=true
TRUSTED_PROXY_NETWORKS=127.0.0.1/32,::1/128
```

普通反向代理本身不等于认证。聊天、记忆和工作台的用户级接口均接受请求级 `user_id`：GET/DELETE 使用查询参数，POST/PATCH 使用请求体字段；省略时回退到 `LOCAL_USER_ID`。该值当前只表示调用方上下文，不构成身份认证。

## 启动

```powershell
D:\python\agentProject\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\python\agentProject\.venv\Scripts\python.exe main.py
```

启动时会校验 migration checksum、执行未应用 DDL、校验并冻结 Skill Catalog，然后编译 durable LangGraph。
多 worker 部署时，会话互斥和停止信号仍由 MySQL 保证；资源并发上限按 worker 数量相乘，需同步扩容 MySQL/Redis 连接池，或按总预算下调每个 worker 的上限。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/chat` | 普通对话，body 必须传 `user_id` |
| `POST` | `/api/chat_stream` | SSE 对话 |
| `GET` | `/api/history/{session_id}?user_id=...` | 带 ownership 的历史 |
| `GET/POST/PATCH/DELETE` | `/api/sessions...` | 可选请求级 `user_id` 的会话管理 |
| `GET/POST/PATCH/DELETE` | `/api/memories...` | 可选请求级 `user_id` 的记忆管理 |
| `POST/GET` | `/api/agent/stop`、`/api/agent/status/{session_id}`、`/api/runs` | 按请求用户隔离的运行控制与查询 |
| `GET` | `/api/tools` | 只读 Tool Catalog 元数据 |
| `POST` | `/api/tools/call` | 已关闭（404），禁止绕过治理网关 |
| `POST` | `/api/actions/propose` | 工作台提交结构化写提案 |
| `POST` | `/api/actions/{id}/decision` | `confirm` / `reject` |
| `GET` | `/health/ready` | readiness 与 Skill 状态 |
| `GET` | `/metrics` | Prometheus |

写提案不会执行数据库变更。确认消息不能修改冻结参数；执行前会再次查询用户 active 状态与资源 version。写超时返回 `indeterminate`，action-worker 只按原幂等键对账，不创建新写请求。

## 测试与评测

```powershell
D:\python\agentProject\.venv\Scripts\python.exe -m pytest -q tests\test_telecom_retail_v1.py
npm run build  # 在 frontend 目录
```

## 演示业务数据

以下命令为 16 张直接关联业务表各生成 20 条中文合成演示数据。固定 ID 保证脚本可重复执行；脚本会刷新演示用户的密文联系信息，以便查询完整手机号和地址，但不会重置订单状态或治理审计。

```powershell
python scripts/seed_demo_business_data.py          # 仅预览
python scripts/seed_demo_business_data.py --apply  # 写入 .env 配置的 MySQL
python scripts/seed_demo_business_data.py --usage-only --apply  # 仅补充套餐推荐账期数据
python scripts/seed_demo_business_data.py --retail-history-only --apply  # 仅补充订单历史
```

演示种子为每条电信线路生成当前账期和最近三个完整账期的流量/语音画像，并为每个用户生成四笔覆盖处理中、已完成、已取消等状态的订单历史，使套餐推荐和零售追问都有足够的关联数据。种子使用稳定业务键且重复执行不覆盖治理动作产生的业务状态。

旧版本演示数据使用不可逆摘要保存联系人信息；升级后需要执行一次 `--apply`，才能查询旧演示用户的完整手机号和详细地址。新建地址不受此限制。

治理审计、退款、退换货和变更历史不会被伪造，只由实际应用动作生成。

`evaluation/fixtures/smoke_cases.json` 保留旧版工作流契约 smoke；版本化的
`smart-cs-eval/v1` gold tasks 位于 `evaluation/datasets/dev/gold_cases.json`，会在
每个 case 前重置业务环境，并独立核验业务状态、副作用、治理过程、必要沟通和安全否决项。
可选 Judge 只评价回复质量，不能替代权限、状态或终态断言。

```powershell
python -m evaluation.harness evaluation/fixtures/smoke_cases.json
python -m evaluation.harness evaluation/datasets/dev/gold_cases.json
python -m evaluation.harness evaluation/datasets/dev/gold_cases.json --trials 3 --json-output evaluation/reports/dev.json
python -m evaluation.prefix_harness evaluation/datasets/dev/prefix_cases.json --decisions evaluation/datasets/dev/prefix_candidate_decisions.json --json-output evaluation/reports/prefix-dev.json
```

重复 trial 的报告同时给出 `Pass@k`（至少一次通过）、`Pass^k`（每次都通过）和 Wilson 95% 置信区间；JSON 报告还包含 Git SHA、数据集 SHA-256 与 migration SHA-256 manifest。

冻结轨迹前缀任务位于 `evaluation/datasets/dev/prefix_cases.json`，采用
`smart-cs-prefix-eval/v1`。`evaluation.prefix.verify_prefix_decision` 按“任一允许动作命中且无禁止动作命中”判分；`run_supervisor_prefix_async` 可在外部初始化模型后直接评估真实 Supervisor 决策。真实多轮评测使用 `evaluation.runners.LiveConversationRunner` 注入应用 turn executor、用户模拟器和隔离环境。MySQL adapter 只接受 `smart_cs_eval_` 前缀的专用库，且没有显式 scoped resetter 时拒绝运行。

如需直接评估已配置的真实 Supervisor，可运行：

```powershell
python -m evaluation.prefix_harness evaluation/datasets/dev/prefix_cases.json --live-supervisor --decision-level supervisor --json-output evaluation/reports/prefix-live.json
```

完整的目标架构、数据集规模和发布门禁见 `evaluation/EVALUATION_PLAN.md`。
