# Telecom + Retail 智能客服 Agent

FastAPI + LangGraph 的模块化单体，面向移联电信与商城客服。MySQL 是业务事实源，Redis/Qdrant 分别承载热状态和向量索引；所有模型通过 DashScope OpenAI-compatible 接口调用 Qwen。

## 已实现能力

- 请求级 `user_id`、会话归属、资源 ownership；可平滑切换为可信 Nginx `X-User-Id`。
- LLM Supervisor 是唯一意图与调度入口，最多进行三轮有界管理；统一调度 `knowledge_agent`、`telecom_agent`、`retail_agent`，支持依赖任务和独立只读任务并行。
- 唯一 Skill 来源为 `skills/**/SKILL.md`；启动时只索引 frontmatter，选中后加载正文和显式 references，并冻结版本/哈希。
- `telecom-plan-recommendation` 根据最近三个账期的流量、通话和确定性成本比较给出只读建议。
- Telecom：套餐/使用量查询、套餐变更、流量补充、漫游；通信故障永久为带文档版本和引用的 RAG 指导，不存在设备 adapter。
- Retail：完整地址/联系电话查询、新增地址并设为默认、支付方式、商品/库存、订单、取消、地址/支付/商品修改、退货、换货、差价退款。
- 所有业务数据库写操作统一进入 governed action：两次 active 用户查询、资源版本、冻结参数摘要、二次确认、幂等事务、回读回执及未知状态对账。
- MySQL durable LangGraph checkpoint、会话串行/有界队列、全局 LLM 20 并发与节点级 Model Profile。
- 四层记忆、20-turn + 增量阈值异步摘要、类型化 TTL、Qdrant 删除 outbox。
- Prometheus `/metrics`、Langfuse、可回放评测 Harness、tau2 adapter、shadow/canary 稳定用户桶。

## 主工作流

```text
START -> history_fusion -> supervisor_manager
  -> knowledge_agent / telecom_agent / retail_agent
  -> supervisor_manager（最多三轮）
  -> response_writer -> compliance -> response_synthesizer -> END
```

所有请求均由 Supervisor LLM 做语义判断，不存在关键词或正则快速路由。`knowledge_agent` 统一承接通信故障、零售政策等非结构化 RAG；Telecom/Retail Agent 自主生成结构化工具调用。只读无依赖任务可并行；每批最多一个写任务，Agent 只能生成冻结提案，用户确认后才由 governed action 执行。最终合规节点是所有路径的必经出口。

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
```

各节点可分别通过 `QWEN_SUPERVISOR_MODEL`、`QWEN_KNOWLEDGE_AGENT_MODEL`、`QWEN_QUERY_REWRITER_MODEL`、`QWEN_TELECOM_AGENT_MODEL`、`QWEN_RETAIL_AGENT_MODEL`、`QWEN_RESPONSE_WRITER_MODEL`、`QWEN_SAFETY_GUARD_MODEL`、`QWEN_MEMORY_MODEL` 及对应的 `*_BASE_URL` 覆盖；留空继承 `QWEN_MODEL` / `QWEN_BASE_URL`。旧 `QWEN_TASK_PLANNER_*` 配置仍可作为 Supervisor 兼容回退，但不再存在规则分类器或独立规则 Planner 节点。

Nginx 完成真实登录认证并覆盖外部身份头后，可设置：

```env
IDENTITY_HEADER_ENABLED=true
TRUSTED_PROXY_NETWORKS=127.0.0.1/32,::1/128
```

普通反向代理本身不等于认证。

## 启动

```powershell
D:\python\agentProject\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\python\agentProject\.venv\Scripts\python.exe main.py
```

启动时会校验 migration checksum、执行未应用 DDL、校验并冻结 Skill Catalog，然后编译 durable LangGraph。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/chat` | 普通对话，body 必须传 `user_id` |
| `POST` | `/api/chat_stream` | SSE 对话 |
| `GET` | `/api/history/{session_id}?user_id=...` | 带 ownership 的历史 |
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
```

旧版本演示数据使用不可逆摘要保存联系人信息；升级后需要执行一次 `--apply`，才能查询旧演示用户的完整手机号和详细地址。新建地址不受此限制。

治理审计、退款、退换货和变更历史不会被伪造，只由实际应用动作生成。

`evaluation/fixtures/smoke_cases.json` 提供首批路由、工具序列、确认与越权用例。`EvaluationHarness` 支持确定性断言，可选 Judge 只评价回复质量，不能替代权限、状态或终态断言。
