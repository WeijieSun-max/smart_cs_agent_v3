### domain/# 领域层，包含客服 Agent、工作流、领域服务和工具定义


#### shared
---

domain/shared/checkpoint/checkpoint_saver_service.py 的作用是：保存并提供 LangGraph 工作流的 checkpoint saver，同时统一生成 LangGraph 的 thread_id。

它主要服务于这个项目的 LangGraph 客服工作流。

checkpoint_saver_service.py 是 LangGraph checkpoint 的共享服务封装，负责：
保存 checkpoint saver
  ↓
提供给 customer_service_workflow.compile(checkpointer=...)
  ↓
根据 user_id + session_id 生成 thread_id
  ↓
让 LangGraph 能按会话隔离工作流状态


---

domain/shared/checkpoint/llm.py 的作用，提供大模型

---

#### customer_service_agent/service
---

domain/customer_service_agent/service/knowledge_service.py 的作用是：定义知识库服务的统一接口，并保存当前项目使用的知识库实例。
它本身不负责具体检索算法，真正的本地知识库实现是在：
infra/knowledge/local_knowledge_store.py
这个文件更像是“领域层的知识库服务入口”。

这个函数用于把具体知识库实现注入到领域服务中。
好处是：上层业务不需要知道底层是：
本地 FAISS；
Elasticsearch；
Milvus；
其他远程知识库。
只要它实现 search 和 add_document，就能替换。

调用方式
knowledge_service.get_service().search(...)

---

domain/customer_service_agent/service/order_service.py 的作用是：提供订单查询服务，给客服 Agent 或工具系统查询订单状态、物流、金额、产品等信息。

目前它是一个模拟实现，没有真正连接数据库或外部订单系统。

提供一个模拟订单查询接口
  ↓
返回订单状态、物流、金额、产品、创建时间
  ↓
供 Agent 节点和 MCP 工具调用

---

domain/customer_service_agent/service/short_term_memory_service.py 的作用是：保存和读取短期会话记忆，也就是用户和助手的历史聊天记录。
它主要给多轮对话使用。

---

#### customer_service_agent/service/tools/mcp_server.py
mcp_server.py 是项目里的工具调用框架。
它负责：
定义工具结构
  ↓
注册工具
  ↓
列出工具
  ↓
根据工具名调用工具
  ↓
捕获工具异常
  ↓
记录调用耗时和日志
  ↓
提供简单 JSON-RPC 风格处理

---
#### 历史消息融合节点 只将context 简单的拼接
D:\python\agentProject\smart-cs-fc-agent\domain\customer_service_agent\workflow\nodes\history_fusion_node.py

