### 基础设施层，封装 Redis、MySQL、Qdrant、LLM、知识库等外部依赖

---

infra/cache/redis_client.py 的作用是：初始化并提供 Redis 客户端，供项目里的缓存/短期记忆等模块使用。
Redis 主要用于短期会话记忆。如果 Redis 不可用，项目仍然可以启动，只是记忆会退化为进程内内存，服务重启后历史会丢失

---

infra/db/mysql_client.py 的作用是：初始化并提供 MySQL 客户端连接池，供项目里的工单仓储等模块访问数据库。
主要服务于项目里的客服工单存储

mysql_client 是单例类
单例类是什么：一个 Python 进程里，这个类只允许创建一个对象实例

普通类每次调用都会创建新对象：
a = User()
b = User()
print(a is b)  # False

单例类则希望做到：
a = MySQLClient(...)
b = MySQLClient(...)
print(a is b)  # True

也就是说，a 和 b 指向的是同一个对象。

__new__ 是 Python 创建对象时最先执行的方法。
创建对象大概分两步：
第 1 步：__new__ 创建对象
第 2 步：__init__ 初始化对象

__new__ 负责“要不要创建新对象 / 返回哪个对象”
__init__ 负责“给这个对象设置属性”

---

infra/checkpoint/ 的作用是：初始化 LangGraph 的 checkpoint 保存器。
简单说，它负责给 Agent 工作流提供“状态保存能力”。
它和普通“聊天历史记忆”不完全一样：
* 短期记忆在 short_term_memory_service.py，主要保存用户/助手消息。
* checkpoint 是 LangGraph 用来保存工作流执行状态的，比如某个 thread_id 下的图状态、中间节点状态等。

所以 checkpoint 只保存在当前 Python 进程内存里：
* 服务重启后会丢失；
* 多进程之间不共享；
* 适合本地开发和示例项目；
* 生产环境通常会换成 Redis、数据库或其他持久化 checkpoint saver。

---

配置大模型
D:\python\agentProject\smart-cs-fc-agent\infra\llm

---

infra/knowledge/local_knowledge_store.py 的作用是：实现一个本地知识库，用于添加文档、检索文档，并可选地用 FAISS 做向量检索

faiss
它的全称是 Facebook AI Similarity Search，是 Meta 开源的一个高性能向量检索库，主要用于对大量 embedding 向量 做快速查找、聚类和近邻搜索。官方描述里也说，Faiss 用于高效的 dense vector similarity search 和 clustering，并且提供 Python/Numpy 接口，部分算法支持 GPU 加速

知识库：
"退款政策：用户在购买后7天内可申请无理由退款，超过7天需提供合理原因。退款将在3-5个工作日内原路退回。",

---

infra/customer_service/ticket_repository.py 的作用是：封装客服工单的数据存储逻辑。
它负责把工单保存到 MySQL 表 cs_tickets，也支持在 MySQL 不可用时退回到内存字典保存。

存储工单
首先判断mysql服务是否正常/ 表是否存在，不存在先创建表格
实现了三种操作 / 创建/查询/更新

---

