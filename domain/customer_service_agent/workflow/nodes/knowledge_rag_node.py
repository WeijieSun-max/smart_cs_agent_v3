from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from domain.customer_service_agent.service import knowledge_service, order_service
from domain.customer_service_agent.workflow.entity.chat_state import ChatState
from domain.shared.llm.llm_service import invoke_llm

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答Agent，负责根据检索到的文档回答用户问题。
回答规则：
1. 严格基于检索到的文档内容回答，不要编造信息。
2. 如果文档中没有相关信息，明确告知用户并建议转人工。
3. 回答要简洁专业，适合客服场景。
4. 对于金融产品信息，必须标注“以上信息仅供参考，具体以合同条款为准”。
5. 在回答末尾标注引用来源。
"""

QUERY_REWRITE_PROMPT = """请将用户的口语化问题改写为适合向量检索的查询语句。
保留核心语义，去除口语化表达，补充专业术语。
只返回改写后的查询。

用户原始问题: {query}
"""


def knowledge_rag_node(state: ChatState) -> dict:
    query = state["raw_query"]
    entities = state.get("entities") or {}
    if "order_id" in entities or "订单" in query:
        order = order_service.get_service().query_order(entities.get("order_id", ""), state.get("user_id") or "")
        answer = f"订单 {order['order_id']} 当前状态：{order['status']}，物流信息：{order['logistics']}。"
        return {
            "sub_results": {**state.get("sub_results", {}), "knowledge_rag": answer},
            "current_agent": "knowledge_rag",
            "node_logs": ["已调用本地订单查询工具"],
        }

    rewritten = invoke_llm(
        [HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=query))],
        run_name="rag.query_rewrite",
    ).content.strip()
    docs = knowledge_service.get_service().search(rewritten, top_k=5)
    reranked = _rerank_documents(rewritten, docs, top_k=3)
    answer = _generate_answer(query, reranked)
    return {
        "sub_results": {**state.get("sub_results", {}), "knowledge_rag": answer},
        "current_agent": "knowledge_rag",
        "node_logs": [f"知识库检索完成，命中文档 {len(reranked)} 条"],
    }


def _rerank_documents(query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    if not documents:
        return []
    return documents[:top_k]


def _generate_answer(query: str, context_docs: list[dict]) -> str:
    if not context_docs:
        return "抱歉，知识库中暂未找到与您问题相关的信息。建议您联系人工客服获取帮助。"
    context = "\n\n---\n\n".join(
        f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')}"
        for doc in context_docs
    )
    response = invoke_llm([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"用户问题: {query}\n\n检索到的参考文档:\n{context}"),
    ], run_name="rag.answer")
    return str(response.content)
