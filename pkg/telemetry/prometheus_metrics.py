from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

http_requests_total=Counter("http_requests_total","HTTP requests",("method","route","status"))
http_duration_seconds=Histogram("http_request_duration_seconds","HTTP request duration",("method","route"))
agent_route_total=Counter("agent_route_total","Agent routes",("domain","intent","outcome"))
agent_task_total=Counter("agent_task_total","Agent tasks",("domain","capability","status"))
tool_calls_total=Counter("tool_calls_total","Tool calls",("tool","effect","status"))
tool_duration_seconds=Histogram("tool_call_duration_seconds","Tool duration",("tool","effect"))
pending_actions_total=Counter("pending_actions_total","Pending actions",("status",))
llm_inflight=Gauge("llm_inflight","Executing LLM calls")
llm_queue_depth=Gauge("llm_queue_depth","Queued LLM calls")
llm_queue_wait_seconds=Histogram("llm_queue_wait_seconds","LLM queue wait")
memory_outbox_lag_seconds=Gauge("memory_outbox_lag_seconds","Memory outbox lag")


def metrics_payload() -> tuple[bytes,str]:
    return generate_latest(),CONTENT_TYPE_LATEST
