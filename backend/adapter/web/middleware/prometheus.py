from __future__ import annotations

import time
from starlette.middleware.base import BaseHTTPMiddleware
from pkg.telemetry.prometheus_metrics import http_duration_seconds,http_requests_total


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        started=time.perf_counter(); status="500"
        try:
            response=await call_next(request); status=str(response.status_code); return response
        finally:
            route=getattr(request.scope.get("route"),"path",None) or "unmatched"
            http_requests_total.labels(request.method,route,status).inc()
            http_duration_seconds.labels(request.method,route).observe(time.perf_counter()-started)
