from __future__ import annotations

from application.customer_service.action_reconcile_worker import ActionReconcileWorker
from domain.business.service import get_service
from pkg.config.settings import get_settings

_worker: ActionReconcileWorker|None=None
def start_action_worker() -> None:
    global _worker
    if not get_settings().action_reconcile_worker_enabled: return
    _worker=ActionReconcileWorker(get_service().store); _worker.start()
def stop_action_worker() -> None:
    global _worker
    if _worker: _worker.stop(); _worker=None
