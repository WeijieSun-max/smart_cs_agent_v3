from __future__ import annotations

from application.customer_service.action_reconcile_worker import ActionReconcileWorker


class Store:
    def __init__(self, *, receipt=None, fail=False) -> None:
        self.receipt = receipt
        self.fail = fail
        self.transitions = []
        self.receipt_lookups = 0

    def list_indeterminate(self, limit):
        assert limit == 20
        if self.fail:
            raise RuntimeError("storage unavailable")
        return [{
            "action_id": "action-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "idempotency_key": "key-1",
        }]

    def get_receipt(self, key):
        assert key == "key-1"
        self.receipt_lookups += 1
        return self.receipt

    def transition_pending(self, *args, **kwargs):
        self.transitions.append((args, kwargs))


def test_reconcile_marks_indeterminate_action_succeeded_from_existing_receipt() -> None:
    store = Store(receipt={"status": "succeeded"})
    worker = ActionReconcileWorker(store)

    count = worker.run_once()

    assert count == 1
    assert store.receipt_lookups == 1
    assert len(store.transitions) == 1
    args, kwargs = store.transitions[0]
    assert args[3:5] == ("indeterminate", "succeeded")
    assert kwargs["receipt_json"] == {"status": "succeeded"}


def test_reconcile_without_receipt_never_replays_or_transitions() -> None:
    store = Store(receipt=None)
    worker = ActionReconcileWorker(store)

    count = worker.run_once()

    assert count == 0
    assert store.receipt_lookups == 1
    assert store.transitions == []


def test_reconcile_storage_error_is_observable_and_non_fatal() -> None:
    worker = ActionReconcileWorker(Store(fail=True))

    assert worker._run_safely() == 0
    status = worker.health_status()

    assert status["status"] == "degraded"
    assert status["last_error_code"]


def test_reconcile_worker_starts_and_stops_cleanly() -> None:
    worker = ActionReconcileWorker(Store(receipt=None), poll_seconds=0.01)

    worker.start()
    assert worker.health_status()["running"] is True
    worker.stop()

    assert worker.health_status()["running"] is False
