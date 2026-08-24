from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING,Any

from pkg.exceptions.exception import RequestConflictError, StorageOperationError

if TYPE_CHECKING:
    from infra.db.mysql_client import MySQLClient


class SessionOwnershipService:
    def __init__(self, client: "MySQLClient | None"):
        self.client=client; self._owners: dict[str,str]={}; self._lock=threading.Lock()

    def bind(self,user_id: str,session_id: str) -> None:
        if self.client is None:
            with self._lock:
                owner=self._owners.setdefault(session_id,user_id)
                if owner != user_id: raise RequestConflictError()
            return
        ok,row=self.client.execute_query("SELECT user_id FROM cs_sessions WHERE session_id=%s",(session_id,),fetch_one=True)
        if not ok: raise StorageOperationError()
        if row and row["user_id"] != user_id: raise RequestConflictError()
        if not row:
            ok,_=self.client.execute_update("INSERT INTO cs_sessions(session_id,user_id,status,version,created_at,updated_at) VALUES (%s,%s,'active',1,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))",(session_id,user_id))
            if not ok:
                ok,row=self.client.execute_query("SELECT user_id FROM cs_sessions WHERE session_id=%s",(session_id,),fetch_one=True)
                if not ok or not row or row["user_id"] != user_id: raise RequestConflictError()

    def require_owner(self,user_id: str,session_id: str) -> None:
        self.bind(user_id,session_id)


_service: SessionOwnershipService | None=None
def initialize_session_ownership(client: "MySQLClient | None") -> None:
    global _service; _service=SessionOwnershipService(client)
def get_session_ownership() -> SessionOwnershipService:
    global _service
    if _service is None: _service=SessionOwnershipService(None)
    return _service
