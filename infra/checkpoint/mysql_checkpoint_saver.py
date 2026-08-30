from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple, ChannelVersions, get_checkpoint_id, get_checkpoint_metadata

from infra.db.mysql_client import MySQLClient
from pkg.exceptions.exception import StorageOperationError


class MySQLCheckpointSaver(BaseCheckpointSaver):
    """Durable LangGraph saver using the platform MySQL pool."""

    def __init__(self, client: MySQLClient):
        super().__init__()
        self.client = client

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        cfg=config["configurable"]; thread_id=cfg["thread_id"]; namespace=cfg.get("checkpoint_ns",""); checkpoint_id=get_checkpoint_id(config)
        sql="SELECT * FROM cs_durable_checkpoints WHERE thread_id=%s AND checkpoint_ns=%s"; args=[thread_id,namespace]
        if checkpoint_id: sql+=" AND checkpoint_id=%s"; args.append(checkpoint_id)
        sql+=" ORDER BY checkpoint_id DESC LIMIT 1"
        ok,row=self.client.execute_query(sql,tuple(args),fetch_one=True)
        if not ok: raise StorageOperationError()
        if not row: return None
        resolved_id=row["checkpoint_id"]
        ok,writes=self.client.execute_query("SELECT task_id,channel,value_blob FROM cs_checkpoint_writes WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s ORDER BY task_id,write_index",(thread_id,namespace,resolved_id))
        if not ok: raise StorageOperationError()
        resolved_config: RunnableConfig={"configurable":{"thread_id":thread_id,"checkpoint_ns":namespace,"checkpoint_id":resolved_id}}
        parent=row.get("parent_checkpoint_id")
        return CheckpointTuple(config=resolved_config,checkpoint=self._loads(row["checkpoint_blob"]),metadata=self._loads(row["metadata_blob"]),pending_writes=[(item["task_id"],item["channel"],self._loads(item["value_blob"])) for item in writes],parent_config={"configurable":{"thread_id":thread_id,"checkpoint_ns":namespace,"checkpoint_id":parent}} if parent else None)

    def list(self, config: RunnableConfig | None, *, filter: dict[str,Any] | None=None, before: RunnableConfig | None=None, limit: int | None=None) -> Iterator[CheckpointTuple]:
        del filter
        if config is None: return iter(())
        cfg=config["configurable"]; sql="SELECT checkpoint_id FROM cs_durable_checkpoints WHERE thread_id=%s AND checkpoint_ns=%s"; args=[cfg["thread_id"],cfg.get("checkpoint_ns","")]
        before_id=get_checkpoint_id(before) if before else None
        if before_id: sql+=" AND checkpoint_id<%s"; args.append(before_id)
        sql+=" ORDER BY checkpoint_id DESC"
        if limit is not None: sql+=" LIMIT %s"; args.append(limit)
        ok,rows=self.client.execute_query(sql,tuple(args))
        if not ok: raise StorageOperationError()
        def iterator() -> Iterator[CheckpointTuple]:
            for row in rows:
                item=self.get_tuple({"configurable":{"thread_id":cfg["thread_id"],"checkpoint_ns":cfg.get("checkpoint_ns",""),"checkpoint_id":row["checkpoint_id"]}})
                if item is not None: yield item
        return iterator()

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        del new_versions
        cfg=config["configurable"]; thread_id=cfg["thread_id"]; namespace=cfg.get("checkpoint_ns",""); checkpoint_id=checkpoint["id"]
        ok,_=self.client.execute_update("INSERT INTO cs_durable_checkpoints(thread_id,checkpoint_ns,checkpoint_id,parent_checkpoint_id,checkpoint_blob,metadata_blob,created_at) VALUES (%s,%s,%s,%s,%s,%s,UTC_TIMESTAMP(6)) ON DUPLICATE KEY UPDATE checkpoint_blob=VALUES(checkpoint_blob),metadata_blob=VALUES(metadata_blob)",(thread_id,namespace,checkpoint_id,cfg.get("checkpoint_id"),self._dumps(checkpoint),self._dumps(get_checkpoint_metadata(config,metadata))))
        if not ok: raise StorageOperationError()
        return {"configurable":{"thread_id":thread_id,"checkpoint_ns":namespace,"checkpoint_id":checkpoint_id}}

    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple[str,Any]], task_id: str, task_path: str="") -> None:
        del task_path
        cfg=config["configurable"]; checkpoint_id=get_checkpoint_id(config)
        if checkpoint_id is None: raise StorageOperationError()
        statements=[("INSERT INTO cs_checkpoint_writes(thread_id,checkpoint_ns,checkpoint_id,task_id,write_index,channel,value_blob) VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE channel=VALUES(channel),value_blob=VALUES(value_blob)",(cfg["thread_id"],cfg.get("checkpoint_ns",""),checkpoint_id,task_id,index,channel,self._dumps(value))) for index,(channel,value) in enumerate(writes)]
        if statements:
            ok,_=self.client.execute_transaction(statements)
            if not ok: raise StorageOperationError()

    def delete_thread(self, thread_id: str) -> None:
        ok,_=self.client.execute_transaction([("DELETE FROM cs_checkpoint_writes WHERE thread_id=%s",(thread_id,)),("DELETE FROM cs_durable_checkpoints WHERE thread_id=%s",(thread_id,))])
        if not ok: raise StorageOperationError()

    def delete_thread_namespace(self, thread_id: str, checkpoint_ns: str) -> None:
        ok,_=self.client.execute_transaction([
            ("DELETE FROM cs_checkpoint_writes WHERE thread_id=%s AND checkpoint_ns=%s",(thread_id,checkpoint_ns)),
            ("DELETE FROM cs_durable_checkpoints WHERE thread_id=%s AND checkpoint_ns=%s",(thread_id,checkpoint_ns)),
        ])
        if not ok: raise StorageOperationError()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: return await asyncio.to_thread(self.get_tuple,config)
    async def alist(self, config: RunnableConfig | None, *, filter: dict[str,Any] | None=None, before: RunnableConfig | None=None, limit: int | None=None) -> AsyncIterator[CheckpointTuple]:
        items=await asyncio.to_thread(lambda:list(self.list(config,filter=filter,before=before,limit=limit)))
        for item in items: yield item
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig: return await asyncio.to_thread(self.put,config,checkpoint,metadata,new_versions)
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str,Any]], task_id: str, task_path: str="") -> None: await asyncio.to_thread(self.put_writes,config,writes,task_id,task_path)
    async def adelete_thread(self, thread_id: str) -> None: await asyncio.to_thread(self.delete_thread,thread_id)

    def _dumps(self,value: Any) -> bytes:
        kind,payload=self.serde.dumps_typed(value)
        return json.dumps({"type":kind,"data":base64.b64encode(payload).decode("ascii")},separators=(",",":")).encode()

    def _loads(self,value: bytes|bytearray|str) -> Any:
        raw=value.decode() if isinstance(value,(bytes,bytearray)) else value; item=json.loads(raw)
        return self.serde.loads_typed((item["type"],base64.b64decode(item["data"])))
