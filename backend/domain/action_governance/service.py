"""对业务工具执行统一的确定性治理，隔离模型决策与真实副作用。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from domain.shared.identity import RequestIdentityContext
from domain.business.service import BusinessService
from domain.customer_service_agent.file_skills.models import LoadedSkill
from domain.customer_service_agent.tools.mcp_server import MCPToolServer, ToolDefinition
from pkg.exceptions.exception import RequestConflictError, ToolValidationError
from pkg.telemetry.prometheus_metrics import pending_actions_total

from .models import ActionEnvelope

_RESERVED = frozenset({"user_id", "tenant_id", "session_id", "turn_id", "request_id", "action_id", "idempotency_key"})


class GovernedActionService:
    """所有业务读工具和写工具的确定性执行边界。

    模型只能选择工具并生成候选参数；本服务负责工具/Skill 白名单、JSON
    Schema、用户状态、参数摘要、确认策略、幂等键和超时状态。任何校验失败
    都会关闭执行路径，而不会尝试猜测或修复参数。
    """

    def __init__(self, server: MCPToolServer, business: BusinessService, *, ttl_seconds: int = 900, tool_timeout_seconds: float = 30.0):
        self.server = server
        self.business = business
        self.ttl_seconds = ttl_seconds
        self.tool_timeout_seconds = tool_timeout_seconds

    async def execute_read(self, tool_name: str, arguments: dict[str, Any], identity: RequestIdentityContext, *, skill: LoadedSkill | None = None) -> Any:
        """校验并执行无副作用工具，瞬时失败最多重试三次。"""

        definition, normalized = self._validate(tool_name, arguments, identity, skill)
        if definition.effect != "read":
            raise ToolValidationError()
        last = None
        for attempt in range(3):
            result = await self.server.call_tool(tool_name, normalized, trusted_context=self._trusted(identity))
            if result.success:
                return deepcopy(result.result)
            last = result
            if attempt < 2:
                await asyncio.sleep(0)
        raise ToolValidationError() if last and last.error_code == "tool.validation" else RuntimeError("read tool unavailable")

    def propose_write(self, tool_name: str, arguments: dict[str, Any], identity: RequestIdentityContext, *, impact_summary: str, skill: LoadedSkill | None = None) -> ActionEnvelope:
        """冻结写工具参数并创建待确认动作，但不执行任何业务变更。

        提议阶段会第一次检查用户是否有效。存储层同时保证同一用户会话只能
        有一个活跃动作，避免多个确认问题互相覆盖。
        """

        definition, normalized = self._validate(tool_name, arguments, identity, skill)
        if definition.effect != "write" or definition.confirmation_policy != "always" or not definition.supports_idempotency:
            raise ToolValidationError()
        self.business.require_active_user(identity.user_id)  # first mandatory lookup
        now = datetime.now(timezone.utc)
        action_id = uuid4().hex[:26]
        payload = {
            "user_id": identity.user_id,
            "session_id": identity.session_id,
            "tool_name": definition.name,
            "tool_version": definition.version,
            "skill_name": skill.metadata.name if skill else None,
            "skill_version": skill.metadata.version if skill else None,
            "arguments": normalized,
        }
        envelope = ActionEnvelope(
            action_id=action_id,
            user_id=identity.user_id,
            session_id=identity.session_id,
            turn_id=identity.turn_id,
            tool_name=definition.name,
            tool_version=definition.version,
            skill_name=payload["skill_name"],
            skill_version=payload["skill_version"],
            arguments=normalized,
            arguments_digest=_digest(payload),
            impact_summary=impact_summary,
            status="awaiting_confirmation",
            idempotency_key=f"business-action:{action_id}",
            resource_version=_expected_version(normalized),
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
        )
        stored = self.business.store.create_pending(envelope.model_dump(mode="python"))
        pending_actions_total.labels("awaiting_confirmation").inc()
        return ActionEnvelope.model_validate(stored)

    def get_active(self, identity: RequestIdentityContext) -> ActionEnvelope | None:
        """读取当前用户、当前会话下尚未终结的治理动作。"""

        item = self.business.store.get_active_pending(identity.user_id, identity.session_id)
        return ActionEnvelope.model_validate(item) if item else None

    def reject(self, identity: RequestIdentityContext) -> ActionEnvelope:
        """将仍在等待确认的动作原子地转换为已拒绝。"""

        active = self.get_active(identity)
        if active is None or active.status != "awaiting_confirmation":
            raise RequestConflictError()
        item = self.business.store.transition_pending(active.action_id, identity.user_id, identity.session_id, "awaiting_confirmation", "rejected")
        pending_actions_total.labels("rejected").inc()
        return ActionEnvelope.model_validate(item)

    async def confirm(self, identity: RequestIdentityContext) -> ActionEnvelope:
        """重新验证已冻结动作并至多执行一次真实写操作。

        确认时再次检查用户、工具版本、参数摘要和 Schema。执行超时不能证明
        写入失败，因此状态转为 `indeterminate`，交由对账任务查询幂等回执，
        而不是盲目重试。
        """

        active = self.get_active(identity)
        if active is None or active.status != "awaiting_confirmation":
            raise RequestConflictError()
        self.business.require_active_user(identity.user_id)  # mandatory second lookup after confirmation
        definition = self.server.get_tool(active.tool_name)
        if definition is None or definition.version != active.tool_version or definition.effect != "write":
            raise RequestConflictError()
        payload = {
            "user_id": active.user_id, "session_id": active.session_id,
            "tool_name": active.tool_name, "tool_version": active.tool_version,
            "skill_name": active.skill_name, "skill_version": active.skill_version,
            "arguments": active.arguments,
        }
        if _digest(payload) != active.arguments_digest:
            raise RequestConflictError()
        self.server.validate_arguments(active.tool_name, active.arguments)
        self.business.store.transition_pending(active.action_id, active.user_id, active.session_id, "awaiting_confirmation", "executing")
        try:
            async with asyncio.timeout(self.tool_timeout_seconds):
                result = await self.server.call_tool(
                    active.tool_name,
                    deepcopy(active.arguments),
                    trusted_context={**self._trusted(identity), "governed_action": True, "action_id": active.action_id, "idempotency_key": active.idempotency_key},
                )
        except TimeoutError:
            item = self.business.store.transition_pending(active.action_id, active.user_id, active.session_id, "executing", "indeterminate", error_code="action.write_status_unknown")
            return ActionEnvelope.model_validate(item)
        if not result.success:
            item = self.business.store.transition_pending(active.action_id, active.user_id, active.session_id, "executing", "failed", error_code=result.error_code or "action.execution_failed")
            return ActionEnvelope.model_validate(item)
        item = self.business.store.transition_pending(
            active.action_id, active.user_id, active.session_id, "executing", "succeeded",
            receipt_json=result.result, executed_at=datetime.now(timezone.utc),
        )
        pending_actions_total.labels("succeeded").inc()
        return ActionEnvelope.model_validate(item)

    def _validate(self, tool_name: str, arguments: dict[str, Any], identity: RequestIdentityContext, skill: LoadedSkill | None) -> tuple[ToolDefinition, dict[str, Any]]:
        """校验工具及 Skill 授权，并剥离模型不得控制的保留身份参数。"""

        definition = self.server.get_tool(tool_name)
        if definition is None:
            raise ToolValidationError()
        if definition.allowed_agent_types and skill and not set(skill.metadata.allowed_agent_types).intersection(definition.allowed_agent_types):
            raise ToolValidationError()
        if skill and tool_name not in skill.metadata.allowed_tools:
            raise ToolValidationError()
        normalized = _strip_reserved(arguments)
        self.server.validate_arguments(tool_name, normalized)
        return definition, normalized

    # 提取identity的用户id, 会话id等消息, 构造仅由应用侧注入、不会暴露给模型修改的工具上下文
    @staticmethod
    def _trusted(identity: RequestIdentityContext) -> dict[str, Any]:
        """构造仅由应用侧注入、不会暴露给模型修改的工具上下文。"""

        return {"user_id": identity.user_id, "session_id": identity.session_id, "turn_id": identity.turn_id, "auth_strength": identity.auth_strength}


def _strip_reserved(arguments: dict[str, Any]) -> dict[str, Any]:
    """删除根层保留字段，并拒绝把保留身份字段藏入嵌套参数。"""

    if not isinstance(arguments, dict):
        raise ToolValidationError()
    def visit(value: Any, *, root: bool = False) -> Any:
        """递归复制参数，并在任意嵌套对象中检测保留字段。"""

        if isinstance(value, dict):
            if not root and any(key in _RESERVED for key in value):
                raise ToolValidationError()
            return {key: visit(item) for key, item in value.items() if root is False or key not in _RESERVED}
        if isinstance(value, list): return [visit(item) for item in value]
        return value
    return visit(deepcopy(arguments), root=True)


def _digest(payload: dict[str, Any]) -> str:
    """对规范化 JSON 计算稳定摘要，作为确认前后的防篡改凭据。"""

    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _expected_version(arguments: dict[str, Any]) -> int | None:
    """提取合法的乐观锁版本；布尔值虽是 int 子类但不属于有效版本。"""

    value = arguments.get("expected_version")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


_action_service: GovernedActionService | None = None


def initialize_action_service(service: GovernedActionService) -> None:
    """在基础设施启动完成后安装进程级治理服务。"""

    global _action_service
    _action_service = service


def get_action_service() -> GovernedActionService:
    """返回已初始化服务；启动顺序错误时立即失败。"""

    if _action_service is None:
        raise RuntimeError("governed action service is not initialized")
    return _action_service
