"""Legacy routing exports.

Text routing was intentionally removed. All intent and agent selection now comes
from ``agents.supervisor_agent``. Only the registry-derived write vocabulary is
kept here temporarily for callers that import it as metadata.
"""

from __future__ import annotations

from domain.customer_service_agent.orchestration.capability_index import get_capability_index

WRITE_CAPABILITIES = get_capability_index().write_capabilities


def route_request(_query: str):
    raise RuntimeError("rule routing was removed; invoke the LLM supervisor")
