"""电信线路、套餐与零售订单的确定性业务能力。"""

from .service import BusinessService, get_service, initialize_service
from .store import InMemoryBusinessStore

__all__ = ["BusinessService", "InMemoryBusinessStore", "get_service", "initialize_service"]
