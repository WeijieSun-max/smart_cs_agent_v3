from application.customer_service.session_ownership import SessionOwnershipService


class Client:
    def __init__(self) -> None:
        self.updates = []

    def execute_query(self, sql, args, fetch_one=False):
        del sql, args, fetch_one
        return True, None

    def execute_update(self, sql, args):
        self.updates.append((sql, args))
        return True, 1


def test_session_ownership_creates_complete_chat_session_row() -> None:
    client = Client()
    service = SessionOwnershipService(client)

    service.bind("user-1", "session-1")

    sql, args = client.updates[0]
    assert "title,agent_id" in sql
    assert "status,version" not in sql
    assert args == ("session-1", "user-1", "新会话", "general")
