import unittest
import uuid
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError

from api.chat import delete_conversations_batch
from models.schemas import ConversationBatchDeleteRequest


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _BatchDeleteDB:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executions = []
        self.commits = 0

    async def execute(self, statement):
        self.executions.append(statement)
        return _Result(self.rows)

    async def commit(self):
        self.commits += 1


class ChatBatchDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_delete_is_atomic_and_deduplicates_ids(self) -> None:
        user_id = uuid.uuid4()
        first = SimpleNamespace(id=uuid.uuid4(), user_id=user_id)
        second = SimpleNamespace(id=uuid.uuid4(), user_id=user_id)
        db = _BatchDeleteDB([second, first])
        payload = ConversationBatchDeleteRequest(
            conversation_ids=[first.id, second.id, first.id]
        )

        result = await delete_conversations_batch(
            payload=payload,
            db=db,
            user=SimpleNamespace(id=user_id, is_superadmin=False),
        )

        self.assertEqual(len(db.executions), 2)
        self.assertIn("conversations.user_id", str(db.executions[0]))
        self.assertEqual(db.commits, 1)
        self.assertEqual(result["deleted_count"], 2)
        self.assertEqual(result["deleted_ids"], [str(first.id), str(second.id)])

    async def test_batch_delete_rejects_whole_request_when_any_id_is_unavailable(self) -> None:
        user_id = uuid.uuid4()
        accessible = SimpleNamespace(id=uuid.uuid4(), user_id=user_id)
        unavailable_id = uuid.uuid4()
        db = _BatchDeleteDB([accessible])
        payload = ConversationBatchDeleteRequest(
            conversation_ids=[accessible.id, unavailable_id]
        )

        with self.assertRaises(HTTPException) as captured:
            await delete_conversations_batch(
                payload=payload,
                db=db,
                user=SimpleNamespace(id=user_id, is_superadmin=False),
            )

        self.assertEqual(captured.exception.status_code, 404)
        self.assertEqual(len(db.executions), 1)
        self.assertEqual(db.commits, 0)

    def test_batch_delete_requires_at_least_one_bounded_id(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationBatchDeleteRequest(conversation_ids=[])
        with self.assertRaises(ValidationError):
            ConversationBatchDeleteRequest(
                conversation_ids=[uuid.uuid4() for _ in range(101)]
            )


if __name__ == "__main__":
    unittest.main()
