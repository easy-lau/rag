"""Rebuild structured knowledge records for already-ingested documents."""

from __future__ import annotations

import asyncio

from core.knowledge_records import rebuild_knowledge_records
from database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as db:
        count = await rebuild_knowledge_records(db)
        await db.commit()
        print(f"knowledge_records rebuilt: {count}")


if __name__ == "__main__":
    asyncio.run(main())
