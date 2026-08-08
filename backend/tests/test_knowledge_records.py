import asyncio
import uuid

from core.knowledge_records import (
    extract_knowledge_records,
    search_knowledge_records_for_chunks,
)


class _Rows:
    def __init__(self, values):
        self._values = values

    def mappings(self):
        return self

    def all(self):
        return self._values


class _CapturingDatabase:
    def __init__(self):
        self.statement = None
        self.parameters = None

    async def execute(self, statement, parameters):
        self.statement = statement
        self.parameters = parameters
        return _Rows([
            {
                "record_id": uuid.UUID("44444444-4444-4444-8444-444444444444"),
                "content": "分页查询组织下的员工 Code /mozi/employee/page",
            }
        ])


def test_extracts_generic_markdown_table_rows() -> None:
    records = extract_knowledge_records(
        """
        【接口目录】
        | 接口说明 | 接口地址 |
        | :--- | :--- |
        | 批量根据员工账号ID获取员工Code | /mozi/employee/listGovEmployeeCodesByAccountIds |
        | 根据部门编码列表获取部门信息 | /mozi/organization/listOrganizationsByCodes |
        """
    )
    assert len(records) == 2
    assert records[0]["record_type"] == "table_row"
    assert records[0]["subject"] == "批量根据员工账号ID获取员工Code"
    assert records[0]["object_value"] == "/mozi/employee/listGovEmployeeCodesByAccountIds"


def test_ignores_non_table_text_and_separator_rows() -> None:
    assert extract_knowledge_records("普通段落，没有结构化表格。") == []


def test_chunk_record_search_is_scoped_deduplicated_and_bounded() -> None:
    first = uuid.UUID("11111111-1111-4111-8111-111111111111")
    second = uuid.UUID("22222222-2222-4222-8222-222222222222")
    db = _CapturingDatabase()

    rows = asyncio.run(
        search_knowledge_records_for_chunks(
            db,
            "查询用户列表接口",
            [first, second, first],
            top_k=99,
        )
    )

    assert len(rows) == 1
    assert "kr.chunk_id = ANY(:chunk_ids)" in str(db.statement)
    assert db.parameters == {
        "query": "查询用户列表接口",
        "chunk_ids": [first, second],
        "top_k": 20,
    }
