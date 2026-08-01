import unittest

from models.schemas import SearchConfig, SearchRequest


class SearchSchemaTests(unittest.TestCase):
    def test_user_selected_tags_are_not_part_of_search_contract(self) -> None:
        config = SearchConfig.model_validate(
            {
                "method": "hybrid",
                "rerank": True,
                "top_k": 5,
                # Old clients may still send the retired field during a rolling
                # deployment.  It must be ignored rather than affect ranking.
                "tags": ["重点"],
            }
        )
        request = SearchRequest.model_validate(
            {
                "query": "普通员工餐补标准",
                "tags": ["重点"],
            }
        )

        self.assertNotIn("tags", SearchConfig.model_fields)
        self.assertNotIn("tags", SearchRequest.model_fields)
        self.assertNotIn("tags", config.model_dump())
        self.assertNotIn("tags", request.model_dump())
