"""Ensure every knowledge/document mutation uses its exact CRUD capability."""

import ast
import inspect
import textwrap
import unittest

from fastapi.routing import APIRoute

from api.document import router as document_router
from api.knowledge import router as knowledge_router
from core.permissions import (
    DOC_CREATE,
    DOC_DELETE,
    DOC_READ,
    DOC_UPDATE,
    KB_CREATE,
    KB_DELETE,
    KB_UPDATE,
)


def _route(router, method: str, path: str) -> APIRoute:
    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route
    raise AssertionError(f"route not found: {method} {path}")


def _required_permission_keys(route: APIRoute) -> set[str]:
    """Read permission keys from the nested FastAPI dependency graph."""
    keys: set[str] = set()

    def visit(dependant) -> None:
        for child in dependant.dependencies:
            call = child.call
            if inspect.isfunction(call):
                key = inspect.getclosurevars(call).nonlocals.get("key")
                if isinstance(key, str):
                    keys.add(key)
            visit(child)

    visit(route.dependant)
    return keys


class CrudRouteAuthorizationTests(unittest.TestCase):
    def test_knowledge_mutations_use_independent_capabilities(self) -> None:
        expected = {
            ("POST", "/knowledge/create"): KB_CREATE,
            ("PUT", "/knowledge/{kb_id}"): KB_UPDATE,
            ("DELETE", "/knowledge/{kb_id}"): KB_DELETE,
        }
        for (method, path), permission in expected.items():
            with self.subTest(method=method, path=path):
                self.assertEqual(
                    _required_permission_keys(_route(knowledge_router, method, path)),
                    {permission},
                )

    def test_document_mutations_use_independent_capabilities(self) -> None:
        expected = {
            ("POST", "/knowledge/{kb_id}/documents"): DOC_CREATE,
            ("POST", "/knowledge/{kb_id}/documents/image"): DOC_CREATE,
            ("POST", "/knowledge/{kb_id}/documents/text"): DOC_CREATE,
            ("PUT", "/knowledge/{kb_id}/documents/{doc_id}"): DOC_UPDATE,
            ("PATCH", "/knowledge/{kb_id}/documents/{doc_id}/tags"): DOC_UPDATE,
            ("PATCH", "/knowledge/{kb_id}/documents/{doc_id}/toggle"): DOC_UPDATE,
            ("DELETE", "/knowledge/{kb_id}/documents/{doc_id}"): DOC_DELETE,
        }
        for (method, path), permission in expected.items():
            with self.subTest(method=method, path=path):
                self.assertEqual(
                    _required_permission_keys(_route(document_router, method, path)),
                    {permission},
                )

    def test_document_reads_still_require_read_capability(self) -> None:
        for path in (
            "/knowledge/{kb_id}/documents",
            "/knowledge/{kb_id}/documents/{doc_id}",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    _required_permission_keys(_route(document_router, "GET", path)),
                    {DOC_READ},
                )

    def test_every_document_mutation_enforces_object_action(self) -> None:
        expected = {
            ("PUT", "/knowledge/{kb_id}/documents/{doc_id}"): "update",
            ("PATCH", "/knowledge/{kb_id}/documents/{doc_id}/tags"): "update",
            ("PATCH", "/knowledge/{kb_id}/documents/{doc_id}/toggle"): "update",
            ("DELETE", "/knowledge/{kb_id}/documents/{doc_id}"): "delete",
        }
        for (method, path), action in expected.items():
            with self.subTest(method=method, path=path):
                endpoint = _route(document_router, method, path).endpoint
                tree = ast.parse(textwrap.dedent(inspect.getsource(endpoint)))
                enforced_actions = {
                    node.args[2].value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_require_document_action"
                    and len(node.args) >= 3
                    and isinstance(node.args[2], ast.Constant)
                }
                self.assertEqual(enforced_actions, {action})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
