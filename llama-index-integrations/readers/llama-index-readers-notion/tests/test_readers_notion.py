"""Unit tests for NotionPageReader with mocked Notion API requests."""

from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader, BaseReader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.notion.base import PRINT_PREFIX, iterate_paginated_api


def _mock_response(
    json_data: Dict[str, Any],
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
    text: str = "",
) -> MagicMock:
    """Build a mocked requests.Response object."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"HTTP {status_code}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _paragraph_block(
    block_id: str,
    text: str,
    has_children: bool = False,
) -> Dict[str, Any]:
    """Build a Notion paragraph block payload."""
    rich_text: List[Dict[str, Any]] = (
        [{"type": "text", "text": {"content": text}}] if text else []
    )
    return {
        "id": block_id,
        "type": "paragraph",
        "has_children": has_children,
        "paragraph": {"rich_text": rich_text},
    }


@pytest.fixture(autouse=True)
def _block_unmocked_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any accidental live HTTP calls during unit tests."""

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"Unmocked network request attempted: {args} {kwargs}")

    monkeypatch.setattr(requests, "request", _forbid)


# -------------------------------------------------------------------------
# Class & Initialization Tests


def test_class() -> None:
    """Verify reader class hierarchy and identifier."""
    names_of_base_classes = [b.__name__ for b in NotionPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
    assert BasePydanticReader.__name__ in names_of_base_classes
    assert NotionPageReader.class_name() == "NotionPageReader"


def test_init_with_token_and_default_flags() -> None:
    """Verify default flags preserve upstream behaviour."""
    reader = NotionPageReader(integration_token="secret_tok")
    assert reader.token == "secret_tok"
    assert reader.headers == {
        "Authorization": "Bearer secret_tok",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    assert reader.is_remote is True
    assert reader.print_feedback is False
    assert reader.include_db_content is False
    assert reader.include_pages_in_databases is True
    assert reader.filter_empty_pages is False

    serialized = reader.to_dict()
    assert serialized["print_feedback"] is False
    assert serialized["include_db_content"] is False
    assert serialized["include_pages_in_databases"] is True
    assert serialized["filter_empty_pages"] is False


def test_init_from_env_and_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify token fallback to NOTION_INTEGRATION_TOKEN and error when unset."""
    monkeypatch.delenv("NOTION_INTEGRATION_TOKEN", raising=False)
    with pytest.raises(ValueError, match="NOTION_INTEGRATION_TOKEN"):
        NotionPageReader()

    monkeypatch.setenv("NOTION_INTEGRATION_TOKEN", "env_secret")
    reader = NotionPageReader()
    assert reader.token == "env_secret"


# -------------------------------------------------------------------------
# Pagination (>100 items) Tests


def test_iterate_paginated_api_beyond_100_items() -> None:
    """Verify generic pagination helper iterates across >100 results."""
    calls: List[Optional[str]] = []

    def fake_endpoint(
        start_cursor: Optional[str] = None, extra: str = ""
    ) -> Dict[str, Any]:
        assert extra == "ok"
        calls.append(start_cursor)
        if start_cursor is None:
            return {
                "results": [{"id": f"item-{i}"} for i in range(100)],
                "has_more": True,
                "next_cursor": "cursor-100",
            }
        if start_cursor == "cursor-100":
            return {
                "results": [{"id": f"item-{i}"} for i in range(100, 200)],
                "has_more": True,
                "next_cursor": "cursor-200",
            }
        return {
            "results": [{"id": f"item-{i}"} for i in range(200, 235)],
            "has_more": False,
            "next_cursor": None,
        }

    items = list(iterate_paginated_api(fake_endpoint, extra="ok"))
    assert len(items) == 235
    assert items[0]["id"] == "item-0"
    assert items[-1]["id"] == "item-234"
    assert calls == [None, "cursor-100", "cursor-200"]


def test_read_page_pagination_beyond_100_blocks_and_nested_children() -> None:
    """
    Verify _read_block passes start_cursor as a query param on the same block URL
    (fixing upstream's bug of substituting next_cursor into the block_id path)
    and indents nested child blocks.
    """
    reader = NotionPageReader(integration_token="tok")

    first_100 = [_paragraph_block(f"b-{i}", f"Line {i}") for i in range(100)]
    # Give b-0 a nested child and include a non-text rich_text span (e.g. equation)
    first_100[0]["has_children"] = True
    first_100[1]["paragraph"]["rich_text"].append(
        {"type": "equation", "equation": {"expression": "E=mc^2"}}
    )
    remaining_25 = [_paragraph_block(f"b-{i}", f"Line {i}") for i in range(100, 125)]

    def mock_http(
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        assert method == "GET"
        assert json is None
        if url == "https://api.notion.com/v1/blocks/b-0/children":
            return _mock_response(
                {
                    "results": [_paragraph_block("child-1", "Nested child")],
                    "has_more": False,
                    "next_cursor": None,
                }
            )
        assert url == "https://api.notion.com/v1/blocks/page-big/children"
        assert params is not None
        if params.get("start_cursor") is None:
            return _mock_response(
                {
                    "results": first_100,
                    "has_more": True,
                    "next_cursor": "cursor-block-100",
                }
            )
        assert params["start_cursor"] == "cursor-block-100"
        return _mock_response(
            {
                "results": remaining_25,
                "has_more": False,
                "next_cursor": None,
            }
        )

    with patch("requests.request", side_effect=mock_http) as mock_req:
        text = reader.read_page("page-big")

    assert mock_req.call_count == 3
    lines = text.split("\n")
    assert lines[0] == "Line 0"
    assert lines[1] == "\tNested child"
    assert lines[2] == "Line 1"
    assert lines[-1] == "Line 124"
    assert len(lines) == 126


def test_list_pages_and_list_databases_paginate_beyond_100() -> None:
    """Verify list_pages and list_databases fetch >100 items and deduplicate in order."""
    reader = NotionPageReader(integration_token="tok")

    def mock_http(
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        assert method == "POST"
        assert url == "https://api.notion.com/v1/search"
        assert json is not None
        obj_type = json["filter"]["value"]
        cursor = json.get("start_cursor")
        if cursor is None:
            return _mock_response(
                {
                    "results": [{"id": f"{obj_type}-{i}"} for i in range(100)],
                    "has_more": True,
                    "next_cursor": f"{obj_type}-c1",
                }
            )
        if cursor == f"{obj_type}-c1":
            # Include a duplicate item-99 to verify order-preserving deduplication
            batch = [{"id": f"{obj_type}-99"}] + [
                {"id": f"{obj_type}-{i}"} for i in range(100, 145)
            ]
            return _mock_response(
                {
                    "results": batch,
                    "has_more": False,
                    "next_cursor": None,
                }
            )
        raise AssertionError(f"Unexpected cursor: {cursor}")

    with patch("requests.request", side_effect=mock_http):
        pages = reader.list_pages()
        dbs = reader.list_databases()

    assert pages == [f"page-{i}" for i in range(145)]
    assert dbs == [f"database-{i}" for i in range(145)]


def test_query_database_and_search_pagination_beyond_100() -> None:
    """
    Verify query_database paginates >100 rows without mutating query_dict across
    subsequent database queries, and search paginates >100 results.
    """
    reader = NotionPageReader(integration_token="tok")
    caller_query: Dict[str, Any] = {
        "filter": {"property": "Done", "checkbox": {"equals": True}}
    }

    def mock_http(
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        assert method == "POST"
        assert json is not None
        cursor = json.get("start_cursor")

        if url == "https://api.notion.com/v1/databases/db-1/query":
            if cursor is None:
                return _mock_response(
                    {
                        "results": [{"id": f"db1-p-{i}"} for i in range(100)],
                        "has_more": True,
                        "next_cursor": "db1-cursor",
                    }
                )
            assert cursor == "db1-cursor"
            return _mock_response(
                {
                    "results": [{"id": f"db1-p-{i}"} for i in range(100, 130)],
                    "has_more": False,
                    "next_cursor": None,
                }
            )

        if url == "https://api.notion.com/v1/databases/db-2/query":
            # Ensure db-1's cursor never leaked into db-2's default query_dict
            assert "start_cursor" not in json
            return _mock_response(
                {
                    "results": [{"id": "db2-p-0"}],
                    "has_more": False,
                    "next_cursor": None,
                }
            )

        if url == "https://api.notion.com/v1/search":
            assert json["query"] == "architecture"
            if cursor is None:
                return _mock_response(
                    {
                        "results": [{"id": f"s-{i}"} for i in range(100)],
                        "has_more": True,
                        "next_cursor": "s-cursor",
                    }
                )
            return _mock_response(
                {
                    "results": [{"id": f"s-{i}"} for i in range(100, 115)],
                    "has_more": False,
                    "next_cursor": None,
                }
            )

        raise AssertionError(f"Unexpected URL: {url}")

    with patch("requests.request", side_effect=mock_http):
        db1_ids = reader.query_database("db-1", query_dict=caller_query)
        db2_ids = reader.query_database("db-2")
        search_ids = reader.search("architecture")

    assert "start_cursor" not in caller_query
    assert len(db1_ids) == 130
    assert db2_ids == ["db2-p-0"]
    assert len(search_ids) == 115


# -------------------------------------------------------------------------
# Database Content Download & Formatting Tests


def test_database_content_download_and_formatting_beyond_100_rows() -> None:
    """Verify Notion database row formatting, >100 row pagination, and custom formatters."""
    reader = NotionPageReader(integration_token="tok")

    def make_row(idx: int) -> Dict[str, Any]:
        return {
            "id": f"row-{idx}",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": f"Task {idx}"}],
                },
                "Done": {"type": "checkbox", "checkbox": idx % 2 == 0},
                "Tags": {"type": "multi_select", "multi_select": [{"name": "eng"}]},
                "Score": {"type": "number", "number": idx},
            },
        }

    def mock_http(
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        assert url == "https://api.notion.com/v1/databases/db-tasks/query"
        assert json is not None
        if json.get("start_cursor") is None:
            return _mock_response(
                {
                    "results": [make_row(i) for i in range(100)],
                    "has_more": True,
                    "next_cursor": "rows-100",
                }
            )
        return _mock_response(
            {
                "results": [make_row(i) for i in range(100, 112)],
                "has_more": False,
                "next_cursor": None,
            }
        )

    with patch("requests.request", side_effect=mock_http):
        db_doc = reader.get_notion_database("db-tasks")
        custom_doc = reader.get_notion_database(
            "db-tasks",
            format_db_json=lambda data: f"rows={len(data['results'])}",
        )

    assert db_doc.id_ == "db-tasks"
    assert db_doc.metadata == {"database_id": "db-tasks"}
    assert "Notion Database Start" in db_doc.text
    assert "Notion Database End" in db_doc.text
    assert "Name,type:title,data:[{'plain_text': 'Task 0'}]" in db_doc.text
    assert "Name,type:title,data:[{'plain_text': 'Task 111'}]" in db_doc.text
    assert "Done,type:checkbox,data:True" in db_doc.text
    # Unknown type ("number") falls back to full prop dict stringification
    assert "Score,type:number,data:{'type': 'number', 'number': 111}" in db_doc.text
    assert db_doc.text.count("New row") == 112

    assert custom_doc.text == "rows=112"


# -------------------------------------------------------------------------
# Feature Flags (Both States) & Backwards Compatibility Tests


def _workspace_mock_http(
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    """Shared mock Notion workspace with 1 database, 1 standalone page, 1 db page, and 1 empty page."""
    if url == "https://api.notion.com/v1/search":
        assert json is not None
        obj_value = json["filter"]["value"]
        if obj_value == "database":
            return _mock_response(
                {"results": [{"id": "db-1"}], "has_more": False, "next_cursor": None}
            )
        return _mock_response(
            {
                "results": [
                    {"id": "page-standalone", "parent": {"type": "workspace"}},
                    {
                        "id": "page-in-db",
                        "parent": {"type": "database_id", "database_id": "db-1"},
                    },
                ],
                "has_more": False,
                "next_cursor": None,
            }
        )

    if url == "https://api.notion.com/v1/databases/db-1/query":
        return _mock_response(
            {
                "results": [
                    {
                        "id": "page-in-db",
                        "properties": {
                            "Title": {
                                "type": "title",
                                "title": [{"plain_text": "Row 1"}],
                            },
                            "Owner": {
                                "type": "email",
                                "email": "dev@example.com",
                            },
                        },
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            }
        )

    if url == "https://api.notion.com/v1/blocks/page-standalone/children":
        return _mock_response(
            {
                "results": [
                    _paragraph_block("b1", "Hello"),
                    _paragraph_block("b2", "World"),
                ],
                "has_more": False,
                "next_cursor": None,
            }
        )

    if url == "https://api.notion.com/v1/blocks/page-in-db/children":
        return _mock_response(
            {
                "results": [_paragraph_block("b3", "Database child page body")],
                "has_more": False,
                "next_cursor": None,
            }
        )

    if url == "https://api.notion.com/v1/blocks/page-empty/children":
        return _mock_response({"results": [], "has_more": False, "next_cursor": None})

    if url == "https://api.notion.com/v1/blocks/page-whitespace/children":
        return _mock_response(
            {
                "results": [_paragraph_block("b4", "   ")],
                "has_more": False,
                "next_cursor": None,
            }
        )

    raise AssertionError(f"Unexpected request: {method} {url}")


def test_backwards_compatible_default_load_data() -> None:
    """
    Verify default load_data matches upstream NotionPageReader behaviour:
    - empty args with load_all_if_empty=False raises ValueError
    - page_ids preserves exact line-joined text and empty-page Documents
    - database_ids loads child pages without emitting a database table Document
    - overlapping page_ids and database child page_ids are deduplicated
    """
    reader = NotionPageReader(integration_token="tok")

    with pytest.raises(
        ValueError, match="Must specify either `page_ids` or `database_ids`"
    ):
        reader.load_data()

    with patch("requests.request", side_effect=_workspace_mock_http):
        page_docs = reader.load_data(page_ids=["page-standalone", "page-empty"])
        assert len(page_docs) == 2
        assert page_docs[0].id_ == "page-standalone"
        assert page_docs[0].text == "Hello\nWorld"
        assert page_docs[0].metadata == {"page_id": "page-standalone"}
        assert page_docs[1].id_ == "page-empty"
        assert page_docs[1].text == ""

        db_docs = reader.load_data(database_ids=["db-1"])
        assert len(db_docs) == 1
        assert db_docs[0].id_ == "page-in-db"
        assert db_docs[0].metadata == {"page_id": "page-in-db"}
        assert db_docs[0].text == "Database child page body"

        dedup_docs = reader.load_data(
            page_ids=["page-in-db", "page-standalone"],
            database_ids=["db-1"],
        )
        assert [d.id_ for d in dedup_docs] == ["page-in-db", "page-standalone"]

        all_default = reader.load_data(load_all_if_empty=True)
        assert [d.id_ for d in all_default] == ["page-standalone", "page-in-db"]


@pytest.mark.parametrize("via_init", [True, False])
def test_flag_include_db_content_both_states(via_init: bool) -> None:
    """Verify include_db_content in both True and False states (via __init__ and load_data)."""
    reader = NotionPageReader(
        integration_token="tok",
        include_db_content=via_init,
    )

    with patch("requests.request", side_effect=_workspace_mock_http):
        if via_init:
            docs_enabled = reader.load_data(database_ids=["db-1"])
            docs_overridden_off = reader.load_data(
                database_ids=["db-1"], include_db_content=False
            )
            assert [d.id_ for d in docs_enabled] == ["db-1", "page-in-db"]
            assert docs_enabled[0].metadata == {"database_id": "db-1"}
            assert "Owner,type:email,data:dev@example.com" in docs_enabled[0].text
            assert [d.id_ for d in docs_overridden_off] == ["page-in-db"]
        else:
            docs_disabled = reader.load_data(database_ids=["db-1"])
            docs_overridden_on = reader.load_data(
                database_ids=["db-1"], include_db_content=True
            )
            assert [d.id_ for d in docs_disabled] == ["page-in-db"]
            assert [d.id_ for d in docs_overridden_on] == ["db-1", "page-in-db"]


@pytest.mark.parametrize("via_init", [True, False])
def test_flag_include_pages_in_databases_both_states(via_init: bool) -> None:
    """Verify include_pages_in_databases in both True and False states."""
    reader = NotionPageReader(
        integration_token="tok",
        include_db_content=True,
        include_pages_in_databases=not via_init,
    )

    with patch("requests.request", side_effect=_workspace_mock_http) as mock_req:
        if via_init:
            # Constructor set include_pages_in_databases=False: only db table is fetched
            db_only = reader.load_data(database_ids=["db-1"])
            assert [d.id_ for d in db_only] == ["db-1"]
            called_urls = [c.args[1] for c in mock_req.call_args_list]
            assert (
                "https://api.notion.com/v1/blocks/page-in-db/children"
                not in called_urls
            )

            # Workspace load skips database child pages and keeps standalone pages + db table
            ws_docs = reader.load_data(load_all_if_empty=True)
            assert [d.id_ for d in ws_docs] == ["db-1", "page-standalone"]

            # Per-call override back to True includes child pages again
            with_children = reader.load_data(
                database_ids=["db-1"], include_pages_in_databases=True
            )
            assert [d.id_ for d in with_children] == ["db-1", "page-in-db"]
        else:
            # Constructor kept default True; per-call override to False skips child pages
            default_docs = reader.load_data(database_ids=["db-1"])
            assert [d.id_ for d in default_docs] == ["db-1", "page-in-db"]
            overridden_docs = reader.load_data(
                database_ids=["db-1"], include_pages_in_databases=False
            )
            assert [d.id_ for d in overridden_docs] == ["db-1"]


@pytest.mark.parametrize("via_init", [True, False])
def test_flag_filter_empty_pages_both_states(via_init: bool) -> None:
    """Verify filter_empty_pages in both False (default) and True states."""
    reader = NotionPageReader(
        integration_token="tok",
        filter_empty_pages=via_init,
    )
    target_pages = ["page-standalone", "page-empty", "page-whitespace"]

    with patch("requests.request", side_effect=_workspace_mock_http):
        if via_init:
            filtered = reader.load_data(page_ids=target_pages)
            assert [d.id_ for d in filtered] == ["page-standalone"]

            unfiltered_override = reader.load_data(
                page_ids=target_pages, filter_empty_pages=False
            )
            assert [d.id_ for d in unfiltered_override] == target_pages
        else:
            unfiltered = reader.load_data(page_ids=target_pages)
            assert [d.id_ for d in unfiltered] == target_pages

            filtered_override = reader.load_data(
                page_ids=target_pages, filter_empty_pages=True
            )
            assert [d.id_ for d in filtered_override] == ["page-standalone"]


def test_flag_print_feedback_both_states(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify print_feedback=False produces no stdout and print_feedback=True logs progress."""
    silent_reader = NotionPageReader(integration_token="tok", print_feedback=False)
    verbose_reader = NotionPageReader(
        integration_token="tok",
        print_feedback=True,
        filter_empty_pages=True,
    )

    with patch("requests.request", side_effect=_workspace_mock_http):
        silent_reader.load_data(page_ids=["page-standalone", "page-empty"])
        assert capsys.readouterr().out == ""

        verbose_reader.load_data(page_ids=["page-standalone", "page-empty"])
        out = capsys.readouterr().out
        assert PRINT_PREFIX in out
        assert "load_data" in out
        assert "WARNING : page page-empty is empty" in out


def test_lazy_and_convenience_methods() -> None:
    """Verify lazy_load_data and convenience helper methods operate lazily and accurately."""
    reader = NotionPageReader(integration_token="tok")

    with patch("requests.request", side_effect=_workspace_mock_http) as mock_req:
        lazy_iter: Iterator[Any] = iter(
            reader.lazy_load_data(page_ids=["page-standalone", "page-in-db"])
        )
        # Generator should not make HTTP requests before iteration starts
        assert mock_req.call_count == 0
        first_doc = next(lazy_iter)
        assert first_doc.id_ == "page-standalone"
        assert mock_req.call_count == 1
        second_doc = next(lazy_iter)
        assert second_doc.id_ == "page-in-db"

        assert reader.read_page_text("page-standalone") == "Hello\nWorld"
        assert list(reader.get_all_notion_database_ids()) == ["db-1"]
        assert [d.id_ for d in reader.get_all_notion_databases()] == ["db-1"]
        assert list(reader.get_all_page_ids()) == ["page-standalone", "page-in-db"]

        non_empty = list(
            reader.get_pages_filter_empty(["page-empty", "page-standalone"])
        )
        assert [d.id_ for d in non_empty] == ["page-standalone"]

        all_pages = list(reader.get_pages(["page-empty", "page-standalone"]))
        assert len(all_pages) == 2

        from_ids = list(
            reader.get_pages_and_databases_from_ids(
                page_ids=["page-empty", "page-standalone"],
                database_ids=["db-1"],
            )
        )
        assert [d.id_ for d in from_ids] == ["db-1", "page-standalone"]

        ignore_dbs = list(reader.get_all_pages_ignore_databases())
        assert [d.id_ for d in ignore_dbs] == ["page-standalone"]

        all_ws = list(reader.get_all())
        assert [d.id_ for d in all_ws] == ["db-1", "page-standalone", "page-in-db"]


# -------------------------------------------------------------------------
# Retry & Error Handling Tests


def test_request_with_retry_handles_rate_limits_and_errors() -> None:
    """Verify 429 exponential backoff retry, max-retry exhaustion, and non-429 errors."""
    reader = NotionPageReader(integration_token="tok")

    # 1) Succeeds after two 429 rate-limit responses
    rate_limited = _mock_response(
        {}, status_code=429, headers={"Retry-After": "2"}, text="rate limited"
    )
    ok_resp = _mock_response({"results": [], "has_more": False, "next_cursor": None})

    with (
        patch("requests.request", side_effect=[rate_limited, rate_limited, ok_resp]),
        patch("time.sleep") as mock_sleep,
    ):
        assert reader.read_page("p-retry") == ""
        assert mock_sleep.call_count == 2
        # Attempt 0: 1*(2^0)+2 = 3; Attempt 1: 1*(2^1)+2 = 4
        assert [c.args[0] for c in mock_sleep.call_args_list] == [3, 4]

    # 2) Exhausts max_retries (5) on persistent 429
    with (
        patch("requests.request", return_value=rate_limited),
        patch("time.sleep"),
        pytest.raises(Exception, match="Maximum retries exceeded"),
    ):
        reader.read_page("p-exhausted")

    # 3) Non-429 HTTPError raises immediately without sleeping
    unauthorized = _mock_response({}, status_code=401, text="invalid token")
    with (
        patch("requests.request", return_value=unauthorized),
        patch("time.sleep") as mock_sleep_err,
        pytest.raises(requests.exceptions.HTTPError, match="invalid token"),
    ):
        reader.read_page("p-unauth")
    assert mock_sleep_err.call_count == 0

    # 4) Network-level RequestException wraps and raises immediately
    with (
        patch(
            "requests.request",
            side_effect=requests.exceptions.ConnectionError("dns failure"),
        ),
        pytest.raises(requests.exceptions.RequestException, match="dns failure"),
    ):
        reader.read_page("p-conn-err")
