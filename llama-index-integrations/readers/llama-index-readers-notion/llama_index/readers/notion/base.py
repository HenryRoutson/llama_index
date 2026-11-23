"""Notion reader."""

import datetime
import os
import time
from itertools import chain
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set

import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

# -------------------------------------------------

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"

PRINT_PREFIX = "Notion reader : "

PAGE_SIZE = 100
PAGE_PREVIEW_LENGTH = 150
DATABASE_PREVIEW_LENGTH = 200
RATE_LIMIT_ERROR_CODE = 429

KNOWN_DB_PROP_TYPES = frozenset(
    {
        "relation",
        "multi_select",
        "rich_text",
        "title",
        "checkbox",
        "url",
        "text",
        "email",
    }
)

# AI : Explicit semantic type aliases keep str IDs compatible with callers while self-documenting intent
json_t = Dict[str, Any]
format_json_f = Callable[[json_t], str]
page_id_t = str
notion_db_id_t = str


# -------------------------------------------------
# Notes
#
# This code interacts with two distinct concepts:
# 1. Notion as a workspace/knowledge store
# 2. Notion databases: https://www.notion.com/help/intro-to-databases
#
# Pagination (`start_cursor`):
#   A cursor string passed back to the same endpoint to retrieve the next page
#   of results when `has_more` is True (max 100 items per request):
#   https://developers.notion.com/reference/intro#pagination
# -------------------------------------------------


def iterate_paginated_api(
    function: Callable[..., json_t], **kwargs: Any
) -> Generator[json_t, None, None]:
    """Return an iterator over the results of any paginated Notion API endpoint."""
    # Adapted from notion_client.helpers.iterate_paginated_api without adding an extra dependency
    next_cursor: Optional[str] = kwargs.pop("start_cursor", None)

    while True:
        response = function(**kwargs, start_cursor=next_cursor)
        yield from response.get("results", [])

        next_cursor = response.get("next_cursor")
        if not response.get("has_more") or not next_cursor:
            return


class NotionPageReader(BasePydanticReader):
    """
    Notion Page reader.

    Reads pages and, optionally, structured database row content from Notion.

    Args:
        integration_token (Optional[str]): Notion integration token. Falls back
            to the ``NOTION_INTEGRATION_TOKEN`` environment variable.
        print_feedback (bool): Whether to print timestamped progress messages
            during Notion API operations. Defaults to ``False``.
        include_db_content (bool): Whether ``load_data`` / ``lazy_load_data``
            should also emit a formatted ``Document`` containing each Notion
            database's rows and properties. Defaults to ``False``.
        include_pages_in_databases (bool): Whether ``load_data`` /
            ``lazy_load_data`` should fetch and load individual pages inside
            Notion databases. Defaults to ``True``.
        filter_empty_pages (bool): Whether ``load_data`` / ``lazy_load_data``
            should omit pages whose body text is empty or whitespace-only.
            Defaults to ``False``.

    """

    is_remote: bool = True
    token: str
    headers: Dict[str, str]
    print_feedback: bool = False
    include_db_content: bool = False
    include_pages_in_databases: bool = True
    filter_empty_pages: bool = False

    def __init__(
        self,
        integration_token: Optional[str] = None,
        print_feedback: bool = False,
        include_db_content: bool = False,
        include_pages_in_databases: bool = True,
        filter_empty_pages: bool = False,
    ) -> None:
        """Initialize with parameters."""
        if integration_token is None:
            integration_token = os.getenv(INTEGRATION_TOKEN_NAME)
            if integration_token is None:
                raise ValueError(
                    "Must specify `integration_token` or set environment "
                    "variable `NOTION_INTEGRATION_TOKEN`."
                )

        headers = {
            "Authorization": f"Bearer {integration_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        super().__init__(
            token=integration_token,
            headers=headers,
            print_feedback=print_feedback,
            include_db_content=include_db_content,
            include_pages_in_databases=include_pages_in_databases,
            filter_empty_pages=filter_empty_pages,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "NotionPageReader"

    def lazy_load_data(
        self,
        page_ids: Optional[Iterable[page_id_t]] = None,
        database_ids: Optional[Iterable[notion_db_id_t]] = None,
        load_all_if_empty: bool = False,
        include_db_content: Optional[bool] = None,
        include_pages_in_databases: Optional[bool] = None,
        filter_empty_pages: Optional[bool] = None,
        format_db_json: Optional[format_json_f] = None,
    ) -> Iterable[Document]:
        """
        Lazily load Notion pages and/or database content as Documents.

        Args:
            page_ids (Optional[Iterable[str]]): Page IDs to load.
            database_ids (Optional[Iterable[str]]): Database IDs to load.
            load_all_if_empty (bool): If True and neither ``page_ids`` nor
                ``database_ids`` is given, discover all pages and databases in
                the workspace. Defaults to ``False``.
            include_db_content (Optional[bool]): Override ``self.include_db_content``
                for this call. When True, yields a formatted ``Document`` for each
                database's row/property content.
            include_pages_in_databases (Optional[bool]): Override
                ``self.include_pages_in_databases`` for this call. When True,
                expands databases into their child page IDs and loads each page.
            filter_empty_pages (Optional[bool]): Override
                ``self.filter_empty_pages`` for this call. When True, skips
                pages with empty or whitespace-only text.
            format_db_json (Optional[Callable[[Dict[str, Any]], str]]): Custom
                formatter converting a Notion database query dict into text.

        Yields:
            Document: Loaded Notion database and/or page documents.

        """
        self._print("lazy_load_data")
        # AI : Fall back to instance-level flag defaults when per-call overrides are None
        use_db_content = (
            self.include_db_content
            if include_db_content is None
            else include_db_content
        )
        use_pages_in_dbs = (
            self.include_pages_in_databases
            if include_pages_in_databases is None
            else include_pages_in_databases
        )
        use_filter_empty = (
            self.filter_empty_pages
            if filter_empty_pages is None
            else filter_empty_pages
        )

        resolved_page_ids: List[page_id_t] = (
            list(page_ids) if page_ids is not None else []
        )
        resolved_db_ids: List[notion_db_id_t] = (
            list(database_ids) if database_ids is not None else []
        )

        if not resolved_page_ids and not resolved_db_ids:
            if not load_all_if_empty:
                raise ValueError(
                    "Must specify either `page_ids` or `database_ids` if "
                    "`load_all_if_empty` is False."
                )
            resolved_db_ids = self.list_databases()
            if use_pages_in_dbs:
                resolved_page_ids = self.list_pages()
            else:
                resolved_page_ids = list(self.get_page_ids_not_in_databases())

        if use_db_content:
            for database_id in resolved_db_ids:
                yield self.get_notion_database(
                    database_id, format_db_json=format_db_json
                )

        all_page_ids: List[page_id_t] = list(dict.fromkeys(resolved_page_ids))
        if use_pages_in_dbs and resolved_db_ids:
            seen: Set[page_id_t] = set(all_page_ids)
            for database_id in resolved_db_ids:
                for db_page_id in self.get_page_ids_from_database(database_id):
                    if db_page_id not in seen:
                        seen.add(db_page_id)
                        all_page_ids.append(db_page_id)

        for page_id in all_page_ids:
            doc = self.get_page(page_id, filter_empty=use_filter_empty)
            if doc is not None:
                yield doc

    def load_data(
        self,
        page_ids: Optional[Iterable[page_id_t]] = None,
        database_ids: Optional[Iterable[notion_db_id_t]] = None,
        load_all_if_empty: bool = False,
        include_db_content: Optional[bool] = None,
        include_pages_in_databases: Optional[bool] = None,
        filter_empty_pages: Optional[bool] = None,
        format_db_json: Optional[format_json_f] = None,
    ) -> List[Document]:
        """
        Load Notion pages and/or database content as a list of Documents.

        Args:
            page_ids (Optional[Iterable[str]]): Page IDs to load.
            database_ids (Optional[Iterable[str]]): Database IDs to load.
            load_all_if_empty (bool): If True and neither ``page_ids`` nor
                ``database_ids`` is given, discover all pages and databases in
                the workspace. Defaults to ``False``.
            include_db_content (Optional[bool]): Override ``self.include_db_content``
                for this call. When True, includes a formatted ``Document`` for
                each database's row/property content.
            include_pages_in_databases (Optional[bool]): Override
                ``self.include_pages_in_databases`` for this call. When True,
                expands databases into their child page IDs and loads each page.
            filter_empty_pages (Optional[bool]): Override
                ``self.filter_empty_pages`` for this call. When True, skips
                pages with empty or whitespace-only text.
            format_db_json (Optional[Callable[[Dict[str, Any]], str]]): Custom
                formatter converting a Notion database query dict into text.

        Returns:
            List[Document]: List of loaded documents.

        """
        self._print("load_data")
        return list(
            self.lazy_load_data(
                page_ids=page_ids,
                database_ids=database_ids,
                load_all_if_empty=load_all_if_empty,
                include_db_content=include_db_content,
                include_pages_in_databases=include_pages_in_databases,
                filter_empty_pages=filter_empty_pages,
                format_db_json=format_db_json,
            )
        )

    # -------------------------------------------------------------------------
    # NETWORK API FUNCTIONS

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make an HTTP request with exponential backoff on rate limits."""
        max_retries = 5
        backoff_factor = 1

        for attempt in range(max_retries):
            try:
                response: requests.Response = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError:
                if response.status_code == RATE_LIMIT_ERROR_CODE:
                    retry_after = int(response.headers.get("Retry-After", 1))
                    time.sleep(backoff_factor * (2**attempt) + retry_after)
                else:
                    raise requests.exceptions.HTTPError(
                        f"Request failed: {response.text}"
                    )
            except requests.exceptions.RequestException as err:
                raise requests.exceptions.RequestException(f"Request failed: {err}")
        raise Exception("Maximum retries exceeded")

    def _request_block(
        self,
        block_id: str,
        start_cursor: Optional[str] = None,
        page_size: int = PAGE_SIZE,
    ) -> json_t:
        """Fetch one page of child blocks for a Notion block or page."""
        self._print("_request_block")
        # AI : Retrieve-block-children is a GET endpoint that takes start_cursor as a query param
        block_url = BLOCK_CHILD_URL_TMPL.format(block_id=block_id)
        params: json_t = {"page_size": page_size}
        if start_cursor is not None:
            params["start_cursor"] = start_cursor

        res = self._request_with_retry(
            "GET",
            block_url,
            headers=self.headers,
            params=params,
        )
        return res.json()

    def _request_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Optional[json_t] = None,
        start_cursor: Optional[str] = None,
    ) -> json_t:
        """Fetch one page of rows from a Notion database query."""
        self._print("_request_database")
        # AI : Copy query_dict so pagination cursors never mutate caller or default dicts
        payload: json_t = {"page_size": PAGE_SIZE, **(query_dict or {})}
        if start_cursor is not None:
            payload["start_cursor"] = start_cursor

        res = self._request_with_retry(
            "POST",
            DATABASE_URL_TMPL.format(database_id=database_id),
            headers=self.headers,
            json=payload,
        )
        return res.json()

    def _request_search(
        self,
        query_dict: Optional[json_t] = None,
        start_cursor: Optional[str] = None,
    ) -> json_t:
        """Fetch one page of results from the Notion search endpoint."""
        self._print("_request_search")
        payload: json_t = {"page_size": PAGE_SIZE, **(query_dict or {})}
        if start_cursor is not None:
            payload["start_cursor"] = start_cursor

        res = self._request_with_retry(
            "POST",
            SEARCH_URL,
            headers=self.headers,
            json=payload,
        )
        return res.json()

    # -------------------------------------------------------------------------
    # BLOCK & PAGE PARSING

    def page_json_list_to_text(
        self, page_json_iter: Iterable[json_t], num_tabs: int = 0
    ) -> str:
        """Convert an iterable of Notion block JSON objects into indented text."""
        result_lines_arr: List[str] = []
        for result in page_json_iter:
            result_type: str = result["type"]
            result_obj: json_t = result[result_type]

            cur_result_text_arr: List[str] = []
            if "rich_text" in result_obj:
                for rich_text in result_obj["rich_text"]:
                    if "text" in rich_text:
                        text: str = rich_text["text"]["content"]
                        prefix: str = "\t" * num_tabs
                        cur_result_text_arr.append(prefix + text)

            result_block_id: str = result["id"]
            if result.get("has_children"):
                children_text: str = self._read_block(
                    result_block_id, num_tabs=num_tabs + 1
                )
                cur_result_text_arr.append(children_text)

            result_lines_arr.append("\n".join(cur_result_text_arr))

        return "\n".join(result_lines_arr)

    def _read_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block and recursively read its children across all pages."""
        blocks = iterate_paginated_api(self._request_block, block_id=block_id)
        return self.page_json_list_to_text(blocks, num_tabs=num_tabs)

    def read_page(self, page_id: page_id_t) -> str:
        """Read a Notion page's full text content."""
        page_text = self._read_block(page_id)
        self._print(
            f"Reading page {page_id} of len {len(page_text)}: \n"
            f" {page_text[:PAGE_PREVIEW_LENGTH]}..."
        )
        return page_text

    def read_page_text(self, page_id: page_id_t) -> str:
        """Read a Notion page's full text content (alias for ``read_page``)."""
        return self.read_page(page_id)

    # -------------------------------------------------------------------------
    # SEARCH & WORKSPACE LISTING

    @staticmethod
    def _extract_ids_from_results(results: Iterable[json_t]) -> Iterable[str]:
        """Extract object IDs from Notion API result items."""
        return (result["id"] for result in results)

    def search(self, query: str) -> List[str]:
        """Search Notion pages/databases matching a text query and return IDs."""
        self._print("search")
        results = iterate_paginated_api(
            self._request_search, query_dict={"query": query}
        )
        return list(self._extract_ids_from_results(results))

    def _search_objects_by_type(self, object_type: str) -> Iterable[json_t]:
        """Iterate over all workspace objects of a given type (``page`` or ``database``)."""
        return iterate_paginated_api(
            self._request_search,
            query_dict={"filter": {"property": "object", "value": object_type}},
        )

    def _list_ids(self, function_name: str, value: str) -> List[str]:
        """List all unique IDs for a Notion object type while preserving order."""
        self._print(f"{function_name} -- getting new data")
        raw_ids = list(
            self._extract_ids_from_results(self._search_objects_by_type(value))
        )
        self._print(f"_list_ids {function_name} -- found {len(raw_ids)} {value}s")
        unique_ids = list(dict.fromkeys(raw_ids))
        self._print(
            f"_list_ids {function_name} -- found {len(unique_ids)} unique {value}s"
        )
        return unique_ids

    def list_databases(self) -> List[notion_db_id_t]:
        """List all database IDs in the Notion workspace."""
        return self._list_ids("list_databases", "database")

    def list_pages(self) -> List[page_id_t]:
        """List all page IDs in the Notion workspace."""
        return self._list_ids("list_pages", "page")

    # -------------------------------------------------------------------------
    # NOTION DATABASES

    def get_pages_from_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Optional[Dict[str, Any]] = None,
    ) -> Iterable[json_t]:
        """Lazily iterate over all page row objects in a Notion database."""
        self._print("get_pages_from_database")
        return iterate_paginated_api(
            self._request_database,
            database_id=database_id,
            query_dict=query_dict,
        )

    def get_page_ids_from_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Optional[Dict[str, Any]] = None,
    ) -> Iterable[page_id_t]:
        """Lazily iterate over all page IDs in a Notion database."""
        self._print("get_page_ids_from_database")
        return self._extract_ids_from_results(
            self.get_pages_from_database(database_id, query_dict=query_dict)
        )

    def query_database(
        self,
        database_id: notion_db_id_t,
        query_dict: Optional[Dict[str, Any]] = None,
    ) -> List[page_id_t]:
        """Get all page IDs from a Notion database."""
        return list(self.get_page_ids_from_database(database_id, query_dict=query_dict))

    def get_page_ids_from_databases(
        self, database_id_list: Iterable[notion_db_id_t]
    ) -> Set[page_id_t]:
        """Collect all unique page IDs across multiple Notion databases."""
        self._print("get_page_ids_from_databases")
        page_ids: Set[page_id_t] = set()
        for database_id in database_id_list:
            page_ids.update(self.get_page_ids_from_database(database_id))
        return page_ids

    @staticmethod
    def default_format_db_json(json_database: json_t) -> str:
        """Format a Notion database query response into human-readable row text."""
        database_text: str = "\n Notion Database Start  -----------------\n"
        for row in json_database.get("results", []):
            properties: json_t = row.get("properties", {})

            database_text += "\nNew row\n"
            for prop_name, prop_value in properties.items():
                # AI : Strip noisy property metadata for known Notion property types
                prop_type: str = prop_value["type"]
                database_text += f"{prop_name},type:{prop_type},data:"

                if prop_type in KNOWN_DB_PROP_TYPES:
                    database_text += str(prop_value.get(prop_type, []))
                else:
                    database_text += str(prop_value)

                database_text += "\n"

        return database_text + "\nNotion Database End  -----------------\n"

    def get_all_notion_database_ids(self) -> Iterable[notion_db_id_t]:
        """Return all database IDs in the Notion workspace."""
        self._print("get_all_notion_database_ids")
        return iter(self.list_databases())

    def get_notion_database_text(
        self,
        database_id: notion_db_id_t,
        format_db_json: Optional[format_json_f] = None,
    ) -> str:
        """Read all rows of a Notion database and format them as text."""
        self._print(f"get_notion_database_text {database_id}")
        # AI : Paginate all database rows before formatting so databases >100 rows are complete
        rows = list(self.get_pages_from_database(database_id))
        formatter = format_db_json or self.default_format_db_json
        return formatter({"results": rows})

    def get_notion_database(
        self,
        database_id: notion_db_id_t,
        format_db_json: Optional[format_json_f] = None,
    ) -> Document:
        """Load a Notion database's row content as a single Document."""
        self._print(f"get_notion_database {database_id}")
        database_text = self.get_notion_database_text(
            database_id, format_db_json=format_db_json
        )
        self._print(f"get_notion_database {database_id} -- len {len(database_text)}")
        self._print(
            f"get_notion_database {database_id} -- text "
            f"{database_text[:DATABASE_PREVIEW_LENGTH]}"
        )
        return Document(
            text=database_text,
            id_=database_id,
            metadata={"database_id": database_id},
        )

    def get_notion_databases(
        self,
        databases: Iterable[notion_db_id_t],
        format_db_json: Optional[format_json_f] = None,
    ) -> Iterable[Document]:
        """Lazily load multiple Notion databases as Documents."""
        self._print("get_notion_databases")
        return (
            self.get_notion_database(database_id, format_db_json=format_db_json)
            for database_id in databases
        )

    def get_all_notion_databases(
        self,
        format_db_json: Optional[format_json_f] = None,
    ) -> Iterable[Document]:
        """Lazily load all databases in the Notion workspace as Documents."""
        self._print("get_all_notion_databases")
        return self.get_notion_databases(
            databases=self.list_databases(), format_db_json=format_db_json
        )

    # -------------------------------------------------------------------------
    # PAGES & CONVENIENCE HELPERS

    def get_page_ids_not_in_databases(self) -> Iterable[page_id_t]:
        """List workspace page IDs whose parent is not a Notion database."""
        self._print("get_page_ids_not_in_databases")
        standalone_ids = [
            result["id"]
            for result in self._search_objects_by_type("page")
            if result.get("parent", {}).get("type") != "database_id"
        ]
        return iter(dict.fromkeys(standalone_ids))

    def get_all_page_ids(self) -> Iterable[page_id_t]:
        """List all unique page IDs across standalone pages and databases."""
        combined = chain(
            self.get_page_ids_not_in_databases(),
            self.get_page_ids_from_databases(self.list_databases()),
        )
        return iter(dict.fromkeys(combined))

    @staticmethod
    def _is_page_text_empty(page_text: Optional[str]) -> bool:
        """Check whether extracted page text is None or whitespace-only."""
        return page_text is None or page_text.strip() == ""

    def get_page(
        self,
        page_id: page_id_t,
        filter_empty: Optional[bool] = None,
    ) -> Optional[Document]:
        """Load a single Notion page as a Document, optionally skipping empty pages."""
        should_filter = (
            self.filter_empty_pages if filter_empty is None else filter_empty
        )
        page_text = self.read_page(page_id)

        if should_filter and self._is_page_text_empty(page_text):
            self._print(f"WARNING : page {page_id} is empty")
            return None

        return Document(text=page_text, id_=page_id, metadata={"page_id": page_id})

    def get_pages(
        self,
        pages: Iterable[page_id_t],
        filter_empty: Optional[bool] = None,
    ) -> Iterable[Optional[Document]]:
        """Lazily load multiple Notion pages."""
        self._print("get_pages")
        return (self.get_page(page_id, filter_empty=filter_empty) for page_id in pages)

    def get_pages_filter_empty(self, pages: Iterable[page_id_t]) -> Iterable[Document]:
        """Lazily load multiple Notion pages, omitting empty pages."""
        return (
            doc
            for page_id in pages
            if (doc := self.get_page(page_id, filter_empty=True)) is not None
        )

    def get_pages_and_databases_from_ids(
        self,
        page_ids: Optional[Iterable[page_id_t]] = None,
        database_ids: Optional[Iterable[notion_db_id_t]] = None,
        format_db_json: Optional[format_json_f] = None,
    ) -> Iterable[Document]:
        """Lazily load database content Documents and non-empty page Documents from IDs."""
        self._print("get_pages_and_databases_from_ids")
        return chain(
            self.get_notion_databases(
                databases=database_ids or [], format_db_json=format_db_json
            ),
            self.get_pages_filter_empty(pages=page_ids or []),
        )

    def get_all_pages_ignore_databases(
        self,
        include_pages_in_databases: bool = False,
    ) -> Iterable[Document]:
        """Lazily load all non-empty workspace pages without database table Documents."""
        return self.lazy_load_data(
            load_all_if_empty=True,
            include_db_content=False,
            include_pages_in_databases=include_pages_in_databases,
            filter_empty_pages=True,
        )

    def get_all_pages_and_databases(
        self,
        format_db_json: Optional[format_json_f] = None,
        include_pages_in_databases: bool = False,
    ) -> Iterable[Document]:
        """Lazily load all workspace database content and non-empty pages."""
        return self.lazy_load_data(
            load_all_if_empty=True,
            include_db_content=True,
            include_pages_in_databases=include_pages_in_databases,
            filter_empty_pages=True,
            format_db_json=format_db_json,
        )

    def get_all(
        self, format_db_json: Optional[format_json_f] = None
    ) -> Iterable[Document]:
        """Lazily load all workspace databases and all non-empty pages (including database pages)."""
        return self.get_all_pages_and_databases(
            format_db_json=format_db_json, include_pages_in_databases=True
        )

    def _print(self, message: str) -> None:
        """Print a timestamped progress message when ``print_feedback`` is enabled."""
        if self.print_feedback:
            timestamp = datetime.datetime.now().strftime(" %H:%M")
            print(f"{PRINT_PREFIX}{message}{timestamp}")


if __name__ == "__main__":  # pragma: no cover
    reader = NotionPageReader()
    print(reader.search("What I"))
    databases = reader.list_databases()
