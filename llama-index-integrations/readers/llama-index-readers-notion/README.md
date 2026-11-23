# LlamaIndex Readers Integration: Notion

## Overview

`NotionPageReader` loads pages and, optionally, structured Notion database row/property tables from a Notion workspace. All Notion endpoints (page blocks, database queries, search, and workspace page/database listing) automatically paginate beyond Notion's 100-item per-request limit.

### Installation

```bash
pip install llama-index-readers-notion
```

### Usage

#### Default Usage (Backwards-Compatible)

```python
from llama_index.readers.notion import NotionPageReader

# Reads NOTION_INTEGRATION_TOKEN from the environment if omitted
reader = NotionPageReader(integration_token="<Integration Token>")

# Load specific pages and/or all child pages inside specified databases
documents = reader.load_data(
    page_ids=["<Page ID 1>", "<Page ID 2>"],
    database_ids=["<Database ID>"],
)
```

#### Downloading Notion Database Row Content & Large Workspaces

```python
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(
    integration_token="<Integration Token>",
    include_db_content=True,
    include_pages_in_databases=False,
    filter_empty_pages=True,
    print_feedback=True,
)

# Stream database table Documents and standalone pages lazily
for doc in reader.lazy_load_data(load_all_if_empty=True):
    print(doc.id_, doc.metadata)
```

You can also customize how database JSON rows are rendered into text by passing `format_db_json`:

```python
db_doc = reader.get_notion_database(
    "<Database ID>",
    format_db_json=lambda data: f"Total rows: {len(data['results'])}",
)
```

### Feature Flags

All flags can be set on `NotionPageReader(...)` at initialization, and `include_db_content`, `include_pages_in_databases`, and `filter_empty_pages` can also be overridden per call in `load_data(...)` / `lazy_load_data(...)`. Defaults match prior `NotionPageReader` behaviour:

| Flag                         | Default | Description                                                                                                                                                  |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `include_db_content`         | `False` | When `True`, emits a formatted `Document` (`metadata={"database_id": ...}`) containing the rows and properties of each Notion database.                      |
| `include_pages_in_databases` | `True`  | When `True`, expands each database into its child page IDs and loads each child page. Set to `False` to skip per-row page block requests on large databases. |
| `filter_empty_pages`         | `False` | When `True`, skips pages whose extracted body text is empty or whitespace-only instead of returning empty `Document` objects.                                |
| `print_feedback`             | `False` | When `True`, prints timestamped progress messages during Notion API requests.                                                                                |

Implementation examples for Notion reader can be found [here](https://docs.llamaindex.ai/en/stable/examples/data_connectors/NotionDemo/).

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
