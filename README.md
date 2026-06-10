# Data Cleaning & Preprocessing Pipeline

A modular, production-minded Python pipeline that takes raw, messy tabular data
and carries it through ingestion, profiling, standardization, cleaning, outlier
handling, and export — with optional MySQL integration. Every step is wrapped in
defensive error handling so a failure in one stage logs a clear message and the
program degrades gracefully instead of crashing.

---

## Features

The whole pipeline lives in a single `DataPipeline` class, organized into stages.
You can call the stages individually, or run them all at once with `run()`.

### Orchestrator
- **`run(config, source, destination, ...)`** — executes the whole pipeline
  end-to-end from a single config dict. Stages run in a fixed, safe order and
  are **opt-in**: any stage whose config key is missing or empty is skipped.
  Returns `(cleaned, outliers)`.

### Ingestion & health report
- **`load_data(path)`** — read a `.csv` or `.json` file with targeted error handling.
- **`health_report()`** — a "big picture" summary: row/column counts, duplicate
  rows, a per-column null table (count + %), and numeric/categorical describe().

### Standardization
- **`standardize_column_names()`** — lowercases, trims, snake_cases, strips
  punctuation, and splits camelCase (`airConditioning` → `air_conditioning`).
- **`standardize_numeric(cols)`** — strips currency symbols, thousands
  separators, and unit suffixes (`€106,500`, `51 m²`) into clean numbers; bad
  values become `NaN`.
- **`standardize_categorical(cols, casing)`** — trims whitespace and unifies text
  casing (lower / upper / title).
- **`standardize_dates()`** — auto-detects date-like columns by name and parses
  mixed per-row formats into a uniform `DD/MM/YYYY`.
- **`standardize_booleans(cols=None)`** — detects boolean-like columns and
  converts many spellings (`Yes/No`, `Y/N`, `1/0`, `Positive/Negative`,
  `on/off`, `T/F`) to a real nullable `boolean` dtype.

### Data cleaning
- **`remove_duplicates(subset=None, keep="first")`** — drops duplicate rows and
  logs how many were removed.
- **`handle_missing(column, strategy, fill_value=None)`** — fills or drops gaps
  in a chosen column using `mean`, `median`, `mode`, `constant`, `ffill`,
  `bfill`, or `drop`.

### Outlier handling
- **`clean_outliers(df, column)`** — tests the column's distribution and applies
  the appropriate method (Z-score if roughly normal, IQR if skewed), returning
  separate cleaned and outlier DataFrames. Guards against non-numeric input.

### Export
- **`export_csv(df, path)`** — writes Excel-friendly `utf-8-sig` CSV, creating any
  missing directories and handling real-world errors (file open in Excel, etc.).

### MySQL integration (SQLAlchemy)
- **`get_db_engine(...)`** — builds a `mysql+pymysql` engine. Credentials are
  never hardcoded; they come from environment variables or a runtime login
  prompt (host/port/database/username are visible, **password stays hidden**).
- **`store_dataframe(df, table, engine, if_exists="replace")`** — writes a
  DataFrame to a table.
- **`query_to_dataframe(query, engine)`** — runs SQL and returns a DataFrame.
- **`infer_sql_schema(df, table_name)`** — generates a `CREATE TABLE` statement
  inferred from the DataFrame's dtypes and values (a starting point to review).

---

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `scipy`, `sqlalchemy`, `pymysql`

```bash
pip install pandas numpy scipy sqlalchemy pymysql
```

---

## Project layout

```
.
├── data_pipeline.py     # the pipeline
├── input/               # put your raw .csv / .json here
├── output/              # cleaned results are written here
├── README.md
└── LICENSE
```

---

## Usage

The execution block at the bottom of `data_pipeline.py` is driven by two
switches so you can pick where data comes from and where it goes:

```python
SOURCE = "file"        # "file" -> input folder | "database" -> MySQL
DESTINATION = "file"   # "file" -> output folder | "database" -> MySQL
```

Then run it from anywhere (paths resolve relative to the script):

```bash
python data_pipeline.py
```

The four source/destination combinations all work: file→file, file→database,
database→file, and database→database. In between, the pipeline runs the health
report, standardization, and outlier cleaning, and prints a **before/after**
preview of the data.

### Using it as a library

```python
from data_pipeline import DataPipeline

p = DataPipeline()
p.load_data("input/data.csv")
p.standardize_column_names()
p.standardize_numeric(["price"])
p.standardize_booleans()
p.remove_duplicates()
p.handle_missing("bathrooms", strategy="median")
cleaned, outliers = p.clean_outliers(p.df, "price")
DataPipeline.export_csv(cleaned, "output/cleaned_data.csv")
```

### One-call run with the `run()` orchestrator

Instead of calling each stage by hand, describe what you want in a config and
let `run()` execute the whole sequence. Any stage whose key is missing or empty
is skipped, so the same call works for a 3-column dataset or a 30-column one.

```python
from data_pipeline import DataPipeline

config = {
    "numeric_cols":     ["price", "area_sqm"],
    "categorical_cols": ["location", "energy_class"],
    "categorical_case": "title",
    "standardize_dates": True,
    "boolean_cols":     None,                  # None = auto-detect; omit to skip
    "drop_duplicates":  True,                  # or {"subset": ["id"], "keep": "first"}
    "missing":          {"bathrooms": "median"},
    "fill_values":      {"energy_class": "UNKNOWN"},  # used by "constant" strategy
    "outlier_col":      "price",
}

p = DataPipeline()
cleaned, outliers = p.run(
    config,
    source="file",            # or "database"
    destination="file",       # or "database"
    input_path="input/data.csv",
    output_path="output/cleaned_data.csv",
    # source_query=..., dest_table=..., engine=...   # for database I/O
)
```

Recognized `config` keys (all optional):

| Key | Stage | Notes |
|-----|-------|-------|
| `numeric_cols` | `standardize_numeric` | list of columns |
| `categorical_cols` | `standardize_categorical` | list of columns |
| `categorical_case` | `standardize_categorical` | `"lower"` (default) / `"upper"` / `"title"` |
| `standardize_dates` | `standardize_dates` | `True` (default) / `False` |
| `boolean_cols` | `standardize_booleans` | `None` = auto-detect; omit key to skip |
| `drop_duplicates` | `remove_duplicates` | `True`, or `{"subset": [...], "keep": "first"}` |
| `missing` | `handle_missing` | `{column: strategy}` per column |
| `fill_values` | `handle_missing` | `{column: value}` for the `"constant"` strategy |
| `outlier_col` | `clean_outliers` | single column to split outliers on |
| `outlier_path` | export outliers | CSV path for the flagged outliers (file destination) |
| `outlier_table` | store outliers | table name for the flagged outliers (database destination) |

The `__main__` block at the bottom of `data_pipeline.py` already uses `run()` —
edit its `CONFIG` and the `SOURCE` / `DESTINATION` switches to match your data.

---

## Database credentials & security

No connection details are stored in the source code. You can supply them in
either of two ways:

1. **Environment variables** — `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
   `DB_PASSWORD`.
2. **Interactive login prompt** — `get_db_engine(prompt_credentials=True)` asks
   for each field at runtime; the password is entered with hidden input.

---

## How this was built

This project was built iteratively with the help of Claude Code.
The design, requirements, feature decisions, and testing were directed by me;
the assistant helped implement and refine the code along the way.

---

## License

Released under the MIT License. See `LICENSE`.
