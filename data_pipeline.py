from __future__ import annotations

import logging
import os
import re
from getpass import getpass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_pipeline")


# --------------------------------------------------------------------------- #
# Database configuration
# --------------------------------------------------------------------------- #
# No connection details are hardcoded here.
# Every field is read from an environment variable if you
# choose to set one; otherwise it defaults to ``None`` and is requested at
# runtime via the hidden, interactive login prompt (prompt_credentials=True).
# The only literal is the driver name, which is not sensitive.
#
# Optional env vars: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
DB_CONFIG = {
    "drivername": "mysql+pymysql",
    "host": os.environ.get("DB_HOST"),
    "port": int(os.environ["DB_PORT"]) if os.environ.get("DB_PORT") else None,
    "database": os.environ.get("DB_NAME"),
    "username": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
}


class DataPipeline:
    """End-to-end data pipeline.

    The class holds a single :class:`pandas.DataFrame` (``self.df``) that is
    progressively transformed by the module methods. Each public method is
    self-contained and fails gracefully, logging a clear message instead of
    raising to the caller.
    """

    # Keywords used to auto-detect date-like columns by name.
    DATE_KEYWORDS = ("date", "time", "created", "updated", "timestamp", "dob")

    # Vocabulary used to recognize boolean-like values (case-insensitive).
    TRUE_TOKENS = {"true", "t", "yes", "y", "1", "positive", "pos", "on"}
    FALSE_TOKENS = {"false", "f", "no", "n", "0", "negative", "neg", "off"}

    def __init__(self, dataframe: Optional[pd.DataFrame] = None) -> None:
        self.df: Optional[pd.DataFrame] = dataframe

    # ===================================================================== #
    # DATA INGESTION & HEALTH REPORT
    # ===================================================================== #
    def load_data(self, path: str) -> Optional[pd.DataFrame]:
        """Load a dataset from ``path`` (CSV or JSON) into ``self.df``.

        Args:
            path: Filesystem path to a ``.csv`` or ``.json`` file.

        Returns:
            The loaded DataFrame, or ``None`` if loading failed.
        """
        logger.info("Loading data from: %s", path)
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".csv":
                self.df = pd.read_csv(path)
            elif suffix == ".json":
                self.df = pd.read_json(path)
            else:
                raise ValueError(
                    f"Unsupported file type '{suffix}'. Use .csv or .json."
                )
            logger.info("Successfully loaded %d rows.", len(self.df))
            return self.df

        except FileNotFoundError:
            logger.error("File not found: %s", path)
        except pd.errors.EmptyDataError:
            logger.error("File is empty: %s", path)
        except pd.errors.ParserError as exc:
            logger.error("Failed to parse file '%s': %s", path, exc)
        except ValueError as exc:
            logger.error("Value error while loading '%s': %s", path, exc)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            logger.error("Unexpected error loading '%s': %s", path, exc)
        return None

    def health_report(self) -> None:
        """Print a 'big picture' health report for the current DataFrame."""
        if not self._has_data():
            return

        df = self.df
        try:
            print("\n" + "=" * 60)
            print("DATA HEALTH REPORT")
            print("=" * 60)

            # --- Big picture ------------------------------------------------
            rows, cols = df.shape
            duplicate_count = int(df.duplicated().sum())
            print("\n[Big Picture]")
            print(f"  Total rows         : {rows}")
            print(f"  Total columns      : {cols}")
            print(f"  Duplicate rows     : {duplicate_count}")

            # --- Column summary table --------------------------------------
            null_counts = df.isna().sum()
            null_pct = (null_counts / rows * 100).round(2) if rows else null_counts
            summary = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Dtype": [str(t) for t in df.dtypes],
                    "Nulls": null_counts.values,
                    "Null_%": null_pct.values,
                }
            )
            print("\n[Column Summary]")
            print(summary.to_string(index=False))

            # --- Statistical summaries -------------------------------------
            print("\n[Numerical Summary]")
            numeric_desc = df.describe()
            if numeric_desc.empty:
                print("  No numerical columns to describe.")
            else:
                print(numeric_desc.to_string())

            print("\n[Categorical Summary]")
            try:
                # exclude="number" selects all non-numeric (text/categorical)
                # columns without the deprecated include="O" behavior that
                # pulls in 'str' dtype and triggers a Pandas4Warning.
                cat_desc = df.describe(exclude="number")
                if cat_desc.empty:
                    print("  No categorical columns found - skipping.")
                else:
                    print(cat_desc.to_string() + "\n")
            except ValueError:
                # Raised by pandas when there are no non-numeric columns.
                print("  No categorical columns found - skipping.")

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build health report: %s", exc)

    # ===================================================================== #
    # STANDARDIZATION PIPELINE
    # ===================================================================== #
    def standardize_column_names(self) -> None:
        """Lowercase, trim, snake_case and de-punctuate all column names."""
        if not self._has_data():
            return
        try:
            cleaned = []
            for col in self.df.columns:
                name = str(col).strip()
                # Split camelCase boundaries BEFORE lowercasing, so
                # "airConditioning" -> "air_conditioning" (not "airconditioning").
                name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
                name = name.lower()
                # Treat any run of whitespace/punctuation as a single separator
                # so "Annual-Salary($)" -> "annual_salary" (not "annualsalary").
                name = re.sub(r"[^a-z0-9]+", "_", name)
                name = name.strip("_")
                cleaned.append(name or "unnamed")
            self.df.columns = cleaned
            logger.info("Standardized column names: %s", cleaned)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to standardize column names: %s", exc)

    def standardize_numeric(self, columns: Iterable[str]) -> None:
        """Coerce dirty string columns into clean numeric dtypes.

        Strips currency symbols, thousands separators and text suffixes
        (e.g. ``"$1,200"`` -> ``1200.0``, ``"100px"`` -> ``100.0``). Values
        that cannot be parsed become ``NaN`` (``errors='coerce'``).

        Args:
            columns: Iterable of column names to convert.
        """
        if not self._has_data():
            return
        for col in columns:
            try:
                if col not in self.df.columns:
                    logger.warning("Numeric standardize: column '%s' missing.", col)
                    continue
                # Keep only digits, sign and decimal point.
                series = (
                    self.df[col]
                    .astype(str)
                    .str.replace(r"[^0-9.\-]", "", regex=True)
                    .replace("", np.nan)
                )
                self.df[col] = pd.to_numeric(series, errors="coerce")
                logger.info("Numeric column standardized: '%s'", col)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to standardize numeric '%s': %s", col, exc)

    def standardize_categorical(
        self, columns: Iterable[str], casing: str = "lower"
    ) -> None:
        """Trim whitespace and unify casing for text columns.

        Args:
            columns: Iterable of column names to clean.
            casing: One of ``"lower"``, ``"upper"`` or ``"title"``.
        """
        if not self._has_data():
            return
        for col in columns:
            try:
                if col not in self.df.columns:
                    logger.warning("Categorical standardize: column '%s' missing.", col)
                    continue
                series = self.df[col].astype(str).str.strip()
                if casing == "lower":
                    series = series.str.lower()
                elif casing == "upper":
                    series = series.str.upper()
                elif casing == "title":
                    series = series.str.title()
                else:
                    raise ValueError(f"Unknown casing option '{casing}'.")
                # Restore genuine missing values lost by astype(str).
                series = series.replace({"nan": np.nan, "none": np.nan})
                self.df[col] = series
                logger.info("Categorical column standardized: '%s'", col)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to standardize categorical '%s': %s", col, exc)

    def standardize_dates(self) -> None:
        """Auto-detect date-like columns by name and parse mixed formats.

        Detected columns are parsed into pandas datetime objects (handling
        mixed formats per row) and then rendered as ``DD/MM/YYYY`` strings.
        """
        if not self._has_data():
            return
        try:
            candidates = [
                col
                for col in self.df.columns
                if any(kw in col.lower() for kw in self.DATE_KEYWORDS)
            ]
            if not candidates:
                logger.info("No date-like columns detected.")
                return

            for col in candidates:
                try:
                    # format="mixed" lets pandas infer per-row, so columns
                    # mixing '2026-05-30' and '30/05/2026' parse correctly.
                    parsed = pd.to_datetime(
                        self.df[col],
                        errors="coerce",
                        dayfirst=True,
                        format="mixed",
                    )
                    self.df[col] = parsed.dt.strftime("%d/%m/%Y")
                    logger.info("Date column standardized: '%s'", col)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to parse dates in '%s': %s", col, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Date standardization step failed: %s", exc)

    def standardize_booleans(self, columns: Optional[Iterable[str]] = None) -> None:
        """Detect boolean-like columns and convert them to real booleans.

        Recognizes common spellings (case-insensitive, whitespace-trimmed):

            True  : true, t, yes, y, 1, positive, pos, on
            False : false, f, no, n, 0, negative, neg, off

        Converted columns use the pandas nullable ``boolean`` dtype, so genuine
        missing values stay as ``<NA>`` rather than becoming ``False``.

        Args:
            columns: Specific columns to convert. If ``None`` (default), every
                non-numeric column whose values are *entirely* boolean-like is
                auto-detected and converted. Numeric columns are skipped in
                auto mode so that genuine 0/1 counts are not misread as flags;
                pass them explicitly to force conversion.
        """
        if not self._has_data():
            return

        token_map = {t: True for t in self.TRUE_TOKENS}
        token_map.update({f: False for f in self.FALSE_TOKENS})

        if columns is None:
            auto = True
            targets = [
                col
                for col in self.df.columns
                if not pd.api.types.is_numeric_dtype(self.df[col])
                and not pd.api.types.is_bool_dtype(self.df[col])
            ]
        else:
            auto = False
            targets = list(columns)

        converted = []
        for col in targets:
            try:
                if col not in self.df.columns:
                    logger.warning("Boolean standardize: column '%s' missing.", col)
                    continue

                series = self.df[col]
                non_null = series.dropna()
                if non_null.empty:
                    continue

                # Normalize: lowercase, trim, and collapse whole-number floats
                # so a numeric 1.0/0.0 column reads as "1"/"0" (and matches the
                # token vocabulary) instead of "1.0"/"0.0".
                def _norm(s: pd.Series) -> pd.Series:
                    return (
                        s.astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
                    )

                normalized = _norm(non_null)
                distinct = set(normalized.unique())
                unmapped = distinct - token_map.keys()

                # Auto mode: only convert when EVERY value is boolean-like.
                if auto and unmapped:
                    continue
                if unmapped:  # explicit mode: warn, then coerce the rest
                    logger.warning(
                        "Column '%s' has non-boolean values %s; they become <NA>.",
                        col, sorted(unmapped),
                    )

                # Map normalized text to True/False; unknown -> <NA>.
                mapped = _norm(series).map(token_map)
                # Preserve genuinely missing inputs as <NA>.
                mapped = mapped.where(series.notna())
                self.df[col] = mapped.astype("boolean")
                converted.append(col)
                logger.info("Boolean column standardized: '%s'", col)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to standardize boolean '%s': %s", col, exc)

        if auto and not converted:
            logger.info("No boolean-like columns detected.")

    # ===================================================================== #
    # DATA CLEANING (duplicates & missing values)
    # ===================================================================== #
    def remove_duplicates(
        self, subset: Optional[Iterable[str]] = None, keep: str = "first"
    ) -> None:
        """Drop duplicate rows from the DataFrame.

        Args:
            subset: Columns to consider when identifying duplicates. ``None``
                (default) compares whole rows. Pass a list (e.g.
                ``["listing_id"]``) to dedupe on key columns only.
            keep: Which duplicate to keep - ``"first"``, ``"last"`` or
                ``False`` (drop every duplicated row entirely).
        """
        if not self._has_data():
            return
        try:
            before = len(self.df)
            self.df = (
                self.df.drop_duplicates(subset=subset, keep=keep)
                .reset_index(drop=True)
            )
            removed = before - len(self.df)
            logger.info(
                "Removed %d duplicate row(s); %d remain.", removed, len(self.df)
            )
        except KeyError as exc:
            logger.error("remove_duplicates: unknown column(s) %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to remove duplicates: %s", exc)

    def handle_missing(
        self,
        column: str,
        strategy: str = "median",
        fill_value: object = None,
    ) -> None:
        """Fill or drop missing values in a single column of your choice.

        Args:
            column: The column to operate on.
            strategy: How to handle the gaps:
                ``"mean"`` / ``"median"`` - numeric columns only.
                ``"mode"``                - most frequent value (any dtype).
                ``"constant"``            - fill with ``fill_value``.
                ``"ffill"`` / ``"bfill"`` - carry the previous/next value.
                ``"drop"``                - drop rows missing this column.
            fill_value: The value used when ``strategy="constant"``.
        """
        if not self._has_data():
            return
        try:
            if column not in self.df.columns:
                logger.warning("handle_missing: column '%s' missing.", column)
                return

            series = self.df[column]
            missing = int(series.isna().sum())
            if missing == 0:
                logger.info("Column '%s' has no missing values.", column)
                return

            if strategy in ("mean", "median"):
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(
                        f"strategy '{strategy}' requires a numeric column."
                    )
                value = series.mean() if strategy == "mean" else series.median()
                self.df[column] = series.fillna(value)
            elif strategy == "mode":
                modes = series.mode(dropna=True)
                if modes.empty:
                    raise ValueError("No mode available (column is all-null).")
                value = modes.iloc[0]
                self.df[column] = series.fillna(value)
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("strategy 'constant' requires fill_value.")
                value = fill_value
                self.df[column] = series.fillna(fill_value)
            elif strategy in ("ffill", "bfill"):
                value = strategy
                self.df[column] = (
                    series.ffill() if strategy == "ffill" else series.bfill()
                )
            elif strategy == "drop":
                before = len(self.df)
                self.df = self.df[series.notna()].reset_index(drop=True)
                logger.info(
                    "Dropped %d row(s) missing '%s'.", before - len(self.df), column
                )
                return
            else:
                raise ValueError(f"Unknown strategy '{strategy}'.")

            logger.info(
                "Filled %d missing value(s) in '%s' using %s (value=%r).",
                missing, column, strategy, value,
            )
        except ValueError as exc:
            logger.error("handle_missing aborted for '%s': %s", column, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to handle missing in '%s': %s", column, exc)

    # ===================================================================== #
    # DISTRIBUTION-BASED OUTLIER CLEANER
    # ===================================================================== #
    def clean_outliers(
        self, df: pd.DataFrame, column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split ``df`` into cleaned and outlier sets for ``column``.

        The distribution of ``column`` is tested for normality. The Z-score
        method is used when the data looks normal; the IQR method is used when
        it is skewed.

        Args:
            df: Source DataFrame.
            column: Numerical column to analyze.

        Returns:
            ``(cleaned_df, outlier_df)``. On failure the original frame is
            returned as ``cleaned`` with an empty ``outlier`` frame.
        """
        print("\n" + "=" * 60)
        print(f"OUTLIER CLEANER -> column '{column}'")
        print("=" * 60 + "\n")

        empty = df.iloc[0:0]
        try:
            # --- Guardrail: must be numeric --------------------------------
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(
                    f"Column '{column}' is not numeric "
                    f"(dtype={df[column].dtype}). Aborting outlier step."
                )

            series = df[column].dropna()
            if series.empty:
                logger.warning("Column '%s' has no usable values.", column)
                return df.copy(), empty.copy()

            # --- Distribution check ----------------------------------------
            skewness = float(series.skew())
            # normaltest needs a reasonable sample size; fall back to skew.
            is_normal = False
            if len(series) >= 8:
                _, p_value = stats.normaltest(series)
                is_normal = p_value > 0.05 and abs(skewness) < 0.5
                print(f"  normaltest p-value : {p_value:.4f}")
            else:
                is_normal = abs(skewness) < 0.5
                print("  Sample < 8: relying on skewness only.")
            print(f"  Skewness           : {skewness:.4f}")
            dist_type = "NORMAL" if is_normal else "SKEWED"
            print(f"  Distribution type  : {dist_type}")

            # --- Flag outliers ---------------------------------------------
            if is_normal:
                z_scores = np.abs(stats.zscore(df[column], nan_policy="omit"))
                outlier_mask = pd.Series(z_scores, index=df.index) > 3
                print("  Method             : Z-Score (|z| > 3)")
            else:
                q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (df[column] < lower) | (df[column] > upper)
                print(f"  Method             : IQR  bounds=[{lower:.2f}, {upper:.2f}]")

            outlier_mask = outlier_mask.fillna(False)
            cleaned_df = df[~outlier_mask].copy()
            outlier_df = df[outlier_mask].copy()

            removed_pct = (len(outlier_df) / len(df) * 100) if len(df) else 0.0
            print(f"\n  Cleaned shape       : {cleaned_df.shape}")
            print(f"  Outlier shape       : {outlier_df.shape}")
            print(f"  Data removed        : {removed_pct:.2f}%")
            return cleaned_df, outlier_df

        except ValueError as exc:
            logger.error("Outlier guardrail triggered: %s", exc)
            return df.copy(), empty.copy()
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error in outlier cleaner: %s", exc)
            return df.copy(), empty.copy()

    # ===================================================================== #
    # ROBUST EXCEL-COMPATIBLE EXPORT
    # ===================================================================== #
    @staticmethod
    def export_csv(df: pd.DataFrame, path: str) -> bool:
        """Save ``df`` to ``path`` as Excel-friendly UTF-8-SIG CSV.

        Creates any missing parent directories and reports each step. Returns
        ``True`` on success, ``False`` otherwise.
        """
        print("\n" + "=" * 60)
        print("EXPORT")
        print("=" * 60 + "\n")
        try:
            print("  Step 1: Validating path...")
            target = Path(path)

            print("  Step 2: Ensuring parent directories exist...")
            target.parent.mkdir(parents=True, exist_ok=True)

            print("  Step 3: Saving with 'utf-8-sig' encoding (Excel BOM)...")
            df.to_csv(target, index=False, encoding="utf-8-sig")

            print(f"  Step 4: Done. File written to: {target.resolve()}" + "\n")
            logger.info("Export successful (%d rows).", len(df))
            return True

        except PermissionError:
            logger.error(
                "Permission denied for '%s'. Is the file open in Excel?", path
            )
        except FileNotFoundError:
            logger.error("Path could not be resolved: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during export: %s", exc)
        return False

    # ===================================================================== #
    # DATABASE INTEGRATION (SQLAlchemy + MySQL)
    # ===================================================================== #
    @staticmethod
    def get_db_engine(
        config: Optional[dict] = None,
        prompt_credentials: bool = False,
        test_connection: bool = True,
    ) -> Optional[Engine]:
        """Securely build a SQLAlchemy engine for MySQL (``mysql+pymysql``).

        The connection string is assembled with :meth:`sqlalchemy.engine.URL.create`,
        which URL-encodes special characters in the credentials so they cannot
        break the URL. No connection details are hardcoded: they come from
        environment variables (``DB_CONFIG``) or are typed at runtime.

        Args:
            config: Optional overrides merged on top of the module-level
                ``DB_CONFIG`` (e.g. ``{"database": "other_db"}``).
            prompt_credentials: If ``True``, show an interactive login prompt
                for all five fields. Host, port, database and username are
                visible as typed; the password is hidden via getpass.
            test_connection: If ``True``, run a ``SELECT 1`` to confirm the
                engine can actually reach the server before returning it.

        Returns:
            A live :class:`sqlalchemy.engine.Engine`, or ``None`` on failure.
        """
        print("\n" + "=" * 60)
        print("DATABASE ENGINE")
        print("=" * 60 + "\n")
        cfg = {**DB_CONFIG, **(config or {})}
        try:
            print("  Step 1: Resolving connection details...")
            if prompt_credentials:
                cfg = DataPipeline._prompt_connection_details(cfg)

            # Validate every required field; report all that are missing.
            missing = [
                field
                for field in ("host", "port", "database", "username", "password")
                if not cfg.get(field)
            ]
            if missing:
                raise ValueError(
                    "Missing connection field(s): "
                    f"{', '.join(missing)}. Set the matching env var "
                    "(DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD) or enter it "
                    "via get_db_engine(prompt_credentials=True)."
                )

            print("  Step 2: Building connection URL (mysql+pymysql)...")
            url = URL.create(
                drivername=cfg["drivername"],
                username=cfg["username"],
                password=cfg["password"],
                host=cfg["host"],
                port=cfg["port"],
                database=cfg["database"],
            )

            print("  Step 3: Creating engine...")
            engine = create_engine(url, pool_pre_ping=True)

            if test_connection:
                print("  Step 4: Testing connection (SELECT 1)...")
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info(
                    "Connected as '%s' to %s/%s",
                    cfg["username"], cfg["host"], cfg["database"],
                )
            else:
                logger.info("Engine created (connection not tested).")
            return engine

        except SQLAlchemyError as exc:
            logger.error("Database connection failed: %s", exc)
        except ValueError as exc:
            logger.error("Configuration error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error creating DB engine: %s", exc)
        return None

    @staticmethod
    def store_dataframe(
        df: pd.DataFrame,
        table_name: str,
        engine: Optional[Engine],
        if_exists: str = "replace",
    ) -> bool:
        """Write ``df`` to a MySQL table via the SQLAlchemy ``engine``.

        Args:
            df: DataFrame to persist.
            table_name: Destination table name.
            engine: Engine from :meth:`get_db_engine`.
            if_exists: ``'replace'`` (default, good for testing) drops and
                recreates the table each run. See the commented ``'append'``
                line below for daily incremental loads that keep old rows.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        print("\n" + "=" * 60)
        print(f"STORE -> table '{table_name}'")
        print("=" * 60)
        try:
            if engine is None:
                raise ValueError("No database engine provided.")
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None - nothing to store.")

            print("  Step 1: Writing rows to MySQL...")
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,   # 'replace' = wipe & rewrite (initial testing)
                # if_exists="append",  # <-- switch to this for daily incremental loads
                index=False,           # don't write the DataFrame index as a column
            )
            print(f"  Step 2: Done. {len(df)} rows written to '{table_name}'.")
            logger.info(
                "Stored %d rows in '%s' (if_exists='%s').",
                len(df), table_name, if_exists,
            )
            return True

        except SQLAlchemyError as exc:
            logger.error("Failed to write to '%s': %s", table_name, exc)
        except ValueError as exc:
            logger.error("Store aborted: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error storing DataFrame: %s", exc)
        return False

    @staticmethod
    def query_to_dataframe(
        query: str, engine: Optional[Engine]
    ) -> Optional[pd.DataFrame]:
        """Run an SQL ``query`` against MySQL and return a clean DataFrame.

        Args:
            query: SQL statement to execute (e.g. ``"SELECT * FROM t LIMIT 10"``).
            engine: Engine from :meth:`get_db_engine`.

        Returns:
            A :class:`pandas.DataFrame` with the results, or ``None`` on failure.
        """
        print("\n" + "=" * 60)
        print("QUERY")
        print("=" * 60 + "\n")
        try:
            if engine is None:
                raise ValueError("No database engine provided.")
            if not query or not str(query).strip():
                raise ValueError("Query string is empty.")

            print("  Step 1: Executing query...")
            result = pd.read_sql(text(query), con=engine)

            print(f"  Step 2: Retrieved {len(result)} rows.")
            logger.info("Query returned %d rows.", len(result))
            return result

        except SQLAlchemyError as exc:
            logger.error("Query failed: %s", exc)
        except ValueError as exc:
            logger.error("Query aborted: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error running query: %s", exc)
        return None

    @staticmethod
    def infer_sql_schema(df: pd.DataFrame, table_name: str = "my_table") -> Optional[str]:
        """Infer a MySQL ``CREATE TABLE`` statement from a DataFrame.

        Each column is mapped to a MySQL type based on its dtype and values:

            bool / boolean       -> TINYINT(1)
            integer              -> INT, or BIGINT if values exceed INT range
            float                -> DOUBLE
            datetime             -> DATETIME
            text / object        -> VARCHAR(n) sized to the longest value,
                                    or TEXT when very long

        Columns that contain nulls are declared ``NULL``; otherwise ``NOT NULL``.
        This is a starting point - review and adjust types (e.g. DECIMAL for
        money) before running it on your server.

        Args:
            df: The DataFrame to inspect (e.g. the cleaned output).
            table_name: Name for the generated table.

        Returns:
            The CREATE TABLE statement as a string, or ``None`` on failure.
        """
        print("\n" + "=" * 60)
        print(f"INFER SQL SCHEMA -> table '{table_name}'")
        print("=" * 60)
        try:
            if df is None or df.shape[1] == 0:
                raise ValueError("DataFrame has no columns to describe.")

            lines = []
            for col in df.columns:
                series = df[col]
                dtype = series.dtype

                if pd.api.types.is_bool_dtype(dtype):
                    sql_type = "TINYINT(1)"
                elif pd.api.types.is_integer_dtype(dtype):
                    non_null = series.dropna()
                    lo = int(non_null.min()) if not non_null.empty else 0
                    hi = int(non_null.max()) if not non_null.empty else 0
                    in_int = -2147483648 <= lo and hi <= 2147483647
                    sql_type = "INT" if in_int else "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "DOUBLE"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "DATETIME"
                else:  # text / object / category
                    lengths = series.dropna().astype(str).str.len()
                    max_len = int(lengths.max()) if not lengths.empty else 1
                    if max_len > 1000:
                        sql_type = "TEXT"
                    else:
                        # Size with headroom, clamped to a sensible range.
                        bound = min(max(max_len * 2, 16), 1000)
                        sql_type = f"VARCHAR({bound})"

                null_clause = "NULL" if series.isna().any() else "NOT NULL"
                lines.append(f"  `{col}` {sql_type} {null_clause}")

            ddl = (
                f"CREATE TABLE `{table_name}` (\n"
                + ",\n".join(lines)
                + "\n);"
            )
            print(ddl)
            logger.info("Inferred SQL schema for %d column(s).", df.shape[1])
            return ddl

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to infer SQL schema: %s", exc)
            return None

    # ===================================================================== #
    # ORCHESTRATOR
    # ===================================================================== #
    def run(
        self,
        config: Optional[dict] = None,
        source: str = "file",
        destination: str = "file",
        input_path: Optional[object] = None,
        output_path: Optional[object] = None,
        source_query: Optional[str] = None,
        dest_table: Optional[str] = None,
        engine: "Optional[Engine]" = None,
        preview: bool = True,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run the whole pipeline end-to-end from a single config.

        Stages run in a fixed, safe order. Each stage is OPT-IN: if its
        ``config`` key is missing or empty, that stage is skipped. The
        individual methods are reused unchanged - ``run`` only orchestrates.

        Recognized ``config`` keys (all optional):
            numeric_cols      : list[str]          -> standardize_numeric
            categorical_cols  : list[str]          -> standardize_categorical
            categorical_case  : "lower"/"upper"/"title" (default "lower")
            standardize_dates : bool (default True) -> standardize_dates
            boolean_cols      : list[str] | None    -> standardize_booleans
                                (omit key to skip; None = auto-detect)
            drop_duplicates   : bool | dict         -> remove_duplicates
                                (True, or {"subset": [...], "keep": "first"})
            missing           : dict[col, strategy] -> handle_missing per column
            fill_values       : dict[col, value]    -> fill_value for "constant"
            outlier_col       : str                 -> clean_outliers
            outlier_path      : path                -> export outliers (file dest)
            outlier_table     : str                 -> store outliers (db dest)

        Args:
            config: The stage configuration described above.
            source: "file" (read ``input_path``) or "database" (run
                ``source_query`` on ``engine``).
            destination: "file" (write ``output_path``) or "database" (write
                ``dest_table`` via ``engine``).
            input_path / output_path: File locations for file source/destination.
            source_query / dest_table: SQL query / table for database I/O.
            engine: A SQLAlchemy engine, required if either side is "database".
            preview: Print a before/after head() preview around standardization.

        Returns:
            ``(cleaned, outliers)``. ``outliers`` is ``None`` when no
            ``outlier_col`` is configured. On a fatal ingest error, returns
            ``(None, None)``.
        """
        config = config or {}
        valid = {"file", "database"}
        if source not in valid or destination not in valid:
            logger.error("source/destination must each be 'file' or 'database'.")
            return None, None
        if "database" in (source, destination) and engine is None:
            logger.error("A database engine is required for database I/O.")
            return None, None

        # --- 1) Ingest ----------------------------------------------------
        if source == "database":
            self.df = DataPipeline.query_to_dataframe(source_query, engine)
            if self.df is None:
                logger.error("run(): could not read data from the database.")
                return None, None
        else:
            if self.load_data(input_path) is None:
                logger.error("run(): could not load input file: %s", input_path)
                return None, None

        # --- 2) Health report --------------------------------------------
        self.health_report()

        before_preview = self.df.head().copy() if preview else None

        # --- 3) Standardization (opt-in per stage) -----------------------
        self.standardize_column_names()
        if config.get("numeric_cols"):
            self.standardize_numeric(config["numeric_cols"])
        if config.get("categorical_cols"):
            self.standardize_categorical(
                config["categorical_cols"],
                config.get("categorical_case", "lower"),
            )
        if config.get("standardize_dates", True):
            self.standardize_dates()
        if "boolean_cols" in config:
            self.standardize_booleans(config["boolean_cols"])

        if preview:
            print("\n[Before Standardization]")
            print(before_preview.to_string(index=False))
            print("\n[After Standardization]")
            print(self.df.head().to_string(index=False) + "\n")

        # --- 4) Cleaning --------------------------------------------------
        dd = config.get("drop_duplicates")
        if dd:
            if isinstance(dd, dict):
                self.remove_duplicates(dd.get("subset"), dd.get("keep", "first"))
            else:
                self.remove_duplicates()
        fill_values = config.get("fill_values", {})
        for col, strategy in config.get("missing", {}).items():
            self.handle_missing(col, strategy, fill_values.get(col))

        # --- 5) Outliers (optional) --------------------------------------
        outlier_col = config.get("outlier_col")
        if outlier_col:
            cleaned, outliers = self.clean_outliers(self.df, outlier_col)
        else:
            cleaned, outliers = self.df, None

        # --- 6) Store -----------------------------------------------------
        if destination == "database":
            DataPipeline.store_dataframe(cleaned, dest_table, engine)
        else:
            DataPipeline.export_csv(cleaned, output_path)

        # Optionally store the flagged outliers alongside the cleaned data.
        if outliers is not None and not outliers.empty:
            outlier_path = config.get("outlier_path")
            outlier_table = config.get("outlier_table")
            if destination == "database" and outlier_table:
                DataPipeline.store_dataframe(outliers, outlier_table, engine)
            elif destination == "file" and outlier_path:
                DataPipeline.export_csv(outliers, outlier_path)

        logger.info("run(): pipeline complete.")
        return cleaned, outliers

    # ===================================================================== #
    # Internal helpers
    # ===================================================================== #
    @staticmethod
    def _prompt_connection_details(cfg: dict) -> dict:
        """Collect the five connection fields at runtime.

        Host, port, database and username are read with :func:`input`, so they
        are visible as you type. The password is read with
        :func:`getpass.getpass`, so its keystrokes stay hidden for safety. For
        any field left blank, the existing value from ``cfg`` (an environment
        variable, if set) is kept. Returns a new dict; the input is not mutated.

        Args:
            cfg: Current connection config used as the fallback for blanks.

        Returns:
            A copy of ``cfg`` updated with whatever the user entered.
        """
        cfg = dict(cfg)  # work on a copy so callers' config is untouched

        print("\n  --- Database Login ---")
        print("  (leave a field blank to use its env var; password stays hidden)")

        host = input("  Host: ").strip()
        if host:
            cfg["host"] = host

        port_raw = input("  Port: ").strip()
        if port_raw:
            try:
                cfg["port"] = int(port_raw)
            except ValueError:
                logger.warning(
                    "Invalid port entered; keeping the existing/env value."
                )

        database = input("  Database: ").strip()
        if database:
            cfg["database"] = database

        username = input("  Username: ").strip()
        if username:
            cfg["username"] = username

        # Hidden input. Password is never stripped (it may contain spaces).
        password = getpass("  Password: ")
        if password:
            cfg["password"] = password

        return cfg

    def _has_data(self) -> bool:
        """Return ``True`` if a DataFrame is loaded, else log and return False."""
        if self.df is None:
            logger.error("No DataFrame loaded. Aborting step.")
            return False
        return True


# --------------------------------------------------------------------------- #
# EXECUTION BLOCK
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ===================================================================== #
    # CHOOSE SOURCE & DESTINATION (edit these two switches)
    # ===================================================================== #
    SOURCE = "file"        # "file"     -> read from the input folder
                           # "database" -> read from MySQL (SOURCE_QUERY)
    DESTINATION = "file"   # "file"     -> write to the output folder
                           # "database" -> write to MySQL (DEST_TABLE)

    # Resolve file paths relative to THIS script's folder, so the input/output
    # folders are found no matter which directory you launch Python from.
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_PATH = BASE_DIR / "input" / "house_listings_messy_dataset.csv"
    OUTPUT_PATH = BASE_DIR / "output" / "house_listings_clean_output.csv"

    # Used only when SOURCE / DESTINATION is "database":
    SOURCE_QUERY = "SELECT * FROM house_listings_clean_output"   # what to read in
    DEST_TABLE = "house_listings_clean_output"                   # where to write out

    # --- Validate the switches -------------------------------------------
    valid = {"file", "database"}
    if SOURCE not in valid or DESTINATION not in valid:
        raise SystemExit("SOURCE and DESTINATION must each be 'file' or 'database'.")

    # --- Open one DB engine if either side needs it ----------------------
    engine = None
    if "database" in (SOURCE, DESTINATION):
        engine = DataPipeline.get_db_engine(prompt_credentials=True)
        if engine is None:
            raise SystemExit("A database connection is required but could not be made.")

    # --- Stage configuration (edit to match your dataset's columns) ------
    # Any stage whose key is missing/empty is skipped. Adjust the column names to your own data.
    
    CONFIG = {
        "numeric_cols": ["area_sqm", "price", "price_per_sqm"],
        "categorical_cols": ["energy_class", "location"],
        "categorical_case": "title",
        "standardize_dates": True,
        "boolean_cols": None,          # None = auto-detect; omit key to skip
        "drop_duplicates": True,
        "missing": {"air_conditioning": "mode","alarm": "mode", "bathrooms": "median", "open_parking_spots": "constant", "closed_parking_spots": "constant"},                 # e.g. {"purchase_amount": "median"}
        "fill_values": {"open_parking_spots": 0, "closed_parking_spots": 0},             # e.g. {"energy_class": "UNKNOWN"}
        "outlier_col": "price_per_sqm",
        # Where to save the flagged outliers (only used if outliers exist):
        "outlier_path": BASE_DIR / "output" / "house_listings_outliers.csv",  # used when DESTINATION == "file"
        "outlier_table": "outlier_listings",                                  # used when DESTINATION == "database"
    }

    # --- Run the whole pipeline in one call ------------------------------
    pipeline = DataPipeline()
    cleaned, outliers = pipeline.run(
        CONFIG,
        source=SOURCE,
        destination=DESTINATION,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        source_query=SOURCE_QUERY,
        dest_table=DEST_TABLE,
        engine=engine,
    )
    if cleaned is None:
        raise SystemExit("Pipeline failed during ingestion. See logs above.")

    print("\nPipeline complete.")
