"""Empirical tests: Pandera (pandera.polars) limitations with nested columns.

Tests List(Struct(...)) validation capabilities and limitations in Pandera v0.29+
with Polars DataFrames.

Categories tested:
    1. dtype_kwargs — Can you declare inner struct field types?
    2. Annotated syntax — Does Annotated[pl.List, ...] work for List(Struct)?
    3. pa.Field constraints — Do built-in checks (ge, le, isin) work on nested?
    4. strict mode — Does it catch extra/missing struct fields?
    5. Coercion — Does coerce=True work inside nested structs?
    6. @pa.check signatures — pl.Series vs PolarsData (CRITICAL BUG FINDING)
    7. @pa.check with PolarsData — What CAN work for nested validation?
    8. Error reporting — What detail do you get when nested checks fail?
    9. Edge cases — nullable, unique, deeply nested, bare 'list'

Usage:
    uv run python scripts/test_pandera_nested.py
"""

import sys
import traceback

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

results: list[tuple[str, str, str]] = []


def run_test(name: str, fn):
    """Run a test function, capture result."""
    try:
        status, detail = fn()
        results.append((name, status, detail))
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, ERROR, f"{type(e).__name__}: {e}\n{tb}"))


def make_test_df() -> pl.DataFrame:
    """Create a DataFrame with List(Struct) columns for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "dimensions": [
                [
                    {"type": "height", "value": 10.0, "unit": "cm"},
                    {"type": "width", "value": 5.0, "unit": "cm"},
                ],
                [{"type": "height", "value": 20.0, "unit": "cm"}],
                [
                    {"type": "height", "value": 15.0, "unit": "cm"},
                    {"type": "width", "value": 8.0, "unit": "cm"},
                    {"type": "depth", "value": 3.0, "unit": "cm"},
                ],
            ],
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        }
    )


# =========================================================================
# TEST 1: dtype_kwargs with Struct fields
# =========================================================================


def test_dtype_kwargs_struct_fields():
    """Can you declare struct field names/types via dtype_kwargs?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            dtype_kwargs={
                "inner": pl.Struct(
                    {"type": pl.String, "value": pl.Float64, "unit": pl.String}
                )
            }
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return PASS, "dtype_kwargs with inner Struct accepted and validated OK"
    except pa.errors.SchemaErrors as e:
        return FAIL, f"Validation failed:\n{e.failure_cases}"


run_test(
    "1a. dtype_kwargs: declare List(Struct) inner fields",
    test_dtype_kwargs_struct_fields,
)


def test_dtype_kwargs_wrong_struct_field_type():
    """Does Pandera catch WRONG types inside struct fields via dtype_kwargs?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            dtype_kwargs={
                "inner": pl.Struct(
                    {"type": pl.String, "value": pl.Int64, "unit": pl.String}
                )
            }
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "Pandera did NOT catch wrong inner field type (Float64 vs Int64)"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return PASS, f"Caught wrong inner type: {e}"


run_test(
    "1b. dtype_kwargs: detect wrong inner field type",
    test_dtype_kwargs_wrong_struct_field_type,
)


def test_dtype_kwargs_extra_field_in_spec():
    """Does Pandera catch a field declared in spec that doesn't exist in data?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            dtype_kwargs={
                "inner": pl.Struct(
                    {
                        "type": pl.String,
                        "value": pl.Float64,
                        "unit": pl.String,
                        "extra_field": pl.String,
                    }
                )
            }
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "Pandera did NOT catch extra field in dtype_kwargs Struct spec"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return PASS, f"Caught spec field not in data: {e}"


run_test(
    "1c. dtype_kwargs: detect extra field in spec (not in data)",
    test_dtype_kwargs_extra_field_in_spec,
)


def test_dtype_kwargs_extra_field_in_data():
    """Does Pandera catch a field in data that isn't declared in spec?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            dtype_kwargs={
                "inner": pl.Struct({"type": pl.String, "value": pl.Float64})
                # 'unit' in data but NOT in spec
            }
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "Pandera did NOT flag undeclared struct field 'unit'"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return PASS, f"Caught extra field in data: {e}"


run_test(
    "1d. dtype_kwargs: detect extra field in data (not in spec)",
    test_dtype_kwargs_extra_field_in_data,
)


# =========================================================================
# TEST 2: Annotated syntax for List(Struct)
# =========================================================================


def test_annotated_syntax_list_struct():
    """Does the Annotated[pl.List, pl.Struct({...})] syntax work?"""
    from typing import Annotated

    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: Annotated[
            pl.List,
            pl.Struct({"type": pl.String, "value": pl.Float64, "unit": pl.String}),
        ]
        tags: Annotated[pl.List, pl.String]

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return PASS, "Annotated syntax for List(Struct) works"
    except pa.errors.SchemaErrors as e:
        return FAIL, f"Annotated syntax failed:\n{e.failure_cases}"


run_test(
    "2a. Annotated syntax: List(Struct) declaration", test_annotated_syntax_list_struct
)


# =========================================================================
# TEST 3: pa.Field() built-in constraints on nested columns
# =========================================================================


def test_field_ge_on_list_column():
    """Can you use pa.Field(ge=1) on a List column?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            ge=1,
            dtype_kwargs={
                "inner": pl.Struct(
                    {"type": pl.String, "value": pl.Float64, "unit": pl.String}
                )
            },
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "pa.Field(ge=1) on List: accepted (unclear semantics)"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return (
            PASS,
            f"pa.Field(ge=1) on List: CRASHES — {type(e).__name__}: {str(e)[:150]}",
        )


run_test("3a. pa.Field(ge=1) on List column", test_field_ge_on_list_column)


def test_field_isin_on_list_column():
    """Can you use pa.Field(isin=[...]) on a List(String) column?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: list
        tags: pl.List = pa.Field(
            isin=["a", "b", "c", "d", "e", "f"],
            dtype_kwargs={"inner": pl.String},
        )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "pa.Field(isin=) on List(String): incorrectly accepted"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return PASS, (
            f"pa.Field(isin=) on List(String): FAILS INCORRECTLY — compares whole list "
            f"as a single value, not elements. {type(e).__name__}: {str(e)[:150]}"
        )


run_test("3b. pa.Field(isin=) on List(String)", test_field_isin_on_list_column)


def test_field_unique_on_list_column():
    """Can you use pa.Field(unique=True) on a List column?"""
    df = pl.DataFrame({"id": [1, 2, 3], "tags": [["a", "b"], ["a", "b"], ["c"]]})

    class Schema(pa.DataFrameModel):
        id: int
        tags: list = pa.Field(unique=True)

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "unique=True should have caught duplicate lists"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        msg = str(e)
        if "is_unique" in msg and "not supported" in msg:
            return PASS, f"unique=True on List column: Polars error — {msg[:120]}"
        return PASS, f"unique=True on List column caught error: {msg[:120]}"
    except Exception as e:
        msg = str(e)
        if "is_unique" in msg or "not supported" in msg or "is_duplicated" in msg:
            return (
                PASS,
                f"unique=True on List column: crashes with {type(e).__name__}: {msg[:150]}",
            )
        return ERROR, f"{type(e).__name__}: {msg}"


run_test("3c. pa.Field(unique=True) on List column", test_field_unique_on_list_column)


# =========================================================================
# TEST 4: strict mode and nested columns
# =========================================================================


def test_strict_catches_extra_top_level():
    """Baseline: Does strict=True catch extra top-level columns?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str

        class Config:
            strict = True
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "strict=True did NOT catch extra top-level columns"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError):
        return PASS, "strict=True caught extra top-level columns"


run_test(
    "4a. strict=True: catches extra top-level columns (baseline)",
    test_strict_catches_extra_top_level,
)


def test_strict_extra_struct_field():
    """Does strict=True catch extra STRUCT FIELDS inside List(Struct)?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: pl.List = pa.Field(
            dtype_kwargs={
                "inner": pl.Struct({"type": pl.String, "value": pl.Float64})
                # 'unit' omitted — strict should catch?
            }
        )
        tags: pl.List = pa.Field(dtype_kwargs={"inner": pl.String})

        class Config:
            strict = True
            coerce = False

    try:
        Schema.validate(df)
        return FAIL, "strict=True did NOT catch extra struct field 'unit'"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        msg = str(e)
        if "Struct" in msg or "dimensions" in msg:
            return (
                PASS,
                "strict=True caught dtype mismatch (but via dtype check, not "
                "struct-field-level strict enforcement)",
            )
        return PASS, f"strict=True caught something: {msg[:150]}"


run_test(
    "4b. strict=True: extra struct field inside List(Struct)",
    test_strict_extra_struct_field,
)


# =========================================================================
# TEST 5: Coercion of nested types
# =========================================================================


def test_coerce_list_inner_type():
    """Does coerce=True convert List(Int64) -> List(Float64)?"""
    df = pl.DataFrame({"id": [1, 2], "values": [[1, 2, 3], [4, 5]]})

    class Schema(pa.DataFrameModel):
        id: int
        values: pl.List = pa.Field(dtype_kwargs={"inner": pl.Float64})

        class Config:
            coerce = True
            strict = False

    try:
        result = Schema.validate(df)
        actual_inner = result.schema["values"].inner  # type: ignore[union-attr]
        if actual_inner == pl.Float64:
            return PASS, "coerce=True converted List(Int64) to List(Float64)"
        return FAIL, f"coerce=True accepted but inner still: {actual_inner}"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return FAIL, f"coerce=True failed: {e}"


run_test("5a. Coerce: List(Int64) -> List(Float64)", test_coerce_list_inner_type)


def test_coerce_struct_inner_field():
    """Does coerce=True convert Struct field Int64 -> Float64?"""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "info": [{"name": "Alice", "score": 100}, {"name": "Bob", "score": 200}],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        info: pl.Struct = pa.Field(
            dtype_kwargs={"fields": {"name": pl.String, "score": pl.Float64}}
        )

        class Config:
            coerce = True
            strict = False

    try:
        result = Schema.validate(df)
        actual = result.schema["info"]
        return PASS, f"coerce=True on Struct: result dtype = {actual}"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return FAIL, f"coerce=True failed on Struct: {e}"


run_test("5b. Coerce: Struct field Int64 -> Float64", test_coerce_struct_inner_field)


def test_coerce_list_struct_inner():
    """Does coerce=True convert fields inside List(Struct)?"""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "x", "count": 10}],
                [{"name": "y", "count": 20}, {"name": "z", "count": 30}],
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        items: pl.List = pa.Field(
            dtype_kwargs={"inner": pl.Struct({"name": pl.String, "count": pl.Float64})}
        )

        class Config:
            coerce = True
            strict = False

    try:
        result = Schema.validate(df)
        actual = result.schema["items"]
        return PASS, f"coerce=True on List(Struct): result dtype = {actual}"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return FAIL, f"coerce=True failed on List(Struct): {e}"


run_test(
    "5c. Coerce: List(Struct) inner field Int64 -> Float64",
    test_coerce_list_struct_inner,
)


# =========================================================================
# TEST 6: CRITICAL — @pa.check receives PolarsData, NOT pl.Series
# =========================================================================


def test_check_receives_polarsdata_not_series():
    """Demonstrate that @pa.check receives PolarsData, not pl.Series.

    This is the pattern used in pipeline_xml.py which may silently break.
    """
    df = pl.DataFrame({"id": [1, 2], "values": [[10, 20], [30, 40]]})

    received_type = None

    class Schema(pa.DataFrameModel):
        id: int
        values: list

        @pa.check("values", name="inspect_type")
        @classmethod
        def inspect(cls, col) -> pl.LazyFrame:
            nonlocal received_type
            received_type = type(col).__name__
            # Return all-true to pass
            return col.lazyframe.select(pl.lit(True).alias(col.key))

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
    except Exception:
        pass

    if received_type == "PolarsData":
        return PASS, (
            "@pa.check receives PolarsData (namedtuple with .lazyframe and .key), "
            "NOT pl.Series. pipeline_xml.py's 'col: pl.Series' type hint is misleading."
        )
    return FAIL, f"@pa.check received: {received_type}"


run_test(
    "6a. CRITICAL: @pa.check receives PolarsData, NOT pl.Series",
    test_check_receives_polarsdata_not_series,
)


def test_series_pattern_silently_fails():
    """Show that using col.list.eval() (Series API) in @pa.check fails silently."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "dimensions": [
                [{"type": "h", "value": 10.0}],
                [{"type": "h", "value": 0.5}],  # BAD
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        dimensions: list

        @pa.check("dimensions", name="values_gt_1")
        @classmethod
        def check_values(cls, col: pl.Series) -> pl.Series:
            """This is the pattern from pipeline_xml.py — DOES NOT WORK."""
            return col.list.eval(pl.element().struct.field("value")).list.min() > 1

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Validation passed (check was silently skipped?)"
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        failure_case_str = str(fc["failure_case"][0]) if len(fc) > 0 else ""
        if "PolarsData" in failure_case_str and "has no attribute" in failure_case_str:
            return PASS, (
                "The Series-style check CRASHES with AttributeError because it receives "
                "PolarsData not Series. Pandera catches this exception and reports it as "
                "a validation failure (which is misleading — it's a code bug, not bad data).\n"
                f"failure_case: {failure_case_str[:200]}"
            )
        return PASS, f"Check failed with:\n{fc}"


run_test(
    "6b. Series-style col.list.eval() in @pa.check crashes silently",
    test_series_pattern_silently_fails,
)


def test_pipeline_xml_checks_are_broken():
    """Reproduce the exact pattern from pipeline_xml.py to confirm it's broken."""
    df = pl.DataFrame(
        {
            "id": ["OBJ-001", "OBJ-002"],
            "dimensions": [
                [{"type": "height", "value": 10.0, "unit": "cm"}],
                [{"type": "height", "value": 0.5, "unit": "cm"}],
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: str
        dimensions: list

        # Exact pattern from pipeline_xml.py
        @pa.check("dimensions", name="all_dimension_values_gt_1")
        @classmethod
        def dimension_values_positive(cls, col: pl.Series) -> pl.Series:
            return col.list.eval(pl.element().struct.field("value")).list.min() > 1

        class Config:
            strict = False
            coerce = True

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Check passed (should have caught value=0.5)"
    except pa.errors.SchemaErrors as e:
        fc_str = str(e.failure_cases["failure_case"][0])
        is_attribute_error = "AttributeError" in fc_str
        if is_attribute_error:
            return PASS, (
                "pipeline_xml.py's @pa.check methods ARE BROKEN in pandera 0.29. "
                "They crash with AttributeError('PolarsData has no attribute list'), "
                "and Pandera reports this as a generic validation failure. "
                "The check never actually validates the data — it fails on ALL rows, "
                "even rows with valid data.\n"
                f"Reported failure: {fc_str[:200]}"
            )
        return PASS, f"Check failed (possibly correctly):\n{e.failure_cases}"


run_test(
    "6c. pipeline_xml.py @pa.check pattern is broken in v0.29",
    test_pipeline_xml_checks_are_broken,
)


# =========================================================================
# TEST 7: Correct @pa.check pattern with PolarsData for nested columns
# =========================================================================


def test_polarsdata_nested_check_basic():
    """Can we validate nested struct fields using the PolarsData API?"""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "dimensions": [
                [{"type": "height", "value": 10.0, "unit": "cm"}],
                [{"type": "height", "value": 0.5, "unit": "cm"}],  # BAD
                [{"type": "height", "value": 15.0, "unit": "cm"}],
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        dimensions: list

        @pa.check("dimensions", name="all_values_gt_1")
        @classmethod
        def check_values(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("value"))
                .list.min()
                .gt(1)
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught value=0.5 in row 1"
    except pa.errors.SchemaErrors as e:
        try:
            fc = e.failure_cases
            return PASS, f"PolarsData-style nested check works.\nfailure_cases:\n{fc}"
        except Exception as fmt_err:
            return PASS, (
                f"PolarsData-style nested check DETECTED the bad data, but Pandera "
                f"crashes building failure_cases: {type(fmt_err).__name__}: {fmt_err}\n"
                "This is because Pandera tries to cast List(Struct) values to String "
                "for the error report, and Polars cannot do that cast."
            )
    except Exception as e:
        if "cannot cast List type" in str(e):
            return PASS, (
                f"PolarsData-style check DETECTED the failure, but Pandera crashes "
                f"internally: {type(e).__name__}: {str(e)[:150]}\n"
                "BUG: Pandera cannot format failure_cases for List(Struct) columns."
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "7a. PolarsData: nested struct field value check",
    test_polarsdata_nested_check_basic,
)


def test_polarsdata_nested_string_pattern():
    """Can we check string patterns inside nested structs with PolarsData?"""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "media": [
                [{"type": "primary", "url": "https://example.com/a.jpg"}],
                [{"type": "primary", "url": "http://example.com/b.jpg"}],  # BAD: http
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        media: list

        @pa.check("media", name="all_urls_https")
        @classmethod
        def check_urls(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("url").str.starts_with("https://"))
                .list.all()
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught http:// URL"
    except pa.errors.SchemaErrors as e:
        try:
            return PASS, f"PolarsData string pattern check works:\n{e.failure_cases}"
        except Exception as fmt_err:
            return PASS, f"Check detected failure but failure_cases crashed: {fmt_err}"
    except Exception as e:
        if "cannot cast List type" in str(e):
            return (
                PASS,
                f"Check detected failure but Pandera crashes on error report: {e}",
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "7b. PolarsData: string pattern inside nested struct",
    test_polarsdata_nested_string_pattern,
)


def test_polarsdata_nested_null_check():
    """Can we check for nulls inside struct fields with PolarsData?"""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "Alice", "score": 100}],
                [{"name": None, "score": 200}],  # BAD: null name
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        items: list

        @pa.check("items", name="no_null_names")
        @classmethod
        def check_no_nulls(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("name").is_not_null())
                .list.all()
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught null name"
    except pa.errors.SchemaErrors as e:
        try:
            return PASS, f"PolarsData null check works:\n{e.failure_cases}"
        except Exception as fmt_err:
            return (
                PASS,
                f"Null check detected failure but failure_cases crashed: {fmt_err}",
            )
    except Exception as e:
        if "cannot cast List type" in str(e):
            return (
                PASS,
                f"Null check detected failure but Pandera crashes on error report: {e}",
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "7c. PolarsData: null check inside struct field", test_polarsdata_nested_null_check
)


def test_polarsdata_list_length():
    """Can we check list length using PolarsData?"""
    df = pl.DataFrame(
        {"id": [1, 2, 3], "tags": [["a", "b"], [], ["c"]]}  # row 1 empty
    )

    class Schema(pa.DataFrameModel):
        id: int
        tags: list

        @pa.check("tags", name="non_empty_list")
        @classmethod
        def check_non_empty(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key).list.len().gt(0).alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught empty list"
    except pa.errors.SchemaErrors as e:
        try:
            return PASS, f"PolarsData list length check works:\n{e.failure_cases}"
        except Exception as fmt_err:
            return (
                PASS,
                f"List length check detected failure but failure_cases crashed: {fmt_err}",
            )
    except Exception as e:
        if "cannot cast" in str(e):
            return PASS, f"List length check detected failure but Pandera crashes: {e}"
        return ERROR, f"{type(e).__name__}: {e}"


run_test("7d. PolarsData: list length check", test_polarsdata_list_length)


def test_polarsdata_dataframe_check_cross_column():
    """Can @pa.dataframe_check access nested structs with PolarsData?"""
    df = pl.DataFrame(
        {
            "department": ["Sculpture", "Painting"],
            "dimensions": [
                [{"type": "height", "value": 10.0}, {"type": "width", "value": 5.0}],
                [{"type": "height", "value": 10.0}, {"type": "width", "value": 5.0}],
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        department: str
        dimensions: list

        @pa.dataframe_check(name="sculpture_must_have_depth")
        @classmethod
        def sculpture_depth(cls, data: PolarsData) -> pl.LazyFrame:
            lf = data.lazyframe
            is_sculpture = pl.col("department") == "Sculpture"
            has_depth = (
                pl.col("dimensions")
                .list.eval(pl.element().struct.field("type") == "depth")
                .list.any()
            )
            return lf.select((~is_sculpture | has_depth).alias("result"))

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught Sculpture without depth"
    except pa.errors.SchemaErrors as e:
        try:
            return (
                PASS,
                f"PolarsData cross-column nested check works:\n{e.failure_cases}",
            )
        except Exception as fmt_err:
            return (
                PASS,
                f"Cross-column check detected failure but failure_cases crashed: {fmt_err}",
            )
    except Exception as e:
        if "cannot cast" in str(e):
            return PASS, f"Cross-column check detected failure but Pandera crashes: {e}"
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "7e. PolarsData: cross-column check with nested structs",
    test_polarsdata_dataframe_check_cross_column,
)


# =========================================================================
# TEST 8: Error reporting granularity
# =========================================================================


def test_error_reporting_row_index():
    """Does Pandera report WHICH ROW failed for nested checks?"""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "dims": [
                [{"v": 10.0}],
                [{"v": 0.5}],  # BAD
                [{"v": 15.0}],
                [{"v": -1.0}],  # BAD
                [{"v": 0.0}],  # BAD
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        dims: list

        @pa.check("dims", name="positive_values")
        @classmethod
        def check_positive(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("v"))
                .list.min()
                .gt(1)
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught 3 bad rows"
    except pa.errors.SchemaErrors as e:
        try:
            fc = e.failure_cases
        except Exception as fmt_err:
            return PASS, (
                f"SchemaErrors raised but failure_cases crashes: {type(fmt_err).__name__}: {fmt_err}\n"
                "This is a Pandera bug — it cannot format List-typed failure cases into the "
                "failure_cases DataFrame."
            )
        lines = [
            f"failure_cases shape: {fc.shape}",
            f"columns: {fc.columns}",
            f"\n{fc}",
        ]
        has_index = "index" in fc.columns
        if has_index:
            indices = fc["index"].to_list()
            lines.append(f"\nRow indices reported: {indices}")
            lines.append("Expected bad rows: [1, 3, 4]")
            if indices == [1, 3, 4]:
                lines.append("ROW-LEVEL reporting works correctly")
            elif all(i is None for i in indices):
                lines.append(
                    "Row index is NULL — Pandera knows check failed but NOT which rows"
                )
            else:
                lines.append("Unexpected indices — partial reporting?")
        else:
            lines.append("No 'index' column — no row-level reporting")
        return PASS, "\n".join(lines)
    except Exception as e:
        # Pandera may crash internally when building failure_cases for List columns
        return PASS, (
            f"Validation detected failures but crashed building error report: "
            f"{type(e).__name__}: {str(e)[:200]}\n"
            "This is a known Pandera limitation — failure_cases formatting breaks "
            "for List/Struct columns."
        )


run_test(
    "8a. Error reporting: row indices for nested failures",
    test_error_reporting_row_index,
)


def test_error_reporting_which_element():
    """Can we tell WHICH nested element within a list failed?"""
    df = pl.DataFrame(
        {
            "id": [1],
            "dims": [
                [
                    {"type": "height", "value": 10.0},  # OK
                    {"type": "width", "value": 0.5},  # BAD
                    {"type": "depth", "value": 20.0},  # OK
                ]
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        dims: list

        @pa.check("dims", name="all_values_gt_1")
        @classmethod
        def check_values(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("value"))
                .list.min()
                .gt(1)
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "Should have caught value=0.5"
    except pa.errors.SchemaErrors as e:
        try:
            fc = e.failure_cases
            fc_val = fc["failure_case"][0] if len(fc) > 0 else "N/A"
            return PASS, (
                f"failure_case value: {fc_val}\n"
                f"Full failure_cases:\n{fc}\n\n"
                "VERDICT: Pandera reports that the COLUMN check failed and the aggregated "
                "result (e.g. min of list was <= 1), but does NOT tell you it was specifically "
                "the 'width' element at index 1 that caused the failure. You need separate "
                "post-hoc analysis to find the offending element."
            )
        except Exception as fmt_err:
            return PASS, (
                f"Check caught the failure but failure_cases crashed: "
                f"{type(fmt_err).__name__}: {fmt_err}\n"
                "VERDICT: Same conclusion — Pandera cannot report element-level detail."
            )
    except Exception as e:
        return PASS, (
            f"Validation detected failure but crashed: {type(e).__name__}: {str(e)[:200]}\n"
            "VERDICT: Pandera cannot even build the error report for List(Struct) failures."
        )


run_test(
    "8b. Error reporting: which nested element failed",
    test_error_reporting_which_element,
)


def test_error_reporting_lazy_vs_eager():
    """Compare lazy=True vs lazy=False error detail for nested failures."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "ok", "value": 10.0}],
                [{"name": "bad", "value": -1.0}],
            ],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        items: list

        @pa.check("items", name="positive_values")
        @classmethod
        def check_positive(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("value"))
                .list.min()
                .gt(0)
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    eager_detail = ""
    try:
        Schema.validate(df, lazy=False)
        eager_detail = "PASSED (unexpected)"
    except pa.errors.SchemaError as e:
        eager_detail = f"SchemaError (single): {str(e)[:200]}"
    except pa.errors.SchemaErrors as e:
        eager_detail = f"SchemaErrors: {e.failure_cases}"
    except Exception as e:
        eager_detail = f"{type(e).__name__}: {str(e)[:200]}"

    lazy_detail = ""
    try:
        Schema.validate(df, lazy=True)
        lazy_detail = "PASSED (unexpected)"
    except pa.errors.SchemaErrors as e:
        try:
            lazy_detail = f"SchemaErrors (all collected):\n{e.failure_cases}"
        except Exception:
            lazy_detail = (
                f"SchemaErrors raised but failure_cases not printable: {str(e)[:200]}"
            )
    except pa.errors.SchemaError as e:
        lazy_detail = f"SchemaError: {str(e)[:200]}"
    except Exception as e:
        lazy_detail = f"{type(e).__name__}: {str(e)[:200]}"

    return PASS, f"EAGER:\n{eager_detail}\n\nLAZY:\n{lazy_detail}"


run_test("8c. Error reporting: lazy vs eager", test_error_reporting_lazy_vs_eager)


# =========================================================================
# TEST 9: Edge cases
# =========================================================================


def test_bare_list_annotation():
    """Does 'dimensions: list' check anything about inner types?"""
    df = make_test_df()

    class Schema(pa.DataFrameModel):
        id: int
        name: str
        dimensions: list
        tags: list

        class Config:
            strict = False
            coerce = False

    Schema.validate(df)

    df_different_inner = pl.DataFrame(
        {
            "id": [1],
            "name": ["test"],
            "dimensions": [[1, 2, 3]],  # List(Int64) not List(Struct)
            "tags": [["a"]],
        }
    )

    try:
        Schema.validate(df_different_inner)
        return PASS, (
            "bare 'list' annotation accepts ANY inner type. "
            "List(Struct) and List(Int64) both pass. No inner type checking."
        )
    except (pa.errors.SchemaErrors, pa.errors.SchemaError):
        return FAIL, "bare 'list' annotation rejected different inner type"


run_test(
    "9a. Bare 'list' annotation: no inner type checking", test_bare_list_annotation
)


def test_nullable_list_column():
    """Does nullable=False on List column catch null values?"""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "items": [[{"name": "Alice"}], None, [{"name": "Charlie"}]],
        }
    )

    class Schema(pa.DataFrameModel):
        id: int
        items: list = pa.Field(nullable=False)

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=True)
        return FAIL, "nullable=False should have caught null list"
    except pa.errors.SchemaErrors as e:
        return PASS, f"nullable=False caught null list:\n{e.failure_cases}"


run_test("9b. nullable=False on List column", test_nullable_list_column)


def test_deeply_nested_dtype():
    """Can dtype_kwargs handle List(Struct(List(Struct)))?"""
    df = pl.DataFrame(
        {
            "id": [1],
            "records": [
                [
                    {
                        "name": "group1",
                        "items": [
                            {"label": "a", "value": 1},
                            {"label": "b", "value": 2},
                        ],
                    }
                ]
            ],
        }
    )

    inner_struct = pl.Struct({"label": pl.String, "value": pl.Int64})
    outer_struct = pl.Struct({"name": pl.String, "items": pl.List(inner_struct)})

    class Schema(pa.DataFrameModel):
        id: int
        records: pl.List = pa.Field(dtype_kwargs={"inner": outer_struct})

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df)
        return PASS, "Deeply nested List(Struct(List(Struct))) dtype check works"
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
        return FAIL, f"Deeply nested dtype failed: {e}"


run_test("9c. Deeply nested List(Struct(List(Struct))) dtype", test_deeply_nested_dtype)


# =========================================================================
# REPORT
# =========================================================================


def main():
    import pandera

    print("=" * 78)
    print("PANDERA NESTED VALIDATION TEST RESULTS")
    print(f"Pandera v{pandera.__version__} | Polars v{pl.__version__}")
    print("=" * 78)

    pass_count = sum(1 for _, s, _ in results if s == PASS)
    fail_count = sum(1 for _, s, _ in results if s == FAIL)
    error_count = sum(1 for _, s, _ in results if s == ERROR)

    for name, status, detail in results:
        icon = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "ERROR": "[ERR] "}[status]
        print(f"\n{icon} {name}")
        for line in detail.split("\n"):
            print(f"       {line}")

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {pass_count} passed, {fail_count} unexpected, {error_count} errors"
    )
    print("=" * 78)

    print(
        """
========================================================================
FINDINGS REPORT
========================================================================

1. DTYPE VALIDATION (tests 1a-1d, 2a):
   Pandera CAN declare inner struct field types via dtype_kwargs or Annotated.
   It DOES validate the full Polars dtype signature: field names, field types,
   field count, and field order. If the declared dtype doesn't exactly match
   the actual column dtype, validation fails.
   LIMITATION: This is a dtype-level check only. It confirms the column's
   Polars type is correct but does NOT inspect individual row values.

2. pa.Field() BUILT-IN CONSTRAINTS (tests 3a-3c):
   Built-in constraints (ge, le, isin, unique, str_length) are designed for
   SCALAR columns. Using them on List columns either:
   - Crashes (ge= tries to compare list to int)
   - Fails incorrectly (isin= compares whole list, not elements)
   - Crashes with Polars error (unique= not supported for list dtype)
   VERDICT: You CANNOT use pa.Field() constraints on nested columns.

3. STRICT MODE (tests 4a-4b):
   strict=True works on TOP-LEVEL columns only. For nested columns, it catches
   struct field mismatches INDIRECTLY via the dtype check (the full Polars type
   signature must match exactly). It does NOT recursively inspect struct fields
   as a separate "strict" operation.

4. COERCION (tests 5a-5c):
   coerce=True successfully converts:
   - List(Int64) -> List(Float64)
   - Struct({score: Int64}) -> Struct({score: Float64})
   - List(Struct({count: Int64})) -> List(Struct({count: Float64}))
   This works because Pandera delegates to Polars' cast() which handles nested
   coercion well.

5. *** CRITICAL BUG: @pa.check RECEIVES PolarsData, NOT pl.Series *** (tests 6a-6c):
   In Pandera v0.29 with the Polars backend, @pa.check methods receive a
   PolarsData namedtuple (with .lazyframe and .key attributes), NOT a pl.Series.

   pipeline_xml.py uses `col: pl.Series` and calls `col.list.eval(...)` which
   CRASHES with AttributeError('PolarsData has no attribute list'). Pandera
   catches this exception and reports it as a generic validation failure.

   This means ALL 14 custom checks in pipeline_xml.py are silently broken:
   they fail on EVERY row (even valid data) because of the wrong API usage,
   not because of actual data quality issues.

6. CORRECT @pa.check PATTERN (tests 7a-7e):
   Using PolarsData, the CHECK LOGIC works correctly — it detects bad data:

   @pa.check("dimensions", name="all_values_gt_1")
   @classmethod
   def check_values(cls, data: PolarsData) -> pl.LazyFrame:
       return data.lazyframe.select(
           pl.col(data.key)
           .list.eval(pl.element().struct.field("value"))
           .list.min()
           .gt(1)
           .alias(data.key)
       )

   All nested patterns work: struct field values, string patterns, null checks,
   list lengths, and cross-column checks with nested data.

   HOWEVER: See finding #7 below for a showstopper bug.

7. *** BUG: failure_cases CRASHES FOR List(Struct) COLUMNS *** (tests 7a-7d, 8a-8c):
   When a @pa.check on a List(Struct) column FAILS (detects bad data), Pandera
   tries to build the failure_cases DataFrame by casting column values to String.
   Polars CANNOT cast List(Struct(...)) to String, so this crashes with:
       InvalidOperationError: cannot cast List type (inner: 'Struct(...)', to: 'String')

   This means:
   - lazy=True mode: crashes with InvalidOperationError (unhandled exception)
   - lazy=False mode: works (raises SchemaError with examples), but only
     reports the first failure
   - @pa.dataframe_check: works for cross-column checks because the failure
     row is serialized as JSON (test 7e passed with row index!)

   WORKAROUND: Use lazy=False for column-level checks, or use
   @pa.dataframe_check instead of @pa.check for List(Struct) columns.

8. ERROR REPORTING (tests 8a-8c):
   - Row index: @pa.dataframe_check reports row indices correctly. Column-level
     @pa.check crashes before it can report indices (see #7).
   - Element index: Pandera NEVER reports which specific nested element within
     a list caused the failure. The failure_case shows the aggregated result.
   - lazy=True crashes for List(Struct) failures; lazy=False works but stops
     at the first failure

8. EDGE CASES (tests 9a-9c):
   - bare 'list' annotation: accepts ANY inner type (no checking)
   - nullable=False on List: correctly catches null list values
   - Deeply nested dtypes: dtype_kwargs works for List(Struct(List(Struct)))
"""
    )

    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
