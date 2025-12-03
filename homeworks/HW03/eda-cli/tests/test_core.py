from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )

def _sample_constant_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "constant_col": [1, 1, 1, 1],
            "mixed_col": [1, 2, 1, 2],
        }
    )

def _sample_id_dublicates_df() -> pd.DataFrame:
    return pd.DataFrame(
        { 
            "user_id": [10,10,20,10,40],
            "country": ["US", "US", "RU", "US", "PL"],
        }
    )

def _sample_zero_value_df() -> pd.DataFrame: 
    return pd.DataFrame(
        { 
            "user_id": [10,20,30,40,50,60],
            "user_score": [0,0,0,10,20,30],
            "level": [1,0,0,0,1,0],
        }
    )

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


#новые тесты для эвристик качества из HW03 
def test_constant_columns(): 
    df = _sample_constant_df() 
    summarize_const = summarize_dataset(df) 
    missing_const = missing_table(df)
    quality_flags = compute_quality_flags(summarize_const,missing_const)
    assert quality_flags["has_constant_columns"] is True


def test_suspicious_id_duplicates(): 
    df = _sample_id_dublicates_df()
    summarize_id = summarize_dataset(df) 
    missing_id = missing_table(df)
    quality_flags = compute_quality_flags(summarize_id,missing_id)
    assert quality_flags["has_suspicious_id_duplicates"] is True


def test_many_zero_values(): 
    df = _sample_zero_value_df()     
    summarize_zero = summarize_dataset(df) 
    missing_zero = missing_table(df)
    quality_flags = compute_quality_flags(summarize_zero,missing_zero)
    assert quality_flags["has_many_zero_values"] is True