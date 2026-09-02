"""데이터 적재와 롱 포맷 변환.

wide parquet(116열)을 격자 × 카테고리 롱 포맷으로 눕히고,
지도용 GeoJSON은 따로 만든다. 질의 계층은 geometry를 만지지 않는다.
"""

from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARQUET_PATH = DATA_DIR / "output_with_weekend.parquet"
REVISIT_CSV = DATA_DIR / "revisit_output_syn.csv"

INDUSTRIES = [
    "내구재(가전·가구)", "문화·레저(용품)", "문화·레저(활동)", "뷰티",
    "생활서비스", "식료품", "여행·숙박·교통", "외식(일반)",
    "유통(오프라인)", "유흥", "자동차", "주유",
    "카페·간편식", "패션·잡화", "헬스케어",
]
AGES = ["20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "99"]
SEXES = ["F", "M"]
ORIGINS = ["foreign", "incheon", "notincheon"]
DAYS = ["평일", "주말"]

CATEGORY_TYPES: Dict[str, List[str]] = {
    "industry": INDUSTRIES,
    "age": AGES,
    "sex": SEXES,
    "origin": ORIGINS,
    "day": DAYS,
}
PREFIX_TO_TYPE = {p: t for t, ps in CATEGORY_TYPES.items() for p in ps}

LABELS: Dict[str, str] = {
    "F": "여성", "M": "남성",
    "foreign": "외국인", "incheon": "인천 거주", "notincheon": "타지역 거주",
    "99": "70대 이상·미상",
}
for _a in AGES[:-1]:
    LABELS.setdefault(_a, f"{_a}대")
for _i in INDUSTRIES:
    LABELS.setdefault(_i, _i)
for _d in DAYS:
    LABELS.setdefault(_d, _d)

TYPE_LABELS = {
    "industry": "업종", "age": "연령", "sex": "성별",
    "origin": "거주지", "day": "요일",
}


def label_of(code: str) -> str:
    return LABELS.get(code, code)


def _read_wide() -> gpd.GeoDataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} 를 찾을 수 없습니다. data/ 폴더에 데이터를 넣으십시오."
        )
    gdf = gpd.read_parquet(PARQUET_PATH)
    dead = [c for c in gdf.columns if c.startswith("None_")]
    if dead:
        gdf = gdf.drop(columns=dead)
    return gdf


@st.cache_data(show_spinner="데이터를 준비하는 중입니다...")
def load_tables():
    """(metric_df, meta_df) 롱 포맷 두 개를 돌려준다. geometry 없음."""
    gdf = _read_wide()

    id_cols = ["grid_id", "gu_nm", "dong_nm"]
    blocks = []
    missing = []
    for prefix, ctype in PREFIX_TO_TYPE.items():
        cols = (f"{prefix}_amt_sum", f"{prefix}_ratio", f"{prefix}_Gi*")
        if any(c not in gdf.columns for c in cols):
            missing.append(prefix)
            continue
        b = pd.DataFrame({c: gdf[c] for c in id_cols})
        b["category"] = prefix
        b["category_type"] = ctype
        b["amt_sum"] = gdf[cols[0]].values
        b["ratio"] = gdf[cols[1]].values
        b["gi_star"] = gdf[cols[2]].values
        blocks.append(b)

    if not blocks:
        raise ValueError("롱 포맷으로 변환할 수 있는 카테고리 컬럼이 없습니다.")

    metric_df = pd.concat(blocks, ignore_index=True)

    meta_cols = ["grid_id", "gu_nm", "dong_nm"]
    meta_df = pd.DataFrame({c: gdf[c] for c in meta_cols})
    meta_df["total_amt_sum"] = gdf.get("all_amt_sum")
    meta_df["total_gi_star"] = gdf.get("total_Gi*")

    if REVISIT_CSV.exists():
        revisit = pd.read_csv(REVISIT_CSV, dtype={"grid_id": str})
        meta_df = meta_df.merge(revisit, on="grid_id", how="left")
    if "revisit_rate" not in meta_df.columns:
        meta_df["revisit_rate"] = pd.NA

    return metric_df, meta_df, sorted(missing)


@st.cache_data(show_spinner=False)
def load_geojson(gu: Optional[str] = None) -> dict:
    """지도용 GeoJSON. grid_id를 properties에 실어 점수와 연결한다."""
    gdf = _read_wide()[["grid_id", "gu_nm", "dong_nm", "geometry"]]
    if gu:
        gdf = gdf[gdf["gu_nm"] == gu]
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf.__geo_interface__


@st.cache_data(show_spinner=False)
def gu_options(_meta_df: pd.DataFrame) -> List[str]:
    return sorted(x for x in _meta_df["gu_nm"].dropna().unique())
