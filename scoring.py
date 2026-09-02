"""입지 후보지 점수 계산.

Streamlit이나 파일 입출력에 의존하지 않는 순수 함수만 둔다.
그래야 합성 데이터로 로직을 검증할 수 있다.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 점수 구성 요소. 키는 내부 식별자, 값은 화면 표시명과 설명.
COMPONENTS: Dict[str, Dict[str, str]] = {
    "market_size": {
        "label": "시장 규모",
        "help": "해당 업종의 결제 금액이 큰 정도. 이미 돈이 도는 자리인가.",
    },
    "concentration": {
        "label": "상권 집중도",
        "help": "Gi* 기준. 주변 격자까지 묶어 통계적으로 뭉쳐 있는 정도.",
    },
    "opportunity": {
        "label": "성장 여지",
        "help": "주변은 뜨거운데 본인 매출은 아직 낮은 정도. 진입 여지의 대리 지표.",
    },
    "specialization": {
        "label": "업종 특화도",
        "help": "인천 평균 대비 이 업종 비중(LQ). 1을 넘으면 특화 상권.",
    },
    "retention": {
        "label": "재방문율",
        "help": "다시 찾는 손님의 비율. 뜨내기 상권인지 정착 상권인지.",
    },
    "target_fit": {
        "label": "타겟 적합도",
        "help": "선택한 고객층(연령·성별·거주지)의 비중이 높은 정도.",
    },
}

# 전략별 가중치 프리셋. 합은 1.0.
PRESETS: Dict[str, Dict[str, float]] = {
    "성숙 상권 진입": {
        "market_size": 0.35, "concentration": 0.25, "opportunity": 0.00,
        "specialization": 0.15, "retention": 0.25, "target_fit": 0.00,
    },
    # 집중도 가중치를 낮게 유지해야 한다. 높이면 이미 다 큰 상권이
    # 그대로 1위로 올라와 '기회 탐색'이라는 이름과 반대 결과가 나온다.
    "기회 상권 탐색": {
        "market_size": 0.05, "concentration": 0.15, "opportunity": 0.55,
        "specialization": 0.10, "retention": 0.15, "target_fit": 0.00,
    },
    "단골 중심 업종": {
        "market_size": 0.20, "concentration": 0.15, "opportunity": 0.05,
        "specialization": 0.15, "retention": 0.45, "target_fit": 0.00,
    },
    "특정 고객층 겨냥": {
        "market_size": 0.20, "concentration": 0.15, "opportunity": 0.10,
        "specialization": 0.10, "retention": 0.10, "target_fit": 0.35,
    },
}

MIN_GRIDS_FOR_PERCENTILE = 20


def pct_rank(s: pd.Series) -> pd.Series:
    """0~1 백분위. 전부 NaN이거나 표본이 너무 적으면 중립값 0.5."""
    valid = s.notna().sum()
    if valid == 0 or valid < 2:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, na_option="keep").fillna(0.5)


def location_quotient(cat_amt: pd.Series, total_amt: pd.Series) -> pd.Series:
    """LQ = (격자 내 업종 비중) / (전체 평균 업종 비중).

    분모는 인천 전체에서 그 업종이 차지하는 비중이므로 스칼라 하나다.
    """
    grid_share = cat_amt / total_amt.replace(0, np.nan)
    overall = cat_amt.sum() / total_amt.sum() if total_amt.sum() > 0 else np.nan
    if not overall or np.isnan(overall):
        return pd.Series(np.nan, index=cat_amt.index)
    return grid_share / overall


def build_candidates(
    metric_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    category: str,
    gu: Optional[str] = None,
    target_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """한 업종에 대한 격자별 후보 지표 표를 만든다.

    metric_df: grid_id, gu_nm, dong_nm, category, amt_sum, ratio, gi_star
    meta_df:   grid_id, total_amt_sum, total_gi_star, revisit_rate
    """
    base = metric_df[metric_df["category"] == category].copy()
    if base.empty:
        return base

    base = base.merge(
        meta_df[["grid_id", "total_amt_sum", "revisit_rate"]],
        on="grid_id", how="left",
    )

    # 타겟 적합도: 선택한 인구 축 카테고리들의 비중 합
    if target_categories:
        tgt = (
            metric_df[metric_df["category"].isin(target_categories)]
            .groupby("grid_id", as_index=False)["ratio"].sum()
            .rename(columns={"ratio": "target_share"})
        )
        base = base.merge(tgt, on="grid_id", how="left")
    else:
        base["target_share"] = np.nan

    base["lq"] = location_quotient(base["amt_sum"], base["total_amt_sum"])

    # 지역을 좁히면 그 안에서 다시 줄을 세운다. 인천 전체 기준 백분위는
    # 강화군 같은 저밀도 지역에서 전부 하위권이 되어 비교가 무의미해진다.
    scope = base
    if gu:
        narrowed = base[base["gu_nm"] == gu]
        if len(narrowed) >= MIN_GRIDS_FOR_PERCENTILE:
            scope = narrowed
        else:
            scope = base  # 표본이 적으면 전체 기준을 유지

    scope = scope.copy()
    scope["p_market_size"] = pct_rank(scope["amt_sum"])
    scope["p_concentration"] = pct_rank(scope["gi_star"])
    scope["p_specialization"] = pct_rank(scope["lq"])
    scope["p_retention"] = pct_rank(scope["revisit_rate"])
    scope["p_target_fit"] = pct_rank(scope["target_share"])

    # 성장 여지: 집중도 백분위가 매출 백분위보다 얼마나 앞서는가.
    # -1~1 범위를 0~1로 옮긴다.
    gap = scope["p_concentration"] - scope["p_market_size"]
    scope["p_opportunity"] = (gap + 1.0) / 2.0

    if gu and scope is not base:
        return scope
    if gu:
        return scope[scope["gu_nm"] == gu].copy()
    return scope


def apply_weights(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """가중치를 적용해 0~100 점수를 매긴다. 가중치 합으로 정규화한다."""
    if df.empty:
        return df

    total_w = sum(max(0.0, w) for w in weights.values())
    out = df.copy()
    if total_w <= 0:
        out["score"] = 50.0
        return out

    score = pd.Series(0.0, index=out.index)
    for key, w in weights.items():
        if w <= 0:
            continue
        col = "p_" + key
        if col not in out.columns:
            continue
        score = score + out[col].fillna(0.5) * w
    out["score"] = (score / total_w * 100).round(1)
    return out


def rank_candidates(
    df: pd.DataFrame,
    weights: Dict[str, float],
    top_n: int = 20,
    min_amt: float = 0.0,
) -> pd.DataFrame:
    """점수를 매기고 상위 N개를 돌려준다."""
    if df.empty:
        return df
    scored = apply_weights(df, weights)
    if min_amt > 0:
        scored = scored[scored["amt_sum"].fillna(0) >= min_amt]
    return scored.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


STRENGTH_FLOOR = 0.6


def explain_row(row: pd.Series, weights: Dict[str, float], top_k: int = 2) -> str:
    """점수를 끌어올린 실제 강점만 문장으로 설명한다.

    기여도(가중치 × 백분위)가 커도 백분위 자체가 평범하면 강점이 아니다.
    가중치가 높다는 이유로 '상위 47%'를 강점이라 부르면 설명이 거짓이 된다.
    """
    strengths = []
    for key, w in weights.items():
        if w <= 0:
            continue
        col = "p_" + key
        if col not in row.index or pd.isna(row[col]):
            continue
        p = float(row[col])
        if p >= STRENGTH_FLOOR:
            strengths.append((COMPONENTS[key]["label"], w * p, p))

    if not strengths:
        return "뚜렷한 강점 없이 전반적으로 평균 수준"

    strengths.sort(key=lambda t: t[1], reverse=True)
    parts = [
        f"{label} 상위 {max(1, round((1 - p) * 100))}%"
        for label, _, p in strengths[:top_k]
    ]
    return " · ".join(parts)
