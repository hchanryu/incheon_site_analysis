"""인천 입지 분석 — 업종과 전략을 정하면 후보 격자를 점수로 줄 세운다."""

from bisect import bisect_right

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import site_data as sd
from scoring import (
    COMPONENTS, PRESETS, build_candidates, explain_row, rank_candidates,
)

st.set_page_config(page_title="인천 입지 분석", layout="wide")

# ColorBrewer YlOrRd 9단계. 색상표 하나 때문에 matplotlib을 끌어오지 않는다.
RAMP = [
    (255, 255, 204), (255, 237, 160), (254, 217, 118),
    (254, 178, 76), (253, 141, 60), (252, 78, 42),
    (227, 26, 28), (189, 0, 38), (128, 0, 38),
]
_STOPS = [i / (len(RAMP) - 1) for i in range(len(RAMP))]


def ramp_color(t: float):
    """0~1 값을 YlOrRd 색으로 선형 보간한다."""
    t = min(max(float(t), 0.0), 1.0)
    i = min(bisect_right(_STOPS, t) - 1, len(RAMP) - 2)
    i = max(i, 0)
    span = _STOPS[i + 1] - _STOPS[i]
    f = 0.0 if span == 0 else (t - _STOPS[i]) / span
    lo, hi = RAMP[i], RAMP[i + 1]
    return [int(round(lo[c] + (hi[c] - lo[c]) * f)) for c in range(3)]


INCHEON_CENTER = (37.45, 126.70)


# ------------------------------------------------------------------ helpers
def fmt_amt(v) -> str:
    if v is None or pd.isna(v):
        return "-"
    v = float(v)
    if v >= 1e8:
        return f"{v / 1e8:,.1f}억원"
    if v >= 1e4:
        return f"{v / 1e4:,.0f}만원"
    return f"{v:,.0f}원"


def fmt_pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{float(v) * 100:.1f}%"


def score_colors(geojson: dict, score_by_grid: dict, dim: bool):
    """점수를 색으로 바꾼다. 점수 없는 격자는 회색."""
    for feat in geojson.get("features", []):
        gid = str(feat.get("properties", {}).get("grid_id"))
        s = score_by_grid.get(gid)
        if s is None:
            feat["properties"]["fill"] = [200, 205, 210, 40 if dim else 90]
            feat["properties"]["score_txt"] = "대상 아님"
        else:
            feat["properties"]["fill"] = ramp_color(s / 100.0) + [205]
            feat["properties"]["score_txt"] = f"{s:.1f}점"
    return geojson


def weight_editor(preset_name: str) -> dict:
    """프리셋을 기본값으로 두되 직접 조정할 수 있게 한다."""
    base = PRESETS[preset_name]
    with st.expander("가중치 직접 조정", expanded=False):
        st.caption("합이 1이 아니어도 됩니다. 비율만 반영합니다.")
        w = {}
        for key, meta in COMPONENTS.items():
            w[key] = st.slider(
                meta["label"], 0.0, 1.0, float(base.get(key, 0.0)), 0.05,
                help=meta["help"], key=f"w_{key}_{preset_name}",
            )
    return w


# ------------------------------------------------------------------ load
try:
    metric_df, meta_df, missing = sd.load_tables()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다.\n\n{e}")
    st.stop()

if missing:
    st.warning(f"parquet에 없어 제외한 카테고리: {', '.join(missing)}")

gus = sd.gu_options(meta_df)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("분석 조건")

    industry = st.selectbox(
        "업종", sd.INDUSTRIES,
        index=sd.INDUSTRIES.index("카페·간편식"),
        help="이 업종의 결제 데이터로 후보지를 평가합니다.",
    )
    gu = st.selectbox("지역", ["인천 전체"] + gus)
    gu_val = None if gu == "인천 전체" else gu

    st.divider()
    preset = st.radio("전략", list(PRESETS.keys()), help="가중치 조합이 달라집니다.")

    with st.expander("타겟 고객층", expanded=(preset == "특정 고객층 겨냥")):
        ages = st.multiselect(
            "연령", sd.AGES, format_func=sd.label_of,
            help="선택한 연령대의 매출 비중이 높은 격자를 우대합니다.",
        )
        sexes = st.multiselect("성별", sd.SEXES, format_func=sd.label_of)
        origins = st.multiselect("거주지", sd.ORIGINS, format_func=sd.label_of)
        target_cats = ages + sexes + origins

    weights = weight_editor(preset)

    st.divider()
    top_n = st.slider("후보 개수", 5, 50, 15, 5)
    min_amt_eok = st.number_input(
        "최소 결제 금액 (억원)", 0.0, 500.0, 0.0, 0.5,
        help="너무 한산한 격자를 걸러냅니다.",
    )

if weights.get("target_fit", 0) > 0 and not target_cats:
    st.sidebar.info("타겟 적합도 가중치가 있지만 고객층을 고르지 않아 중립값으로 계산합니다.")


# ------------------------------------------------------------------ compute
cand = build_candidates(
    metric_df, meta_df, category=industry, gu=gu_val,
    target_categories=target_cats or None,
)

st.title("인천 입지 분석")
st.caption(
    f"{gu} · {industry} · {preset} — 결제 데이터만으로 매긴 상대 순위입니다. "
    "임대료, 경쟁 점포 수, 유동인구는 반영되지 않습니다."
)

if cand.empty:
    st.warning("해당 조건에 맞는 격자가 없습니다. 업종이나 지역을 바꿔보십시오.")
    st.stop()

ranked = rank_candidates(cand, weights, top_n=top_n, min_amt=min_amt_eok * 1e8)

if ranked.empty:
    st.warning(
        f"최소 결제 금액 {min_amt_eok}억원 조건에 맞는 격자가 없습니다. "
        "기준을 낮춰보십시오."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("평가 대상 격자", f"{len(cand):,}개")
c2.metric("후보 격자", f"{len(ranked):,}개")
c3.metric("최고 점수", f"{ranked['score'].max():.1f}")
c4.metric("후보 평균 결제액", fmt_amt(ranked["amt_sum"].mean()))


# ------------------------------------------------------------------ tabs
tab_map, tab_list, tab_detail, tab_about = st.tabs(
    ["지도", "후보 목록", "격자 상세", "지표 설명"]
)

with tab_map:
    score_by_grid = dict(zip(ranked["grid_id"].astype(str), ranked["score"]))
    geo = score_colors(sd.load_geojson(gu_val), score_by_grid, dim=True)

    lat, lon = INCHEON_CENTER
    zoom = 10 if gu_val is None else 12
    if gu_val:
        sub = [
            f for f in geo["features"]
            if str(f["properties"].get("grid_id")) in score_by_grid
        ]
        if sub:
            coords = [
                c for f in sub
                for ring in (f["geometry"]["coordinates"] or [])
                for c in (ring if isinstance(ring[0], (list, tuple)) else [ring])
            ]
            flat = [c for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
            if flat:
                lon = float(np.mean([c[0] for c in flat]))
                lat = float(np.mean([c[1] for c in flat]))

    st.pydeck_chart(
        pdk.Deck(
            layers=[
                pdk.Layer(
                    "GeoJsonLayer", data=geo, pickable=True, stroked=True, filled=True,
                    get_fill_color="properties.fill",
                    get_line_color=[255, 255, 255, 80],
                    line_width_min_pixels=0.5,
                )
            ],
            initial_view_state=pdk.ViewState(
                latitude=lat, longitude=lon, zoom=zoom,
            ),
            map_style="light",
            tooltip={"text": "{dong_nm}\n{grid_id}\n{score_txt}"},
        ),
        height=560,
    )
    st.caption("진한 색일수록 점수가 높습니다. 회색은 후보에 들지 못한 격자입니다.")

with tab_list:
    view = ranked.copy()
    view["설명"] = view.apply(lambda r: explain_row(r, weights), axis=1)
    view["순위"] = range(1, len(view) + 1)
    show = pd.DataFrame({
        "순위": view["순위"],
        "점수": view["score"],
        "구": view["gu_nm"],
        "동": view["dong_nm"],
        "격자": view["grid_id"],
        "결제액": view["amt_sum"].map(fmt_amt),
        "업종 비중": view["ratio"].map(fmt_pct),
        "LQ": view["lq"].round(2),
        "Gi*": view["gi_star"].round(2),
        "재방문율": view["revisit_rate"].map(fmt_pct),
        "강점": view["설명"],
    })
    st.dataframe(show, hide_index=True, width="stretch", height=520)
    st.download_button(
        "CSV로 내려받기",
        show.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"입지후보_{gu}_{industry}.csv",
        mime="text/csv",
    )

with tab_detail:
    pick = st.selectbox(
        "격자 선택",
        ranked["grid_id"].tolist(),
        format_func=lambda g: (
            f"{g} · {ranked.loc[ranked.grid_id == g, 'dong_nm'].iloc[0]} "
            f"({ranked.loc[ranked.grid_id == g, 'score'].iloc[0]:.1f}점)"
        ),
    )
    row = ranked[ranked["grid_id"] == pick].iloc[0]

    st.subheader(f"{row['dong_nm']} · {pick}")
    st.write(f"**강점:** {explain_row(row, weights)}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{industry} 결제액", fmt_amt(row["amt_sum"]))
    m2.metric("업종 비중", fmt_pct(row["ratio"]))
    m3.metric("LQ (특화도)", f"{row['lq']:.2f}" if pd.notna(row["lq"]) else "-")
    m4.metric("재방문율", fmt_pct(row["revisit_rate"]))

    st.markdown("##### 구성 요소별 백분위")
    comp = pd.DataFrame({
        "요소": [COMPONENTS[k]["label"] for k in COMPONENTS],
        "백분위": [
            round(float(row.get("p_" + k, np.nan)) * 100, 1)
            if pd.notna(row.get("p_" + k, np.nan)) else np.nan
            for k in COMPONENTS
        ],
        "가중치": [weights.get(k, 0.0) for k in COMPONENTS],
    })
    st.dataframe(comp, hide_index=True, width="stretch")

    st.markdown("##### 이 격자의 고객 구성")
    prof = metric_df[
        (metric_df["grid_id"] == pick)
        & (metric_df["category_type"].isin(["age", "sex", "origin", "day"]))
    ].copy()
    if prof.empty:
        st.caption("고객 구성 데이터가 없습니다.")
    else:
        prof["구분"] = prof["category_type"].map(sd.TYPE_LABELS)
        prof["항목"] = prof["category"].map(sd.label_of)
        prof["비중"] = (prof["ratio"] * 100).round(1)
        for t in ["연령", "성별", "거주지", "요일"]:
            part = prof[prof["구분"] == t]
            if part.empty:
                continue
            st.caption(t)
            st.bar_chart(part.set_index("항목")["비중"], height=180)

    st.info(
        "고객 구성은 이 격자 **전체 매출** 기준입니다. "
        f"{industry} 업종만의 고객 구성이 아닙니다 — 원 데이터에 교차 집계가 없습니다."
    )

with tab_about:
    st.markdown("##### 점수 구성 요소")
    st.dataframe(
        pd.DataFrame({
            "요소": [v["label"] for v in COMPONENTS.values()],
            "의미": [v["help"] for v in COMPONENTS.values()],
        }),
        hide_index=True, width="stretch",
    )

    st.markdown("##### 점수 계산 방식")
    st.markdown(
        "각 요소를 **백분위(0~1)** 로 바꾼 뒤 가중 평균해 100점 만점으로 환산합니다. "
        "단위가 다른 지표를 그대로 더하지 않기 위해서입니다. "
        "지역을 좁히면 그 지역 안에서 다시 백분위를 매깁니다. "
        "인천 전체 기준으로 줄을 세우면 저밀도 지역이 전부 하위권이 되어 비교가 무의미해집니다."
    )

    st.markdown("##### Gi\\* 해석")
    st.markdown(
        "Gi\\*는 금액이 아니라 **z-score**입니다. 1.96을 넘으면 95% 신뢰수준에서, "
        "2.58을 넘으면 99% 신뢰수준에서 통계적으로 유의한 공간 군집입니다. "
        "\"Gi\\*가 3이면 매출이 3배\"라는 해석은 잘못된 것입니다."
    )

    st.markdown("##### 성장 여지의 의미")
    st.markdown(
        "Gi\\*는 이웃 격자를 포함한 국지 통계이므로, "
        "**Gi\\* 백분위는 높은데 본인 결제액 백분위는 낮은** 격자가 존재합니다. "
        "주변은 활발한데 아직 이 자리는 약하다는 뜻이며, 진입 여지의 대리 지표로 씁니다. "
        "다만 그 자리가 약한 데는 접근성이나 건물 사정 같은 이유가 있을 수 있습니다."
    )

    st.warning(
        "**이 도구의 한계**\n\n"
        "- 임대료, 권리금, 경쟁 점포 수, 유동인구, 배후 인구가 데이터에 없습니다.\n"
        "- 연령·성별·업종의 **교차 집계가 없습니다.** "
        "\"20대 여성의 카페 매출\"은 계산할 수 없습니다.\n"
        "- 점수는 절대 평가가 아니라 선택한 범위 안에서의 **상대 순위**입니다.\n"
        "- 결제 결과 데이터이므로 과거를 설명할 뿐 미래를 보장하지 않습니다.\n\n"
        "후보를 좁히는 데까지만 쓰고, 최종 판단은 현장 확인과 임대 조건 검토를 거치십시오."
    )
