"""
Eva Causal AI Domain Demo
제조 / 공공 / 교육 / 금융 4개 업종 인과추론 데모 웹앱
"""

import warnings

import pathlib
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

import explain

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causal AI Domain Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1000px; }
    h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; }
    h2 { font-size: 1.1rem; font-weight: 600; border-bottom: 1px solid #ddd;
         padding-bottom: 0.3rem; margin-top: 1.5rem; }
    h3 { font-size: 0.95rem; font-weight: 600; margin-top: 1rem; }
    .info-box {
        background: #f5f5f5;
        border-left: 3px solid #999;
        padding: 0.6rem 0.9rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        border-radius: 0 3px 3px 0;
    }
    .metric-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #666; margin-bottom: 2px; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .step-bar { margin: 0.4rem 0 1rem 0; font-size: 0.78rem; line-height: 2.2; }
    .step-item  { display: inline-block !important; padding: 3px 12px; border-radius: 12px;
                  background: #eee; color: #888; margin-right: 4px; white-space: nowrap; }
    .step-active{ display: inline-block !important; padding: 3px 12px; border-radius: 12px;
                  background: #1a1a2e; color: #fff; margin-right: 4px; white-space: nowrap;
                  font-weight: 600; }
    .step-done  { display: inline-block !important; padding: 3px 12px; border-radius: 12px;
                  background: #2d6a4f; color: #fff; margin-right: 4px; white-space: nowrap; }
    div[data-testid="stSidebar"] { background: #f8f8f8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Korean font — cache_resource 대신 매 실행마다 rcParams 직접 설정
# ---------------------------------------------------------------------------
# repo에 번들된 폰트 (클라우드/로컬 공통으로 사용)
_BUNDLE_FONT = pathlib.Path(__file__).parent / "fonts" / "MalgunGothic.ttf"

def _setup_font():
    matplotlib.rcParams["axes.unicode_minus"] = False
    if _BUNDLE_FONT.exists():
        fm.fontManager.addfont(str(_BUNDLE_FONT))
        prop = fm.FontProperties(fname=str(_BUNDLE_FONT))
        matplotlib.rcParams["font.family"] = prop.get_name()
        return
    # 번들 폰트 없으면 시스템 폰트 탐색
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                  "NanumSquare", "Hancom Gothic", "AppleGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for f in candidates:
        if f in available:
            matplotlib.rcParams["font.family"] = f
            return

_setup_font()

# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------
DOMAINS = {
    "public": {
        "name": "공공",
        "subtitle": "직업훈련 프로그램 → 소득",
        "desc": "정부 직업훈련(NSW) 참여가 1978년 실질소득에 미치는 인과적 효과 분석",
        "source": "LaLonde (1986) NSW 실험 데이터",
        "T": "treat",
        "T_label": "직업훈련 참여",
        "T_binary": True,
        "W": ["age", "educ", "black", "hisp", "married", "re74", "re75"],
        "W_labels": {
            "age": "나이(세)", "educ": "교육연수(년)", "black": "흑인 여부",
            "hisp": "히스패닉 여부", "married": "기혼 여부",
            "re74": "1974년 소득(달러)", "re75": "1975년 소득(달러)",
        },
        "Y": "re78",
        "Y_label": "1978년 실질소득 (달러)",
        "Y_binary": False,
        "W_value_labels": {
            "black":   {0: "아님", 1: "흑인"},
            "hisp":    {0: "아님", 1: "히스패닉"},
            "married": {0: "미혼",  1: "기혼"},
        },
        "dag_desc": "직업훈련(T)이 나이·학력·과거소득 등(W)을 통제한 상태에서 1978년 소득(Y)에 미치는 순수 효과를 추정합니다.",
    },
    "edu": {
        "name": "교육",
        "subtitle": "유료 보충학습 → 최종 성적",
        "desc": "포르투갈 중고교생의 유료 보충학습 참여가 최종 수학 성적에 미치는 인과적 효과 분석",
        "source": "UCI Student Performance Dataset",
        "T": "paid_bin",
        "T_label": "유료 보충학습 참여",
        "T_binary": True,
        "W": ["age", "Medu", "Fedu", "studytime", "failures", "absences", "G1", "G2"],
        "W_labels": {
            "age": "나이(세)", "Medu": "어머니 학력(0-4)", "Fedu": "아버지 학력(0-4)",
            "studytime": "주간 공부시간(1-4)", "failures": "과거 낙제 횟수",
            "absences": "결석 일수", "G1": "1학기 성적(0-20)", "G2": "2학기 성적(0-20)",
        },
        "Y": "G3",
        "Y_label": "최종 성적 (0-20점)",
        "Y_binary": False,
        "W_value_labels": {
            "Medu":      {0: "없음", 1: "초등", 2: "중학", 3: "고교", 4: "대학"},
            "Fedu":      {0: "없음", 1: "초등", 2: "중학", 3: "고교", 4: "대학"},
            "studytime": {1: "2시간 미만", 2: "2~5시간", 3: "5~10시간", 4: "10시간 초과"},
            "failures":  {0: "0회", 1: "1회", 2: "2회", 3: "3회 이상"},
        },
        "dag_desc": "유료 보충학습(T)이 가정환경·이전 성적 등(W)을 통제한 상태에서 최종 성적(Y)에 미치는 순수 효과를 추정합니다.",
    },
    "finance": {
        "name": "금융",
        "subtitle": "카드 연체 → 다음달 채무불이행",
        "desc": "신용카드 연체 이력이 다음달 채무불이행(부도)에 미치는 인과적 효과 분석",
        "source": "UCI Default of Credit Card Clients Dataset",
        "T": "PAY_0_bin",
        "T_label": "9월 연체 여부",
        "T_binary": True,
        "W": ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "BILL_AMT1", "PAY_AMT1"],
        "W_labels": {
            "LIMIT_BAL": "신용한도(원)", "EDUCATION": "학력(1=대학원 2=대학 3=고교)",
            "MARRIAGE": "결혼상태(1=기혼 2=미혼)", "AGE": "나이(세)",
            "BILL_AMT1": "9월 청구금액(원)", "PAY_AMT1": "9월 납부금액(원)",
        },
        "Y": "default_payment_next_month",
        "Y_label": "다음달 채무불이행 (0=정상, 1=부도)",
        "Y_binary": True,
        "W_value_labels": {
            "EDUCATION": {1: "대학원", 2: "대학", 3: "고교", 4: "기타", 5: "기타", 6: "기타"},
            "MARRIAGE":  {0: "기타",   1: "기혼",  2: "미혼",  3: "기타"},
        },
        "dag_desc": "9월 연체 여부(T)가 신용한도·나이·청구금액 등(W)을 통제한 상태에서 다음달 부도 여부(Y)에 미치는 순수 효과를 추정합니다.",
    },
    "mfg": {
        "name": "제조",
        "subtitle": "공구 과다 마모 → 기계 고장",
        "desc": "공구 마모 수준이 기계 고장에 미치는 인과적 효과 분석",
        "source": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "T": "tool_wear_high",
        "T_label": "공구 과다 마모",
        "T_binary": True,
        "W": ["air_temp", "process_temp", "rpm", "torque"],
        "W_labels": {
            "air_temp": "공기 온도(°C)", "process_temp": "공정 온도(°C)",
            "rpm": "회전 속도(RPM)", "torque": "토크(Nm)",
        },
        "Y": "machine_failure",
        "Y_label": "기계 고장 여부 (0=정상, 1=고장)",
        "Y_binary": True,
        "W_value_labels": {},
        "dag_desc": "공구 과다 마모(T)가 온도·회전속도·토크 등(W)을 통제한 상태에서 기계 고장(Y)에 미치는 순수 효과를 추정합니다.",
    },
}

# ---------------------------------------------------------------------------
# 한국어 조사 헬퍼
# ---------------------------------------------------------------------------
def _josa_i_ga(text: str) -> str:
    """받침 있으면 '이', 없으면 '가'"""
    c = ord(text[-1]) if text else 0
    if 0xAC00 <= c <= 0xD7A3:
        return "이" if (c - 0xAC00) % 28 != 0 else "가"
    return "이"

def _josa_eul_reul(text: str) -> str:
    """받침 있으면 '을', 없으면 '를'"""
    c = ord(text[-1]) if text else 0
    if 0xAC00 <= c <= 0xD7A3:
        return "을" if (c - 0xAC00) % 28 != 0 else "를"
    return "을"

STEP_LABELS = ["업종 선택", "데이터 확인", "DAG", "ATE / CATE", "반사실 시뮬레이션"]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for _k, _v in [("step", 0), ("domain", None), ("df", None),
                ("estimator", None), ("results", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _step_bar(current: int) -> None:
    parts = []
    for i, label in enumerate(STEP_LABELS):
        if i < current:
            cls = "step-done"
        elif i == current:
            cls = "step-active"
        else:
            cls = "step-item"
        parts.append(f'<span class="{cls}">{label}</span>')
    st.markdown(f'<div class="step-bar">{"".join(parts)}</div>', unsafe_allow_html=True)


def _metric(label: str, value: str, color: str = "#1a1a2e") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'</div>'
    )


def _nav(back_step: int = None, next_step: int = None,
         back_label: str = "이전", next_label: str = "다음",
         next_home: bool = False) -> None:
    cols = st.columns([5, 1, 1])
    with cols[1]:
        if back_step is not None:
            if st.button(back_label, key=f"nav_back_{back_step}"):
                st.session_state["step"] = back_step
                st.rerun()
    with cols[2]:
        if next_step is not None:
            if st.button(next_label, key=f"nav_next_{next_step}", type="primary"):
                if next_home:
                    _reset()
                else:
                    st.session_state["step"] = next_step
                st.rerun()


def _reset() -> None:
    st.session_state["step"] = 0
    st.session_state["domain"] = None
    for k in ["df", "estimator", "results", "cate", "ate", "ate_ci"]:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if k.startswith("_ai_"):
            del st.session_state[k]


# ===========================================================================
# Step 0 — 업종 선택
# ===========================================================================
def page_domain_select() -> None:
    _show_usage_badge()
    st.markdown("## EVA Causal AI  /  업종별 인과추론 데모")
    st.markdown(
        '<div class="info-box">분석할 업종을 선택하세요. 선택 후 데이터 로드 → DAG → ATE/CATE → 반사실 시뮬레이션 순으로 진행됩니다.</div>',
        unsafe_allow_html=True,
    )
    _step_bar(0)

    cols = st.columns(4)
    for col, (key, cfg) in zip(cols, DOMAINS.items()):
        with col:
            st.markdown(f"**{cfg['name']}**")
            st.caption(cfg["subtitle"])
            st.caption(cfg["desc"])
            st.caption(f"출처: {cfg['source']}")
            if st.button(f"{cfg['name']} 선택", key=f"sel_{key}", use_container_width=True):
                _reset()
                st.session_state["domain"] = key
                st.session_state["step"] = 1
                st.rerun()


# ===========================================================================
# Step 1 — 데이터 확인
# ===========================================================================
def page_data() -> None:
    domain = st.session_state["domain"]
    cfg = DOMAINS[domain]

    st.markdown(f"## {cfg['name']}  /  데이터 확인")
    _step_bar(1)
    st.markdown(
        f'<div class="info-box">{cfg["desc"]}<br>'
        f'처치변수(T), 결과변수(Y), 교란변수(W)를 확인합니다.'
        f'<br><small>출처: {cfg["source"]}</small></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("df") is None:
        with st.spinner("데이터셋 다운로드 및 로드 중..."):
            try:
                from data_loader import load_domain_data
                df = load_domain_data(domain)
                st.session_state["df"] = df
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")
                _nav(back_step=0, back_label="처음으로")
                return

    df: pd.DataFrame = st.session_state["df"]

    # Variable table
    st.markdown("### 변수 구성  /  T · W · Y")
    rows = [{"역할": "T (처치)", "변수": cfg["T"], "설명": cfg["T_label"]}]
    for w in cfg["W"]:
        rows.append({"역할": "W (교란)", "변수": w, "설명": cfg["W_labels"].get(w, w)})
    rows.append({"역할": "Y (결과)", "변수": cfg["Y"], "설명": cfg["Y_label"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Summary metrics
    t_rate = float(df[cfg["T"]].mean()) if cfg["T"] in df.columns else 0.0
    cols_show = [cfg["T"]] + [c for c in cfg["W"] if c in df.columns] + [cfg["Y"]]
    cols_exist = [c for c in cols_show if c in df.columns]

    c1, c2, c3 = st.columns(3)
    c1.markdown(_metric("샘플 수", f"{len(df):,}"), unsafe_allow_html=True)
    c2.markdown(_metric("분석 변수 수 (T+W+Y)", str(len(cols_exist))), unsafe_allow_html=True)
    c3.markdown(_metric("처치 비율 (T=1)", f"{t_rate:.1%}"), unsafe_allow_html=True)

    st.markdown("### 기술 통계")
    st.dataframe(df[cols_exist].describe().round(3), use_container_width=True)

    # AI 설명
    w_kr = ", ".join(cfg["W_labels"].get(w, w) for w in cfg["W"] if w in df.columns)
    explain.show(
        f"data_{domain}_{len(df)}",
        f"'{cfg['name']}' 업종 인과 분석 데이터셋이다. 총 {len(df):,}개 샘플이 있다. "
        f"처치변수(T)는 '{cfg['T_label']}'이고 처치 비율은 {t_rate:.1%}다. "
        f"교란변수(W)는 [{w_kr}]이며, 결과변수(Y)는 '{cfg['Y_label']}'이다. "
        f"이 데이터가 무엇을 분석하는 것인지, T·W·Y의 역할을 포함해 일반인에게 2~3문장으로 설명하라.",
        domain=domain,
    )

    st.divider()
    _nav(back_step=0, next_step=2, back_label="처음으로")


# ===========================================================================
# Step 2 — DAG
# ===========================================================================
def page_dag() -> None:
    domain = st.session_state["domain"]
    cfg = DOMAINS[domain]
    df: pd.DataFrame = st.session_state.get("df")

    if df is None:
        st.session_state["step"] = 1
        st.rerun()
        return

    st.markdown(f"## {cfg['name']}  /  DAG  인과 구조 그래프")
    _step_bar(2)
    st.markdown(
        f'<div class="info-box">{cfg["dag_desc"]}</div>',
        unsafe_allow_html=True,
    )

    W_labels = cfg["W_labels"]
    T_node = cfg["T_label"]
    Y_node = cfg["Y_label"].split(" (")[0]
    W_nodes = [W_labels.get(w, w) for w in cfg["W"] if w in df.columns]
    n_w = len(W_nodes)

    # 한글 폰트 파일 직접 로드
    _setup_font()
    # 번들 폰트 우선, 없으면 시스템 탐색
    _kr_font_path = str(_BUNDLE_FONT) if _BUNDLE_FONT.exists() else None
    if not _kr_font_path:
        for _fname in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "Hancom Gothic"]:
            try:
                _fp = fm.findfont(fm.FontProperties(family=_fname), fallback_to_default=False)
                if _fp and "DejaVu" not in _fp and "ttf" in _fp.lower():
                    _kr_font_path = _fp
                    break
            except Exception:
                continue
    from matplotlib.font_manager import FontProperties
    import textwrap

    def _fp(size):
        return FontProperties(fname=_kr_font_path, size=size) if _kr_font_path else FontProperties(size=size)

    # 긴 라벨 → 줄바꿈 (한글 1자=2폭으로 계산, 최대 표시폭 10)
    def _display_width(s):
        """한글/CJK는 2폭, 나머지는 1폭으로 계산"""
        return sum(2 if '가' <= c <= '힣' or '一' <= c <= '鿿' else 1 for c in s)

    def _wrap(text, max_w=10):
        import re
        # 괄호 앞 분리: "나이(세)" → "나이\n(세)"
        text = re.sub(r"([가-힣a-zA-Z0-9]+)\(", r"\1\n(", text, count=1)
        result = []
        for part in text.split("\n"):
            if _display_width(part) <= max_w:
                result.append(part)
            else:
                # 폭 기준으로 직접 분할
                line, w = "", 0
                for ch in part:
                    cw = 2 if '가' <= ch <= '힣' else 1
                    if w + cw > max_w:
                        result.append(line)
                        line, w = ch, cw
                    else:
                        line += ch
                        w += cw
                if line:
                    result.append(line)
        return "\n".join(result)

    # 라벨 줄바꿈 적용
    W_wrapped = {wn: _wrap(wn) for wn in W_nodes}
    T_wrapped  = _wrap(T_node, max_w=12)
    Y_wrapped  = _wrap(Y_node, max_w=12)

    # 노드 반지름: 표시폭 기준으로 계산
    def _radius(wrapped_text, base=0.38):
        lines = wrapped_text.split("\n")
        max_dw = max(_display_width(l) for l in lines)
        n_lines = len(lines)
        return base + max(0, (max_dw - 6) * 0.025) + (n_lines - 1) * 0.09

    node_r = {wn: _radius(W_wrapped[wn], base=0.32) for wn in W_nodes}
    node_r[T_node] = _radius(T_wrapped, base=0.38)
    node_r[Y_node] = _radius(Y_wrapped, base=0.38)

    # Layout — 전체 높이를 최대 4.2인치로 고정해 한 화면에 들어오게
    MAX_H = 4.2
    y_spacing = min(1.2, (MAX_H - 0.6) / max(n_w - 1, 1))
    # 노드 반지름도 간격에 비례 축소
    _r_scale = min(1.0, y_spacing / 1.2)
    for _k in list(node_r.keys()):
        node_r[_k] = round(node_r[_k] * _r_scale, 3)
    total_h   = (n_w - 1) * y_spacing
    center_y  = total_h / 2.0
    x_t = 3.0
    x_y = 6.0
    pos = {}
    for i, wn in enumerate(W_nodes):
        pos[wn] = (0.0, total_h - i * y_spacing)
    pos[T_node] = (x_t, center_y)
    pos[Y_node] = (x_y, center_y)

    fig_w = 9
    fig_h = max(2.5, total_h + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 노드 색상
    node_color = {T_node: "#1a1a2e", Y_node: "#2d6a4f"}
    for wn in W_nodes:
        node_color[wn] = "#6b6b8a"

    # 노드 그리기
    label_map = {**W_wrapped, T_node: T_wrapped, Y_node: Y_wrapped}
    for node, (x, y) in pos.items():
        r = node_r[node]
        circle = plt.Circle((x, y), r, color=node_color[node], zorder=3, clip_on=False)
        ax.add_patch(circle)
        n_lines = label_map[node].count("\n") + 1
        fs = max(5, 8 - n_lines)
        ax.text(x, y, label_map[node], color="white",
                ha="center", va="center",
                fontproperties=_fp(fs), zorder=4,
                multialignment="center", linespacing=1.2)

    # 엣지 그리기
    def draw_edge(src, tgt, color, rad, lw=1.4):
        ax.annotate(
            "", xy=pos[tgt], xytext=pos[src],
            arrowprops=dict(
                arrowstyle="->, head_width=0.22, head_length=0.16",
                color=color, lw=lw,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=node_r[src] * 72,
                shrinkB=node_r[tgt] * 72,
            ),
            zorder=2,
        )

    for wn in W_nodes:
        draw_edge(wn, T_node, "#4a7abf", rad=0.25,  lw=1.4)
        draw_edge(wn, Y_node, "#999999", rad=-0.18, lw=1.0)
    draw_edge(T_node, Y_node, "#c0392b", rad=0.0, lw=2.2)

    # 범례
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(color="#1a1a2e", label=f"T  {T_node}"),
        Patch(color="#2d6a4f", label=f"Y  {Y_node}"),
        Patch(color="#6b6b8a", label="W  교란변수"),
        Line2D([0],[0], color="#4a7abf", lw=1.5, label="W → T"),
        Line2D([0],[0], color="#999",    lw=1.0, label="W → Y"),
        Line2D([0],[0], color="#c0392b", lw=2.0, label="T → Y"),
    ]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.9, prop=_fp(8))
    ax.set_title(f"{cfg['name']} 인과 구조 DAG", fontproperties=_fp(10))
    margin = 0.8
    ax.set_xlim(-node_r[W_nodes[0]] - margin, x_y + node_r[Y_node] + margin)
    ax.set_ylim(-margin, total_h + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # AI 설명
    w_str = ", ".join(W_nodes)
    explain.show(
        f"dag_{domain}",
        f"'{cfg['name']}' 업종의 인과 구조 DAG를 그렸다. "
        f"교란변수들({w_str})은 원인({T_node})과 결과({Y_node}) 양쪽에 화살표가 연결되어 있고, "
        f"원인({T_node})은 결과({Y_node})에 직접 연결된다. "
        f"이 화살표 구조가 무엇을 의미하는지, 교란변수의 역할을 포함해 일반인에게 2~3문장으로 설명하라.",
        domain=domain,
    )

    st.divider()
    _nav(back_step=1, next_step=3)


# ===========================================================================
# Step 3 — ATE / CATE
# ===========================================================================
def page_ate_cate() -> None:
    _show_usage_badge()
    domain = st.session_state["domain"]
    cfg = DOMAINS[domain]
    df: pd.DataFrame = st.session_state.get("df")

    if df is None:
        st.session_state["step"] = 1
        st.rerun()
        return

    st.markdown(f"## {cfg['name']}  /  ATE · CATE  처치 효과 추정")
    _step_bar(3)
    st.markdown(
        '<div class="info-box">'
        'CausalForestDML로 평균 처치 효과(ATE)와 개별 처치 효과(CATE)를 추정합니다.<br>'
        'ATE: 전체 집단의 평균 효과 &nbsp;|&nbsp; CATE: 개인별 효과 이질성'
        '</div>',
        unsafe_allow_html=True,
    )

    # Run estimation
    if st.session_state.get("estimator") is None:
        if st.button("추정 실행  /  Run Estimation", type="primary"):
            with st.spinner("CausalForestDML 학습 중... (30~60초 소요)"):
                try:
                    from causal_engine import run_ate_cate
                    res = run_ate_cate(df, cfg)
                    st.session_state.update({
                        "results": res,
                        "estimator": res["estimator"],
                        "cate": res["cate"],
                        "ate": res["ate"],
                        "ate_ci": res["ate_ci"],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"추정 실패: {e}")
                    return
        else:
            st.info("버튼을 눌러 추정을 실행하세요.")
            _nav(back_step=2)
            return

    res = st.session_state.get("results", {})
    ate: float = res.get("ate", 0.0)
    ate_lb, ate_ub = res.get("ate_ci", (0.0, 0.0))
    cate: np.ndarray = res.get("cate", np.array([]))
    sig = ate_lb * ate_ub > 0

    # Metrics
    sig_txt = "유의 (0 미포함)" if sig else "비유의 (0 포함)"
    sig_color = "#2d6a4f" if sig else "#b5461a"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric("ATE  평균 처치 효과", f"{ate:.4f}"), unsafe_allow_html=True)
    c2.markdown(_metric("95% CI 하한", f"{ate_lb:.4f}"), unsafe_allow_html=True)
    c3.markdown(_metric("95% CI 상한", f"{ate_ub:.4f}"), unsafe_allow_html=True)
    c4.markdown(_metric("통계적 유의성", sig_txt, sig_color), unsafe_allow_html=True)

    direction = "높인다" if ate >= 0 else "낮춘다"
    t_name  = cfg["T_label"]
    y_name  = cfg["Y_label"].split(" (")[0]
    josa_t  = _josa_i_ga(t_name)
    josa_y  = _josa_eul_reul(y_name)
    if cfg.get("Y_binary"):
        ate_desc = f"평균 <b>{abs(ate):.4f}</b> (약 <b>{abs(ate)*100:.1f}%p</b>) {direction}"
        ate_note = "(이진 결과이므로 ATE는 확률 변화량입니다.)"
    else:
        ate_desc = f"평균 <b>{abs(ate):.4f}</b>만큼 {direction}"
        ate_note = ""
    st.markdown(
        f'<div class="info-box">'
        f'{t_name}{josa_t} {y_name}{josa_y} {ate_desc}. {ate_note}<br>'
        f'95% CI: [{ate_lb:.4f}, {ate_ub:.4f}]  →  {"통계적으로 유의합니다." if sig else "통계적으로 확실하지 않습니다."}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # CATE distribution
    st.markdown("### CATE Distribution  /  개별 처치 효과 분포")
    _setup_font()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(cate, bins=30, color="#1a1a2e", edgecolor="white", alpha=0.85)
    ax.axvline(ate, color="#c0392b", lw=2, ls="--", label=f"ATE = {ate:.3f}")
    ax.axvline(0, color="#aaa", lw=1, ls=":")
    ax.set_xlabel("CATE  (개인별 처치 효과)")
    ax.set_ylabel("샘플 수")
    ax.set_title(f"CATE 분포:  {cfg['T_label']}  →  {cfg['Y_label'].split(' (')[0]}", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # CATE by confounders (최대 4개)
    W_cols = res.get("W_cols", cfg["W"])[:4]
    W_cols_exist = [c for c in W_cols if c in df.columns]
    if W_cols_exist and len(cate) > 0:
        st.markdown("### CATE by Confounder  /  교란변수별 처치 효과")
        n_w = len(W_cols_exist)
        fig2, axes = plt.subplots(1, n_w, figsize=(3.5 * n_w, 3.5))
        if n_w == 1:
            axes = [axes]
        n_cate = len(cate)
        for ax2, w_col in zip(axes, W_cols_exist):
            w_vals = df[w_col].values[:n_cate]
            ax2.scatter(w_vals, cate, alpha=0.3, s=12, color="#1a1a2e")
            ax2.axhline(0, color="#aaa", lw=1, ls=":")
            ax2.axhline(ate, color="#c0392b", lw=1, ls="--")
            ax2.set_xlabel(cfg["W_labels"].get(w_col, w_col), fontsize=8)
            ax2.set_ylabel("CATE", fontsize=8)
            ax2.set_title(cfg["W_labels"].get(w_col, w_col), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # AI 설명
    dir_kr = "높아진다" if ate >= 0 else "낮아진다"
    conf_kr = "통계적으로 확실하다" if sig else "아직 통계적으로 확실하지 않다"
    explain.show(
        f"ate_{domain}_{round(ate, 4)}",
        f"'{cfg['name']}' 업종에서 '{cfg['T_label']}'이 '{cfg['Y_label'].split(' (')[0]}'에 주는 순수 인과 효과를 추정했다. "
        f"처치를 받으면 결과가 평균 {abs(ate):.4f}만큼 {dir_kr}. "
        f"이 효과는 {conf_kr} (95% CI: [{ate_lb:.3f}, {ate_ub:.3f}]). "
        f"CATE 분포를 보면 사람마다 효과 크기가 다르다. "
        f"이 결과를 방향과 크기를 그대로 유지하면서 일반인에게 2~3문장으로 설명하라. 효과가 생긴 이유를 지어내지 마라.",
        domain=domain,
    )

    st.divider()
    _nav(back_step=2, next_step=4)


# ===========================================================================
# Step 4 — 반사실 시뮬레이션
# ===========================================================================
def page_counterfactual() -> None:
    _show_usage_badge()
    domain = st.session_state["domain"]
    cfg = DOMAINS[domain]
    df: pd.DataFrame = st.session_state.get("df")
    estimator = st.session_state.get("estimator")

    if df is None or estimator is None:
        st.warning("ATE/CATE 추정을 먼저 실행해주세요.")
        st.session_state["step"] = 3
        st.rerun()
        return

    st.markdown(f"## {cfg['name']}  /  반사실 시뮬레이션")
    _step_bar(4)
    st.markdown(
        '<div class="info-box">'
        '교란변수(W) 값을 슬라이더로 조절하면, 처치(T=1 vs T=0)에 따른 결과(Y) 차이가 실시간으로 변합니다.<br>'
        '"만약 처치를 받았다면 / 받지 않았다면?" 을 개인 조건별로 시뮬레이션합니다.'
        '</div>',
        unsafe_allow_html=True,
    )

    W_cols = [c for c in cfg["W"] if c in df.columns]
    W_labels = cfg["W_labels"]
    W_value_labels: dict = cfg.get("W_value_labels", {})
    ate: float = st.session_state.get("ate", 0.0)
    cate_all: np.ndarray = st.session_state.get("cate", np.array([]))

    # W sliders
    st.markdown("### 개입 조건 설정  /  배경 변수 값 선택")
    n_cols = min(len(W_cols), 4)
    slider_cols = st.columns(n_cols)
    w_vals: dict = {}
    for i, w_col in enumerate(W_cols):
        with slider_cols[i % n_cols]:
            col_data = df[w_col].dropna()
            unique_vals = sorted(col_data.unique())
            label = W_labels.get(w_col, w_col)
            vl = W_value_labels.get(w_col, {})
            if len(unique_vals) <= 6:
                if vl:
                    # 값+의미 함께 표시
                    opt_display = [f"{int(v)}  ({vl.get(int(v), v)})" for v in unique_vals]
                    default_idx = len(unique_vals) // 2
                    sel = st.selectbox(label, options=opt_display,
                                       index=default_idx, key=f"cf_{w_col}")
                    val = float(unique_vals[opt_display.index(sel)])
                else:
                    val = float(st.selectbox(label, options=unique_vals,
                                             index=len(unique_vals) // 2, key=f"cf_{w_col}"))
            else:
                mn, mx, mean = float(col_data.min()), float(col_data.max()), float(col_data.mean())
                val = st.slider(label, min_value=mn, max_value=mx, value=mean,
                                step=(mx - mn) / 100, key=f"cf_{w_col}")
            w_vals[w_col] = float(val)

    # Compute CATE for this point
    try:
        from causal_engine import point_cate
        X_pt = np.array([[w_vals[w] for w in W_cols]])
        cate_pt = point_cate(estimator, X_pt)
    except Exception as e:
        st.error(f"반사실 계산 실패: {e}")
        _nav(back_step=3)
        return

    # T=0 그룹의 Y 평균을 기준값으로 사용 (전체 평균 X)
    t_col = cfg["T"]
    y_col = cfg["Y"]
    y_is_binary = cfg.get("Y_binary", False)
    df_t0 = df[df[t_col] == 0]
    y_base = float(df_t0[y_col].mean()) if len(df_t0) > 0 else float(df[y_col].mean())
    y_treated = y_base + cate_pt

    # 이진 Y는 확률로 클리핑
    if y_is_binary:
        y_base    = float(np.clip(y_base, 0, 1))
        y_treated = float(np.clip(y_treated, 0, 1))

    # Metrics
    st.markdown("### 선택한 조건에서의 처치 효과")
    y_unit = "확률" if y_is_binary else cfg["Y_label"].split(" (")[0]
    c1, c2, c3 = st.columns(3)
    base_label = f"T=0 미처치  {y_unit} 기준"
    treat_label = f"T=1 처치 시  {y_unit} 예측"
    c1.markdown(_metric(base_label,
                        f"{y_base:.3f}" + (f"  ({y_base*100:.1f}%)" if y_is_binary else "")),
                unsafe_allow_html=True)
    c2.markdown(_metric(treat_label,
                        f"{y_treated:.3f}" + (f"  ({y_treated*100:.1f}%)" if y_is_binary else "")),
                unsafe_allow_html=True)
    pt_color = "#2d6a4f" if cate_pt >= 0 else "#c0392b"
    cate_disp = f"{cate_pt:+.3f}" + (f"  ({cate_pt*100:+.1f}%p)" if y_is_binary else "")
    c3.markdown(_metric("이 조건의 CATE", cate_disp, pt_color), unsafe_allow_html=True)

    # Charts
    _setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: T=0 vs T=1 bar
    ax = axes[0]
    labels_bar = ["T=0  미처치", "T=1  처치 적용"]
    vals_bar = [y_base, y_treated]
    bar_colors = ["#6b6b8a", "#1a1a2e"]
    bars = ax.bar(labels_bar, vals_bar, color=bar_colors, edgecolor="white", width=0.5)
    y_range = abs(y_treated - y_base) or 0.1
    for bar, v in zip(bars, vals_bar):
        label_txt = f"{v:.3f}" + (f"\n({v*100:.1f}%)" if y_is_binary else "")
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + y_range * 0.05, label_txt,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    y_min = max(0, min(vals_bar) - y_range * 0.4) if y_is_binary else min(vals_bar) - y_range * 0.3
    y_max = min(1.1, max(vals_bar) + y_range * 0.5) if y_is_binary else max(vals_bar) + y_range * 0.4
    ax.set_ylim(y_min, y_max)
    y_axis_label = f"{cfg['Y_label'].split(' (')[0]}  {'(확률)' if y_is_binary else ''}"
    ax.set_ylabel(y_axis_label)
    ax.set_title("처치 전/후 결과 비교  (T=0 그룹 기준)", fontsize=10)

    # Right: CATE distribution with this point highlighted
    ax2 = axes[1]
    if len(cate_all) > 0:
        ax2.hist(cate_all, bins=30, color="#ccc", edgecolor="white", alpha=0.8, label="전체 CATE 분포")
    ax2.axvline(cate_pt, color="#c0392b", lw=2.5, label=f"선택 조건 CATE: {cate_pt:+.3f}")
    ax2.axvline(ate, color="#1a1a2e", lw=1.5, ls="--", label=f"전체 평균 ATE: {ate:+.3f}")
    ax2.axvline(0, color="#aaa", lw=1, ls=":")
    ax2.set_xlabel("CATE (처치 효과)")
    ax2.set_ylabel("샘플 수")
    ax2.set_title("선택 조건의 효과 위치  vs  전체 분포", fontsize=10)
    ax2.legend(fontsize=7.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # AI 설명 — W 값 레이블 포함, 이진 Y는 %p 표현
    vl_map = cfg.get("W_value_labels", {})
    def _fmt_w(w, v):
        vl = vl_map.get(w, {})
        lbl = W_labels.get(w, w)
        if vl and int(v) in vl:
            return f"{lbl}={vl[int(v)]}({int(v)})"
        return f"{lbl}={v:.1f}" if v != int(v) else f"{lbl}={int(v)}"
    w_cond_str = ", ".join(_fmt_w(w, v) for w, v in w_vals.items())

    dir_kr  = "높아진다" if cate_pt >= 0 else "낮아진다"
    vs_avg  = "더 큰" if abs(cate_pt) > abs(ate) else "더 작은"
    y_name  = cfg["Y_label"].split(" (")[0]
    if y_is_binary:
        effect_str = f"{abs(cate_pt)*100:.1f}%p({abs(cate_pt):.3f})만큼 {dir_kr} (확률 변화)"
        base_str   = f"미처치 시 {y_base*100:.1f}%, 처치 시 {y_treated*100:.1f}%로 추정된다"
    else:
        effect_str = f"{abs(cate_pt):.3f}만큼 {dir_kr}"
        base_str   = f"미처치 시 {y_base:.3f}, 처치 시 {y_treated:.3f}로 추정된다"
    explain.show(
        f"cf_{domain}_{round(cate_pt, 3)}",
        f"'{cfg['name']}' 업종 반사실 시뮬레이션. "
        f"조건[{w_cond_str}]에서 '{cfg['T_label']}'을 받으면 '{y_name}'이 {effect_str}. "
        f"{base_str}. "
        f"전체 평균(ATE={ate:.3f})보다 이 조건에서는 {vs_avg} 효과. "
        f"이 시뮬레이션의 의미와 왜 조건마다 효과가 다른지 2~3문장으로 설명하라.",
        domain=domain,
    )

    st.divider()
    _nav(back_step=3, next_step=0, next_label="처음으로", next_home=True)


# ===========================================================================
# LLM 사용량 — 우상단 고정 표시
# ===========================================================================
def _show_usage_badge() -> None:
    """모든 페이지 상단에서 호출 — AI 사용량 배지 표시."""
    try:
        _ui = llm.usage_info()
        if llm.detect_mode() == "gemini":
            _cnt, _lim = _ui["count"], _ui["limit"]
            _pct = min(int(_cnt / _lim * 100), 100) if _lim else 0
            if _cnt >= _lim:
                color, msg = "#c0392b", f"AI 해석 한도 소진 ({_cnt}/{_lim})"
            elif _pct >= 80:
                color, msg = "#e67e22", f"AI 해석 {_cnt}/{_lim}회"
            else:
                color, msg = "#2d6a4f", f"AI 해석 {_cnt}/{_lim}회"
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:{color};'
                f'margin-bottom:4px">{msg}  ({_ui["mode"]})</div>',
                unsafe_allow_html=True,
            )
        elif llm.detect_mode() == "ollama":
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:#888;margin-bottom:4px">'
                f'{_ui["mode"]}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

# ===========================================================================
# Router
# ===========================================================================
_step = st.session_state.get("step", 0)
if _step == 0:
    page_domain_select()
elif _step == 1:
    page_data()
elif _step == 2:
    page_dag()
elif _step == 3:
    page_ate_cate()
elif _step == 4:
    page_counterfactual()
else:
    _reset()
    st.rerun()
