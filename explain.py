"""explain.py — 업종별 AI 해석 생성 및 Streamlit 렌더링"""

import hashlib
import streamlit as st
import llm

# ---------------------------------------------------------------------------
# 업종별 맥락 + 변수 사전
# ---------------------------------------------------------------------------
_DOMAIN_CTX = {
    "public": {
        "context": "정부 직업훈련(NSW) 프로그램 참여가 1978년 실질소득에 미치는 효과를 분석하는 인과추론 데모다.",
        "vars": {
            "treat": "직업훈련 참여 여부",
            "age": "나이(세)",
            "educ": "교육연수(년)",
            "black": "흑인 여부",
            "hisp": "히스패닉 여부",
            "married": "기혼 여부",
            "re74": "1974년 소득(달러)",
            "re75": "1975년 소득(달러)",
            "re78": "1978년 실질소득(달러)",
        },
    },
    "edu": {
        "context": "포르투갈 중고교생의 유료 보충학습 참여가 최종 수학 성적(G3)에 미치는 효과를 분석하는 인과추론 데모다.",
        "vars": {
            "paid_bin": "유료 보충학습 참여",
            "age": "나이(세)",
            "Medu": "어머니 학력(0-4)",
            "Fedu": "아버지 학력(0-4)",
            "studytime": "주간 공부시간(1-4)",
            "failures": "과거 낙제 횟수",
            "absences": "결석 일수",
            "G1": "1학기 성적(0-20)",
            "G2": "2학기 성적(0-20)",
            "G3": "최종 성적(0-20)",
        },
    },
    "finance": {
        "context": "신용카드 9월 연체 여부가 다음달 채무불이행(부도)에 미치는 효과를 분석하는 인과추론 데모다.",
        "vars": {
            "PAY_0_bin": "9월 연체 여부",
            "LIMIT_BAL": "신용한도(원)",
            "EDUCATION": "학력(1=대학원, 2=대학, 3=고교)",
            "MARRIAGE": "결혼상태(1=기혼, 2=미혼)",
            "AGE": "나이(세)",
            "BILL_AMT1": "9월 청구금액(원)",
            "PAY_AMT1": "9월 납부금액(원)",
            "default_payment_next_month": "다음달 채무불이행",
        },
    },
    "mfg": {
        "context": "공구 과다 마모 여부가 기계 고장에 미치는 효과를 분석하는 AI4I 2020 예측정비 인과추론 데모다.",
        "vars": {
            "tool_wear_high": "공구 과다 마모 여부",
            "air_temp": "공기 온도(°C)",
            "process_temp": "공정 온도(°C)",
            "rpm": "회전 속도(RPM)",
            "torque": "토크(Nm)",
            "machine_failure": "기계 고장 여부",
        },
    },
}

_SYSTEM_BASE = """당신은 인과추론(Causal Inference) 분석 결과를 통계를 전혀 모르는 일반인에게 쉽게 풀어 설명하는 전문가다.

[절대 규칙]
- 주어진 변수 사전에 없는 소재를 절대 만들어내지 마라.
- 변수명 영문 약어를 그대로 쓰지 말고 반드시 한글 이름으로 바꿔라.
- 주어진 숫자와 방향(증가/감소)을 절대 바꾸지 마라.
- 효과가 생긴 이유를 지어내지 마라.

[용어 풀이]
- 처치(T): 효과를 알아보려고 바꿔 보는 원인 변수
- 결과(Y): 측정하려는 목표 지표
- 교란변수(W): 원인과 결과 양쪽에 동시에 영향을 주는 배경 요인
- ATE: 평균 처치 효과 (전체 집단 평균)
- CATE: 개별 처치 효과 (개인마다 다른 효과)
- 반사실: "만약 다르게 했다면 어땠을까"를 추정하는 것
- 통계적으로 유의: 신뢰구간이 0을 포함하지 않아 효과가 확실함
- 통계적으로 비유의: 신뢰구간이 0을 포함해 효과가 불확실함

[작성 규칙]
- 반드시 2~3문장으로 끝내라. 절대 4문장 이상 쓰지 마라.
- 중학생도 이해할 수준으로 쉽게 쓰라.
- 불릿/제목/이모지 없이 자연스러운 문단으로만 답하라."""


def _system(domain: str = None) -> str:
    if domain and domain in _DOMAIN_CTX:
        ctx = _DOMAIN_CTX[domain]
        var_lines = "\n".join(f"- {k}: {v}" for k, v in ctx["vars"].items())
        return f"{_SYSTEM_BASE}\n\n[이 분석의 맥락]\n{ctx['context']}\n\n[변수 사전]\n{var_lines}"
    return _SYSTEM_BASE


def _box(text: str, label: str) -> None:
    st.markdown(
        f'<div style="border:1px solid #d8e0ea;border-radius:8px;margin:0.6rem 0 0.2rem;overflow:hidden">'
        f'<div style="background:#2C3E6B;padding:6px 12px;color:#fff;font-size:0.8rem;font-weight:600">'
        f'AI 해석'
        f'<span style="float:right;font-weight:400;opacity:0.75;font-size:0.72rem">{label}</span>'
        f'</div>'
        f'<div style="padding:11px 14px;font-size:0.9rem;line-height:1.65;color:#1a1a2e;background:#fafbfd">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_PREFIX = "【규칙】반드시 3문장 이내로만 답하라.\n\n"


def show(cache_key: str, prompt: str, temperature: float = 0.3, domain: str = None) -> None:
    """AI 해석 생성 및 렌더링. 동일 캐시 키는 재호출하지 않는다."""
    h = hashlib.md5((cache_key + "||" + prompt).encode("utf-8")).hexdigest()
    sk = f"_ai_{h}"
    if sk not in st.session_state:
        try:
            with st.spinner("AI가 결과를 해석하는 중..."):
                result = llm.generate(
                    _PREFIX + prompt,
                    system=_system(domain),
                    temperature=temperature,
                    max_tokens=600,
                )
                st.session_state[sk] = result
        except Exception:
            # LLM 미연결 또는 모델 없음 — AI 해석 없이 조용히 스킵
            return
    _box(st.session_state[sk], llm.status_label())
