"""llm.py — LLM 라우터

우선순위:
  1. Ollama (로컬, gemma3:4b)  — PC 실행 시
  2. Gemini API (Google)       — 클라우드 배포 시 기본
  3. Claude API (Anthropic)    — Gemini 키 없을 때 폴백
  4. none                      — LLM 없이 조용히 스킵
"""

import os
import requests

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_mode = None


# ---------------------------------------------------------------------------
# 가용 모드 감지
# ---------------------------------------------------------------------------
def _ollama_model_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
        names = [m.get("name", "").split(":")[0] for m in tags]
        model_base = OLLAMA_MODEL.split(":")[0]
        return any(model_base in n for n in names) or any(
            OLLAMA_MODEL in m.get("name", "") for m in tags
        )
    except Exception:
        return False


def _get_gemini_key() -> str:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _get_claude_key() -> str:
    try:
        import streamlit as st
        if "ANTHROPIC_API_KEY" in st.secrets:
            return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def detect_mode(force: bool = False) -> str:
    global _mode
    if _mode is not None and not force:
        return _mode
    if _ollama_model_available():
        _mode = "ollama"
    elif _get_gemini_key():
        _mode = "gemini"
    elif _get_claude_key():
        _mode = "claude"
    else:
        _mode = "none"
    return _mode


def status_label() -> str:
    return {
        "ollama": f"로컬 {OLLAMA_MODEL}",
        "gemini": f"Google {GEMINI_MODEL}",
        "claude": "Claude API",
        "none":   "LLM 미연결",
    }.get(detect_mode(), "알 수 없음")


# ---------------------------------------------------------------------------
# 생성 함수
# ---------------------------------------------------------------------------
def _gen_ollama(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _gen_gemini(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=_get_gemini_key())
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system or "You are a helpful assistant.",
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def _gen_claude(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_get_claude_key())
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return msg.content[0].text.strip()


def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.4,
    max_tokens: int = 600,
) -> str:
    mode = detect_mode()
    if mode == "ollama":
        return _gen_ollama(prompt, system, temperature, max_tokens)
    if mode == "gemini":
        return _gen_gemini(prompt, system, temperature, max_tokens)
    if mode == "claude":
        return _gen_claude(prompt, system, temperature, max_tokens)
    raise RuntimeError("LLM 사용 불가: Ollama 미구동, Gemini/Claude API 키 없음.")
