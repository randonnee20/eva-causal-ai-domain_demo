"""data_loader.py — 4개 업종 데이터셋 다운로드 및 전처리
다운로드 실패 시 합성 데이터로 자동 fallback.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

_TIMEOUT = 30


def _get(url: str) -> requests.Response:
    r = requests.get(url, timeout=_TIMEOUT)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# 공공 — LaLonde (NSW)
# ---------------------------------------------------------------------------
_LALONDE_URLS = [
    "https://raw.githubusercontent.com/py-why/dowhy/main/docs/source/example_notebooks/datasets/nsw_dw.dta",
    "https://raw.githubusercontent.com/gsbDBI/ExperimentData/master/Welfare/ProcessedData/lalonde_nsw.csv",
]

_LALONDE_COLS = ["treat", "age", "educ", "black", "hisp", "married", "re74", "re75", "re78"]


def _load_lalonde() -> pd.DataFrame:
    # Try Stata file
    try:
        r = _get(_LALONDE_URLS[0])
        df = pd.read_stata(io.BytesIO(r.content))
        df.columns = [c.lower().strip() for c in df.columns]
        missing = [c for c in _LALONDE_COLS if c not in df.columns]
        if not missing:
            return df[_LALONDE_COLS].dropna().reset_index(drop=True)
    except Exception:
        pass
    # Try CSV
    try:
        r = _get(_LALONDE_URLS[1])
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.lower().strip() for c in df.columns]
        available = [c for c in _LALONDE_COLS if c in df.columns]
        if len(available) >= 5:
            return df[available].dropna().reset_index(drop=True)
    except Exception:
        pass
    return _synthetic_lalonde()


def _synthetic_lalonde() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 722
    age = rng.integers(17, 55, n).astype(float)
    educ = rng.integers(3, 16, n).astype(float)
    black = rng.binomial(1, 0.84, n).astype(float)
    hisp = rng.binomial(1, 0.06, n).astype(float)
    married = rng.binomial(1, 0.19, n).astype(float)
    re74 = np.maximum(0, rng.normal(2095, 4886, n))
    re75 = np.maximum(0, rng.normal(1532, 3219, n))
    treat = rng.binomial(1, 0.43, n).astype(float)
    # ATE ≈ 1794 (LaLonde 1986)
    re78 = np.maximum(0, 2000 + 1794 * treat + 50 * age + 300 * educ + rng.normal(0, 3000, n))
    return pd.DataFrame({
        "treat": treat, "age": age, "educ": educ, "black": black,
        "hisp": hisp, "married": married, "re74": re74, "re75": re75, "re78": re78,
    })


# ---------------------------------------------------------------------------
# 교육 — Student Performance (UCI)
# ---------------------------------------------------------------------------
_STUDENT_URLS = [
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
]

_STUDENT_COLS = ["paid_bin", "age", "Medu", "Fedu", "studytime", "failures", "absences", "G1", "G2", "G3"]


def _load_student() -> pd.DataFrame:
    for url in _STUDENT_URLS:
        try:
            r = _get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_names = [n for n in z.namelist() if "student-mat" in n.lower() and n.endswith(".csv")]
            if not csv_names:
                csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if csv_names:
                df = pd.read_csv(z.open(csv_names[0]), sep=";")
                return _preprocess_student(df)
        except Exception:
            continue
    return _synthetic_student()


def _preprocess_student(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "paid" in df.columns:
        df["paid_bin"] = (df["paid"].astype(str).str.lower() == "yes").astype(float)
    else:
        df["paid_bin"] = 0.0
    avail = [c for c in _STUDENT_COLS if c in df.columns]
    df = df[avail].copy()
    for c in avail:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _synthetic_student() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 395
    age = rng.integers(15, 22, n).astype(float)
    Medu = rng.integers(0, 5, n).astype(float)
    Fedu = rng.integers(0, 5, n).astype(float)
    studytime = rng.integers(1, 5, n).astype(float)
    failures = rng.integers(0, 4, n).astype(float)
    absences = rng.integers(0, 30, n).astype(float)
    G1 = np.clip(rng.normal(10.5, 3.0, n), 0, 20)
    G2 = np.clip(G1 + rng.normal(0, 1.5, n), 0, 20)
    paid_bin = rng.binomial(1, 0.32, n).astype(float)
    G3 = np.clip(10.0 + 1.5 * paid_bin + 0.5 * G2 + 0.3 * studytime - 0.4 * failures + rng.normal(0, 2, n), 0, 20)
    return pd.DataFrame({
        "paid_bin": paid_bin, "age": age, "Medu": Medu, "Fedu": Fedu,
        "studytime": studytime, "failures": failures, "absences": absences,
        "G1": G1, "G2": G2, "G3": G3,
    })


# ---------------------------------------------------------------------------
# 금융 — Credit Default (UCI)
# ---------------------------------------------------------------------------
_CREDIT_CSV_URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/UCI_Credit_Card.csv",
]
_CREDIT_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
]

_CREDIT_COLS = ["PAY_0_bin", "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE", "BILL_AMT1", "PAY_AMT1", "default_payment_next_month"]


def _load_credit() -> pd.DataFrame:
    # Try CSV first
    for url in _CREDIT_CSV_URLS:
        try:
            r = _get(url)
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = [c.strip() for c in df.columns]
            return _preprocess_credit(df)
        except Exception:
            pass
    # Try ZIP (XLS)
    for url in _CREDIT_ZIP_URLS:
        try:
            r = _get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            xls_names = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))]
            if xls_names:
                df = pd.read_excel(z.open(xls_names[0]), header=1)
                df.columns = [c.strip() for c in df.columns]
                return _preprocess_credit(df)
        except Exception:
            pass
    return _synthetic_credit()


def _preprocess_credit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalize target column name
    for c in df.columns:
        if "default" in c.lower():
            df = df.rename(columns={c: "default_payment_next_month"})
            break

    # Binarize PAY_0: delayed >= 1 month
    if "PAY_0" in df.columns:
        df["PAY_0_bin"] = (pd.to_numeric(df["PAY_0"], errors="coerce") >= 1).astype(float)
    else:
        df["PAY_0_bin"] = 0.0

    avail = [c for c in _CREDIT_COLS if c in df.columns]
    df = df[avail].copy()
    for c in avail:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    if "EDUCATION" in df.columns:
        df = df[df["EDUCATION"].between(1, 6)]
    if "MARRIAGE" in df.columns:
        df = df[df["MARRIAGE"].between(0, 3)]

    return df.head(5000).reset_index(drop=True)


def _synthetic_credit() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    limit_bal = np.maximum(10000, rng.exponential(150000, n))
    education = rng.integers(1, 4, n).astype(float)
    marriage = rng.integers(1, 3, n).astype(float)
    age = rng.integers(21, 75, n).astype(float)
    bill_amt1 = np.maximum(0, rng.exponential(50000, n))
    pay_amt1 = np.maximum(0, rng.exponential(5000, n))
    pay_0_bin = rng.binomial(1, 0.22, n).astype(float)
    prob = np.clip(0.10 + 0.25 * pay_0_bin - 0.01 * (limit_bal / 100000) + rng.normal(0, 0.05, n), 0.01, 0.99)
    default = rng.binomial(1, prob).astype(float)
    return pd.DataFrame({
        "PAY_0_bin": pay_0_bin, "LIMIT_BAL": limit_bal, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age, "BILL_AMT1": bill_amt1,
        "PAY_AMT1": pay_amt1, "default_payment_next_month": default,
    })


# ---------------------------------------------------------------------------
# 제조 — AI4I 2020 Predictive Maintenance
# ---------------------------------------------------------------------------
_AI4I_URLS = [
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "https://raw.githubusercontent.com/praveengururajan/AI4I-2020-Predictive-Maintenance-Dataset/main/ai4i2020.csv",
]

_AI4I_COL_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

_AI4I_COLS = ["tool_wear_high", "air_temp", "process_temp", "rpm", "torque", "machine_failure"]


def _load_ai4i() -> pd.DataFrame:
    # Try ZIP
    try:
        r = _get(_AI4I_URLS[0])
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if csv_names:
            df = pd.read_csv(z.open(csv_names[0]))
            return _preprocess_ai4i(df)
    except Exception:
        pass
    # Try raw CSV
    try:
        r = _get(_AI4I_URLS[1])
        df = pd.read_csv(io.BytesIO(r.content))
        return _preprocess_ai4i(df)
    except Exception:
        pass
    return _synthetic_ai4i()


def _preprocess_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=_AI4I_COL_MAP)
    if "tool_wear" in df.columns:
        threshold = float(df["tool_wear"].quantile(0.80))
        df["tool_wear_high"] = (pd.to_numeric(df["tool_wear"], errors="coerce") >= threshold).astype(float)
    else:
        df["tool_wear_high"] = 0.0

    avail = [c for c in _AI4I_COLS if c in df.columns]
    df = df[avail].copy()
    for c in avail:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 켈빈 → 섭씨 변환
    for col in ["air_temp", "process_temp"]:
        if col in df.columns:
            df[col] = df[col] - 273.15
    return df.dropna().reset_index(drop=True)


def _synthetic_ai4i() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 10000
    air_temp = rng.normal(300 - 273.15, 2, n)
    process_temp = rng.normal(310 - 273.15, 1, n)
    rpm = rng.normal(1538, 179, n)
    torque = rng.normal(40, 10, n)
    tool_wear_high = rng.binomial(1, 0.20, n).astype(float)
    prob = np.clip(0.02 + 0.15 * tool_wear_high + 0.001 * (torque - 40) + rng.normal(0, 0.03, n), 0.001, 0.99)
    machine_failure = rng.binomial(1, prob).astype(float)
    return pd.DataFrame({
        "tool_wear_high": tool_wear_high, "air_temp": air_temp,
        "process_temp": process_temp, "rpm": rpm, "torque": torque,
        "machine_failure": machine_failure,
    })


# ---------------------------------------------------------------------------
# 공개 진입점
# ---------------------------------------------------------------------------
_LOADERS = {
    "public": _load_lalonde,
    "edu": _load_student,
    "finance": _load_credit,
    "mfg": _load_ai4i,
}


def load_domain_data(domain: str) -> pd.DataFrame:
    """도메인 데이터 로드 (로컬 캐시 우선)."""
    cache = DATA_DIR / f"{domain}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    loader = _LOADERS.get(domain)
    if loader is None:
        raise ValueError(f"Unknown domain: {domain}")
    df = loader()
    df.to_csv(cache, index=False)
    return df


def clear_cache(domain: str = None) -> None:
    """캐시 삭제 (domain=None 이면 전체)."""
    if domain:
        p = DATA_DIR / f"{domain}.csv"
        if p.exists():
            p.unlink()
    else:
        for p in DATA_DIR.glob("*.csv"):
            p.unlink()
