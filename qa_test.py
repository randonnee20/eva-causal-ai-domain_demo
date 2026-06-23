"""QA 테스트 스크립트"""
import sys
print("Python:", sys.version[:30])

print("\n[1] 모듈 임포트...")
import llm
import data_loader
import causal_engine
print("    OK")

print("\n[2] LLM 모드:", llm.detect_mode())

print("\n[3] 합성 데이터 fallback 테스트...")
df_pub = data_loader._synthetic_lalonde()
df_edu = data_loader._synthetic_student()
df_fin = data_loader._synthetic_credit()
df_mfg = data_loader._synthetic_ai4i()
print(f"    공공: {df_pub.shape}")
print(f"    교육: {df_edu.shape}")
print(f"    금융: {df_fin.shape}")
print(f"    제조: {df_mfg.shape}")

print("\n[4] 필수 컬럼 확인...")
from app import DOMAINS
checks = [("public", df_pub), ("edu", df_edu), ("finance", df_fin), ("mfg", df_mfg)]
all_ok = True
for domain, df in checks:
    cfg = DOMAINS[domain]
    missing = []
    if cfg["T"] not in df.columns:
        missing.append(f"T={cfg['T']}")
    if cfg["Y"] not in df.columns:
        missing.append(f"Y={cfg['Y']}")
    for c in cfg["W"]:
        if c not in df.columns:
            missing.append(f"W={c}")
    status = "OK" if not missing else f"FAIL: {missing}"
    print(f"    {domain}: {status}")
    if missing:
        all_ok = False

print("\n[5] causal_engine 소규모 실행 (제조 n=300)...")
from causal_engine import run_ate_cate, point_cate
import numpy as np

cfg_mfg = DOMAINS["mfg"]
res = run_ate_cate(df_mfg.head(300), cfg_mfg)
print(f"    ATE: {res['ate']:.4f}  CI: [{res['ate_ci'][0]:.4f}, {res['ate_ci'][1]:.4f}]")
print(f"    CATE shape: {res['cate'].shape}")

# point_cate test
W_cols = res["W_cols"]
x_pt = np.array([[df_mfg[w].mean() for w in W_cols]])
cate_pt = point_cate(res["estimator"], x_pt)
print(f"    point_cate (mean W): {cate_pt:.4f}")

print("\n[6] 교육 도메인 causal_engine 소규모 실행 (n=200)...")
cfg_edu = DOMAINS["edu"]
res_edu = run_ate_cate(df_edu.head(200), cfg_edu)
print(f"    ATE: {res_edu['ate']:.4f}  CI: [{res_edu['ate_ci'][0]:.4f}, {res_edu['ate_ci'][1]:.4f}]")

print("\n[7] explain.py 임포트 확인...")
import explain
print("    OK")

print("\n" + ("=" * 40))
print("ALL QA PASS" if all_ok else "QA FAIL - 위 출력 확인")
