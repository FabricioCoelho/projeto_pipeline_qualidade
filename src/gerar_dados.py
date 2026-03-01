"""
gerar_dados.py
--------------
Gera o dataset sintético de sensores de subestações elétricas
com problemas de qualidade injetados intencionalmente.
"""

import numpy as np
import pandas as pd


def gerar_dataset(n: int = 8000, seed: int = 99) -> pd.DataFrame:
    """
    Gera dataset de leituras de sensores IoT de subestações elétricas.

    Parâmetros
    ----------
    n    : número de registros base
    seed : semente aleatória para reprodutibilidade

    Retorna
    -------
    pd.DataFrame com problemas de qualidade injetados (nulos, outliers, duplicatas)
    """
    np.random.seed(seed)

    datas = pd.date_range("2024-01-01", periods=n, freq="30min")

    df = pd.DataFrame({
        "timestamp"        : datas,
        "subestacao_id"    : np.random.choice([f"SE_{i:02d}" for i in range(1, 9)], n),
        "tensao_kv"        : np.random.normal(138, 2.5, n).round(3),
        "corrente_a"       : np.abs(np.random.normal(450, 80, n)).round(2),
        "potencia_mw"      : np.abs(np.random.normal(95, 18, n)).round(3),
        "temperatura_c"    : np.random.normal(42, 6, n).round(1),
        "status_alarme"    : np.random.choice(
            ["OK", "AVISO", "CRITICO"], n, p=[0.88, 0.09, 0.03]
        ),
        "operador_id"      : np.random.choice(
            [f"OP{i:03d}" for i in range(1, 15)], n
        ),
    })

    # ── Injeção de problemas de qualidade ────────────────────────────────────

    # 1. Nulos (~2.5% por coluna numérica)
    for col in ["tensao_kv", "corrente_a", "potencia_mw"]:
        idx = df.sample(frac=0.025, random_state=7).index
        df.loc[idx, col] = np.nan

    # 2. Outliers físicos impossíveis
    idx_out = df.sample(n=150, random_state=5).index
    df.loc[idx_out[:50],   "tensao_kv"]    = np.random.choice([-5, 0, 999], 50)
    df.loc[idx_out[50:100], "corrente_a"]  = -100
    df.loc[idx_out[100:],  "temperatura_c"] = 250

    # 3. Duplicatas
    duplicatas = df.sample(n=100, random_state=9)
    df = pd.concat([df, duplicatas], ignore_index=True)

    print(f"✅ Dataset gerado: {df.shape[0]:,} registros | {df.shape[1]} colunas")
    print(f"   Período : {datas[0].date()} → {datas[-1].date()}")
    print(f"   Nulos injetados por coluna:")
    print(df[["tensao_kv", "corrente_a", "potencia_mw"]].isna().sum().to_string(
        header=False
    ))

    return df
