"""
visualizacoes.py
----------------
Funções de visualização para o pipeline de qualidade de dados.
Cada função gera e salva uma figura na pasta outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Pasta de saída
OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

plt.rcParams["font.size"] = 11
sns.set_theme(style="whitegrid")


def plot_antes_depois(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Histogramas comparativos antes e depois do pipeline."""
    variaveis = ["tensao_kv", "corrente_a", "temperatura_c"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, var in enumerate(variaveis):
        # Antes
        dados_raw = df_raw[var].dropna()
        axes[0, idx].hist(dados_raw, bins=50, color="#EF5350", alpha=0.75, edgecolor="white")
        axes[0, idx].set_title(f"{var} — RAW", fontweight="bold")
        axes[0, idx].set_ylabel("Frequência" if idx == 0 else "")

        # Depois
        axes[1, idx].hist(df_clean[var], bins=50, color="#42A5F5", alpha=0.75, edgecolor="white")
        axes[1, idx].set_title(f"{var} — LIMPO", fontweight="bold")
        axes[1, idx].set_ylabel("Frequência" if idx == 0 else "")

    fig.suptitle(
        "Distribuições: Antes vs Depois do Pipeline de Qualidade",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    caminho = OUTPUTS / "fig1_antes_depois.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   💾 Salvo: {caminho}")


def plot_dashboard(df_clean: pd.DataFrame, metricas: dict):
    """Dashboard de monitoramento das subestações."""
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Potência média por subestação
    ax1 = fig.add_subplot(gs[0, :])
    pot = df_clean.groupby("subestacao_id")["potencia_mw"].agg(["mean", "std"])
    x   = range(len(pot))
    ax1.bar(x, pot["mean"], yerr=pot["std"], capsize=4,
            color="#1565C0", alpha=0.8, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pot.index)
    ax1.set_title("Potência Média por Subestação (± desvio padrão)", fontweight="bold")
    ax1.set_ylabel("Potência (MW)")

    # Série temporal de tensão (SE_01)
    ax2 = fig.add_subplot(gs[1, :2])
    se01 = df_clean[df_clean["subestacao_id"] == "SE_01"].sort_values("timestamp")
    ax2.plot(
        se01["timestamp"].iloc[:200], se01["tensao_kv"].iloc[:200],
        color="#0097A7", linewidth=1.5, alpha=0.85
    )
    ax2.set_title("Série Temporal — Tensão (SE_01)", fontweight="bold")
    ax2.set_ylabel("Tensão (kV)")
    ax2.tick_params(axis="x", rotation=20)

    # Pizza de alarmes
    ax3 = fig.add_subplot(gs[1, 2])
    alarmes = df_clean["status_alarme"].value_counts()
    cores   = {"OK": "#66BB6A", "AVISO": "#FFA726", "CRITICO": "#EF5350"}
    ax3.pie(
        alarmes.values,
        labels=alarmes.index,
        colors=[cores.get(k, "#9E9E9E") for k in alarmes.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax3.set_title("Status de Alarmes", fontweight="bold")

    # Boxplot temperatura por hora
    ax4 = fig.add_subplot(gs[2, :2])
    df_clean.boxplot(
        column="temperatura_c", by="hora", ax=ax4,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        medianprops={"color": "red", "linewidth": 2},
    )
    ax4.set_title("Temperatura por Hora do Dia", fontweight="bold")
    ax4.set_xlabel("Hora")
    ax4.set_ylabel("Temperatura (°C)")
    plt.sca(ax4)
    plt.title("Temperatura por Hora do Dia")

    # Tabela de métricas
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis("off")
    dados_tabela = [
        ["Registros RAW",    f"{metricas['registros_raw']:,}"],
        ["Registros LIMPOS", f"{metricas['registros_finais']:,}"],
        ["Retenção",         f"{metricas['retencao_pct']}%"],
        ["Completude",       f"{metricas['completude_pct']}%"],
        ["Etapas Pipeline",  str(metricas["etapas_pipeline"])],
    ]
    tabela = ax5.table(
        cellText=dados_tabela,
        colLabels=["Métrica", "Valor"],
        loc="center", cellLoc="center"
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.8)
    ax5.set_title("Sumário do Pipeline", fontweight="bold", pad=10)

    fig.suptitle(
        "Dashboard de Monitoramento — Rede de Subestações",
        fontsize=13, fontweight="bold"
    )
    caminho = OUTPUTS / "fig2_dashboard.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   💾 Salvo: {caminho}")


def plot_features(df_clean: pd.DataFrame):
    """Correlação entre features normalizadas e encoding cíclico da hora."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matriz de correlação
    cols_norm = [c for c in df_clean.columns if c.endswith("_norm")]
    corr = df_clean[cols_norm].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=axes[0], linewidths=0.5,
        cbar_kws={"label": "Correlação de Pearson"},
    )
    axes[0].set_title("Correlação — Features Normalizadas", fontweight="bold")
    axes[0].set_xticklabels(
        [c.replace("_norm", "") for c in cols_norm], rotation=30
    )
    axes[0].set_yticklabels(
        [c.replace("_norm", "") for c in cols_norm], rotation=0
    )

    # Encoding cíclico da hora
    amostra = df_clean.sample(500, random_state=42)
    sc = axes[1].scatter(
        amostra["hora_seno"], amostra["hora_cosseno"],
        c=amostra["hora"], cmap="hsv", alpha=0.6, s=40
    )
    plt.colorbar(sc, ax=axes[1], label="Hora do Dia")
    axes[1].set_xlabel("Seno (Hora Cíclica)")
    axes[1].set_ylabel("Cosseno (Hora Cíclica)")
    axes[1].set_title("Encoding Cíclico da Variável Hora", fontweight="bold")
    axes[1].set_aspect("equal")

    fig.suptitle(
        "Análise de Features Transformadas",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    caminho = OUTPUTS / "fig3_features.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   💾 Salvo: {caminho}")
