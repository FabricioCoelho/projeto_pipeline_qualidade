import json
import time
from pathlib import Path

from src.gerar_dados     import gerar_dataset
from src.pipeline        import DataQualityPipeline
from src.visualizacoes   import plot_antes_depois, plot_dashboard, plot_features

#Configurações
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

SCHEMA = {
    "colunas_obrigatorias": [
        "timestamp", "subestacao_id", "tensao_kv",
        "corrente_a", "potencia_mw", "temperatura_c", "status_alarme",
    ],
    "tipos": {"timestamp": "datetime64[ns]"},
}

ESTRATEGIA_NULOS = {
    "tensao_kv"    : "median",
    "corrente_a"   : "median",
    "potencia_mw"  : "mean",
    "temperatura_c": "mean",
}

REGRAS_NEGOCIO = [
    {
        "nome"     : "tensao_invalida",
        "condicao" : "tensao_kv <= 0 or tensao_kv > 500",
        "acao"     : "drop",
    },
    {
        "nome"     : "corrente_negativa",
        "condicao" : "corrente_a < 0",
        "acao"     : "drop",
    },
    {
        "nome"     : "temp_alta_sem_alarme",
        "condicao" : 'temperatura_c > 80 and status_alarme == "OK"',
        "acao"     : "flag",
    },
]


# Execução
def main():
    print("  PIPELINE DE QUALIDADE DE DADOS — IoT / Subestações")
    print("\nGerando dataset sintético...")
    df_raw = gerar_dataset(n=8000)

    print("\nExecutando pipeline...")
    inicio = time.time()

    pipeline = (
        DataQualityPipeline(df_raw, SCHEMA)
        .validar_schema()
        .remover_duplicatas(subset=["timestamp", "subestacao_id"])
        .tratar_nulos(ESTRATEGIA_NULOS)
        .tratar_outliers(
            ["tensao_kv", "corrente_a", "temperatura_c"],
            metodo="iqr",
            acao="winsorize",
        )
        .validar_regras(REGRAS_NEGOCIO)
        .normalizar(
            ["tensao_kv", "corrente_a", "potencia_mw", "temperatura_c"],
            metodo="minmax",
        )
        .feature_engineering("timestamp")
    )

    duracao  = time.time() - inicio
    metricas = pipeline.calcular_metricas()
    df_clean = pipeline.df

    print(f"Pipeline concluído em {duracao:.3f}s")
    pipeline.log.imprimir()
    pipeline.imprimir_metricas()

    print("\nGerando visualizações...")
    plot_antes_depois(df_raw, df_clean)
    plot_dashboard(df_clean, metricas)
    plot_features(df_clean)

    print("\nExportando resultados...")

    # Dataset limpo
    csv_path = OUTPUTS / "dataset_sensores_limpo.csv"
    df_clean.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"   💾 Dataset salvo: {csv_path}")

    # Relatório JSON
    relatorio = {
        "pipeline_info": {
            "nome"     : "DataQualityPipeline v1.0",
            "dataset"  : "sensores_subestacao_2024",
            "data_exec": __import__("pandas").Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_s": round(duracao, 3),
        },
        "metricas_qualidade": metricas,
        "log_transformacoes": pipeline.log.etapas,
        "estatisticas_finais": {
            col: {
                "media": round(float(df_clean[col].mean()), 4),
                "std"  : round(float(df_clean[col].std()),  4),
                "min"  : round(float(df_clean[col].min()),  4),
                "max"  : round(float(df_clean[col].max()),  4),
            }
            for col in ["tensao_kv", "corrente_a", "potencia_mw", "temperatura_c"]
        },
    }

    json_path = OUTPUTS / "relatorio_qualidade.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"   Relatório salvo: {json_path}")

    print("\n  Processo finalizado com sucesso!")
    print(f"   Arquivos gerados em: {OUTPUTS.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
