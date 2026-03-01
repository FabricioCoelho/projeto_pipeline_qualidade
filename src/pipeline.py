"""
pipeline.py
-----------
Classe DataQualityPipeline — pipeline modular de pré-processamento
e validação de qualidade de dados para sensores IoT/AMI.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# LOG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineLog:
    """Registra cada transformação aplicada no pipeline."""
    etapas: List[Dict] = field(default_factory=list)

    def registrar(self, etapa: str, descricao: str, registros_afetados: int = 0):
        self.etapas.append({
            "etapa"               : etapa,
            "descricao"           : descricao,
            "registros_afetados"  : registros_afetados,
            "timestamp"           : pd.Timestamp.now().strftime("%H:%M:%S"),
        })

    def resumo(self) -> pd.DataFrame:
        return pd.DataFrame(self.etapas)

    def imprimir(self):
        print("\n📋 Log de Transformações:")
        print("-" * 65)
        for e in self.etapas:
            afetados = f"  ({e['registros_afetados']} registros)" if e["registros_afetados"] else ""
            print(f"  [{e['timestamp']}] {e['etapa']:<10} → {e['descricao']}{afetados}")
        print("-" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityPipeline:
    """
    Pipeline modular de pré-processamento e validação de qualidade de dados.

    Uso com method chaining:
        pipeline = (DataQualityPipeline(df, schema)
                    .validar_schema()
                    .remover_duplicatas()
                    .tratar_nulos({"col": "median"})
                    .tratar_outliers(["col1", "col2"])
                    .normalizar(["col1", "col2"])
                    .feature_engineering("timestamp"))
    """

    def __init__(self, df: pd.DataFrame, schema: Dict):
        self.df_raw   = df.copy()
        self.df       = df.copy()
        self.schema   = schema
        self.log      = PipelineLog()
        self.metricas = {}
        self.log.registrar(
            "INIT",
            f"Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas"
        )

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    def validar_schema(self) -> "DataQualityPipeline":
        """Verifica colunas obrigatórias e converte tipos."""
        faltantes = [
            c for c in self.schema["colunas_obrigatorias"]
            if c not in self.df.columns
        ]
        if faltantes:
            raise ValueError(f"Schema inválido — colunas faltantes: {faltantes}")

        for col, dtype in self.schema.get("tipos", {}).items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype)

        self.log.registrar("Stage 1", "Validação de schema OK — todos os campos presentes")
        return self

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    def remover_duplicatas(self, subset: Optional[List[str]] = None) -> "DataQualityPipeline":
        """Remove registros duplicados."""
        n = self.df.duplicated(subset=subset).sum()
        self.df.drop_duplicates(subset=subset, keep="first", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.log.registrar("Stage 2", "Duplicatas removidas", int(n))
        return self

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    def tratar_nulos(self, estrategia: Dict[str, str]) -> "DataQualityPipeline":
        """
        Trata valores nulos por coluna.
        estrategia: {"coluna": "mean" | "median" | "mode" | "drop" | <valor>}
        """
        total = 0
        for col, metodo in estrategia.items():
            if col not in self.df.columns:
                continue
            nulos = int(self.df[col].isna().sum())
            if nulos == 0:
                continue
            total += nulos
            if metodo == "mean":
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif metodo == "median":
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif metodo == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif metodo == "drop":
                self.df.dropna(subset=[col], inplace=True)
            else:
                self.df[col].fillna(metodo, inplace=True)

        self.log.registrar("Stage 3", "Nulos tratados por coluna", total)
        return self

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    def tratar_outliers(
        self,
        colunas : List[str],
        metodo  : str = "iqr",
        acao    : str = "winsorize",
    ) -> "DataQualityPipeline":
        """
        Detecta e trata outliers.
        metodo: "iqr" | "zscore"
        acao  : "winsorize" | "drop"
        """
        total = 0
        self.outlier_bounds: Dict = {}

        for col in colunas:
            if metodo == "iqr":
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR    = Q3 - Q1
                lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:
                mu, sigma = self.df[col].mean(), self.df[col].std()
                lb, ub    = mu - 3 * sigma, mu + 3 * sigma

            mask  = (self.df[col] < lb) | (self.df[col] > ub)
            total += int(mask.sum())
            self.outlier_bounds[col] = (lb, ub)

            if acao == "winsorize":
                self.df[col] = self.df[col].clip(lower=lb, upper=ub)
            else:
                self.df = self.df[~mask]

        self.log.registrar("Stage 4", f"Outliers tratados ({metodo}/{acao})", total)
        return self

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    def validar_regras(self, regras: List[Dict]) -> "DataQualityPipeline":
        """
        Valida regras de domínio.
        regra: {"nome": str, "condicao": str (query), "acao": "flag" | "drop"}
        """
        if "_flags" not in self.df.columns:
            self.df["_flags"] = ""

        for regra in regras:
            violacoes = self.df.query(regra["condicao"]).index
            n = len(violacoes)
            if regra["acao"] == "flag":
                self.df.loc[violacoes, "_flags"] += f"|{regra['nome']}"
            elif regra["acao"] == "drop":
                self.df.drop(index=violacoes, inplace=True)
            self.log.registrar("Stage 5", f"Regra '{regra['nome']}' → {n} violações", n)

        return self

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    def normalizar(self, colunas: List[str], metodo: str = "minmax") -> "DataQualityPipeline":
        """
        Normaliza colunas numéricas, criando novas colunas com sufixo.
        metodo: "minmax" → sufixo _norm | "zscore" → sufixo _std
        """
        scaler = MinMaxScaler() if metodo == "minmax" else StandardScaler()
        sufixo = "_norm" if metodo == "minmax" else "_std"
        novas  = [c + sufixo for c in colunas]
        self.df[novas] = scaler.fit_transform(self.df[colunas])
        self.log.registrar("Stage 6", f"Normalização {metodo} aplicada a {colunas}")
        return self

    # ── Stage 7 ───────────────────────────────────────────────────────────────
    def feature_engineering(self, col_tempo: str) -> "DataQualityPipeline":
        """Extrai features temporais e encoding cíclico de uma coluna datetime."""
        self.df[col_tempo]       = pd.to_datetime(self.df[col_tempo])
        self.df["hora"]          = self.df[col_tempo].dt.hour
        self.df["dia_semana"]    = self.df[col_tempo].dt.dayofweek
        self.df["mes"]           = self.df[col_tempo].dt.month
        self.df["semana_ano"]    = self.df[col_tempo].dt.isocalendar().week.astype(int)
        self.df["is_weekend"]    = (self.df["dia_semana"] >= 5).astype(int)
        self.df["hora_seno"]     = np.sin(2 * np.pi * self.df["hora"] / 24)
        self.df["hora_cosseno"]  = np.cos(2 * np.pi * self.df["hora"] / 24)
        self.log.registrar("Stage 7", "Features temporais e cíclicas criadas")
        return self

    # ── Métricas ──────────────────────────────────────────────────────────────
    def calcular_metricas(self) -> Dict:
        """Calcula e retorna métricas de qualidade do dataset final."""
        self.metricas = {
            "registros_raw"    : len(self.df_raw),
            "registros_finais" : len(self.df),
            "retencao_pct"     : round(len(self.df) / len(self.df_raw) * 100, 2),
            "completude_pct"   : round(
                (1 - self.df.isna().sum().sum() / self.df.size) * 100, 2
            ),
            "n_colunas"        : len(self.df.columns),
            "etapas_pipeline"  : len(self.log.etapas),
        }
        return self.metricas

    def imprimir_metricas(self):
        m = self.calcular_metricas()
        print("\n📊 Métricas de Qualidade:")
        print("-" * 40)
        for k, v in m.items():
            print(f"  {k:<22}: {v}")
        print("-" * 40)
