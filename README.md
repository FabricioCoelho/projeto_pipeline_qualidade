# Pipeline de Qualidade de Dados 

Pipeline modular orientado a objetos para pré-processamento e validação de qualidade de dados de sensores industriais. O projeto simula o fluxo real de dados sujos vindos de dispositivos IoT em subestações elétricas, aplicando um conjunto de transformações encadeadas até produzir um dataset limpo.

---

## 📌 Contexto
Sensores instalados em subestações elétricas geram leituras contínuas de tensão, corrente, potência e temperatura. Na prática, esses dados chegam com uma série de problemas: registros duplicados por falha de transmissão, leituras ausentes por queda de sinal, valores fisicamente impossíveis por falha de sensor. Antes de qualquer análise ou modelagem, os dados precisam passar por um processo rigoroso de validação e limpeza.

Este projeto constrói esse processo do zero, com arquitetura reutilizável e log completo de cada transformação aplicada.
