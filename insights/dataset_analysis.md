# Análise Completa do Dataset — SPR 2026

> **Competição:** Classificação de Laudos Mamográficos (BI-RADS 0–6)  
> **Métrica:** F1-Score Macro  
> **Melhor Score:** 0.82073 (BERTimbau v4 + Focal Loss + Threshold Tuning)  
> **Gerado a partir de:** `notebooks/08_term_analysis.ipynb`

---

## Parte I — Síntese Objetiva dos Dados

### 1. Visão Geral do Dataset

| Campo | Valor |
|-------|-------|
| Amostras de treino | 18.272 |
| Colunas | `ID`, `report`, `target` |
| Classes | 7 (BI-RADS 0 a 6) |
| Tipo de texto | Laudos mamográficos em português |
| Formato | Texto livre semi-estruturado |

### 2. Distribuição de Classes

| Classe | Nome | N | % | Proporção relativa |
|--------|------|---|---|-------------------|
| 0 | Incompleto | 610 | 3,34% | 21,0x menor que classe 2 |
| 1 | Negativo | 693 | 3,79% | 23,0x menor |
| 2 | **Benigno** | **15.968** | **87,39%** | **Dominante** |
| 3 | Provavelmente Benigno | 713 | 3,90% | 22,4x menor |
| 4 | Suspeito | 214 | 1,17% | 74,6x menor |
| 5 | Altamente Sugestivo | 29 | 0,16% | 550,6x menor |
| 6 | Malignidade Comprovada | 45 | 0,25% | 354,8x menor |

**Fator de desbalanceamento máximo:** 550,6x (classe 2 vs classe 5)

> A classe 2 domina com 87,4% dos exemplos. As classes 4, 5 e 6 juntas representam apenas 1,58% do dataset (288 amostras). A classe 5 possui apenas **29 amostras** — insuficiente para qualquer estratégia de data augmentation Text-level confiável.

### 3. Estatísticas Textuais por Classe

| Classe | Palavras (média) | Palavras (mediana) | Observação |
|--------|------------------|--------------------|------------|
| 0 — Incompleto | 57 | 55 | Textos médios — descreve nódulos não resolvidos |
| 1 — Negativo | 35 | 34 | **Mais curtos** — laudos de exclusão ("não se observam") |
| 2 — Benigno | 43 | 40 | Moderados — achados benignos simples |
| 3 — Provavelmente Benigno | 68 | 64 | Longos — descrevem achados + acompanhamento |
| 4 — Suspeito | 66 | 61 | Longos — detalhamento de calcificações/morfologia |
| 5 — Altamente Sugestivo | 68 | 61 | Longos — descrições detalhadas de lesão |
| 6 — Malignidade Comprovada | 80 | 75 | **Mais longos** — incluem resultado histopatológico |

> **Média global:** ~45 palavras | **Mediana global:** ~40 palavras.  
> Há correlação entre severidade e comprimento textual: classes mais graves (3–6) produzem laudos mais detalhados.

### 4. Menções Explícitas ao BI-RADS

| Métrica | Valor |
|---------|-------|
| Laudos com menção explícita ("bi-rads", "birads", "bi rads") | 8 / 18.272 |
| Percentual | **0,04%** |

> Conclusão: a classificação BI-RADS **nunca** aparece explicitamente nos laudos. Modelos precisam inferir a categoria inteiramente a partir da semântica descritiva.

---

## Parte II — Heterogeneidade Inter-Classes

### 5. Distância Cosseno entre Classes (TF-IDF)

**Média global de distância:** 0,3629

#### Pares mais próximos (difíceis de separar)

| Par | Distância Cosseno | Interpretação |
|-----|-------------------|---------------|
| **1 ↔ 2** (Negativo ↔ Benigno) | **0,0827** | Extremamente similares |
| **0 ↔ 3** (Incompleto ↔ Prov. Benigno) | **0,1182** | Muito similares |
| **3 ↔ 4** (Prov. Benigno ↔ Suspeito) | **0,1313** | Similares |
| 0 ↔ 2 (Incompleto ↔ Benigno) | 0,1585 | Moderado |
| 2 ↔ 3 (Benigno ↔ Prov. Benigno) | 0,1675 | Moderado |

#### Pares mais distantes (fáceis de separar)

| Par | Distância Cosseno |
|-----|-------------------|
| 1 ↔ 6 | 0,6632 |
| 1 ↔ 5 | 0,6076 |
| 2 ↔ 6 | 0,5851 |
| 2 ↔ 5 | 0,5289 |
| 1 ↔ 4 | 0,4819 |

> **Insight:** Classes com laudos benignos/negativos (1, 2) são facilmente separáveis de classes malignas (5, 6). O desafio concentra-se na "zona cinzenta" (0, 3, 4) e na confusão 1↔2.

### 6. Sobreposição de Vocabulário (Jaccard)

**Jaccard médio:** 0,5109

#### Maior sobreposição

| Par | Jaccard | % do vocabulário compartilhado |
|-----|---------|-------------------------------|
| 0 ↔ 3 | 0,770 | 77% |
| 0 ↔ 2 | 0,735 | 73,5% |
| 2 ↔ 3 | 0,733 | 73,3% |
| 0 ↔ 4 | 0,704 | 70,4% |
| 3 ↔ 4 | 0,684 | 68,4% |

#### Menor sobreposição

| Par | Jaccard |
|-----|---------|
| 1 ↔ 6 | 0,254 |
| 1 ↔ 5 | 0,267 |
| 1 ↔ 4 | 0,289 |

> **Paradoxo:** Mesmo com 77% de vocabulário compartilhado (0↔3), a diferença está na **frequência relativa** dos termos, não na sua presença/ausência. Isso explica por que métodos bag-of-words binários falham.

### 7. Silhouette Score (TF-IDF)

| Métrica | Valor |
|---------|-------|
| **Silhouette Global** | **-0,058** |

| Classe | Silhouette | Interpretação |
|--------|------------|---------------|
| 0 — Incompleto | -0,037 | Cluster difuso |
| 1 — Negativo | **+0,161** | **Melhor separação** |
| 2 — Benigno | -0,062 | Difuso (domina o espaço) |
| 3 — Prov. Benigno | -0,089 | **Pior cluster** |
| 4 — Suspeito | -0,064 | Difuso |
| 5 — Alt. Sugestivo | +0,114 | Boa separação |
| 6 — Mal. Comprovada | -0,036 | Difuso (N pequeno) |

> **Silhouette negativo global** indica que, no espaço TF-IDF, as classes não formam clusters bem definidos. Apenas as classes 1 (Negativo) e 5 (Altamente Sugestivo) têm coesão positiva — justamente as de vocabulário mais distinto.

### 8. Visualização t-SNE

O gráfico t-SNE (ver notebook, célula 17) mostra:
- **Classe 2 (Benigno):** nuvem massiva dominando todo o espaço
- **Classe 1 (Negativo):** cluster parcialmente identificável na periferia
- **Classes 0, 3, 4:** completamente imbricadas na nuvem da classe 2
- **Classes 5 e 6:** pontos dispersos sem cluster coeso (N insuficiente)

---

## Parte III — Termos Discriminativos

### 9. Chi² — Top Termos por Classe

Os valores Chi² medem a associação estatística entre um n-grama e uma classe.

#### BI-RADS 0 — Incompleto
| Termo | χ² |
|-------|----|
| cm as | 589,8 |
| contornos parcialmente | 325,0 |
| individualizados devido | 324,5 |
| parcialmente individualizados | 322,8 |
| individualizados | 321,7 |
| devido sobreposição | 312,1 |
| tecidual localizado | 294,7 |
| sobreposição tecidual | 283,4 |
| tecidual | 272,3 |
| devido | 267,2 |

> **Semântica:** Laudos incompletos descrevem limitações técnicas — "sobreposição tecidual", nódulos "parcialmente individualizados", necessidade de complementação.

#### BI-RADS 1 — Negativo
| Termo | χ² |
|-------|----|
| nódulos não | 1.125,0 |
| lipossubstituídas não | 319,3 |
| adiposas não | 246,8 |
| mamografia não | 73,4 |
| predominantemente adiposasnão | 73,0 |
| método não | 48,8 |
| calcificações benignas | 38,4 |
| benignas esparsas | 38,4 |

> **Semântica:** Dominado por **negação** — "não se observam nódulos", mamas "adiposas" ou "lipossubstituídas" sem achados. O Chi² mais alto de todo o dataset (1.125) reflete a assinatura quase formulaica desta classe.

#### BI-RADS 2 — Benigno
| Termo | χ² |
|-------|----|
| nódulos não | 348,6 |
| cm | 294,1 |
| da mama | 262,3 |
| de cm | 235,7 |
| medindo | 225,9 |
| nódulo | 209,2 |
| lipossubstituídas não | 207,0 |
| cerca | 203,4 |
| cerca de | 202,0 |
| no | 200,0 |

> **Semântica:** Achados benignos com **medições** ("medindo", "cm", "cerca de") — descreve nódulos conhecidos, calcificações benignas. Vocabulário muito semelhante ao da classe 1, mas com presença de achados descritos.

#### BI-RADS 3 — Provavelmente Benigno
| Termo | χ² |
|-------|----|
| calcificações puntiformes | 226,0 |
| puntiformes | 211,4 |
| puntiformes agrupadas | 184,6 |
| da mama | 163,0 |
| no | 157,2 |
| cm | 151,9 |
| quadrante | 142,7 |
| nódulo | 136,6 |
| mama | 131,2 |
| medindo | 127,9 |

> **Semântica:** Achados intermediários — "calcificações puntiformes agrupadas", localização por quadrante, termos de acompanhamento ("estável"). É a classe mais difusa lexicalmente.

#### BI-RADS 4 — Suspeito
| Termo | χ² |
|-------|----|
| extensão | 456,9 |
| extensão de | 437,4 |
| amorfas | 350,5 |
| calcificações amorfas | 327,6 |
| localizadas | 286,3 |
| com distribuição | 245,6 |
| distribuição | 241,4 |
| cm localizadas | 229,7 |
| amorfas agrupadas | 224,5 |
| localizadas no | 223,6 |

> **Semântica:** Calcificações **amorfas** com extensão e distribuição — termos que indicam suspeita sem malignidade confirmada. "Extensão de" é o discriminador principal (descreve a área afetada).

#### BI-RADS 5 — Altamente Sugestivo
| Termo | χ² |
|-------|----|
| nódulo espiculado | 575,7 |
| espiculado medindo | 502,0 |
| retração | 466,0 |
| espiculado | 458,8 |
| espiculada | 457,6 |
| massa | 363,3 |
| associado retração | 342,1 |
| espiculado no | 308,9 |
| retração espessamento | 270,6 |
| irregular | 204,5 |

> **Semântica:** Descritores morfológicos de **malignidade radiológica** — "espiculado", "retração", "massa", "irregular". Estes termos correspondem diretamente à terminologia BI-RADS ACR para lesões altamente suspeitas.

#### BI-RADS 6 — Malignidade Comprovada
| Termo | χ² |
|-------|----|
| carcinoma | 1.384,8 |
| invasivo | 1.161,9 |
| de carcinoma | 877,8 |
| carcinoma mamário | 874,9 |
| mamário invasivo | 780,9 |
| carcinoma ductal | 473,6 |
| grau | 304,2 |
| espiculado | 269,0 |
| invasivo não | 265,2 |
| irregular | 259,5 |

> **Semântica:** Presença de resultado histopatológico — "carcinoma mamário invasivo", "carcinoma ductal". É a classe mais fácil de detectar lexicalmente, pois inclui diagnóstico definitivo.

### 10. Log-Odds Ratio — Termos Enriquecidos e Ausentes

O log₂ Odds-Ratio mede o quanto um termo é **mais (ou menos) provável** em uma classe comparada ao restante.

#### Termos mais enriquecidos por classe (Top 3)

| Classe | Termo | log₂OR | Interpretação |
|--------|-------|--------|---------------|
| 0 | "manteve após" | +6,74 | Persistência após compressão |
| 0 | "quadrantes medindo" | +6,42 | Localização de achado |
| 0 | "cm as" | +6,15 | Medidas seguidas de limitações |
| 1 | "adiposas não" | +6,24 | Mamas adiposas sem achados |
| 1 | "adiposasnão se" | +6,06 | Variante sem espaço |
| 1 | "método não" | +5,89 | Exclusão por método |
| 2 | "adiposas calcificações" | +4,62 | Mamas adiposas com calc. benignas |
| 2 | "benignos" | +3,19 | Achados benignos |
| 2 | "benignos pelo" | +2,95 | Conclusão benigna |
| 3 | "às horas" | +7,31 | Localização horária |
| 3 | "em meses" | +7,31 | Intervalo de acompanhamento |
| 3 | "com cm" | +7,16 | Medição de achado |
| 4 | "doppler medindo" | +8,26 | Uso de doppler (suspeita) |
| 4 | "heterogêneas com" | +8,26 | Calcificações heterogêneas |
| 4 | "pleomórficas de" | +8,00 | Morfologia suspeita |
| 5 | "espiculada" | +9,87 | **Maior enriquecimento** |
| 5 | "massa" | +8,87 | Lesão sólida grande |
| 5 | "espiculado medindo" | +8,87 | Nódulo espiculado medido |
| 6 | "carcinoma mamário" | +11,23 | **Maior log-OR do dataset** |
| 6 | "mamário invasivo" | +10,14 | Diagnóstico histológico |
| 6 | "invasivo" | +9,84 | Tipo tumoral |

#### Termos mais ausentes por classe (Top 3)

| Classe | Termo | log₂OR | Significado |
|--------|-------|--------|-------------|
| 1 | "benignas" | -6,86 | Negativo ≠ possui achados benignos |
| 1 | "calcificações benignas" | -6,85 | — |
| 2 | "espiculado" | -8,03 | Benigno ≠ possui morfologia maligna |
| 2 | "invasivo" | -7,34 | — |
| 5 | "observam alterações" | -5,40 | Alt. Sugestivo ≠ normal |
| 6 | "suspeitas as" | -4,33 | Comprovado ≠ mera suspeita |

### 11. Termos Exclusivos por Classe (no Top-200 TF-IDF)

Termos que aparecem no top-200 TF-IDF de uma classe mas **não** no top-200 de nenhuma outra:

| Classe | Termos Exclusivos | Destaque |
|--------|-------------------|----------|
| 0 — Incompleto | "contornos parcialmente", "individualizados", "sobreposição tecidual", "parcialmente individualizados" | Limitações técnicas |
| 1 — Negativo | "adiposas não", "apenas", "após estudo", "comparativa primeira" | Exclusão e normalidade |
| 2 — Benigno | "agrupadas linfonodo", "bilaterais", "cirúrgica", "decorrente de", "arquitetura habitual" | Achados benignos conhecidos |
| 3 — Prov. Benigno | "calcificações puntiformes", "estável em", "controle de", "desde data", "puntiformes" | Acompanhamento e estabilidade |
| 4 — Suspeito | "amorfas", "calcificações amorfas", "com distribuição", "extensão de", "direcionada" | Morfologia suspeita de calcificações |
| 5 — Alt. Sugestivo | "associado retração", "cutânea", "cutâneo", "de permeio", "espiculada" | Sinais de malignidade radiológica |
| 6 — Mal. Comprovada | "biópsia", "carcinoma", "core biopsy", "com resultado", "com diagnóstico" | Histopatologia confirmada |

---

## Parte IV — Padrões BI-RADS (Regex sobre Glossário)

### 12. Frequência de Padrões por Classe (%)

Padrões extraídos via regex baseados no [Glossário BI-RADS](../data/glossary/birads_glossary.md).

| Padrão | C0 | C1 | C2 | C3 | C4 | C5 | C6 | Padrão discriminativo? |
|--------|----|----|----|----|----|----|----|-----------------------|
| **Negação** | 96% | **100%** | 100% | 98% | 97% | 97% | 93% | ❌ Ubíquo |
| **Benigno** | 94% | 1% | **99%** | 94% | 92% | 86% | 87% | ⚠️ Fraco — presente em quase tudo |
| Nódulo/massa | 70% | 37% | 32% | 66% | 62% | **97%** | 80% | ✅ Classe 5 |
| Calcificação | 7% | 11% | 9% | 11% | 7% | 3% | 4% | ❌ Não discrimina |
| Assimetria | **27%** | 3% | 5% | **32%** | 10% | 7% | 24% | ⚠️ Classes 0, 3 |
| Distorção arq. | 3% | 2% | 1% | 4% | 4% | **10%** | **31%** | ✅ Classes 5, 6 |
| **Morf. benigna** | 28% | 0% | 6% | **47%** | 24% | 17% | 18% | ✅ Classe 3 |
| **Morf. suspeita** | 4% | 0% | 0% | 1% | **22%** | **48%** | **51%** | ✅✅ Classes 4, 5, 6 |
| **Morf. maligna** | 0% | 0% | 0% | 0% | 7% | **55%** | **33%** | ✅✅✅ Altamente discriminativo |
| Calc. benigna | 3% | 0% | 2% | 4% | **12%** | 0% | 0% | ⚠️ Fraco |
| **Calc. suspeita** | 8% | 11% | 9% | 13% | **44%** | 24% | **31%** | ✅ Classe 4 |
| Retração | 0% | 0% | 0% | 0% | 1% | **35%** | 11% | ✅✅ Classe 5 |
| Espessamento | 0% | 0% | 0% | 0% | 2% | **35%** | 4% | ✅✅ Classe 5 |
| **Adenopatia** | 2% | 3% | 3% | 3% | 9% | **31%** | 24% | ✅ Classes 5, 6 |
| **Biópsia** | 1% | 0% | 1% | 2% | 4% | 3% | **73%** | ✅✅✅ Altamente discriminativo |
| Acompanhamento | 1% | 1% | 2% | **20%** | 5% | 0% | 4% | ✅ Classe 3 |
| Complementar | 2% | 1% | 1% | **11%** | 6% | 10% | 16% | ⚠️ |
| Densidade | 70% | **89%** | 80% | 78% | 87% | 90% | 60% | ❌ Ubíquo |
| Pós-cirúrgico | 1% | 0% | 2% | 2% | 1% | 3% | 2% | ❌ Raro |
| **Maligno** | 2% | 3% | 3% | 3% | 3% | 7% | **76%** | ✅✅✅ Classe 6 |

### 13. Síntese dos Padrões Discriminativos

| Classe | Padrões-chave | Confiança |
|--------|---------------|-----------|
| 0 | Assimetria + ausência de achados resolvidos | Baixa |
| 1 | Negação dominante + sem achados benignos | Média |
| 2 | Benigno + medições + achados conhecidos | Baixa (base rate) |
| 3 | Morf. benigna + acompanhamento + puntiformes | Média |
| 4 | Calc. suspeita + morf. suspeita + extensão | Alta |
| 5 | **Morf. maligna + retração + espessamento** | **Alta** |
| 6 | **Biópsia + maligno + carcinoma** | **Muito alta** |

---

## Parte V — Confusão Vocabular

### 14. Análise dos 3 Pares Mais Confusos

#### Par 1: BI-RADS 1 ↔ 2 (distância = 0,0827)

O par **mais difícil** de todo o dataset.

**Vocabulário compartilhado com pesos quase idênticos:**
| Termo | Peso em C1 | Peso em C2 |
|-------|------------|------------|
| "não" | 0,200 | 0,165 |
| "análise comparativa" | 0,116 | 0,085 |
| "se observam" | 0,088 | 0,084 |
| "não se" | 0,088 | 0,084 |

**O que separa C1 do C2:**
- C1 tem mais: "nódulos não" (0,100 vs 0,002) — negação explícita de achados
- C2 tem mais: "calcificações benignas" (0,059 vs 0,000) — presença de achados benignos

> **Insight:** A diferença crucial é: C1 = "nada foi encontrado" vs C2 = "algo foi encontrado, mas é benigno". Ambos usam fórmulas de negação, mas C2 inclui descrição de achados.

#### Par 2: BI-RADS 0 ↔ 3 (distância = 0,1182)

**Vocabulário compartilhado:**
| Termo | Peso em C0 | Peso em C3 |
|-------|------------|------------|
| "mama" | 0,087 | 0,084 |
| "da mama" | 0,080 | 0,071 |
| "nódulo" | 0,058 | 0,056 |
| "cm" | 0,065 | 0,056 |

**O que separa:**
- C0 tem mais: "cm as" (sobreposição tecidual) — limitação técnica
- C3 tem mais: "estável" (0,030 vs 0,001), "puntiformes" (0,025 vs 0,003) — seguimento temporal

> **Insight:** Ambos descrevem achados de incerteza, mas C0 indica necessidade de refazer exame (técnica) enquanto C3 indica necessidade de acompanhar achado (clínica).

#### Par 3: BI-RADS 3 ↔ 4 (distância = 0,1313)

**Vocabulário compartilhado:**
| Termo | Peso em C3 | Peso em C4 |
|-------|------------|------------|
| "mama" | 0,084 | 0,088 |
| "da mama" | 0,071 | 0,076 |
| "cm" | 0,056 | 0,079 |
| "mama esquerda" | 0,055 | 0,054 |

**O que separa:**
- C3 tem mais: "assimetria focal" (0,033 vs 0,008), "estável" (0,030 vs 0,001)
- C4 tem mais: "extensão de" (0,043 vs 0,007), "amorfas" (0,033 vs 0,003)

> **Insight:** A diferença é na **morfologia das calcificações** — puntiformes (benigna) vs amorfas/pleomórficas (suspeita) — e na presença de "extensão" indicando área afetada.

---

## Parte VI — Exemplos Representativos

### 15. Laudos na Mediana (por comprimento)

#### BI-RADS 0 — Incompleto (55 palavras)
> *Indicação clínica: rastreamento. Achados: Mamas densas pelo predomínio do tecido fibroglandular o que diminui a sensibilidade da mamografia. Assimetrias focais de aspecto nodular localizadas na junção dos quadrantes laterais bilateralmente. Calcificações benignas esparsas. Não se observam calcificações suspeitas agrupadas. As regiões axilares não apresentam alterações significativas. Análise comparativa: Mamografias anteriores não disponíveis para análise comparativa.*

#### BI-RADS 1 — Negativo (34 palavras)
> *Indicação clínica: rastreamento. Achados: Mamas heterogeneamente densas, o que pode ocultar pequenos nódulos. Não se observam calcificações suspeitas. As regiões axilares não apresentam alterações significativas. Análise comparativa: Mamografias anteriores não disponíveis para análise comparativa.*

#### BI-RADS 2 — Benigno (40 palavras)
> *Indicação clínica: rastreamento. Achados: Mamas parcialmente lipossubstituídas. Calcificações benignas esparsas. Não se observam microcalcificações pleomórficas agrupadas. Linfonodo intramamário à direita. As regiões axilares não apresentam alterações significativas. Análise comparativa: Em relação ao exame de \<DATA\>, não se observam alterações significativas.*

#### BI-RADS 3 — Provavelmente Benigno (64 palavras)
> *Indicação clínica: rastreamento. Achados: Mamas parcialmente lipossubstituídas, com tecido fibroglandular remanescente denso e heterogêneo. Assimetria focal localizada na união dos quadrantes laterais da mama direita que persiste após estudo com compressão seletiva, estável desde \<DATA\>. Calcificações benignas esparsas. Não se observam calcificações suspeitas agrupadas. As regiões axilares não apresentam alterações significativas. Análise comparativa: Em relação ao exame de \<DATA\>, não se observam alterações significativas.*

#### BI-RADS 4 — Suspeito (61 palavras)
> *Indicação clínica: rastreamento. Achados: Mamas heterogeneamente densas, o que pode ocultar pequenos nódulos. Calcificações amorfas agrupadas na região periareolar/junção dos quadrantes laterais da mama direita e calcificações pleomórficas segmentares na junção dos quadrantes superiores da mama esquerda. Mama acessória no prolongamento axilar. Calcificações benignas esparsas. As regiões axilares não apresentam alterações significativas. Análise comparativa: Mamografias anteriores não disponíveis para análise comparativa.*

#### BI-RADS 5 — Altamente Sugestivo (61 palavras)
> *Indicação clínica: investigação de câncer de mama. Achados: Mamas densas pelo predomínio do tecido fibroglandular o que diminui a sensibilidade da mamografia. Calcificações benignas esparsas bilaterais. Lesão irregular, espiculada, associada a distorção arquitetural e retração cutânea adjacente, medindo cerca de 2,5 cm, localizada no quadrante superolateral profundo da mama esquerda. As regiões axilares não apresentam alterações significativas. Análise comparativa: Primeira mamografia.*

#### BI-RADS 6 — Malignidade Comprovada (75 palavras)
> *Indicação clínica: reavaliação de alteração observada à mamografia de \<DATA\>. Realizadas incidências craniocaudal e perfil, magnificadas e com compressão localizada na mama esquerda. Achados: Calcificações pleomórficas, agrupadas localizadas no quadrante superolateral da mama esquerda, sendo identificados dois grupamentos, um distando 5,0 cm do complexo areolopapilar e o outro, 3,0 cm. Esses dois grupamentos foram submetidos a biópsia por mamotomia no dia \<DATA\>, com resultado de carcinoma ductal in situ (CDIS) de padrão plano ("clinging carcinoma").*

---

## Parte VII — Mapa de Dificuldade

### 16. Classificação de Dificuldade por Classe

| Classe | Dificuldade | Justificativa |
|--------|------------|---------------|
| 6 — Mal. Comprovada | 🟢 Fácil | Vocabulário histopatológico exclusivo ("carcinoma", "biópsia"). Log₂OR > 11. |
| 5 — Alt. Sugestivo | 🟡 Média | Descritores morfológicos fortes ("espiculado", "retração"). Mas N=29 é crítico. |
| 1 — Negativo | 🟡 Média | Padrão formulaico de negação, mas confunde com classe 2. |
| 4 — Suspeito | 🟠 Difícil | Requer distinção entre morfologias de calcificação (amorfas vs puntiformes). |
| 3 — Prov. Benigno | 🔴 Muito difícil | Pior silhouette (-0,089). Vocabulário 77% idêntico à classe 0. Depende de nuances sutis. |
| 0 — Incompleto | 🔴 Muito difícil | Conceito é "técnico" (precisa refazer) vs "clínico". Vocabulário misturado. |
| 2 — Benigno | ⚫ Especial | Domina o dataset (87%). Modelo enviesado em direção a esta classe. F1 alto aqui mascarando erros nas outras. |

### 17. Hipótese sobre Erros do Modelo Atual

Com base na análise, os erros do BERTimbau v4 (F1=0.82) devem concentrar-se em:

1. **C1 → C2 (Negativo → Benigno):** Distância 0,08. Ambos usam "não se observam". A presença de "calcificações benignas esparsas" no C2 é o único diferenciador, e esse fragmento também aparece como boilerplate em outras classes.

2. **C0 ↔ C3 (Incompleto ↔ Provavelmente Benigno):** Distância 0,12. Ambos descrevem achados de incerteza com vocabulário 77% igual. A pista está em "sobreposição tecidual" (C0) vs "estável desde" (C3).

3. **C3 → C4 (Provavelmente Benigno → Suspeito):** Distância 0,13. A fronteira entre calcificações "puntiformes" (benignas) e "amorfas/pleomórficas" (suspeitas) é uma única palavra.

4. **C5 sobreposta a C4:** Ambas descrevem achados suspeitos, mas C5 adiciona "retração", "espessamento", "massa espiculada".

---

## Parte VIII — Implicações para Modelagem

### 18. Por Que TF-IDF Funciona (até certo ponto)

- Termos exclusivos existem e são altamente discriminativos para C5, C6
- Classes extremas (1, 6) têm assinaturas lexicais claras
- **Teto:** ~0.78 (LinearSVC). Falha nas classes de fronteira (0, 3, 4)

### 19. Por Que Transformers São Superiores

- Capturam **contexto posicional** — "nódulo espiculado" ≠ "nódulo ovalado"
- Lidam melhor com negação — "não se observam nódulos" ≠ "nódulo observado"
- Embeddings contextuais diferenciam mesmas palavras em contextos diferentes
- BERTimbau pré-treinado em português captura nuances da redação médica

### 20. Limitações Identificadas

1. **N insuficiente para C5 (29 amostras):** Qualquer modelo vai sub-representar esta classe
2. **Ruído de tokenização:** Tokens concatenados como "adiposasnão" indicam problemas na extração/OCR original
3. **Boilerplate extenso:** "Indicação clínica: rastreamento", "As regiões axilares não apresentam alterações significativas" — presente em **quase todas as classes**, diluindo sinal
4. **Ausência de BI-RADS explícito:** O modelo deve aprender a "interpretar" em vez de "ler" a classificação

---

## Apêndices

### A. Referência de Padrões Regex Utilizados

Padrões baseados no [Glossário BI-RADS](../data/glossary/birads_glossary.md) com variantes coloquiais em PT-BR.

- **morf_benigna:** oval, ovalad, circunscrit, regular, macrolobulad, microlobulad, homogêne
- **morf_suspeita:** irregular, microlobulad, heterogêne, limites imprecis
- **morf_maligna:** espiculad, estelad, retração, infiltrat
- **calc_benigna:** grosseir, vascula, alarg, grande, amorfa (excluída)
- **calc_suspeita:** amorfa, pleomórf, heterogênea, fina, linear, ramificad

### B. Localização dos Artefatos

| Artefato | Caminho |
|----------|---------|
| Notebook de análise | `notebooks/08_term_analysis.ipynb` |
| Glossário BI-RADS | `data/glossary/birads_glossary.md` |
| Dados exportados | `data/analysis_export.txt` |
| Gráfico t-SNE | Notebook célula 17 |
| Heatmap TF-IDF | Notebook célula 9 |
| Heatmap Padrões BI-RADS | Notebook célula 20 |
| Matriz Cosseno | Notebook célula 13 |
| Matriz Jaccard | Notebook célula 14 |
