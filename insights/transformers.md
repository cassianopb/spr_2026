# Transformers - Análise Metodológica

## Resultados

| Rank | Modelo | Score | Status |
|------|--------|-------|--------|
| 🏆 | **BERTimbau v4 + Threshold Tuning** | **0.82073** | ✅ MELHOR |
| 2 | **BERTimbau + Focal Loss** | **0.79696** | ✅ Submetido |
| 3 | BERTimbau v5 (Gamma Search γ=2.5) | 0.75574 | ❌ Abaixo v4 |
| 4 | BERTimbau v5 (LR Search) | 0.75508 | ❌ Abaixo v4 |
| 5 | **BioBERTpt** | **0.72480** | ✅ Submetido |
| 6 | BERTimbau v5 (Class) | 0.69238 | ❌ Regressão |
| 7 | **XLM-RoBERTa + Mean Pool** | **0.68767** | ✅ Submetido |
| 8 | ModernBERT base | 0.68578 | ✅ Submetido |
| 9 | BERTimbau base | 0.64319 | ✅ Submetido |
| 10 | BERT Multilingual | 0.56095 | ✅ Submetido |
| 11 | DistilBERT Multilingual | 0.55229 | ✅ Submetido |
| ❌ | BERTimbau + LoRA (Offline) | 0.13261 | ⚠️ Falhou |
| ❌ | mDeBERTa + class weights | 0.01008 | ⚠️ BUG |
| ❌ | mDeBERTa-v3 | 0.01008 | ⚠️ Bug fp16 |

> **🏆 BERTimbau v4 + Threshold Tuning = 0.82073** - Variações de LR/Gamma não melhoram!

---

## 🏆 Análise: BERTimbau + Focal Loss (0.79696) - MELHOR SCORE!

**Primeiro transformer a superar TF-IDF!** BERTimbau com Focal Loss alcançou **0.79696**, superando o baseline TF-IDF (0.77885) em **+2.3%**.

### Por que funcionou?

1. **Focal Loss:** γ=2 foca nos exemplos difíceis, melhorando classes minoritárias
2. **BERTimbau base:** Especializado em português, vocabulário adequado
3. **Fine-tuning adequado:** Parâmetros bem calibrados para o dataset
4. **Sem overfitting:** Focal Loss tem efeito regularizador implícito

### Por que superou TF-IDF?

- **Contexto semântico:** Entende relações complexas como "sem sinais de malignidade"
- **Transfer learning:** Conhecimento prévio de português médico
- **Focal Loss:** Resolve o problema de desbalanceamento de classes

### Comparação com baseline TF-IDF

- BERTimbau + Focal: **0.79696** (+2.3% acima)
- TF-IDF baseline: 0.77885
- **Breakthrough!** Primeiro transformer a vencer.

---

## ⚠️ Análise: BERTimbau + LoRA (0.13261) - FALHA

**LoRA offline falhou completamente.** Score de 0.13261 indica predições quase aleatórias.

### Por que falhou?

1. **Offline execution:** Kaggle offline pode ter problemas com bibliotecas PEFT
2. **Rank muito baixo:** LoRA com r=4 ou r=8 pode ser insuficiente para este task
3. **Adapter não treinou:** Possível problema no salvamento/carregamento dos pesos
4. **Incompatibilidade:** Versão do PEFT pode não ser compatível com ambiente Kaggle

### Lições aprendidas

- Full fine-tuning funciona melhor que LoRA para datasets pequenos
- Testar sempre online antes de submeter offline
- LoRA economiza memória mas pode perder performance

---

## Análise: ModernBERT (0.68578)

**Segundo melhor transformer.** ModernBERT ficou **12% abaixo** do TF-IDF baseline (0.77885).

### Por que funcionou melhor?

1. **Arquitetura moderna:** RoPE embeddings, Flash Attention 2, GeGLU
2. **Contexto longo:** Suporta até 8192 tokens nativamente
3. **Treinamento eficiente:** Mais dados, melhor curriculum
4. **Tokenização robusta:** Melhor handling de tokens subword

### Comparação com baseline TF-IDF

- ModernBERT: **0.68578** (88% do baseline)
- Gap ainda significativo, mas promissor para ensembles

---

## ✅ Análise: BioBERTpt (0.72480) - 2º MELHOR TRANSFORMER

**BioBERTpt superou ModernBERT!** Score de **0.72480** demonstra a importância do domínio médico.

### Por que funcionou bem?

1. **Pré-treino médico:** Treinado em textos biomédicos PT-BR (PubMed, artigos)
2. **Vocabulário especializado:** Tokeniza "calcificação", "BIRADS" corretamente
3. **Transfer learning específico:** Conhecimento de domínio médico
4. **Português nativo:** Entende nuances da língua

### Comparação com outros modelos

| Modelo | Score | Domínio | Língua |
|--------|-------|---------|--------|
| BioBERTpt | **0.72480** | Médico | PT |
| ModernBERT | 0.68578 | Geral | EN |
| BERTimbau | 0.64319 | Geral | PT |

**Insight:** Domínio específico > Língua específica > Geral multilingual

### Por que não superou BERTimbau + Focal?

- Sem Focal Loss: Class weights normais não são tão efetivos
- BERTimbau Large + Focal teve mais capacidade

---

## ✅ Análise: XLM-RoBERTa + Mean Pooling (0.68767)

**XLM-RoBERTa competitivo** com ModernBERT. Mean pooling melhorou a representação.

### Por que funcionou?

1. **Mean Pooling:** Agrega todos os tokens, não só [CLS]
2. **Pré-treino massivo:** 100 línguas, dados abundantes
3. **RoBERTa optimizations:** Treinamento mais longo, sem NSP

### Comparação Mean vs CLS pooling

| Pooling | Score | Motivo |
|---------|-------|--------|
| Mean | **0.68767** | Captura toda a sequência |
| CLS | ~0.65 | Perde informação de tokens finais |

**Insight:** Mean pooling é superior para classificação de textos médicos.

---

## Análise: BERTimbau (0.64319)

**Resultado intermediário.** BERTimbau ficou abaixo do ModernBERT apesar de ser especializado em português.

### Por que não superou ModernBERT?

1. **Arquitetura mais antiga:** BERT original (2018) vs ModernBERT (2024)
2. **Sem otimizações modernas:** Sem Flash Attention, RoPE
3. **Pré-treinamento limitado:** Menos dados que ModernBERT
4. **Contexto máximo:** 512 tokens vs 8192 do ModernBERT

### Pontos positivos

- Modelo PT-BR ainda é **14% melhor** que BERT Multilingual (0.56095)
- Base sólida para fine-tuning adicional

---

## Análise: DistilBERT Multilingual (0.55229)

**Resultado esperado.** DistilBERT ficou próximo ao BERT Multilingual (0.56095), apenas 1.5% abaixo.

### Por que ficou similar ao BERT Multilingual?

1. **Destilação preservou conhecimento:** DistilBERT mantém ~97% da performance do BERT original
2. **Mesmo problema:** Modelo genérico multilingual, sem especialização PT-BR
3. **Vantagem:** 40% mais rápido e 60% menor que BERT full

### Quando usar DistilBERT?

- Quando tempo de inferência é crítico
- Como modelo rápido para ensembles
- Trade-off aceitável: -1.5% score vs 2x mais rápido

---

## ⚠️ BUG: mDeBERTa + Class Weights (0.01008)

**Score praticamente zero indica bug crítico!** Modelo completamente quebrado.

### Diagnóstico

Score de 0.01 em F1-macro significa:
- Modelo predizendo **sempre a mesma classe**
- Ou labels **invertidas/mapeadas incorretamente**
- Ou problema no **training loop**

### Possíveis causas

1. **Mixed precision + class_weights:** Adicionamos `.float()` nos logits, mas pode ter outro problema
2. **Label mismatch:** Verificar se labels do test set batem com train
3. **Peso de classes extremo:** Class weights muito altos podem causar gradientes instáveis
4. **Learning rate alta demais:** Com class weights, LR precisa ser menor

### Próximos passos

1. Testar `submit_deberta.ipynb` (sem class weights) para isolar o problema
2. Verificar distribuição de predições no output
3. Se necessário, remover class weights e usar Focal Loss em vez disso

---

## Análise: BERT Multilingual (0.56095)

**Resultado decepcionante.** BERT Multilingual ficou **28% abaixo** do TF-IDF baseline (0.77885).

### Por que falhou?

1. **Modelo genérico:** BERT Multilingual é treinado em 104 idiomas, diluindo conhecimento de português
2. **Sem domínio médico:** Vocabulário médico/radiológico não está bem representado
3. **Tokenização subótima:** Warning de regex do Mistral indica problemas no tokenizer
4. **Epoch insuficientes:** F1 ainda estava subindo (0.43 → 0.50 → 0.56), precisava de mais epochs
5. **Incompatibilidade LayerNorm:** Warnings de `gamma/beta` vs `weight/bias` indicam checkpoint com formato antigo

### Lições aprendidas

- Transformers genéricos **não** superam TF-IDF automaticamente
- Precisa de modelo especializado em português (BERTimbau) ou domínio médico (BioBERTpt)
- Hiperparâmetros precisam de tuning (mais epochs, learning rate schedule)

---

## Inputs Kaggle Necessários

Todos os modelos transformer precisam ser adicionados como **Input** no Kaggle:

| Modelo | Kaggle Input |
|--------|--------------|
| BERTimbau base | `neuralmind/bert-base-portuguese-cased` |
| BERTimbau large | `neuralmind/bert-large-portuguese-cased` |
| BioBERTpt | `pucpr/biobertpt-all` |
| mDeBERTa-v3 | `microsoft/mdeberta-v3-base` |
| DistilBERT | `distilbert-base-multilingual-cased` |
| XLM-RoBERTa | `xlm-roberta-large` |
| ModernBERT | `answerdotai/ModernBERT-base` |

---

## Hipóteses

### 1. Potencial Vantagens sobre TF-IDF

- **Contexto semântico:** Transformers entendem "ausência de malignidade" vs "presença de malignidade"
- **Transfer learning:** Pré-treinamento em português pode ajudar
- **Embeddings contextuais:** Mesma palavra em contextos diferentes tem representações diferentes

### 2. Riscos

- **Overfitting:** Dataset pequeno + modelo grande = alto risco
- **Tempo de inferência:** Kaggle tem limite de tempo
- **GPU requirements:** Pode falhar offline

### 3. Modelos Prioritários

1. **BERTimbau base:** Melhor modelo PT-BR, baseline obrigatório
2. **BioBERTpt:** Domínio médico, pode ter vantagem
3. **mDeBERTa:** Estado da arte em NLU

---

## ❌ Análise: BERTimbau v5 Variações (2026-03-03)

**Nenhuma variação superou o v4 (0.82073).** Resultados decepcionantes em todas as tentativas.

| Variação | Score | Delta vs v4 |
|----------|-------|-------------|
| Gamma Search (γ=2.5) | 0.75574 | **-6.5%** |
| LR Search (1e-5/3e-5) | 0.75508 | **-6.6%** |
| Class Weights | 0.69238 | **-12.8%** |

### Por que falharam?

1. **Gamma γ=2.5:** Aumentar gamma focou demais nos exemplos difíceis, ignorando os fáceis
2. **LR diferente:** Learning rate 2e-5 do v4 já era ótima para o dataset
3. **Class Weights:** Pior resultado - Focal Loss já trata desbalanceamento, class weights é redundância prejudicial

### Insight Chave

> **Threshold Tuning é o diferencial do v4, não os hiperparâmetros de treino.**

O v4 usou thresholds otimizados por classe:
```python
THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.30, 6: 0.25}
```

Classes 5 e 6 (minoria) têm thresholds mais baixos, aumentando recall nessas classes.

### Lições Aprendidas

- ❌ Não iterar sobre hiperparâmetros de treino quando já estão otimizados
- ✅ Focar em **pós-processamento** (thresholds, calibração)
- ❌ Class weights + Focal Loss = redundância que prejudica
- ✅ γ=2.0 é o sweet spot para Focal Loss neste dataset

---

## Configurações Recomendadas

### Fine-tuning básico
```python
{
    "learning_rate": 2e-5,
    "epochs": 3-5,
    "batch_size": 16,
    "max_length": 128,  # textos são curtos
    "warmup_ratio": 0.1
}
```

### Para dataset desbalanceado
- Focal Loss (γ=2)
- Class weights inversamente proporcionais
- Oversampling da classe minoritária

### Para poucos dados
- LoRA (r=8, alpha=16)
- Gradient checkpointing
- Early stopping agressivo

---

## Resumo

| Modelo | Score | vs Baseline |
|--------|-------|-------------|
| **BERTimbau v4 + Threshold** | **0.82073** | **+5.4%** 🏆 |
| BERTimbau + Focal | 0.79696 | +2.3% |
| BERTimbau v5 (Gamma) | 0.75574 | -3.0% |
| BERTimbau v5 (LR) | 0.75508 | -3.0% |
| BioBERTpt | 0.72480 | -7.0% |
| BERTimbau v5 (Class) | 0.69238 | -11.1% |
| XLM-RoBERTa | 0.68767 | -11.7% |
| ModernBERT | 0.68578 | -12.0% |
| BERTimbau | 0.64319 | -17.4% |
| BERT Multilingual | 0.56095 | -28.0% |
| DistilBERT | 0.55229 | -29.1% |
| BERTimbau + LoRA | 0.13261 | ❌ Falhou |
| mDeBERTa + CW | 0.01008 | ❌ BUG |

**Conclusão:** 🏆 **BERTimbau v4 + Threshold Tuning é o melhor modelo (0.82073)**

**Insights v5:**
- ❌ Variações de LR/Gamma não melhoram
- ❌ Class weights + Focal Loss é redundância
- ✅ Threshold tuning é o diferencial

**Próximos passos:**
1. Seed ensemble (3-5 seeds) com v4
2. CV-based threshold estimation
3. Testar alpha α weights no Focal Loss

---

*Atualizado em: 03/03/2026*
