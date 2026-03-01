# Validação: BioBERTpt + Focal Loss

**Modelo:** BioBERTpt - Modelo biomédico PT-BR (`pucpr/biobertpt-all`)

---

## 📊 Histórico
- Score **0.72480** com CrossEntropy (2º melhor transformer)
- NUNCA testado com Focal Loss (potencial maior)

## 🎯 Objetivo
Testar BioBERTpt com Focal Loss e outras otimizações.

## 📊 Configurações Testadas
- **Loss:** CrossEntropy vs Focal Loss (γ=1,2,3)
- **LR:** 1e-5, 2e-5, 3e-5
- **Threshold Tuning:** Por classe

---

## Ambiente
- **Device:** cuda
- **Model:** `/kaggle/input/models/fabianofilho/biobertpt/pytorch/default/1/biobertpt`

## Dataset
- **Total:** 18,272 amostras
- **Train:** 14,617 | **Val:** 3,655

### Distribuição de Classes
| Target | Count |
|--------|-------|
| 0 | 610 |
| 1 | 693 |
| 2 | 15,968 |
| 3 | 713 |
| 4 | 214 |
| 5 | 29 |
| 6 | 45 |

---

## Experimentos

### 1. CrossEntropy (Baseline)

**Config:** `lr=2e-5, batch_size=8, max_length=256, epochs=5`

#### Training History
| Epoch | Training Loss | Validation Loss | F1 Macro |
|-------|---------------|-----------------|----------|
| 1 | 1.090796 | 0.428546 | 0.395102 |
| 2 | 0.343828 | 0.319278 | 0.529922 |
| 3 | 0.261335 | 0.316879 | 0.698019 |
| 4 | 0.196065 | 0.357841 | 0.707935 |
| 5 | 0.150405 | 0.389521 | 0.708971 |

**Resultado:** F1-Macro = **0.70897**  
**Tempo:** ~43:50 min

---

## Status
- [x] Exp 1: CrossEntropy baseline
- [ ] Exp 2: Focal Loss γ=2.0
- [ ] Exp 3: Focal Loss γ=1.0
- [ ] Exp 4: Focal Loss γ=3.0
- [ ] Threshold Tuning nos melhores

---

*Aguardando mais resultados...*
