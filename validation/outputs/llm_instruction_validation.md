# Validação: LLMs Instruction-Tuned (Phi, Mistral, BioGPT)

**Objetivo:** Comparar LLMs instruction-tuned para classificação BI-RADS.

---

## 📊 Modelos Planejados
| Modelo | Params | Descrição |
|--------|--------|-----------|
| Phi-3.5 | 3.8B | Microsoft, eficiente |
| Mistral | 7B | Mistral AI, forte em tarefas |
| BioGPT | ~1.5B | Microsoft, treinado em PubMed |

## 🎯 Setup
- **Device:** cuda
- **Dataset:** 98 train, 42 val (amostra estratificada)

---

## ❌ RESULTADO: FALHA - MODELOS NÃO DISPONÍVEIS

### Phi-3.5 (3.8B)
```
Model: microsoft/Phi-3.5-mini-instruct
Erro: We couldn't connect to 'https://huggingface.co' to load the files, 
and couldn't find them in the cached files.
```

### Mistral (7B)
```
Model: mistralai/Mistral-7B-Instruct-v0.3
Erro: We couldn't connect to 'https://huggingface.co' to load the files, 
and couldn't find them in the cached files.
```

### BioGPT
```
Model: microsoft/BioGPT-Large
Erro: We couldn't connect to 'https://huggingface.co' to load the files, 
and couldn't find them in the cached files.
```

---

## 📊 RESUMO

| Modelo | F1-Macro | Status |
|--------|----------|--------|
| Phi-3.5 (3.8B) | - | ❌ Não disponível |
| Mistral (7B) | - | ❌ Não disponível |
| BioGPT | - | ❌ Não disponível |

### Referências
- TF-IDF: 0.77885
- BERTimbau v4: 0.82073

---

## 📝 INSIGHTS

1. **Problema Principal:**
   - Nenhum modelo LLM estava disponível offline no Kaggle
   - `local_files_only=True` falhou para todos
   - Modelos NÃO foram uploaded como datasets

2. **Ação Necessária:**
   - Criar notebooks de download para cada modelo
   - Upload para Kaggle como datasets privados
   - Ver [skills/models/upload.md](../../skills/models/upload.md)

3. **Modelos a Baixar:**
   ```
   models/llm/
   ├── download_phi3.ipynb
   ├── download_mistral.ipynb
   └── download_biogpt.ipynb
   ```

4. **Alternativa:**
   - Usar Qwen3 (já disponível - testado em resubmit/)
   - Ver resultados de `resubmit_qwen3_zeroshot.ipynb`

---

## ⚠️ STATUS: REQUER AÇÃO

- [ ] Criar notebook download_phi3.ipynb
- [ ] Criar notebook download_mistral.ipynb  
- [ ] Criar notebook download_biogpt.ipynb
- [ ] Re-executar validação com modelos offline

*Validação inconclusiva - repetir após download dos modelos.*
