# 📰 News Summarization using QLoRA Fine-Tuned T5 Model

This project focuses on building a **news article summarization system** using the **T5 Transformer architecture**, fine-tuned with **QLoRA (Quantized LoRA)** on the CNN/DailyMail dataset. The system takes a long-form article and generates a concise, meaningful summary.

> 💡 Fine-tuned with **QLoRA** on **4-bit precision**, making it **efficient** for the  low-resource environments without compromising accuracy.

---

### 📌 Table of Contents

- [Dataset](#-dataset)
- [Model Overview](#-model-overview)
- [Fine-Tuning Technique](#fine-tuning-technique)
- [Project Pipeline](#project-pipeline)
- [Evaluation](#evaluation)
- [Results Summary](#-results-summary)
- [Contact Info](#-contact)
---

## 📂 Dataset

**CNN/DailyMail News Dataset**  
📁 Source: [Kaggle Dataset Link](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

- ~287K Training Articles  
- ~13K Validation Articles  
- ~11K Test Articles

Each entry contains:
- `article`: The full news article.
- `highlights`: The reference summary.

---

## 🤖 Model Overview

- **Base Model**: `t5-small` (Text-To-Text Transfer Transformer)  
- **Tokenizer**: `T5Tokenizer`  
- **Task**: Sequence-to-Sequence Summarization  
- **Precision**: 4-bit quantization (`bitsandbytes`)  
- **Hardware Used**: GPU (Kaggle P100)

---

## Fine-Tuning Technique

I employed **QLoRA** (Quantized Low-Rank Adaptation) for parameter-efficient fine-tuning using:

- `LoRAConfig`: `r=64`, `alpha=16`, `dropout=0.05`  
- Target Modules: `["q", "v"]`  
- Optimizer: `paged_adamw_8bit`  
- Trainer: HuggingFace `Seq2SeqTrainer`

---

## Project Pipeline

1. **Exploratory Data Analysis**  
   - Word count distributions  
   - Length statistics  
   - Missing/duplicate checks

2. **Data Preprocessing**  
   - Tokenization (max input: 1200, max output: 80)  
   - PyTorch-compatible formatting

3. **Model Training**  
   - Trained for 1000 steps using 4-bit precision  
   - Evaluation every 100 steps

4. **Evaluation**  
   - ROUGE and BLEU metrics

5. **Testing**  
   - Summaries was generated for the custom/unseen news articles

---

## Evaluation

The model is evaluated using **ROUGE** and **BLEU** scores over 2000 test samples to save resources, and it showed a simillar performance:

| Metric   | Score (2000 samples)    |
|----------|-----------|
| ROUGE-1  | 0.359   |
| ROUGE-2  | 0.157  | 
| ROUGE-L  | 0.262  | 
| BLEU     | 0.124   |

> 🚀 These scores are competitive for small-scale summarization models and show that QLoRA fine-tuning can match larger models' performance with fewer resources.

---

## 📈 Results Summary

| Aspect         | Description                          |
|----------------|--------------------------------------|
| Model          | T5-small                             |
| Fine-Tuning    | QLoRA + LoRA (PEFT)                  |
| Precision      | 4-bit (QLoRA)                        |
| Dataset        | CNN/DailyMail                        |
| Metrics        | ROUGE-1, ROUGE-2, ROUGE-L, BLEU |
| Use Case       | Real-time news summarization         |


---

## 📫 Contact

For questions, or improvements, feel free to reach out:

**Saba Mohamed**  
Fresh Graduate from the British University in Egypt (BUE).

<sub>📧 Email: saba.mohamed.abdeltawab@gmail.com</sub>

<sub>🌍 LinkedIn: [linkedin.com/in/saba-mohamed-552a102b2](https://www.linkedin.com/in/saba-mohamed-552a102b2/)</sub>
