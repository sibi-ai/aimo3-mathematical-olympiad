# AIMO3 — AI Mathematical Olympiad Progress Prize 3

> **Competition:** [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) (Kaggle, April 2026)
> **Result:** 🥉 **Bronze Medal** | **42 / 50** correct answers
> **Model:** GPT-OSS-120B on single H100 GPU
> **Author:** Haseeb Ahmad | [kaggle.com/hasib007](https://kaggle.com/hasib007)

[![Kaggle](https://img.shields.io/badge/Kaggle-Expert-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/hasib007)
[![Score](https://img.shields.io/badge/Score-42%2F50-brightgreen?style=for-the-badge)](https://kaggle.com/hasib007)
[![Model](https://img.shields.io/badge/Model-GPT--OSS--120B-EE4C2C?style=for-the-badge)](https://huggingface.co/sibi-ai)
[![GPU](https://img.shields.io/badge/GPU-H100%20(single)-76B900?style=for-the-badge)](https://kaggle.com/hasib007)

---

## Task

Solve **50 IMO-level mathematical problems** with integer answers (0–99999). Each problem is solved by an LLM with access to a Python code execution sandbox.

---

## Architecture

```
Problem (text)
      │
      ▼
┌─────────────────────────────────────────────────┐
│              Dynamic Time Budget                │
│  (notebook_limit=17400s ÷ problems_remaining)   │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│         8 Parallel Reasoning Attempts           │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  GPT-OSS-120B (vLLM, fp8 KV cache)     │    │
│  │  context=65536 | temperature=0.7        │    │
│  │  min_p=0.1 | top_logprobs=5            │    │
│  │                                         │    │
│  │  Multi-turn reasoning loop:             │    │
│  │  ┌─ Think ─────────────────────────┐   │    │
│  │  │  Chain-of-thought reasoning      │   │    │
│  │  │  IMO-level structured approach   │   │    │
│  │  └──────────────────────────────────┘   │    │
│  │  ┌─ Code (optional) ────────────────┐   │    │
│  │  │  Jupyter sandbox (persistent)    │   │    │
│  │  │  sympy / numpy / math            │   │    │
│  │  └──────────────────────────────────┘   │    │
│  │         → extract \boxed{N}             │    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  Early stop: 4+ attempts agree → stop           │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│          Entropy-Weighted Voting                │
│                                                 │
│  weight = 1 / mean_token_entropy               │
│  (lower entropy = higher confidence = more weight) │
└─────────────────────────────────────────────────┘
      │
      ▼
   Final Answer (integer 0–99999)
```

---

## Key Design Decisions

| Decision | Detail | Rationale |
|---|---|---|
| **8 parallel attempts** | ThreadPoolExecutor, shared stop_event | Diversity without reranker |
| **Entropy weighting** | weight = 1/H(logprobs) | Confident answers outweigh uncertain ones |
| **Early stopping** | Stop when 4+ agree | Save budget for harder problems |
| **FP8 KV cache** | `fp8_e4m3` | 120B model on single H100 |
| **Persistent kernels** | 16 Jupyter sandboxes, reused | Eliminate cold-start per turn |
| **Dynamic budget** | Per-problem time = f(remaining) | Spend more time on early hard problems |
| **Per-attempt seed** | `seed = (base_seed + attempt)²` | Controlled diversity across attempts |
| **Prefix caching** | `--enable-prefix-caching` | Reuse system prompt KV across turns |

---

## Experiments (7 Iterations)

| Exp | Change | Result |
|---|---|---|
| Baseline | GPT-OSS-120B, default settings | Reference |
| QLoRA fine-tuning | Fine-tune on math data | Degraded (disrupted co-optimization) |
| Multi-temperature sampling | Temperature sweep | Degraded |
| LLM-as-judge | Secondary model for answer selection | Degraded |
| Alternative models | Smaller models tested | Degraded |
| Concise prompt | Shorter system prompt | Degraded |
| GRPO on GPT-OSS-20B | RL fine-tuning | Degraded |

**Central finding:** All 7 modifications degraded the baseline. GPT-OSS-120B's chain-of-thought + tool-use pipeline was tightly co-optimized; external modifications consistently hurt performance.

---

## Results

| Metric | Value |
|---|---|
| **Score** | **42 / 50** 🥉 |
| **Medal** | Bronze |
| **Model** | GPT-OSS-120B |
| **Hardware** | Single H100 GPU |
| **Experiments** | 7 (all modifications degraded baseline) |

---

## Project Structure

```
aimo3-mathematical-olympiad/
├── src/
│   └── solver.py               ← Full inference pipeline (clean, documented)
├── notebooks/
│   └── aimo-improved-version.ipynb   ← Original Kaggle notebook
├── requirements.txt
└── README.md
```

---

## Setup

> **Note:** Runs on Kaggle with GPT-OSS-120B model + H100 GPU.
> Local reproduction requires the model weights and Kaggle competition data.

```bash
git clone https://github.com/sibi-ai/aimo3-mathematical-olympiad
cd aimo3-mathematical-olympiad
pip install -r requirements.txt
```

---

## Dependencies

```
vllm               # LLM serving
openai             # API client for vLLM
openai_harmony     # Conversation template for GPT-OSS
jupyter_client     # Persistent Jupyter kernel management
transformers
polars
pandas
```

---

## Author

**Haseeb Ahmad** — Independent ML/NLP Researcher | Kaggle Expert
- 📊 [kaggle.com/hasib007](https://kaggle.com/hasib007)
- 🤗 [huggingface.co/sibi-ai](https://huggingface.co/sibi-ai)
- 💼 [linkedin.com/in/hasib007](https://linkedin.com/in/hasib007)
- 📧 haseb_ahmad@yahoo.com
