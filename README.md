# NanoGPT: Decoder-Only Transformer Language Model from Scratch

Built a character-level GPT model from scratch in PyTorch, trained on Tiny Shakespeare.

---

## Results

| Model | Val Loss | Val Perplexity |
|-------|----------|----------------|
| Bigram Baseline | 2.49 | 12.08 |
| **NanoGPT (ours)** | **1.47** | **4.35** |

- **2.8× improvement** over bigram baseline
- **10.7M parameters**
- Architecture: 6 layers, 6 heads, 384-dim embeddings, 256 context window

---

## Ablation Study

| Variable | Best | Worst |
|----------|------|-------|
| Model Depth (n_layers) | 6 layers → ppl 5.58 | 2 layers → ppl 6.57 |
| Attention Heads (n_heads) | 4 heads → ppl 5.07 | 6 heads → ppl 5.22 |
| Context Window (block_size) | 64 → ppl 5.89 | 256 → ppl 5.99 |
| Embedding Dim (n_embd) | 384 → ppl 5.04 | 128 → ppl 7.64 |

---

## Files

```
NanoGPT_Implementation.ipynb   — full implementation with explanation
NanoGPT_Reasoning.ipynb        — reasoning behind every decision
```

---

## How to Run

1. Open `NanoGPT_Implementation.ipynb` in Google Colab
2. Set Runtime → GPU (T4)
3. Run all cells top to bottom (~15–20 min)

---

## Architecture

```
Input tokens  (B, T)
     ↓
Token Embedding + Position Embedding  (B, T, 384)
     ↓
6 × Transformer Block
    ├── Masked Multi-Head Self-Attention  (6 heads × 64 dim)
    ├── Residual Connection + LayerNorm
    ├── Feed-Forward Network  (384 → 1536 → 384)
    └── Residual Connection + LayerNorm
     ↓
Final LayerNorm
     ↓
Linear Head  (384 → 65)
     ↓
Logits → Sample with temperature
```

---

## Key Implementation Details

| Component | Detail |
|-----------|--------|
| Tokeniser | Character-level, vocab size = 65 |
| Attention | Scaled dot-product, causal mask |
| Normalisation | Pre-LayerNorm (GPT-2 style) |
| Optimiser | AdamW, β1=0.9, β2=0.95 |
| LR Schedule | Cosine decay with linear warmup |
| Regularisation | Dropout=0.2, weight decay=0.1, gradient clipping=1.0 |
| Weight tying | Token embedding shared with LM head |
| Sampling | Temperature (0.5–1.2) + top-k |

---

## Dataset

- **Tiny Shakespeare** — 1,115,394 characters
- Train: 90% (~1M chars) | Val: 10% (~115K chars)
- Download: automatic on first run

---

## CV Points

- Implemented a 6-layer, 6-head, 384-dim decoder-only Transformer from scratch in PyTorch with masked self-attention
- Trained a 10.7M-parameter model on 1.1M characters, achieving validation perplexity of 4.35 vs. baseline 12.08
- Ran ablation experiments across model depth, attention heads, and context window size to measure each component's impact
- Added temperature-based sampling at inference to control output randomness, tested across temperatures 0.5 to 1.2

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Karpathy (2022) — *Let's build GPT* (inspiration)
