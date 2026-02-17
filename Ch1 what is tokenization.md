# Chapter 1: What Is Tokenization? 🔤

> **"Tokenization is the art of breaking language into the right-sized pieces so a machine can think about it."**

---

## 🧒 Story for a 5-Year-Old

Imagine you're reading a book, but before you can understand it, you need to cut it up with scissors into little pieces. You could cut every single letter separately — but then each piece is TOO small and doesn't mean much. Or you could keep every whole sentence — but then each piece is TOO big and there are too many to remember.

The smartest thing? Cut it into **words and word-parts** — just the right size. That's tokenization. You're deciding WHERE to put the scissors.

---

## 💡 Intuition

**Tokenization** is the process of splitting raw text into a sequence of discrete units called **tokens**, which a language model uses as its basic "atoms" of thought.

Before a neural network can process language, it needs:
1. **Finite vocabulary** — a fixed set of "symbols" it knows
2. **Discrete indices** — each token mapped to an integer ID
3. **Consistent segmentation** — the same text always splits the same way (deterministic) or in a principled way (probabilistic)

Text → Tokenizer → `[token_id_1, token_id_2, ..., token_id_n]` → LLM

---

## 📐 Why Does Tokenization Exist?

### Problem 1: Neural Networks Need Numbers
Text is continuous, but neural nets work on vectors. Tokenization creates the **bridge**: discrete symbols → integer IDs → embedding vectors.

### Problem 2: Vocabulary Must Be Finite
If you tokenize by whole words, English alone has **millions** of word forms. Rare words become OOV (Out-of-Vocabulary). Too big a vocabulary → too many parameters in the embedding table.

### Problem 3: Characters Are Too Fine-Grained
If you tokenize by characters, every word becomes very long. "tokenization" = 12 characters = 12 tokens. The sequence length explodes, destroying compute efficiency.

**Subword tokenization** is the sweet spot.

---

## 📊 Information Theory Perspective

### Entropy & Coding

From Shannon's Information Theory, the **entropy** of a language is:

```
H(X) = -∑ p(x) · log₂ p(x)
```

Where `p(x)` is the probability of token `x`.

A **good tokenizer** minimizes the description length of text — it's essentially a **lossless compression** scheme.

If your tokenizer assigns long codes (many tokens) to frequent patterns, it's wasteful. A perfect tokenizer would approach the Shannon entropy of the language.

### Compression View

Think of tokenization as **step 1 of compression**:
- BPE was literally invented as a data compression algorithm (Gage, 1994) before being adapted for NLP
- A tokenizer that uses fewer tokens for the same text → more compressed → better efficiency

```
Compression ratio ≈ (original chars) / (number of tokens)
```

Higher is better. English GPT-4 tokenizer achieves ~4 chars/token on average text.

---

## 📏 The Core Trade-Off: Vocabulary Size vs Sequence Length

```
┌──────────────────────────────────────────────────────┐
│                  VOCABULARY SIZE                     │
│  Small ←─────────────────────────────────→ Large    │
│                                                      │
│  Char-level        Subword           Word-level      │
│  [h,e,l,l,o]    [hel,lo]           [hello]          │
│                                                      │
│  Sequence: LONG     MEDIUM            SHORT          │
│  OOV risk: NONE     LOW               HIGH           │
│  Morphology: NO     SOME              YES            │
└──────────────────────────────────────────────────────┘
```

| Strategy | Vocab Size | Seq Length | OOV? | Morphology? |
|---|---|---|---|---|
| Character-level | ~256 | Very Long | Never | No |
| Byte-level | 256 | Long | Never | No |
| BPE Subword | 32k–128k | Medium | Rare | Partial |
| Word-level | 100k+ | Short | Frequent | Yes |

---

## 💥 Impact on LLM Compute: O(n²) Attention

This is critical. Transformer self-attention is **O(n²)** in sequence length `n`:

```
Attention FLOPs ≈ 4 · n² · d_model    (per layer)

Where:
  n = number of tokens
  d_model = hidden dimension
```

If tokenization doubles the sequence length → **4× the attention compute cost**.

### Real Example
- English: "The transformer is powerful" → 5 tokens
- Tamil equivalent: → ~15 tokens (due to fragmentation)
- Tamil pays **9× more attention FLOPs** for the same semantic content!

This is why tokenization **directly determines LLM training cost, inference latency, and context window capacity**.

```
Total FLOPs ≈ 4·n²·d·L + 8·n·d²·L

Where L = number of layers
```

---

## 📊 Tokenization Levels — Full Comparison

| Level | Example | Pros | Cons |
|---|---|---|---|
| **Word** | `["hello", "world"]` | Meaningful | OOV, large vocab |
| **Character** | `["h","e","l","l","o"]` | No OOV | Very long sequences |
| **Byte** | `[104, 101, 108, 108, 111]` | Universal | Longest sequences |
| **Subword (BPE)** | `["hel","lo"]` | Balance | Language-biased |
| **Subword (Unigram)** | `["hello"]` or `["hel","lo"]` | Probabilistic | Complex training |

---

## 🔬 Compression Theory Connections

### Zipf's Law
In any natural language corpus, word frequency follows a **power law**:

```
freq(rank r) ∝ 1/r^s   (s ≈ 1 for natural language)
```

This means:
- A tiny fraction of words account for most text
- The long tail is enormous (rare words)
- BPE exploits this: it merges the most *frequent* pairs first

### Tokenization as Huffman Coding
A theoretically optimal tokenizer would assign shorter tokens to more frequent substrings — exactly like **Huffman coding**. BPE approximates this greedily.

---

## 🔢 Token Efficiency Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Fertility** | tokens / words | How many tokens per word (lower = better) |
| **CPT** | chars / token | Average chars per token (higher = better) |
| **Compression** | chars / tokens | Same as CPT |
| **NSL** | lang_tokens / en_tokens | Normalized sequence length vs English |
| **Parity Ratio** | lang_tokens / pivot_tokens | Cross-lingual fairness |

**Example:**
- English: "I love machine learning" → 4 tokens → Fertility ≈ 1.0
- Tamil: "நான் இயந்திர கற்றல் நேசிக்கிறேன்" → ~18 tokens → Fertility ≈ 4.5

---

## 📐 Mathematical Formalization

Given a string `s` over alphabet `Σ`, a tokenizer is a function:

```
τ: Σ* → V*

Where:
  Σ* = set of all strings over alphabet Σ
  V  = finite vocabulary of token symbols
  V* = sequences of vocabulary tokens
```

The tokenizer must be:
1. **Lossless**: original string recoverable from token sequence
2. **Consistent**: same input → same output (or same distribution)
3. **Finite**: vocabulary V is fixed-size

Training a tokenizer = finding V that minimizes `E[|τ(s)|]` (expected sequence length) subject to `|V| ≤ budget`.

---

## ⚙️ System-Level View: LLM Training Pipeline

```
Raw Text Corpus
      │
      ▼
 Pre-tokenization (whitespace, Unicode normalization)
      │
      ▼
 Tokenizer Training (BPE / Unigram / WordPiece)
      │
      ▼
 Vocabulary File  +  Merge Rules / Token Probabilities
      │
      ▼
 Apply Tokenizer to Training Corpus
      │
      ▼
 Token IDs → Embedding Layer → Transformer Layers
      │
      ▼
 LLM Output
```

---

## ⚠️ Common Mistakes

1. **Forgetting normalization**: Tokenizing before Unicode normalization causes identical text to produce different tokens
2. **Treating tokenization as separate from training**: Vocab size affects perplexity, not just speed
3. **Ignoring language bias**: English-trained tokenizers penalize non-Latin scripts massively
4. **Over-large vocabulary**: Embedding table dominates parameter count; vocab > 128k often wasteful
5. **Confusing deterministic vs stochastic**: BPE = deterministic; Unigram = can be stochastic during training

---

## 🔭 Research-Level Insights

- **Tokenization and the Noiseless Channel** (Zouhar et al., 2023): Tokenization can be framed as a noisy communication channel, where longer tokens → less noise
- **CANINE** (Clark et al., 2022): Tokenization-free models operating on raw Unicode characters show that tokenization is not fundamental — but compute cost is real
- **Do All Languages Cost the Same?** (Ahia et al., 2023): Empirically shows that users in some languages pay 2.5–10× more API costs due to tokenization unfairness
- **Language Model Tokenizers Introduce Unfairness** (Petrov et al., 2023): Formal framework for "tokenization premium" and "parity ratio"

---

## 🧪 Open Research Problems

1. **Learned tokenization end-to-end**: Can we train tokenizers jointly with the LLM?
2. **Morphologically-aware tokenization**: Respecting linguistic boundaries (prefixes, stems, suffixes)
3. **Dynamic vocabulary**: Update vocab as new language data arrives
4. **Token-free architectures**: ByT5, CANINE — no tokenization at all
5. **Fairness-constrained tokenizer training**: Explicitly optimize for cross-lingual parity

---

## 💻 Code: Basic Tokenization Demo

```python
import unicodedata
from collections import Counter

def naive_word_tokenize(text):
    """Word-level tokenizer"""
    return text.split()

def char_tokenize(text):
    """Character-level tokenizer"""
    return list(text)

def measure_fertility(tokens, words):
    """Fertility = tokens per word"""
    return len(tokens) / len(words) if words else 0

# Example
text = "The transformer model learns from data"
words = naive_word_tokenize(text)
chars = char_tokenize(text.replace(" ", ""))

print(f"Words: {words}")
print(f"Word count: {len(words)}")
print(f"Char tokens: {len(chars)}")
print(f"Fertility (char): {measure_fertility(chars, words):.2f}")

# Multilingual comparison
english = "The cat sat on the mat"
tamil = "பூனை தரையில் அமர்ந்தது"

en_words = english.split()
ta_words = tamil.split()

# Approximate GPT-4 tokenization (for illustration)
# English typically ~1 token/word, Tamil ~3-5 tokens/word
print(f"\nEnglish words: {len(en_words)}")
print(f"Tamil words: {len(ta_words)}")
print(f"Tamil requires ~3-5x more tokens for similar meaning")
```

---

## 🔗 Key Papers to Read

| Paper | Year | Contribution |
|---|---|---|
| Gage (1994) | 1994 | Original BPE compression algorithm |
| Sennrich et al. | 2016 | BPE for NMT (ACL 2016) |
| Kudo (2018) | 2018 | Unigram LM tokenization |
| Kudo & Richardson | 2018 | SentencePiece system |
| Radford et al. (GPT-2) | 2019 | Byte-level BPE |
| Rust et al. | 2021 | How good is your tokenizer? |
| Ahia et al. | 2023 | Do all languages cost the same? |
| Petrov et al. | 2023 | Language model tokenizers introduce unfairness |
| Zouhar et al. | 2023 | Tokenization and the noiseless channel |

---

*Next: Chapter 2 — Byte Pair Encoding (BPE) in depth →*
