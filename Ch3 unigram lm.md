# Chapter 3: Unigram Language Model Tokenization 🎲

> **"Instead of greedily merging, Unigram LM starts big and prunes — keeping only the most probable subwords."**  
> — Kudo, 2018 ("Subword Regularization")

---

## 🧒 Story for a 5-Year-Old

BPE was like building a Lego tower by gluing pieces together one at a time, always picking the most popular pair. But Unigram works backwards: imagine you START with a huge pile of every possible word piece, then throw away the ones that aren't helping. You keep doing this until you have just the right amount of pieces.

Also, Unigram is a bit unpredictable on purpose — sometimes it splits "playing" into ["play", "ing"], sometimes ["pla", "ying"]. Being a little random makes it smarter in the end!

---

## 💡 Intuition

Unigram Language Model tokenization (Kudo, 2018) takes a fundamentally different approach:

1. **Start with a large seed vocabulary** (all substrings, or BPE vocabulary)
2. **Assign a probability to each token** (how likely is this subword to appear?)
3. **Define the best segmentation** as the one maximizing the probability of the text
4. **Iteratively remove tokens** that contribute least to corpus likelihood
5. **Output multiple possible segmentations** with probabilities (subword regularization)

The key insight: **tokenization is a probabilistic model**, not just a lookup table.

---

## 📐 Mathematical Explanation

### The Model

Given a sentence `X`, define a **segmentation** as `x = (x₁, x₂, ..., xₘ)` where each `xᵢ` is a subword.

The **Unigram model** assumes subwords are independent:

```
P(x) = ∏ᵢ P(xᵢ)

Where P(xᵢ) is the unigram probability of subword xᵢ
```

The **best segmentation** (Viterbi decoding) is:

```
x* = argmax_x P(x) = argmax_x ∏ᵢ P(xᵢ)
   = argmax_x Σᵢ log P(xᵢ)    [taking log for numerical stability]
```

### Training Objective

Given corpus `D = {X₁, X₂, ..., Xₙ}`, we want to find vocabulary `V` and probabilities `{pᵢ}` that maximize the **marginal log-likelihood**:

```
L(V, p) = Σₛ log P(Xₛ)
         = Σₛ log Σ_{x∈S(Xₛ)} P(x)

Where S(Xₛ) = all possible segmentations of sentence Xₛ
```

This is a **sum over all possible segmentations** — not just the best one!

### Why Marginalize?

Because we don't know which segmentation is "correct." We treat all segmentations as latent variables and marginalize them out.

---

## 🔄 EM Training Algorithm

Unigram LM is trained using **Expectation-Maximization (EM)**:

```
Algorithm Unigram_LM_Train(corpus, target_vocab_size):

  1. Initialize V with large vocabulary (e.g., all substrings up to length 16)
     or start from BPE vocabulary
  
  2. Initialize uniform probabilities: p(x) = 1/|V| for all x ∈ V
  
  REPEAT:
    ── E-Step: Compute expected token counts ──
    For each sentence X in corpus:
      Enumerate all segmentations S(X)
      Compute posterior P(x|X) = P(x) / Σ_{x'∈S(X)} P(x')
      Accumulate expected counts: E[count(xᵢ)] += P(xᵢ|X)
    
    ── M-Step: Re-estimate probabilities ──
    p*(xᵢ) = E[count(xᵢ)] / Σⱼ E[count(xⱼ)]
    
    ── Pruning Step ──
    For each subword x ∈ V:
      Compute loss_x = how much L decreases if x is removed
    Remove bottom η% of subwords (e.g., η = 10%)
    (Never remove single characters — they're the fallback)
  
  UNTIL |V| == target_vocab_size
  
  Output: V, {p(x) for x ∈ V}
```

---

## ⚡ Forward-Backward Algorithm

The E-step requires computing `P(x|X)` for all segmentations — exponentially many! The **forward-backward algorithm** (analogous to HMMs) computes this efficiently.

### Forward Pass
```
α(i) = probability that positions 0..i are covered by valid segmentations

α(0) = 1  (empty prefix, trivially segmented)

α(i) = Σ_{x covers [j..i]} α(j) · P(x)
```

### Backward Pass
```
β(i) = probability that positions i..n are covered by valid segmentations

β(n) = 1  (empty suffix)

β(i) = Σ_{x covers [i..j]} P(x) · β(j)
```

### Posterior
```
P(x covers [i..j] | X) = α(i) · P(x) · β(j) / α(n)
```

**Time complexity:** O(n² · |V|) per sentence with trie optimization → O(n · L_max) where L_max = max subword length.

---

## 🎯 Segmentation: Viterbi vs Sampling

### Viterbi (Greedy Best)
```
x* = argmax_x Σᵢ log P(xᵢ)
```
Returns a single deterministic segmentation — the most probable one.

### Sampling (Subword Regularization)
```
x ~ P(x|X)
```
Samples a segmentation proportional to its probability. Different runs → different tokenizations!

**Why sampling helps training:**
- Model sees multiple segmentations of the same word
- Forces robust representations
- Like data augmentation for tokenization

```
"playing" can be tokenized as:
  ["play", "ing"]    p = 0.7
  ["play", "i", "ng"] p = 0.15
  ["p", "laying"]    p = 0.05
  ...
```

During training, we **sample** one of these. During inference, we use Viterbi (most probable).

---

## 📊 Comparison: BPE vs Unigram LM

| Feature | BPE | Unigram LM |
|---|---|---|
| **Algorithm type** | Bottom-up merge | Top-down pruning |
| **Training objective** | Greedy frequency | Maximum likelihood |
| **Segmentation** | Deterministic | Deterministic (Viterbi) or stochastic |
| **Multiple segmentations?** | No (with BPE-Dropout: yes) | Yes, naturally |
| **Handles ambiguity?** | No | Yes |
| **Morphological alignment** | Accidental | Better (probabilistic) |
| **Multilingual** | Biased to high-resource | More fair |
| **Training complexity** | O(N·V) | O(N·n·L_max) per EM iter |
| **Inference speed** | Fast | Moderate (Viterbi) |
| **Used in** | GPT-2/3/4, LLaMA (old) | SentencePiece, T5, LLaMA-2+ |

---

## 🌍 Why Unigram LM is Better for Multilingual Models

### Key Reason 1: Probabilistic Marginalization
BPE always gives ONE segmentation. Unigram gives a **distribution** over segmentations. For morphologically complex words with valid multiple splits, this is more correct.

```
Turkish: "evlerinizden" (from your houses)
BPE:     ["evler", "##inizden"]  (one arbitrary split)
Unigram: ["ev", "ler", "iniz", "den"]  (morpheme-aware, higher prob)
```

### Key Reason 2: Better Vocabulary Coverage
Unigram explicitly optimizes for a vocabulary that best **explains** the corpus. BPE just merges greedily — no global optimization.

### Key Reason 3: Subword Regularization
Seeing multiple segmentations makes the model robust to tokenization errors, especially important for morphologically rich and low-resource languages where word forms vary widely.

---

## 📊 Log-Likelihood Intuition

Consider the word "unhappiness":

```
Segmentation A: ["un", "happiness"]
  log P = log P("un") + log P("happiness")
       = -2.3 + -4.1 = -6.4

Segmentation B: ["un", "happy", "ness"]  
  log P = log P("un") + log P("happy") + log P("ness")
       = -2.3 + -2.8 + -3.1 = -8.2

Segmentation C: ["unhappy", "ness"]
  log P = log P("unhappy") + log P("ness")
       = -5.5 + -3.1 = -8.6

Best (Viterbi): Segmentation A (highest log-prob = -6.4)
```

Unigram LM would select A, but might occasionally sample B or C during training — good for regularization.

---

## 🎨 ASCII Diagram: Unigram vs BPE

```
BPE (bottom-up):                  Unigram (top-down):

Start: [u][n][h][a][p][p][y]      Start: huge vocab with ALL substrings:
                                    [unhappy], [unhapp], [unhap], [unha],
Merge 1: [u][n][h][a][pp][y]       [un], [happy], [happ], [hap], ...
Merge 2: [u][n][ha][pp][y]                        │
Merge 3: [u][n][ha][ppy]                    Prune least useful
Merge 4: [u][n][happy]                             │
Merge 5: [un][happy]                        [unhappy], [un], [happy],
                                             [ness], [h], [u], ...
Final: [un][happy]                Final: [un][happy] (Viterbi)
```

---

## 💻 Code: Unigram LM Segmentation (Simplified Viterbi)

```python
import math
from typing import Dict, List, Tuple, Optional

def viterbi_segment(word: str, vocab_probs: Dict[str, float]) -> List[str]:
    """
    Viterbi decoding for Unigram LM segmentation.
    
    Args:
        word: input word to segment
        vocab_probs: dict {subword: log_probability}
    
    Returns:
        best segmentation as list of subwords
    """
    n = len(word)
    
    # dp[i] = (best_log_prob to reach position i, split_point)
    dp = [(-float('inf'), -1)] * (n + 1)
    dp[0] = (0.0, -1)  # base case: empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            subword = word[j:i]
            if subword in vocab_probs:
                log_prob = dp[j][0] + vocab_probs[subword]
                if log_prob > dp[i][0]:
                    dp[i] = (log_prob, j)
    
    # Backtrack to find segmentation
    segments = []
    i = n
    while i > 0:
        _, j = dp[i]
        if j == -1:
            # OOV: fall back to characters
            segments.append(word[0:i])
            break
        segments.append(word[j:i])
        i = j
    
    return list(reversed(segments))


def sample_segment(word: str, vocab_probs: Dict[str, float], 
                   alpha: float = 0.1) -> List[str]:
    """
    Sample a segmentation (subword regularization).
    
    alpha: smoothing parameter (lower = more uniform sampling)
    """
    import random
    n = len(word)
    
    # Build all valid segmentations (simplified for short words)
    # In practice, use forward-backward algorithm
    def enumerate_segs(s, vocab, memo={}):
        if s in memo:
            return memo[s]
        if len(s) == 0:
            return [[]]
        
        results = []
        for length in range(1, len(s) + 1):
            piece = s[:length]
            if piece in vocab:
                for rest in enumerate_segs(s[length:], vocab, memo):
                    results.append([piece] + rest)
        
        memo[s] = results
        return results
    
    all_segs = enumerate_segs(word, vocab_probs)
    if not all_segs:
        return list(word)  # character fallback
    
    # Score each segmentation
    def score(seg):
        return sum(vocab_probs.get(piece, -20) for piece in seg) / alpha
    
    scores = [score(s) for s in all_segs]
    
    # Softmax sampling
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]
    
    # Sample
    r = random.random()
    cumulative = 0
    for seg, prob in zip(all_segs, probs):
        cumulative += prob
        if r <= cumulative:
            return seg
    
    return all_segs[0]


# ---- Example Usage ----
# Toy vocabulary with log probabilities
vocab = {
    "un":       math.log(0.15),
    "happy":    math.log(0.12),
    "ness":     math.log(0.08),
    "unhappy":  math.log(0.05),
    "happiness":math.log(0.03),
    "hap":      math.log(0.04),
    "py":       math.log(0.06),
    "u":        math.log(0.20),
    "n":        math.log(0.25),
    "h":        math.log(0.18),
    "a":        math.log(0.22),
    "p":        math.log(0.19),
    "y":        math.log(0.14),
    "i":        math.log(0.21),
    "e":        math.log(0.23),
    "s":        math.log(0.17),
}

word = "unhappiness"

# Viterbi (best)
best = viterbi_segment(word, vocab)
print(f"Viterbi: {word} → {best}")

# Sampling (regularization)
print("Sampled segmentations:")
for _ in range(5):
    sample = sample_segment(word[:8], vocab)  # use shorter word for demo
    print(f"  {sample}")
```

---

## 🔬 Subword Regularization in Practice

Used in **SentencePiece** with two hyperparameters:
- `l`: number of candidate segmentations to sample from
- `α` (alpha): smoothing — controls how peaked the distribution is

```
α → 0:  uniform distribution over all segmentations (max randomness)
α → ∞:  approaches Viterbi (greedy best)
α = 0.1: recommended by Kudo (2018)
```

In practice, set `l = 64` and `α = 0.1` for training. At inference, use Viterbi.

---

## 📈 Empirical Results (Kudo, 2018)

On Japanese-English translation (WAT 2017):
```
Method               BLEU
BPE                  28.4
Unigram LM           28.7
Unigram + sampling   29.1  ← best (subword regularization)
```

Improvement is modest for English but larger for morphologically rich languages like Japanese, German, Turkish.

---

## ⚠️ Common Mistakes

1. **Using Viterbi during training**: Should use sampling during training for regularization
2. **Too small α**: If α is too small, distribution is too uniform → model sees incoherent segmentations
3. **Not warming up with BPE**: Unigram often initialized from BPE vocabulary for faster convergence
4. **Forgetting the pruning step**: EM without pruning doesn't reduce vocabulary size
5. **Using Unigram for byte-level**: Unigram doesn't apply at the byte level — bytes don't have probabilities in the same sense

---

## 🔭 Open Research Problems

1. **Better initialization**: Is BPE initialization optimal for Unigram?
2. **Morphology-constrained Unigram**: Can we encode linguistic constraints into the probability model?
3. **Joint tokenizer + model training**: End-to-end optimization with downstream loss
4. **Cross-lingual parity**: Explicitly optimize Unigram for equal fertility across languages
5. **Bayesian tokenization**: Model uncertainty over vocabulary itself

---

*Next: Chapter 4 — SentencePiece →*
