# Chapter 2: Byte Pair Encoding (BPE) 🔧

> **"BPE is a greedy algorithm that keeps merging the most popular couple until it's satisfied."**  
> — Origin: Data Compression → NLP (Sennrich et al., ACL 2016)

---

## 🧒 Story for a 5-Year-Old

Imagine you're writing a diary, but your pencil is slow. You notice you write "the" a LOT. So instead of writing t-h-e every time, you invent a special symbol ★ that means "the". Then you notice you write "ing" a lot too, so you make another symbol ✦ for that.

You keep finding the MOST written pair of letters and replace them with a new symbol. That's BPE. You're making a personal shorthand by learning what's most common.

---

## 💡 Intuition

BPE starts with the smallest possible pieces (characters or bytes) and **repeatedly merges the most frequent adjacent pair** into a new token. It stops when:
- The vocabulary reaches the desired size, OR
- No more merges improve things

The result: common words stay whole ("the", "is"), common morphemes become tokens ("ing", "un", "re"), rare words get split into sub-pieces.

---

## 📜 Historical Origin

**Philip Gage (1994)** invented BPE as a **data compression algorithm** in his paper *"A New Algorithm for Data Compression"* (C Users Journal).

Original use: Replace the most frequent byte pair in binary data with an unused byte. Repeat until no pair occurs more than once. This is lossless compression.

**Rico Sennrich et al. (ACL 2016)** adapted it for NLP:
> *"We encode rare and unknown words as sequences of subword units."*

Key adaptation: Instead of compressing bytes, we compress **character sequences** to create a subword vocabulary for neural machine translation.

---

## 🎯 The BPE Algorithm — Step by Step

### Phase 1: Training (Learning Merge Rules)

**Input**: Training corpus, desired vocabulary size `V`

```
Algorithm BPE_Train(corpus, V):
  1. Pre-tokenize corpus into words (split on whitespace/punctuation)
  2. Split every word into characters + end-of-word marker
  3. Count word frequencies
  4. Initialize vocabulary = {all unique characters} ∪ {</w>}
  
  Repeat until |vocab| == V:
    a. Count all adjacent symbol pairs across entire corpus
    b. Find most frequent pair (A, B)
    c. Merge (A, B) → AB everywhere in corpus
    d. Add AB to vocabulary
    e. Save merge rule: (A, B) → AB
  
  Output: vocabulary + ordered list of merge rules
```

### Phase 2: Encoding (Applying Merge Rules)

```
Algorithm BPE_Encode(word, merge_rules):
  1. Split word into characters
  2. Apply merge rules IN ORDER (left to right priority)
  3. Return final segmentation
```

---

## 📊 Worked Example: `low`, `lower`, `lowest`

### Initial Corpus

```
Word      Frequency
low         5
lower       2
lowest      3
```

### Step 0: Character Initialization

```
Corpus (with </w> end-of-word marker):
l o w </w>        ×5
l o w e r </w>    ×2
l o w e s t </w>  ×3

Initial Vocabulary:
{l, o, w, e, r, s, t, </w>}
```

### Step 1: Count All Pairs

```
Pair         Count
(l, o)       5+2+3 = 10   ← MOST FREQUENT
(o, w)       5+2+3 = 10   ← TIE → pick first
(w, </w>)    5
(w, e)       2+3 = 5
(e, r)       2
(e, s)       3
(s, t)       3
(t, </w>)    3
(r, </w>)    2
```

### Step 1: Merge (l, o) → lo

```
lo w </w>        ×5
lo w e r </w>    ×2
lo w e s t </w>  ×3

Vocabulary adds: "lo"
Merge rule #1: (l, o) → lo
```

### Step 2: Count Pairs Again

```
Pair         Count
(lo, w)      10   ← MOST FREQUENT
(w, </w>)    5
(w, e)       5
...
```

### Step 2: Merge (lo, w) → low

```
low </w>        ×5
low e r </w>    ×2
low e s t </w>  ×3

Merge rule #2: (lo, w) → low
```

### Step 3: Merge (low, </w>) → low</w>

```
Pair         Count
(low, </w>)  5
(low, e)     5
(e, r)       2
(e, s)       3
(s, t)       3
...
```

(Tie — pick first alphabetically or first encountered)

After several more merges:

```
Merge rule #3: (low, </w>) → low</w>  [freq 5]
Merge rule #4: (low, e)    → lowe     [freq 5]
Merge rule #5: (e, s)      → es       [freq 3]
Merge rule #6: (s, t)      → st       ...
Merge rule #7: (lowe, s)   → lowes    ...
```

### Final Vocabulary (after 7 merges)

```
Original chars:  l, o, w, e, r, s, t, </w>   (8)
After merges:    lo, low, low</w>, lowe, es, st, lowes, lowest</w>
```

**Segmentation results:**
```
"low"    → [low</w>]
"lower"  → [lowe, r, </w>]     or [low, e, r, </w>]
"lowest" → [lowest</w>]        or [lowes, t, </w>]
```

---

## 🎨 ASCII Diagram: BPE Merge Process

```
Initial:  l  o  w  e  s  t  </w>
                │
          Merge #1: (l,o) → lo
                │
Step 1:   lo  w  e  s  t  </w>
                │
          Merge #2: (lo,w) → low
                │
Step 2:   low  e  s  t  </w>
                │
          Merge #3: (e,s) → es
                │
Step 3:   low  es  t  </w>
                │
          Merge #4: (es,t) → est
                │
Step 4:   low  est  </w>
                │
          Merge #5: (est,</w>) → est</w>
                │
Step 5:   low  est</w>
                │
Final:    ["low", "est</w>"]  → tokens: ["low", "est"]
```

---

## 📈 Frequency Table Through Training

```
Iteration | Merge Applied    | top pair count
----------|-----------------|----------------
    0     | (init)          | (l,o) = 10
    1     | (l,o) → lo      | (lo,w) = 10
    2     | (lo,w) → low    | tied @ 5
    3     | (low,</w>)      | 5
    4     | (low,e) → lowe  | 5
   ...    | ...             | ...
```

---

## 🧮 Time Complexity

| Step | Complexity |
|---|---|
| Count initial chars | O(N) where N = corpus size |
| Count pairs each iteration | O(N) per merge |
| Total merges = V - |Σ| | O(V) merges |
| **Total training** | **O(N·V)** |
| Encoding a word | O(|word|² · |merge_rules|) naively, O(|word|²) with efficient implementation |

For large corpora (billions of tokens) with V~50k merges, BPE training is expensive but done only once.

---

## 💾 Merge Rule Storage Format

BPE models store merge rules as an **ordered list**:

```
#version: 0.2
l o          <- merge rule 1: "l" + "o" → "lo"
lo w         <- merge rule 2: "lo" + "w" → "low"
low </w>     <- merge rule 3
low e        <- merge rule 4
...
```

During encoding, rules are applied **in order**. Earlier rules take priority.

```python
# Example merge rules as Python dict (with priority)
merge_rules = {
    ('l', 'o'): 0,      # priority 0 = highest
    ('lo', 'w'): 1,
    ('low', '</w>'): 2,
    ('low', 'e'): 3,
}
```

---

## ❓ Why BPE Prefers Frequent Pairs

BPE is a **greedy frequency-based** algorithm. It has no knowledge of:
- Meaning
- Morphology
- Linguistic boundaries

It only sees: **which pair appears most often**?

This means:
- Common English prefixes (`un-`, `re-`, `pre-`) get merged → good!
- Common suffixes (`-ing`, `-ed`, `-tion`) get merged → good!
- But random frequent n-grams also get merged → e.g., `" the"`, `"in "` 
- Morphologically correct splits are **accidental**, not guaranteed

---

## 🌍 Why BPE Struggles with Morphology-Rich Languages

**Turkish example:**
```
"evlerinizden"  = ev + ler + iniz + den
                = house + PLURAL + YOUR + FROM
                = "from your houses"
```

This is one word in Turkish but encodes 4 morphemes. BPE will split it arbitrarily based on frequency in the training corpus — probably into something like `["evler", "iniz", "den"]` if lucky, or `["e", "vler", "ini", "zden"]` if not.

**Why it fails:**
1. BPE doesn't know where morpheme boundaries are
2. It merges based on *surface frequency*, not morphological structure
3. In agglutinative languages, word forms are near-infinite, so most forms are rare

**The fix (partial):** Unigram LM (Chapter 3) handles this better.

---

## 🔡 Why Rare Words Fragment

Rare words haven't built up enough frequency for their character sequences to be merged into tokens. So they remain as individual characters or very small subwords.

```
"Pneumonoultramicroscopicsilicovolcanoconiosis"
→ ["P", "ne", "um", "ono", "ultra", "micro", "sc", "op", "ic", ...]
```

Each fragment individually is a valid token (from other words), but the rare word itself is never in the vocabulary.

---

## 🔠 Subword Regularities BPE Learns

BPE accidentally learns morphologically meaningful units because they're statistically frequent:

```
English suffixes naturally become tokens:
  "ing"    → frequent → merged → token
  "tion"   → frequent → merged → token  
  "ly"     → frequent → merged → token
  "un"     → frequent prefix → token
  "er"     → frequent → token

Code tokens learned by GPT tokenizers:
  "def ", "return ", "import ", "self."
```

This is **emergent structure** from frequency statistics.

---

## 📊 Vocabulary Growth Dynamics

```
Vocabulary size over BPE training:

|V|
│
50k ┤                                          ╭────
    │                                    ╭────╯
40k ┤                              ╭────╯
    │                        ╭────╯
30k ┤                  ╭────╯
    │            ╭────╯
20k ┤       ╭───╯
    │  ╭───╯
10k ┤──╯  (starts at |alphabet| size)
    │
    └──────────────────────────────────────────────→ merges
      0    10k   20k   30k   40k   50k
```

Linear growth by design — each merge adds exactly 1 token.

---

## ⚡ BPE Variants

| Variant | Description | Paper |
|---|---|---|
| **Standard BPE** | Character-level BPE | Sennrich et al., 2016 |
| **Byte-level BPE** | Start from bytes, not chars | GPT-2 (Radford et al., 2019) |
| **BPE-Dropout** | Stochastic BPE — drop merges randomly | Provilkov et al., 2020 |
| **Unigram BPE** | Probabilistic alternative | Kudo, 2018 |
| **SentencePiece BPE** | Language-agnostic BPE | Kudo & Richardson, 2018 |

---

## 🔢 Mathematical Depth

Let corpus `D` be a multiset of words. Each word `w` appears `c(w)` times.

**BPE objective** (informal): Minimize total number of tokens across corpus.

At each step, the merge that provides the greatest reduction is:

```
best_merge = argmax_{(a,b)} count(a,b)

After merge (a,b) → ab:
  Δ_tokens = -count(a,b)   (each merged pair reduces token count by 1)
```

So BPE greedily minimizes token count, one merge at a time. This is a **greedy approximation** to the combinatorially hard problem of optimal subword vocabulary selection.

**Total token reduction** from merge `i`:
```
ΔN_i = count_i(a_i, b_i)
```

**Total tokens after V merges:**
```
N_final = N_initial - Σᵢ ΔN_i
```

---

## 💻 Python Implementation: BPE from Scratch

```python
from collections import defaultdict, Counter

def get_vocab(corpus):
    """Build initial character-level vocabulary with word frequencies."""
    vocab = defaultdict(int)
    for word, freq in corpus.items():
        # Add spaces between characters, append </w> end marker
        chars = list(word) + ['</w>']
        vocab[' '.join(chars)] += freq
    return vocab

def get_pairs(vocab):
    """Count all adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the most frequent pair everywhere in vocabulary."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(corpus, num_merges):
    """
    Train BPE tokenizer.
    
    corpus: dict {word: frequency}
    num_merges: number of merge operations
    
    Returns: list of merge rules
    """
    vocab = get_vocab(corpus)
    merge_rules = []
    
    print(f"Initial vocab: {vocab}\n")
    
    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        
        # Find most frequent pair (break ties alphabetically)
        best_pair = max(pairs, key=lambda x: (pairs[x], x))
        best_count = pairs[best_pair]
        
        print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)} (count: {best_count})")
        
        vocab = merge_vocab(best_pair, vocab)
        merge_rules.append(best_pair)
    
    print(f"\nFinal vocab:")
    for word, freq in vocab.items():
        print(f"  '{word}' (×{freq})")
    
    return merge_rules

# ---- Run the example ----
corpus = {
    'low':    5,
    'lower':  2,
    'lowest': 3,
}

print("="*50)
print("BPE Training on: low(×5), lower(×2), lowest(×3)")
print("="*50 + "\n")

merge_rules = train_bpe(corpus, num_merges=10)
print(f"\nMerge rules learned: {merge_rules}")
```

**Expected output:**
```
Merge 1: ('l', 'o') → lo (count: 10)
Merge 2: ('lo', 'w') → low (count: 10)
Merge 3: ('low', '</w>') → low</w> (count: 5)
Merge 4: ('low', 'e') → lowe (count: 5)
Merge 5: ('e', 's') → es (count: 3)
...
```

---

## 🔍 BPE Encoding Function

```python
def apply_bpe(word, merge_rules):
    """Encode a word using learned BPE merge rules."""
    # Initialize: split into characters
    symbols = list(word) + ['</w>']
    
    for (left, right) in merge_rules:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1 and 
                symbols[i] == left and 
                symbols[i+1] == right):
                new_symbols.append(left + right)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    
    return symbols

# Test encoding
word = "lowest"
tokens = apply_bpe(word, merge_rules)
print(f"'{word}' → {tokens}")
```

---

## ⚠️ Common Mistakes with BPE

1. **Not adding end-of-word marker**: Without `</w>`, "low" and the "low" in "lower" can't be distinguished
2. **Applying merges out of order**: Rules must be applied in training order
3. **Forgetting pre-tokenization**: BPE is usually applied WITHIN words (after splitting on whitespace)
4. **Assuming morphological correctness**: BPE merges are statistical, not linguistic
5. **Not handling Unicode properly**: BPE on bytes (byte-level BPE) vs BPE on Unicode characters differ significantly

---

## 🔭 Research-Level Insights

- **BPE-Dropout** (Provilkov et al., 2020): Randomly drop some merges during training, forcing the model to see multiple segmentations → better robustness
- **Greedy suboptimality**: BPE is provably not optimal for minimum description length; Unigram LM gets closer
- **Vocabulary size scaling**: Optimal BPE vocabulary tends to scale with corpus size as O(N^0.5) (approximate)
- **Cross-lingual BPE**: Training BPE on combined multilingual corpora biases toward high-resource languages

---

## 📊 BPE vs Other Methods — Preview

| Feature | BPE | Unigram LM | WordPiece |
|---|---|---|---|
| **Deterministic?** | Yes | Can be stochastic | Yes |
| **Training objective** | Greedy freq merge | Maximize likelihood | Maximize likelihood |
| **Segmentation** | Greedy forward | Viterbi / sampling | Viterbi |
| **OOV handling** | Byte fallback | Char fallback | [UNK] |
| **Used in** | GPT family | LLaMA, T5 | BERT family |
| **Morphology** | Poor | Better | Moderate |

---

*Next: Chapter 3 — Unigram Language Model Tokenization →*
