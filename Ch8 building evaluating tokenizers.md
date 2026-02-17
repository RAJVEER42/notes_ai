# Chapter 8: How to Build, Evaluate & Think Like a Tokenizer Researcher 🔬

> **"A tokenizer is the single most consequential design decision in an LLM — more than architecture, more than optimizer. Yet it receives the least attention."**

---

## 🧒 Story for a 5-Year-Old

Building a tokenizer is like designing scissors for cutting text. You have to decide: should the scissors cut every single letter? Or cut at word-edges? Or somewhere in between?

To check if your scissors are GOOD, you ask:
- Do they cut fairly — the same amount of work for every language?
- Do they cut efficiently — not too many pieces?
- Do they cut sensibly — at meaningful places?

That's evaluation. And if you want to be a RESEARCHER, you design experiments that prove one pair of scissors is better than another!

---

## 🏗️ Building a Tokenizer from Scratch: Complete Pipeline

```
Step 1: Collect Training Corpus
         │
Step 2: Preprocess (normalize, clean)
         │
Step 3: Pre-tokenize (whitespace/regex split)
         │
Step 4: Train tokenizer (BPE or Unigram)
         │
Step 5: Evaluate (intrinsic + extrinsic)
         │
Step 6: Package & ship
```

---

## 📦 Step 1: Corpus Collection

```python
# Good multilingual corpus sources:
CORPUS_SOURCES = {
    "English": ["Common Crawl", "Wikipedia", "Books3", "C4"],
    "Multilingual": ["mC4", "OSCAR", "CC-100", "CulturaX"],
    "Code": ["GitHub", "The Stack", "Code Parrot"],
    "Scientific": ["ArXiv", "PubMed"],
}

# Crucial: balance languages intentionally
# Formula for sampling probability:
#   p_i ∝ n_i^α   (α = 0.3-0.7 for multilingual fairness)
#   n_i = number of tokens in language i
#   α < 1 → upsample low-resource languages

def compute_sampling_probs(lang_sizes: dict, alpha: float = 0.5) -> dict:
    """
    Compute balanced sampling probabilities.
    
    alpha=1.0: proportional sampling (biases large languages)
    alpha=0.5: square-root sampling (better multilingual balance)
    alpha=0.3: heavily upsamples small languages
    """
    total_adjusted = sum(v**alpha for v in lang_sizes.values())
    return {
        lang: (size**alpha) / total_adjusted
        for lang, size in lang_sizes.items()
    }

lang_sizes = {
    "English": 1_000_000_000,
    "German":   100_000_000,
    "Chinese":  200_000_000,
    "Tamil":      5_000_000,
    "Yoruba":     1_000_000,
}

probs_proportional = compute_sampling_probs(lang_sizes, alpha=1.0)
probs_balanced = compute_sampling_probs(lang_sizes, alpha=0.3)

print("Proportional sampling:")
for lang, p in probs_proportional.items():
    print(f"  {lang}: {p:.4f}")

print("\nBalanced sampling (α=0.3):")
for lang, p in probs_balanced.items():
    print(f"  {lang}: {p:.4f}")
```

---

## 🧹 Step 2: Preprocessing

```python
import unicodedata
import re

class TextPreprocessor:
    def __init__(self, normalization='NFKC', 
                 remove_control_chars=True,
                 normalize_whitespace=True):
        self.normalization = normalization
        self.remove_control_chars = remove_control_chars
        self.normalize_whitespace = normalize_whitespace
    
    def __call__(self, text: str) -> str:
        # 1. Unicode normalization
        if self.normalization:
            text = unicodedata.normalize(self.normalization, text)
        
        # 2. Remove control characters (keep \n and \t)
        if self.remove_control_chars:
            text = ''.join(
                ch for ch in text 
                if unicodedata.category(ch)[0] != 'C' 
                or ch in '\n\t\r'
            )
        
        # 3. Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r' +', ' ', text)     # multiple spaces → one
            text = re.sub(r'\n+', '\n', text)   # multiple newlines → one
            text = text.strip()
        
        return text
    
    def process_file(self, input_path: str, output_path: str):
        """Process entire file."""
        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    processed = self(line)
                    if processed:  # skip empty lines
                        fout.write(processed + '\n')
```

---

## 🔧 Step 3: Pre-tokenization

Pre-tokenization splits text into "words" BEFORE subword tokenization.

```python
import regex  # pip install regex (supports \p{} Unicode properties)

class PreTokenizer:
    """
    Pre-tokenizer: split text into words before BPE/Unigram.
    Different strategies for different use cases.
    """
    
    def whitespace_split(self, text: str) -> list:
        """Basic: split on whitespace."""
        return text.split()
    
    def gpt2_split(self, text: str) -> list:
        """
        GPT-2 style: regex split on punctuation, numbers, whitespace.
        Keeps contractions together, splits numbers.
        """
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return regex.findall(pattern, text)
    
    def bert_split(self, text: str) -> list:
        """
        BERT style: whitespace + punctuation split.
        """
        tokens = []
        for word in text.split():
            # Split punctuation from word
            current = ""
            for char in word:
                if regex.match(r'\p{P}', char):
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(char)
                else:
                    current += char
            if current:
                tokens.append(current)
        return tokens

# Example:
pt = PreTokenizer()
text = "Hello, world! I'm learning tokenization."

print("Whitespace:", pt.whitespace_split(text))
print("GPT-2:", pt.gpt2_split(text))
print("BERT:", pt.bert_split(text))
```

---

## 🎓 Step 4: Training with SentencePiece (Production-Grade)

```python
import sentencepiece as spm
import os

def train_tokenizer(
    corpus_files: list,
    output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'unigram',  # 'bpe' or 'unigram'
    character_coverage: float = 0.9999,
    num_threads: int = 16,
    input_sentence_size: int = 5_000_000,  # subsample for large corpora
):
    """
    Train a production SentencePiece tokenizer.
    
    character_coverage: 
      0.9995 for Latin-only
      0.9999 for multilingual with Indic/CJK scripts
    """
    
    # Build training command
    train_args = dict(
        input=','.join(corpus_files),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        
        # Byte fallback: OOV chars → byte tokens
        byte_fallback=True,
        
        # Unicode normalization
        normalization_rule_name='nfkc',
        
        # Training behavior
        split_digits=True,     # "123" → ["1", "2", "3"]
        split_by_unicode_script=True,  # Split at script boundaries
        
        # User-defined symbols (special tokens)
        user_defined_symbols=['[INST]', '[/INST]', '<<SYS>>', '<</SYS>>'],
    )
    
    spm.SentencePieceTrainer.train(**train_args)
    print(f"Tokenizer saved to {output_prefix}.model and {output_prefix}.vocab")


def load_and_test(model_path: str):
    """Load and test a trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    test_cases = [
        "Hello, world!",
        "The transformer architecture revolutionized NLP.",
        "நான் படிக்கிறேன்",  # Tamil
        "我在学习",            # Chinese
        "🎉🚀💻",             # Emoji
        "def train_model(data, epochs=10):",  # Code
    ]
    
    for text in test_cases:
        tokens = sp.encode(text, out_type=str)
        print(f"\nText: {text!r}")
        print(f"Tokens ({len(tokens)}): {tokens}")
```

---

## 📏 Step 5: Evaluation Framework

### 5.1 Intrinsic Metrics

```python
from collections import defaultdict
import math

class TokenizerEvaluator:
    """Complete tokenizer evaluation suite."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def fertility(self, texts: list) -> float:
        """
        Average tokens per word (lower = better).
        Fertility > 1 means words are being split.
        """
        total_tokens = 0
        total_words = 0
        for text in texts:
            words = text.split()
            tokens = self.tokenizer.encode(text)
            total_words += len(words)
            total_tokens += len(tokens)
        return total_tokens / total_words if total_words > 0 else 0
    
    def characters_per_token(self, texts: list) -> float:
        """
        Average characters per token (higher = better).
        English: ~4 chars/token in good tokenizers.
        """
        total_chars = 0
        total_tokens = 0
        for text in texts:
            tokens = self.tokenizer.encode(text, out_type=str)
            total_chars += sum(len(t.replace('▁', '').replace('##', '')) 
                             for t in tokens)
            total_tokens += len(tokens)
        return total_chars / total_tokens if total_tokens > 0 else 0
    
    def word_fragmentation_rate(self, texts: list) -> float:
        """
        Fraction of words split into multiple tokens (lower = better).
        """
        fragmented = 0
        total_words = 0
        for text in texts:
            for word in text.split():
                total_words += 1
                word_tokens = self.tokenizer.encode(word, 
                              add_special_tokens=False)
                if len(word_tokens) > 1:
                    fragmented += 1
        return fragmented / total_words if total_words > 0 else 0
    
    def parity_ratio(self, reference_texts: list, 
                     other_texts: list) -> float:
        """
        Parity: ratio of token counts between two languages 
        for equivalent texts (Petrov et al., 2023).
        
        Ideal: 1.0 (same tokens for same content)
        English vs Tamil: ~3.5 (Tamil needs 3.5× more tokens)
        """
        ref_tokens = sum(len(self.tokenizer.encode(t)) 
                        for t in reference_texts)
        other_tokens = sum(len(self.tokenizer.encode(t)) 
                          for t in other_texts)
        return other_tokens / ref_tokens if ref_tokens > 0 else 0
    
    def vocabulary_coverage(self, texts: list) -> float:
        """
        Fraction of words that appear as single tokens (no fragmentation).
        """
        single_token_words = 0
        total_words = 0
        for text in texts:
            for word in text.split():
                total_words += 1
                tokens = self.tokenizer.encode(word, 
                         add_special_tokens=False)
                if len(tokens) == 1:
                    single_token_words += 1
        return single_token_words / total_words if total_words > 0 else 0
    
    def renyi_efficiency(self, texts: list) -> float:
        """
        Renyi entropy-based efficiency metric (Limisiewicz et al., 2023).
        Measures information content per token.
        Higher = more efficient tokenizer.
        """
        token_counts = defaultdict(int)
        total = 0
        for text in texts:
            tokens = self.tokenizer.encode(text)
            for tok in tokens:
                token_counts[tok] += 1
                total += 1
        
        if total == 0:
            return 0
        
        # Renyi entropy (order 2)
        probs_sq = sum((c/total)**2 for c in token_counts.values())
        renyi_2 = -math.log2(probs_sq) if probs_sq > 0 else 0
        
        # Normalize by log2(vocab_size)
        vocab_size = max(token_counts.keys()) + 1 if token_counts else 1
        return renyi_2 / math.log2(vocab_size) if vocab_size > 1 else 0
    
    def full_report(self, language_datasets: dict) -> dict:
        """
        Generate full evaluation report across languages.
        
        language_datasets: {lang_name: [text1, text2, ...]}
        """
        results = {}
        reference_lang = "English"
        reference_tokens = None
        
        for lang, texts in language_datasets.items():
            metrics = {
                'fertility': self.fertility(texts),
                'cpt': self.characters_per_token(texts),
                'wfr': self.word_fragmentation_rate(texts),
                'coverage': self.vocabulary_coverage(texts),
                'renyi': self.renyi_efficiency(texts),
            }
            
            total_tokens = sum(len(self.tokenizer.encode(t)) for t in texts)
            if lang == reference_lang:
                reference_tokens = total_tokens
            
            if reference_tokens is not None:
                metrics['parity'] = total_tokens / reference_tokens
            
            results[lang] = metrics
        
        return results
    
    def print_report(self, language_datasets: dict):
        """Pretty-print the evaluation report."""
        results = self.full_report(language_datasets)
        
        print(f"\n{'Language':<12} {'Fertility':>10} {'CPT':>6} {'WFR%':>6} "
              f"{'Coverage%':>10} {'Parity':>8}")
        print("-" * 60)
        
        for lang, m in results.items():
            print(f"{lang:<12} {m['fertility']:>10.2f} {m['cpt']:>6.2f} "
                  f"{m.get('wfr', 0)*100:>5.1f}% "
                  f"{m.get('coverage', 0)*100:>9.1f}% "
                  f"{m.get('parity', 1.0):>8.2f}x")
```

### 5.2 Extrinsic Evaluation

```python
# Extrinsic: evaluate tokenizer via downstream task performance
# Best proxy: per-language perplexity on held-out test set

def compare_tokenizers_by_perplexity(
    tokenizers: dict,  # {name: tokenizer}
    model,             # trained language model
    test_texts: dict,  # {lang: [texts]}
) -> dict:
    """
    Compare tokenizers by downstream perplexity.
    Lower perplexity = tokenizer works better for this language.
    
    NOTE: Requires a trained model — expensive!
    Prefer intrinsic metrics for fast iteration.
    """
    results = {}
    for tok_name, tokenizer in tokenizers.items():
        results[tok_name] = {}
        for lang, texts in test_texts.items():
            total_loss = 0
            total_tokens = 0
            for text in texts:
                tokens = tokenizer.encode(text)
                # Forward pass through LM
                loss = model.compute_loss(tokens)
                total_loss += loss * len(tokens)
                total_tokens += len(tokens)
            
            ppl = math.exp(total_loss / total_tokens)
            results[tok_name][lang] = ppl
    
    return results
```

---

## 🔬 Designing Tokenizer Experiments

### Experiment 1: Vocabulary Size Ablation

```python
# Question: What's the optimal vocabulary size for a multilingual model?

vocab_sizes = [8_000, 16_000, 32_000, 64_000, 128_000, 256_000]

for v in vocab_sizes:
    # 1. Train tokenizer with this vocab size
    # 2. Measure fertility across languages
    # 3. Train small LM (~100M params) with this tokenizer
    # 4. Measure perplexity per language
    # 5. Measure training FLOPs (proportional to sequence length)
    pass

# Expected finding:
# Larger vocab → lower fertility → fewer FLOPs per step
# But larger vocab → larger embedding table → more parameters
# Sweet spot: ~64k-128k for multilingual models
```

### Experiment 2: Training Data Balance Ablation

```python
# Question: How does training data balance affect tokenizer fairness?

alpha_values = [0.3, 0.5, 0.7, 1.0]  # sampling exponents

for alpha in alpha_values:
    # 1. Sample training data with this alpha
    # 2. Train tokenizer (Unigram, 64k vocab)
    # 3. Measure parity ratio across 10 languages
    # 4. Train small LM, measure downstream performance
    pass

# Expected finding:
# alpha=1.0: English dominates, Tamil gets 50-100 vocab slots
# alpha=0.3: More balanced, Tamil gets 1000-2000 slots
# Trade-off: Better Tamil → slightly worse English
```

### Experiment 3: BPE vs Unigram for Morphology

```python
# Question: Which algorithm better respects morphology?

from datasets import load_dataset

def morphological_alignment_score(tokenizer, word_morpheme_pairs: list) -> float:
    """
    MorphScore: fraction of morpheme boundaries that align with token boundaries.
    
    word_morpheme_pairs: [(word, [morpheme1, morpheme2, ...]), ...]
    """
    aligned = 0
    total = 0
    
    for word, morphemes in word_morpheme_pairs:
        tokens = tokenizer.encode(word, add_special_tokens=False, out_type=str)
        
        # Find morpheme boundary positions in word
        morph_boundaries = set()
        pos = 0
        for morpheme in morphemes[:-1]:  # all but last
            pos += len(morpheme)
            morph_boundaries.add(pos)
        
        # Find token boundary positions in word
        token_boundaries = set()
        pos = 0
        for token in tokens[:-1]:
            pos += len(token.replace('▁', '').replace('##', ''))
            token_boundaries.add(pos)
        
        # Count aligned boundaries
        for boundary in morph_boundaries:
            total += 1
            if boundary in token_boundaries:
                aligned += 1
    
    return aligned / total if total > 0 else 0
```

---

## 🐛 Debugging Tokenizer Issues in Production

### Issue 1: Inconsistent Tokenization

```python
def debug_tokenization(tokenizer, text1: str, text2: str):
    """Check why two visually similar texts tokenize differently."""
    
    t1_tokens = tokenizer.encode(text1, out_type=str)
    t2_tokens = tokenizer.encode(text2, out_type=str)
    
    print(f"Text 1: {text1!r}")
    print(f"  Code points: {[f'U+{ord(c):04X}' for c in text1]}")
    print(f"  Bytes: {text1.encode('utf-8').hex()}")
    print(f"  Tokens: {t1_tokens}")
    
    print(f"\nText 2: {text2!r}")
    print(f"  Code points: {[f'U+{ord(c):04X}' for c in text2]}")
    print(f"  Bytes: {text2.encode('utf-8').hex()}")
    print(f"  Tokens: {t2_tokens}")
    
    if t1_tokens != t2_tokens:
        print("\n⚠️  INCONSISTENT TOKENIZATION DETECTED")
        print("Possible causes:")
        print("  - Different Unicode normalization forms")
        print("  - Homoglyph characters")
        print("  - Different whitespace characters")

# Test for the classic é problem
import unicodedata
text_nfc = "café"  # precomposed é
text_nfd = unicodedata.normalize('NFD', text_nfc)  # decomposed é
# debug_tokenization(tokenizer, text_nfc, text_nfd)
```

### Issue 2: Unexpected Token Splits

```python
def explain_tokenization(tokenizer, word: str, merge_rules: list = None):
    """
    Show step-by-step how a word gets tokenized.
    Works for SentencePiece tokenizers.
    """
    tokens = tokenizer.encode(word, out_type=str)
    ids = tokenizer.encode(word)
    
    print(f"\nWord: {word!r}")
    print(f"Final tokens: {tokens}")
    print(f"Token IDs: {ids}")
    
    # Character frequency analysis
    print(f"\nCharacter breakdown:")
    for char in word:
        cp = ord(char)
        cat = unicodedata.category(char)
        print(f"  {char!r} (U+{cp:04X}, {cat}): "
              f"{len(char.encode('utf-8'))} bytes")
    
    print(f"\nFragmentation: {len(tokens)} tokens for {len(word)} chars "
          f"= {len(tokens)/len(word):.1f} tokens/char")
```

### Issue 3: OOV / Unknown Tokens

```python
def find_oov_words(tokenizer, text: str, unk_id: int = 0) -> list:
    """
    Find words that produce unknown tokens.
    (For tokenizers with explicit UNK handling)
    """
    oov_words = []
    for word in text.split():
        tokens = tokenizer.encode(word)
        if unk_id in tokens:
            oov_words.append(word)
    return oov_words
```

---

## 📊 The Researcher's Complete Toolkit

### Metrics Cheat Sheet

```
INTRINSIC (fast, no LM needed):
  ✓ Fertility = tokens/word (lower = better, target < 2.0 for each language)
  ✓ CPT = chars/token (higher = better, target > 3.0 for all languages)
  ✓ WFR = % words fragmented (lower = better, target < 30%)
  ✓ Parity = token ratio vs reference language (closer to 1.0 = fairer)
  ✓ Coverage = % words as single token (higher = better)
  ✓ Renyi efficiency = vocabulary utilization (higher = better)
  ✓ MorphScore = morphological boundary alignment (higher = better)

EXTRINSIC (slow, needs trained LM):
  ✓ Per-language perplexity on test set
  ✓ Downstream task performance (NER, translation, QA) per language
  ✓ Inference latency by language
  ✓ Training cost by language (FLOPs)
```

---

## 🎯 Key Research-Level Insights

### Insight 1: Fertility-Performance Relationship
<cite index="16-1">Fertility (average number of tokens per word) is often considered a necessary condition for better tokenization.</cite>
But it's a necessary, not sufficient condition. A tokenizer can have low fertility but split at semantically wrong boundaries.

### Insight 2: Vocabulary Allocation is Zero-Sum
In a 32k tokenizer:
- Every token slot for English prefixes = one fewer slot for Tamil syllables
- This is a **resource allocation problem** with equity implications

### Insight 3: Training Data = Hidden Variable
<cite index="13-1">Subword-based tokenizers trained on multilingual corpora tend to generate fewer tokens per word for dominant languages. This produces tokenization premiums of up to 10-15× in some cases.</cite>

The fix is NOT algorithmic alone — you must fix the **training data distribution**.

### Insight 4: Tokenizer Fairness ≠ Task Fairness
A tokenizer can have good parity ratio yet still produce poor downstream results for a language if:
- The tokens don't align with linguistic structure
- The model hasn't seen enough data in that language during LM training

---

## 🔭 Open Research Frontiers

| Problem | Current State | Active Research |
|---|---|---|
| **Morphology-aware tokenization** | MorphBPE (2025), rule-based hybrids | MorphScore, guided merges |
| **Token-free architectures** | ByT5, CANINE work but expensive | Efficient byte models |
| **Jointly-trained tokenization** | Early experiments | End-to-end differentiable tokenizers |
| **Dynamic vocabulary** | Not yet solved | VQ-VAE approaches |
| **Tokenizer fairness** | MAGNET (2024), cluster-based | Parity-constrained training |
| **Cross-lingual transfer** | Trans-tokenization (2024) | Zero-shot adaptation |
| **Tokenization for arithmetic** | Known to hurt math | Special digit tokenization |
| **Tokenization for code** | Code-specific models help | Mixed code-text tokenizers |

---

## 📚 Master Reading List

### Foundational Papers (Must Read)
1. **Gage (1994)** — Original BPE compression algorithm
2. **Sennrich et al. (ACL 2016)** — BPE for NMT [*arXiv:1508.07909*]
3. **Kudo (2018)** — Subword Regularization / Unigram LM [*arXiv:1804.10959*]
4. **Kudo & Richardson (2018)** — SentencePiece [*arXiv:1808.06226*]
5. **Radford et al. (2019)** — GPT-2, Byte-Level BPE

### Advanced Papers
6. **Provilkov et al. (2020)** — BPE-Dropout [*arXiv:1910.13267*]
7. **Rust et al. (2021)** — How Good Is Your Tokenizer? [*ACL 2021*]
8. **Xue et al. (2022)** — ByT5: Byte-Level T5 [*arXiv:2105.13626*]
9. **Clark et al. (2022)** — CANINE: Tokenization-Free [*arXiv:2103.06874*]

### Fairness & Multilingual
10. **Ahia et al. (2023)** — Do All Languages Cost the Same? [*EMNLP 2023*]
11. **Petrov et al. (2023)** — Language Model Tokenizers Introduce Unfairness [*arXiv:2305.15425*]
12. **Zouhar et al. (2023)** — Tokenization and the Noiseless Channel [*ACL 2023*]
13. **Blasi et al. (2024)** — MAGNET: Multilingual Fairness [*NeurIPS 2024*]

### Recent (2024-2025)
14. **Remy et al. (2024)** — Trans-Tokenization and Cross-lingual Vocabulary Transfer
15. **Ali et al. (2024)** — Tokenization Influence Study
16. **MorphBPE (2025)** — Morphology-Aware Tokenizer

---

## 💻 Quick-Reference: Tokenizer Selection Guide

```
Which tokenizer should you use?

IF training English-only LLM:
  → GPT-2 style Byte-Level BPE, 50k vocab
  
IF training multilingual LLM:
  → SentencePiece Unigram, 128k-256k vocab
  → character_coverage = 0.9999
  → byte_fallback = True
  → Training data: balanced with α = 0.3-0.5
  
IF training code LLM:
  → BPE with code-specific pre-tokenization
  → Larger vocab (100k+) for identifiers
  → split_digits = True
  
IF training multilingual + code:
  → Large Unigram (256k vocab)
  → Separate code split patterns
  → Balance general + code data

IF you need zero OOV guarantee:
  → Byte-Level BPE (GPT-2/GPT-4 style)
  
IF you need morphological accuracy:
  → MorphBPE or Unigram LM with morphological constraints
```

---

## 🏆 Summary: The Tokenization Mindset

Think of every design choice as a **tradeoff in a multi-dimensional space**:

```
┌─────────────────────────────────────────────────────┐
│            TOKENIZER DESIGN SPACE                   │
│                                                     │
│  Efficiency ←────────────────────→ Fairness         │
│  (few tokens)                   (equal treatment)   │
│                                                     │
│  Robustness ←────────────────────→ Compression      │
│  (never OOV)                    (short sequences)   │
│                                                     │
│  Morphology ←────────────────────→ Speed            │
│  (linguistic)                   (fast training)     │
│                                                     │
│  Monolingual ←───────────────────→ Multilingual     │
│  (best for 1)                   (good for all)      │
└─────────────────────────────────────────────────────┘
```

**No tokenizer wins on all axes simultaneously.** Research is about understanding these tradeoffs and designing systems that make principled choices for their target use case.

The best tokenizer researchers are the ones who:
1. **Measure everything** — don't assume, test
2. **Think multilingually** — every design choice has a different impact across languages
3. **Trace to information theory** — what's the entropy of the task? Is the tokenizer aligned?
4. **Consider fairness** — who loses when you make this design choice?
5. **Stay curious about alternatives** — token-free models may be the future

---

*End of Tokenization Foundations Notes*

*These notes cover: BPE, Unigram LM, SentencePiece, Byte-Level Tokenization, Multilingual Fragmentation, Unicode Normalization, and Tokenizer Evaluation — from story-level to research-level.*
