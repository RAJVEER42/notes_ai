# Chapter 4: SentencePiece 🌐

> **"SentencePiece treats raw text as a sequence of Unicode characters — no language-specific rules needed."**  
> — Kudo & Richardson, 2018

---

## 🧒 Story for a 5-Year-Old

Imagine you're trying to learn to read Chinese, Japanese, Tamil, and English all at once. In English, words have spaces between them. But in Chinese and Japanese, there are NO spaces! How do you split things up?

SentencePiece says: "I won't make any assumptions about spaces. I'll just look at the raw text as a stream of characters and figure out the chunks myself." It's like learning to read without anyone telling you where words start and end.

---

## 💡 Why SentencePiece Exists

Traditional tokenizers assume:
- Text is pre-tokenized (split by whitespace)
- Words exist as meaningful units
- Latin scripts with clear word boundaries

These assumptions **break** for:
- **Japanese**: スシが食べたい (no spaces between words)
- **Chinese**: 我喜欢吃寿司 (no spaces)
- **Thai**: ฉันชอบกินซูชิ (no spaces)
- **Tamil**: நான் சுஷி சாப்பிட விரும்புகிறேன் (spaces, but complex morphology)

**SentencePiece** solves this by:
1. Treating whitespace as a regular character (normalizing it)
2. Working directly on raw Unicode characters
3. Not assuming any pre-tokenization
4. Running BPE or Unigram LM on the resulting character stream

---

## 🔧 Key Design Decision: The ▁ (LOWER ONE EIGHTH BLOCK) Character

SentencePiece replaces whitespace with `▁` (U+2581) and treats it as part of the vocabulary.

```
Input:  "Hello World how are you"
Stored: "▁Hello▁World▁how▁are▁you"

Tokens might be: ["▁Hello", "▁World", "▁how", "▁are", "▁you"]
            or: ["▁Hello", "▁Wor", "ld", "▁how", "▁are", "▁you"]
```

**Why this matters:**
- "World" in the middle of text vs "▁World" at start of a word → different tokens
- Allows perfect reconstruction: tokens → original text (lossless)
- No need for separate pre-tokenization step

```
ASCII Diagram:

Traditional pipeline:          SentencePiece pipeline:
                               
  Raw text                       Raw text
     │                              │
     ▼                              │
  Whitespace split                  │ (raw, no splitting)
     │                              ▼
     ▼                         Replace space with ▁
  Word list                         │
     │                              ▼
     ▼                         BPE or Unigram LM
  Subword tokenizer                 │
     │                              ▼
     ▼                         Tokens (▁ marks word boundaries)
  Tokens
```

---

## 🌏 Language Examples

### Japanese
```
Input:     日本語のトークン化
Raw chars: 日 本 語 の ト ー ク ン 化

Without SentencePiece: ??? (no spaces to split on!)
With SentencePiece:    ["▁日本語", "の", "▁トーク", "ン化"]
                    or ["▁日本", "語のトーク", "ン化"]  (different, still valid)
```

### Chinese
```
Input:     我喜欢吃寿司
Raw chars: 我 喜 欢 吃 寿 司

SentencePiece tokens: ["▁我", "喜欢", "吃", "寿司"]
```

### Tamil
```
Input:     நான் படிக்கிறேன்
SentencePiece: ["▁நான்", "▁படி", "க்கி", "றேன்"]
```

### English
```
Input:    "Hello World"
→ Stored: "▁Hello▁World"
→ Tokens: ["▁Hello", "▁World"]

# Note: ▁Hello ≠ Hello (different tokens!)
# "Hello" in middle of word: "saHello" → ["sa", "Hello"]
```

---

## 🔢 Mathematical Treatment

SentencePiece normalizes training text:

```
normalize(text) = unicode_normalize(replace(' ', '▁'))
```

Then trains BPE or Unigram LM directly on:
```
normalized_corpus = "▁Hello▁World▁how▁are▁you..."
```

The model sees `▁` as a normal character. Tokens that start with `▁` are "word-initial" by convention.

---

## ⚙️ System-Level Insight

### Why Google Used SentencePiece for T5, ALBERT, mT5

1. **Pipeline simplicity**: No language-specific tokenizer needed. One tool for all 101 languages in mT5.
2. **Reproducibility**: No dependency on language-specific tools (MeCab for Japanese, Jieba for Chinese, etc.)
3. **Raw byte-level fallback**: Unknown characters → individual bytes → never OOV
4. **Subword regularization**: Built-in Unigram LM sampling

### SentencePiece Configuration

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='mymodel',
    vocab_size=32000,
    model_type='bpe',          # or 'unigram'
    character_coverage=0.9995, # coverage of Unicode characters
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[BOS]',
    eos_piece='[EOS]',
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

text = "Hello World how are you"
tokens = sp.encode(text, out_type=str)
print(tokens)  # ['▁Hello', '▁World', '▁how', '▁are', '▁you']

ids = sp.encode(text, out_type=int)
print(ids)     # [123, 456, 789, 234, 567]

# Decode back
decoded = sp.decode(ids)
print(decoded)  # "Hello World how are you"

# Subword regularization (sampling)
samples = [sp.encode(text, out_type=str, 
                     enable_sampling=True, alpha=0.1, nbest_size=-1)
           for _ in range(5)]
```

---

## 📊 character_coverage Parameter

Critical for multilingual models:

```
character_coverage = 0.9995  # covers 99.95% of all Unicode characters in corpus

For Latin-script languages: 0.9995 is fine
For Indic languages: use 0.9999 or higher (many unique characters)
For emoji/rare chars: lower is okay (they get byte-fallback)
```

Characters NOT covered → decomposed into individual bytes (byte fallback).

---

## ⚠️ Common Mistakes

1. **Mixing normalized and unnormalized text**: Training on NFKC but testing on NFC → inconsistent tokens
2. **Forgetting ▁ in decoding**: Stripping tokens without replacing ▁ → lost spacing
3. **Wrong `character_coverage` for language**: Too low → OOV; too high → sparse vocabulary
4. **Using `model_type='bpe'` for multilingual**: Unigram is generally better for multilingual
5. **Treating ▁token ≠ token**: `"▁Hello" != "Hello"` — different token IDs!

---

## 🔭 Research Insight

SentencePiece's design enables **zero-shot cross-lingual transfer**: if a model is trained on multiple languages with one SentencePiece model, it can generalize to unseen languages at inference time — because the byte-level fallback ensures all text can be represented.

---

---

# Chapter 5: Byte-Level Tokenization & Byte Fallback 🔢

> **"Every piece of text — in any language, with any emoji — can be represented as a sequence of bytes. Bytes never fail."**  
> — GPT-2 (Radford et al., 2019)

---

## 🧒 Story for a 5-Year-Old

You know how computers secretly store everything as numbers? Even the letter "A" is actually the number 65 inside the computer. And the 🍕 emoji is several numbers: 240, 159, 141, 149.

Byte-level tokenization says: let's just use those raw numbers (0-255) as our "alphabet"! That way, ANYTHING you type — any emoji, any language, any weird symbol — can be expressed as these 256 building blocks. Nothing is ever "unknown."

---

## 💡 Intuition

### Why Byte-Level?

UTF-8 encodes all Unicode text as sequences of bytes (0-255). Every possible string can be expressed as a byte sequence. Therefore:

- **Vocabulary size = 256** (just the bytes)
- **No OOV ever** — anything can be expressed
- **Language agnostic** — bytes don't care about scripts

The downside: bytes are smaller than characters → longer sequences → more compute.

**Byte-level BPE** (GPT-2 style): Start with 256 byte tokens, then run BPE to merge frequent byte sequences into longer tokens.

---

## 🌐 UTF-8 Encoding

UTF-8 is a **variable-length encoding** for Unicode:

```
Character   Unicode    UTF-8 bytes
─────────────────────────────────
A           U+0041     41
é           U+00E9     C3 A9
€           U+20AC     E2 82 AC
中          U+4E2D     E4 B8 AD
😀          U+1F600    F0 9F 98 80

ASCII chars (0-127):  1 byte
Latin extended:       2 bytes
Most Asian scripts:   3 bytes
Emoji / rare:         4 bytes
```

```
ASCII Diagram: UTF-8 byte structure

1 byte:  0xxxxxxx                          (U+0000 to U+007F)
2 bytes: 110xxxxx 10xxxxxx                 (U+0080 to U+07FF)
3 bytes: 1110xxxx 10xxxxxx 10xxxxxx        (U+0800 to U+FFFF)
4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (U+10000 to U+10FFFF)
```

---

## 🔢 Walk-Through Examples

### Example 1: ASCII
```
"hello" → UTF-8 bytes → [104, 101, 108, 108, 111]
         → Byte tokens → ['h', 'e', 'l', 'l', 'o']
         → After BPE merges:
           h+e → he
           he+l → hel  
           hel+l → hell
           hell+o → hello
         → [hello]  (if common enough)
```

### Example 2: Emoji 🍕
```
🍕 → UTF-8 bytes → [0xF0, 0x9F, 0x8D, 0x95]
               → [240, 159, 141, 149]
               → byte tokens: ['Ġ', 'Ł', 'ŉ', 'ŕ']  (GPT-2 byte encoding)
               → might merge to: ['🍕']  (if frequent in training data)
               → or stay as 4 byte tokens (if rare)
```

### Example 3: Mixed Language
```
"Hello 世界 🌍"

English:  H e l l o    → 5 bytes → (merges to ~2 tokens)
Space:    ' '           → 1 byte
Chinese:  世(3B) 界(3B) → 6 bytes → (may merge to 1-2 tokens if common)
Space:    ' '           → 1 byte  
Emoji:    🌍(4B)        → 4 bytes → (if rare, stays 4 tokens)

Total: ~14 bytes → ~8-10 tokens after BPE
```

### Example 4: Rare Character
```
"ꩻ" (Cham script, Myanmar)
→ UTF-8: [0xEA, 0xA9, 0xBB] = [234, 169, 187]
→ 3 byte tokens (almost certainly won't be merged — too rare)
→ Result: 3 tokens for 1 character!
```

---

## 🎨 ASCII Diagram: Byte Fallback

```
Input character: "喜"

Standard tokenizer:
  "喜" in vocab? → YES → token ID 24853
                → NO  → [UNK] token!  ← BAD

Byte-level tokenizer:
  "喜" → UTF-8 bytes: [0xE5, 0x96, 0x9C]
       → byte tokens: [229, 150, 156]
       → 3 tokens (always works, never UNK)
```

---

## 📊 Token Efficiency: Language Comparison

GPT-4 tokenizer (cl100k_base):

```
Script          Chars/token    Example
─────────────────────────────────────────────
English         ~4 chars       "programming" = 1 token
German          ~3.5 chars     "programmierung" = 2 tokens
Chinese (CJK)   ~1.5 chars    "程序设计" = ~3 tokens
Arabic          ~1.2 chars    "برمجة" = ~4 tokens
Tamil           ~0.8 chars    "நிரலாக்கம்" = ~10 tokens
Emoji           ~0.25 chars   "🎉" = 1 token (if in vocab)
               
Relative cost (English = 1.0x):
  English:  1.0x
  German:   1.1x
  Chinese:  2.5x
  Arabic:   3.5x
  Tamil:    5-10x
```

---

## ⚖️ Trade-offs: Robustness vs Token Efficiency

```
┌────────────────────────────────────────────────────────┐
│                   BYTE-LEVEL BPE                       │
│                                                        │
│  PROS:                         CONS:                   │
│  ✓ Zero OOV ever               ✗ Longer sequences      │
│  ✓ Handles all languages        ✗ More compute cost    │
│  ✓ Handles all emoji            ✗ Less semantic per    │
│  ✓ Robust to typos             │  token initially      │
│  ✓ Simple (256 base tokens)    ✗ Byte boundaries may   │
│  ✓ No Unicode normalization    │  cut through chars    │
│    required                                            │
└────────────────────────────────────────────────────────┘
```

---

## 📈 Impact on Sequence Length

For the SAME content:

```
Standard BPE (32k vocab):  ~N tokens
Byte-level BPE (32k vocab): ~1.1N tokens for English
                            ~2-3N tokens for CJK scripts
                            ~4-10N tokens for Tamil/Arabic (unfamiliar chars)
```

This directly impacts:
- Attention FLOPs: O(n²) → longer = much more expensive
- Context window utilization: fewer "ideas" fit in same context
- Generation speed: more steps = slower

---

## 🔍 GPT-2 Byte-Level BPE Details

GPT-2 uses a specific byte encoding where all 256 bytes are mapped to printable Unicode characters:

```python
def bytes_to_unicode():
    """
    GPT-2's mapping of bytes to Unicode characters.
    Ensures all 256 bytes have a unique printable representation.
    """
    bs = (list(range(ord("!"), ord("~")+1)) +
          list(range(ord("¡"), ord("¬")+1)) +
          list(range(ord("®"), ord("ÿ")+1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# Space (byte 32) → 'Ġ'
# 'A' (byte 65) → 'A'
# Tab (byte 9) → 'ĉ'
```

Then BPE merges are run on this "bytified" text.

---

## 💻 Byte Tokenization Demo

```python
def text_to_bytes(text: str) -> list:
    """Convert text to UTF-8 bytes."""
    return list(text.encode('utf-8'))

def bytes_to_text(byte_list: list) -> str:
    """Convert bytes back to text."""
    return bytes(byte_list).decode('utf-8', errors='replace')

def analyze_languages():
    """Compare byte lengths across languages."""
    examples = {
        "English": "Hello World programming",
        "Chinese": "你好世界编程",
        "Arabic":  "مرحبا بالعالم",
        "Tamil":   "வணக்கம் உலகம்",
        "Emoji":   "🌍🎉🚀💻",
        "Japanese": "こんにちは世界",
        "Mixed":   "Hello 世界 🌍",
    }
    
    print(f"{'Language':<12} {'Text':<25} {'Chars':>6} {'Bytes':>6} {'B/C':>6}")
    print("-" * 60)
    
    for lang, text in examples.items():
        chars = len(text)
        byte_len = len(text.encode('utf-8'))
        ratio = byte_len / chars
        print(f"{lang:<12} {text:<25} {chars:>6} {byte_len:>6} {ratio:>6.2f}")

analyze_languages()

# Show UTF-8 byte breakdown
def show_utf8(char):
    """Show UTF-8 bytes for a character."""
    bts = char.encode('utf-8')
    hex_repr = ' '.join(f'{b:02X}' for b in bts)
    dec_repr = ' '.join(str(b) for b in bts)
    print(f"'{char}' (U+{ord(char):04X}): bytes [{hex_repr}] = [{dec_repr}]")

print("\nUTF-8 byte breakdown:")
for char in ['A', 'é', '中', '😀', 'ன']:
    show_utf8(char)
```

---

## 🔬 Research: ByT5 — Pure Byte-Level Transformer

**ByT5** (Xue et al., 2022) operates directly on raw bytes — no tokenization step at all!

```
Architecture:
  Input bytes → [byte embeddings] → long encoder sequence
                                  → compressed latent
                                  → shorter decoder sequence

Performance: Competitive with T5 on many benchmarks
Advantage:   Zero tokenization artifacts, perfect multilingual coverage
Disadvantage: 4-8x longer sequences → much more compute
```

Key finding: byte-level models are **more robust to noise, typos, and character-level attacks** than subword models.

---

## ⚠️ Common Mistakes

1. **Confusing characters and bytes**: `len("🌍")` = 1 char but 4 bytes in Python 3
2. **Naive byte splitting**: Not respecting UTF-8 multi-byte boundaries → garbled characters
3. **Ignoring the efficiency cost**: Byte-level models need longer context windows for same content
4. **Not using BPE on top of bytes**: Pure byte vocabulary (256 tokens) gives terrible efficiency; need BPE merges on top

---

## 🔭 Open Research Problems

1. **Efficient byte models**: Hierarchical byte processing (encode bytes → compress → process)
2. **Adaptive granularity**: Different "zoom levels" per token based on information content
3. **Cross-modal byte models**: Can byte-level models naturally handle binary data, images, audio?
4. **Learned byte groupings**: Learn optimal byte merges language-specifically

---

*Next: Chapter 6 — Why Tamil, Arabic, and CJK Fragment More →*
