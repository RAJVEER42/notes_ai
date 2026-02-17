# Chapter 6: Why Tamil, Arabic, CJK Fragment More 🌏

> **"LLMs speak English natively. Every other language is a second language — some barely spoken at all."**

---

## 🧒 Story for a 5-Year-Old

Imagine the AI went to school in America and only read English books. It learned that "cat" is one piece, "the" is one piece, "running" is one piece. Now you show it a Tamil word like "படிக்கிறேன்" (I am studying). It's never seen this before! So it has to cut it into tiny tiny bits — almost letter by letter. It takes WAY more pieces to write the same thing. That's not fair.

---

## 💡 The Core Problem

Modern LLM tokenizers (BPE, Unigram) are trained on corpora that are:
- **80-95% English and European languages** (Common Crawl, Wikipedia bias)
- **Latin script dominated**
- **Morphologically simple** (compared to Tamil, Arabic, Turkish)

Result: The tokenizer "learns" English-shaped tokens efficiently. Non-Latin scripts get poor representation → **over-fragmentation**.

---

## 📊 The Numbers: Same Meaning, Very Different Token Count

**Sentence: "I am studying machine learning"**

```
Language    Script      Text                              Tokens
─────────────────────────────────────────────────────────────────
English     Latin       "I am studying machine learning"    5
French      Latin       "J'apprends l'apprentissage auto"   8
German      Latin       "Ich lerne maschinelles Lernen"     7
Chinese     CJK         "我在学习机器学习"                   5-7
Japanese    CJK+Kana    "私は機械学習を勉強しています"         9-12
Arabic      Arabic      "أنا أدرس التعلم الآلي"             6-10
Tamil       Tamil       "நான் இயந்திர கற்றல் படிக்கிறேன்"   15-20
Turkish     Latin+rich  "Makine öğrenimini öğreniyorum"     8-12
```

**Tamil requires 3-4x more tokens than English for equivalent content.**

Using GPT-4's tokenizer (Petrov et al., 2023):
```
Language    Tokens per sentence (normalized to English = 1.0)
English     1.0×
Russian     1.3×
Chinese     1.4×
Arabic      1.7×
Hindi       2.0×
Tamil       3.5×
Yoruba      5.0×
```

---

## 🔤 Why CJK (Chinese, Japanese, Korean) Fragments

### 1. No Whitespace Word Boundaries
```
Chinese: 机器学习  (Machine Learning)
         机 器 学 习
         ↑  Each character is a "morpheme" by itself
         
Problem: BPE was trained on languages WHERE spaces separate words.
         For Chinese, BPE has no natural pre-tokenization boundary.
         
Result: Either whole characters become tokens (if frequent enough)
        or they get split at byte level
```

### 2. Large Character Set
Chinese has **~50,000 unique characters** in Unicode (CJK Unified Ideographs).
Only ~3,500 are used regularly, but even that's huge.

In a 32k-vocab tokenizer dominated by English:
```
Space for Chinese chars: ~3,000-5,000 vocabulary slots
Number of common Chinese chars: ~3,500
→ Some characters have no token → byte fallback → fragmentation
```

### 3. No Morphological Composition
In English: "run" + "ning" = "running" → BPE can merge these
In Chinese: 机 + 器 = 机器 → these combine in meaning but not in a systematic way

```
Japanese example:
  食べています (I am eating)
  
  食べ    = eat (stem)
  て      = te-form (connector)
  い      = progressive marker  
  ます    = polite ending
  
  = FOUR morphemes fused into one verb form
  BPE sees this as an opaque string, not morphemes
```

---

## 📜 Why Arabic Fragments

### 1. Diacritics (Harakat/Tashkeel)
Arabic has optional vowel marks called **diacritics**:
```
كتب = k-t-b = "books" (unvocalized)
كُتُبٌ = ku-tu-bun = "books" (vocalized)

With diacritics = different byte sequences = different tokens!
Even though they're the same word in meaning.
```

### 2. Letter Shape Changes (Ligatures)
Arabic letters change shape depending on position in word:
```
ب (ba) has 4 forms:
  Isolated: ب
  Initial: بـ  
  Medial: ـبـ
  Final: ـب

Each is technically the same Unicode code point (U+0628)
but VISUALLY different — rendering engine handles this
```

### 3. Root-Pattern Morphology (Trilateral Roots)
Arabic uses a unique morphological system where 3-consonant roots + patterns = words:
```
Root: ك-ت-ب (k-t-b) = "write"

kataba  = he wrote
kātibu  = writer  
maktaba = library (place of writing)
kitābu  = book
maktūb  = written thing / letter

ALL from same 3 consonants! Each is a different word.
BPE treats each surface form independently → every pattern is rare → fragmentation
```

### 4. Arabic Script Properties
```
Arabic is:
  Right-to-left (RTL)
  Cursive (letters connect)
  28 consonants + vowel markers as separate Unicode combining chars
  No capital/lowercase (unlike Latin)
```

---

## 🔡 Why Tamil Fragments Most

Tamil is particularly poorly served for multiple reasons:

### 1. Agglutinative Morphology
Tamil stacks morphemes onto a base word:
```
வருகிறேன் = varu + kiṟ + ēn
          = come + PRES + 1SG
          = "I am coming"

படிக்கிறேன் = paṭi + kk + iṟ + ēn
            = study + CAUSATIVE + PRESENT + 1SG
            = "I am studying"

These can get even longer:
படித்திருக்கிறேன் = "I have been studying" (complex perfect progressive)
```

A single Tamil word can have **10+ morphemes** and encode a full English sentence.

### 2. Consonant Clusters (Puṇarcci)
Tamil uses special combining characters for consonant clusters:
```
க + ் = க் (ka + pulli = k consonant without vowel)
க் + க = க்க (cluster kk)

These are MULTIPLE Unicode code points rendering as ONE glyph
```

### 3. Abugida Script
Tamil is an **abugida** — consonant-based script where vowels modify the base consonant shape:
```
க = ka
கி = ki  (vowel ி changes the glyph appearance)
கீ = kī
கு = ku
கூ = kū

Each vowel marker is a combining Unicode character
→ Each Tamil "syllable" = 1-3 Unicode code points
→ Each Unicode code point = 1-3 bytes
→ Tamil word = many bytes → many tokens after fragmentation
```

### 4. Training Data Scarcity
Tamil has ~78 million speakers but:
- Common Crawl Tamil data: ~0.1% of corpus
- English data: ~50-60% of corpus
- Tokenizer vocabulary allocates space proportional to training data

```
Vocabulary allocation (approximate for 32k-vocab English-heavy tokenizer):
  English tokens: ~25,000
  European languages: ~5,000
  CJK: ~1,500  
  Arabic: ~300
  Tamil: ~50-100  ← catastrophically underrepresented!
  All other: remaining
```

---

## 🎨 ASCII Diagram: Tokenization Comparison

```
Sentence: "I love learning" ↔ Tamil equivalent

English:
  [I] [▁love] [▁learning]
   ↑      ↑         ↑
  3 tokens — compact, meaningful

Tamil: நான் கற்றலை நேசிக்கிறேன்  
  [▁ந] [ான்] [▁க] [ற்] [ற] [லை] [▁ந] [ேசி] [க்] [கி] [றே] [ன்]
    ↑    ↑    ↑   ↑   ↑   ↑    ↑    ↑    ↑    ↑    ↑    ↑
   12+ tokens — fragmented, loses morphological coherence
```

---

## 📊 Token Fragmentation Metrics

| Metric | English | Chinese | Arabic | Tamil |
|---|---|---|---|---|
| **Fertility** (tokens/word) | 1.1 | 1.8 | 2.5 | 4.2 |
| **CPT** (chars/token) | 4.5 | 1.5 | 1.2 | 0.8 |
| **Parity** (vs English) | 1.0× | 1.4× | 1.7× | 3.5× |
| **WFR** (% words fragmented) | 15% | 60% | 75% | 90% |

*(Approximate values based on Petrov et al. 2023, Ahia et al. 2023)*

---

## 💰 Socio-Technical Implications

### 1. API Cost Inequity
<cite index="12-1">Commercial services charge users per token. These discrepancies lead to users of some languages paying at least 2.5 times more than English users for equivalent content.</cite>

```
Same paragraph in different languages:
  English user pays:  $0.001 (1000 tokens)
  Tamil user pays:    $0.003 (3000 tokens for same content)
  = 3× cost penalty for being Tamil-speaking
```

### 2. Context Window Inequity
With 128k context window:
```
English user can fit: 128k tokens ÷ 1.1 = ~116k words
Tamil user can fit:   128k tokens ÷ 4.2 = ~30k words
= Tamil users have ~¼ the effective context window
```

### 3. Performance Gap
<cite index="13-1">Bias toward high-resource languages: tokens for underrepresented languages may be over-fragmented, inflating sequence lengths, processing time, and computational cost, with downstream consequences for model fairness and accessibility.</cite>

When Tamil is fragmented into meaningless sub-character pieces:
- The model **can't build coherent representations** of Tamil morphology
- Downstream tasks (translation, QA, generation) perform worse
- Tamil NLP remains "second class" despite large speaker population

### 4. Real-Time Applications
Fragmented text = more tokens = more inference time = higher latency for non-English users in time-critical applications (emergency response, real-time translation).

---

## 🔧 Proposed Solutions (Research-Level)

| Solution | Description | Paper |
|---|---|---|
| **Larger vocab** | 128k-256k vocab for multilingual models | LLaMA-2 (32k→128k) |
| **Cluster-based tokenization** | Train separate sub-tokenizers per language cluster | Chung et al., 2020 |
| **Morphology-aware BPE** | Segment at morpheme boundaries | MorphBPE, 2025 |
| **MAGNET** | Adaptive gradient-based tokenization | Blasi et al., 2024 |
| **Byte-level models** | Bypass tokenization entirely | ByT5, CANINE |
| **Balanced training data** | Fix the root cause — more multilingual data | mT5, BLOOM |
| **Language-specific tokenizers** | Train separate tokenizer per language | Krutrim (Indian languages) |

---

## 💻 Code: Measuring Fragmentation

```python
def measure_fragmentation(text: str, tokenizer) -> dict:
    """
    Measure tokenization quality metrics for a text.
    
    tokenizer: any HuggingFace tokenizer
    """
    words = text.split()
    tokens = tokenizer.encode(text)
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    # Filter special tokens
    real_tokens = [t for t in token_strings 
                   if t not in ['<s>', '</s>', '[CLS]', '[SEP]']]
    
    fertility = len(real_tokens) / len(words) if words else 0
    
    # Average chars per token
    total_chars = sum(len(w) for w in words)
    cpt = total_chars / len(real_tokens) if real_tokens else 0
    
    # Word fragmentation rate: % words split into multiple tokens
    fragmented = 0
    for word in words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(word_tokens) > 1:
            fragmented += 1
    wfr = fragmented / len(words) if words else 0
    
    return {
        'text': text,
        'words': len(words),
        'tokens': len(real_tokens),
        'fertility': round(fertility, 2),
        'cpt': round(cpt, 2),
        'wfr': round(wfr * 100, 1),
    }

# Compare across languages (requires transformers library)
try:
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    sentences = {
        "English": "I am studying machine learning every day",
        "German":  "Ich studiere maschinelles Lernen jeden Tag",
        "Chinese": "我每天都在学习机器学习",
        # "Tamil":  "நான் ஒவ்வொரு நாளும் இயந்திர கற்றல் படிக்கிறேன்",
    }
    
    print(f"{'Language':<10} {'Words':>6} {'Tokens':>7} {'Fertility':>10} {'CPT':>6} {'WFR%':>6}")
    print("-" * 50)
    
    for lang, text in sentences.items():
        m = measure_fragmentation(text, tokenizer)
        print(f"{lang:<10} {m['words']:>6} {m['tokens']:>7} {m['fertility']:>10.2f} {m['cpt']:>6.2f} {m['wfr']:>5.1f}%")

except ImportError:
    print("Install: pip install transformers")
    print("Then run this code to see real fragmentation metrics")
```

---

---

# Chapter 7: Unicode Normalization (NFC, NFKC) 🔣

> **"The same visual character can be stored in multiple ways. If you don't normalize, your tokenizer sees different 'characters' that look identical."**

---

## 🧒 Story for a 5-Year-Old

Imagine writing the letter "é" (e with accent). You could write it two ways: as one special symbol (é), or as the letter e followed by a tiny floating accent mark (e + ́). They LOOK the same, but inside the computer they're stored differently!

If the AI sees "é" one day and "e + accent" the next day, it thinks they're different! That's a big problem. Normalization says: "Let's always pick ONE way to write each character, so there's no confusion."

---

## 💡 The Core Problem

Unicode allows multiple representations of the same visual character:

```
"é" can be stored as:

Option 1 (NFC - composed):
  U+00E9 = LATIN SMALL LETTER E WITH ACUTE
  → 1 code point, 2 bytes in UTF-8 (0xC3 0xA9)

Option 2 (NFD - decomposed):
  U+0065 = LATIN SMALL LETTER E
  U+0301 = COMBINING ACUTE ACCENT
  → 2 code points, 3 bytes in UTF-8 (0x65 0xCC 0x81)
```

**If not normalized:**
```
"café" (NFC) → ["caf", "é"]          → token IDs [1234, 5678]
"café" (NFD) → ["caf", "e", "̈"]     → token IDs [1234, 9012, 3456]
                       ↑ completely different tokenization!
```

---

## 📐 Unicode Code Points and Combining Characters

A **Unicode code point** is a number assigned to a character: `U+XXXX`

**Combining characters** are code points that MODIFY the preceding character (they have no standalone visual form):
```
U+0301 = COMBINING ACUTE ACCENT (◌́)
U+0308 = COMBINING DIAERESIS (◌̈)   
U+0300 = COMBINING GRAVE ACCENT (◌̀)
U+0328 = COMBINING OGONEK (◌̨)

These stack with base characters:
  e + U+0301 = é
  a + U+0308 = ä
  e + U+0300 = è
  a + U+0328 = ą
```

---

## 🔢 The Four Normalization Forms

### NFC — Canonical Decomposition, then Canonical Composition
**Meaning**: Decompose, then re-compose into precomposed forms where possible.

```
é (NFD: e + ́) → NFC → é (precomposed U+00E9)
ä (NFD: a + ̈) → NFC → ä (precomposed U+00E4)
```
**Best for**: Storage efficiency, compatibility with most systems. **Used by macOS, most web content.**

### NFD — Canonical Decomposition
**Meaning**: Decompose all precomposed characters into base + combining marks.

```
é (U+00E9) → NFD → e (U+0065) + ́ (U+0301)
ä (U+00E4) → NFD → a (U+0061) + ̈ (U+0308)
```
**Best for**: String comparison, text analysis. **Used by some filesystems (HFS+).**

### NFKC — Compatibility Decomposition, then Canonical Composition
**Meaning**: Like NFC but ALSO normalizes "compatibility" characters (visually similar but semantically different).

```
ﬁ (U+FB01, fi ligature) → NFKC → fi (two separate chars)
① (U+2460, circled 1)   → NFKC → 1
ℌ (U+210C, Fraktur H)   → NFKC → H
Ａ (U+FF21, fullwidth A) → NFKC → A (regular A)
```
**Best for**: Tokenizer training — removes spurious distinctions. **Used by BERT, most LLM tokenizers.**

### NFKD — Compatibility Decomposition
**Meaning**: Like NFD but also normalizes compatibility characters.

```
ﬁ → fi
① → 1
```

---

## 🎨 ASCII Diagram

```
Original: "café résumé"
           │
           ▼
    Analyze code points:
    c  a  f  é(U+00E9)  ▁  r  é  s  u  m  é
           │
     NFD Decompose:
    c  a  f  e + ́(U+0301)  ▁  r  e + ́  s  u  m  e + ́
           │
     NFC Re-compose:
    c  a  f  é(U+00E9)  ▁  r  é  s  u  m  é
           │
     NFKC also converts:
    ﬁ → fi,  ① → 1,  Ａ → A
```

---

## 🔢 Impact on Token Counts

```python
import unicodedata

word = "café"

# NFC version
nfc = unicodedata.normalize('NFC', word)
print(f"NFC: {nfc!r}")
print(f"NFC code points: {[hex(ord(c)) for c in nfc]}")
print(f"NFC bytes: {nfc.encode('utf-8').hex()}")
print(f"NFC length: {len(nfc)} chars, {len(nfc.encode('utf-8'))} bytes\n")

# NFD version
nfd = unicodedata.normalize('NFD', word)
print(f"NFD: {nfd!r}")
print(f"NFD code points: {[hex(ord(c)) for c in nfd]}")
print(f"NFD bytes: {nfd.encode('utf-8').hex()}")
print(f"NFD length: {len(nfd)} chars, {len(nfd.encode('utf-8'))} bytes")
```

Output:
```
NFC: 'café'
NFC code points: ['0x63', '0x61', '0x66', '0xe9']
NFC bytes: 636166c3a9
NFC length: 4 chars, 5 bytes

NFD: 'café'
NFD code points: ['0x63', '0x61', '0x66', '0x65', '0x301']
NFD bytes: 636166650xcc81
NFD length: 5 chars, 6 bytes
```

**Implication for tokenization:**
```
BPE tokenizer trained on NFC text:
  "café" (NFC) → tokens [caf, é]    ← correct, 2 tokens
  "café" (NFD) → tokens [caf, e, ̈]  ← wrong! 3 tokens, middle token may be [UNK]
```

---

## ⚠️ Why Normalization Must Happen BEFORE Training

```
WRONG pipeline:
  Raw corpus (mixed NFC/NFD) → tokenizer training
  Result: tokenizer has BOTH "é" (U+00E9) AND "e+̈" as patterns
          → inconsistent tokenization
          → vocabulary wasted on duplicate entries
          → model confuses identical text

CORRECT pipeline:
  Raw corpus → Unicode normalization (NFKC) → tokenizer training
  Result: all "é" forms consolidated → consistent vocabulary
```

**Standard practice:**
- NFKC normalization before training (used by SentencePiece by default)
- Same normalization at inference time (critical!)
- Store normalization form in tokenizer config

---

## 🔐 Security Implications: Homoglyph Attacks

Unicode normalization is critical for security:

```
Homoglyphs — visually identical but different code points:

'A' (U+0041, Latin A)  vs  'А' (U+0410, Cyrillic A)
'о' (Latin o)          vs  'о' (Cyrillic o)  [same glyph!]
'l' (letter l)         vs  '1' (digit 1)     [similar glyph]

Attack: "pаypal.com" looks like "paypal.com" but Cyrillic 'a'!
```

**LLM security implications:**
```
Prompt: "Ignоre all instructions" (with Cyrillic 'о')
         → Bypasses exact-match filters
         → Tokenized differently than expected
         → May bypass safety measures

NFKC normalization catches some homoglyphs but not all!
```

```python
# Homoglyph detection example
def detect_homoglyphs(text: str) -> list:
    """Find characters that look like common ASCII but aren't."""
    suspicious = []
    for i, char in enumerate(text):
        cp = ord(char)
        name = unicodedata.name(char, 'UNKNOWN')
        
        # Check if it's in suspicious ranges
        if (0x0400 <= cp <= 0x04FF):  # Cyrillic
            if char.lower() in 'aeijopsx':  # Similar to Latin
                suspicious.append((i, char, f'Cyrillic U+{cp:04X}: {name}'))
    
    return suspicious

text = "pаypal.com"  # Contains Cyrillic 'а'
suspects = detect_homoglyphs(text)
print(f"Suspicious characters: {suspects}")
```

---

## 🔢 Tamil-Specific Unicode Complexity

Tamil has particularly complex Unicode behavior:

```
Tamil syllable: "க" = ka
Tamil syllable with vowel: "கி" = ki

"கி" is stored as:
  U+0B95 (க, ka consonant)
  U+0BBF (ி, i vowel sign - combining character)
  
= 2 code points but 1 visual syllable

With NFC/NFD: these don't change (already canonical)
With tokenizer: might be split between the 2 code points → broken syllable!
```

This is why Tamil fragmentation is so bad — even WITH normalization, the underlying character structure maps poorly to token boundaries.

---

## 📊 Normalization Impact Table

| Form | Decomposes? | Compatibility? | Use Case |
|---|---|---|---|
| **NFC** | No (recomposes) | No | Web, macOS storage |
| **NFD** | Yes | No | Text analysis, macOS filesystem |
| **NFKC** | No (recomposes) | Yes | **LLM training (recommended)** |
| **NFKD** | Yes | Yes | Analysis requiring compat normalization |

---

## 💻 Full Normalization Pipeline

```python
import unicodedata
import re

def normalize_for_tokenizer(text: str, form: str = 'NFKC') -> str:
    """
    Normalize text for consistent tokenization.
    
    1. Unicode normalization (NFKC by default)
    2. Remove control characters
    3. Normalize whitespace
    """
    # Step 1: Unicode normalization
    text = unicodedata.normalize(form, text)
    
    # Step 2: Remove control characters (except newlines/tabs)
    text = ''.join(
        ch for ch in text 
        if unicodedata.category(ch) != 'Cc' or ch in '\n\t\r'
    )
    
    # Step 3: Normalize whitespace (optional, language-dependent)
    text = re.sub(r'[ \t]+', ' ', text)  # collapse multiple spaces
    text = text.strip()
    
    return text


def compare_normalizations(text: str):
    """Compare all 4 normalization forms."""
    forms = ['NFC', 'NFD', 'NFKC', 'NFKD']
    
    print(f"Original: {text!r}")
    print(f"Original code points: {len(text)}, bytes: {len(text.encode('utf-8'))}\n")
    
    for form in forms:
        normalized = unicodedata.normalize(form, text)
        print(f"{form}: {normalized!r}")
        print(f"  Code points: {len(normalized)}, bytes: {len(normalized.encode('utf-8'))}")
        print(f"  Code point list: {[f'U+{ord(c):04X}' for c in normalized]}\n")


# Test with examples
test_texts = [
    "café",                    # é can be precomposed or decomposed
    "ﬁle",                     # fi ligature → fi after NFKC
    "①②③",                     # circled numbers → 1 2 3 after NFKC
    "Ａｌｇｏ",                  # fullwidth letters → ASCII after NFKC
    "é",                       # precomposed (NFC form)
]

for text in test_texts:
    print("="*60)
    compare_normalizations(text)
```

---

## 🔭 Open Research Problems

1. **Normalization for code**: Should code tokenizers normalize? `\n\t` vs spaces is meaningful
2. **Script-specific normalization**: Different scripts may need different normalization strategies  
3. **Security-aware tokenization**: Detecting homoglyph attacks before tokenization
4. **Handling denormalized inputs**: Robust tokenization even with mixed normalization in production
5. **Emoji and new Unicode**: Unicode adds ~100-200 new emoji/chars per release — how to handle tokenizer updates?

---

## 📌 Summary: Normalization Best Practices

```
1. ALWAYS normalize with NFKC before training tokenizer
2. Apply SAME normalization at inference time  
3. Document normalization form in tokenizer config
4. Watch out for:
   - macOS NFD filesystem output
   - Web scraping (mixed NFC/NFD)
   - User inputs (may be from any source)
5. Security: NFKC catches SOME homoglyphs but not all
6. For Tamil/Arabic: be especially careful with combining chars
```

---

*Next: Chapter 8 — How to Build and Evaluate a Tokenizer from Scratch →*
