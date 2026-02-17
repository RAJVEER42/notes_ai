# 🔤 Tokenization Foundations — Complete Study Notes
### *From Story Level to Research Level*

---

## 📚 File Index

| File | Topics | Key Concepts |
|---|---|---|
| `ch1_what_is_tokenization.md` | What is tokenization, why it exists | Information theory, compression, O(n²) attention cost, token metrics |
| `ch2_bpe.md` | Byte Pair Encoding | Merge rules, worked example (low/lower/lowest), BPE-Dropout, code impl |
| `ch3_unigram_lm.md` | Unigram Language Model | EM training, Viterbi, subword regularization, forward-backward algorithm |
| `ch4_sentencepiece_ch5_byte_level.md` | SentencePiece + Byte-Level BPE | ▁ marker, UTF-8, byte fallback, GPT-2 byte encoding, ByT5 |
| `ch6_multilingual_ch7_unicode.md` | Multilingual fragmentation + Unicode | Tamil/Arabic/CJK analysis, NFC/NFD/NFKC, homoglyph attacks, fairness |
| `ch8_building_evaluating_tokenizers.md` | Building + evaluating tokenizers | Full pipeline, 8 metrics, experiment design, debugging, research frontiers |

---

## ⚡ Quick Reference

### Algorithm Comparison
| Feature | BPE | Unigram LM | SentencePiece | Byte-Level BPE |
|---|---|---|---|---|
| Direction | Bottom-up merge | Top-down prune | Wrapper (BPE or Unigram) | Bottom-up merge |
| Objective | Greedy frequency | Max likelihood | Depends on model_type | Greedy frequency |
| Deterministic? | Yes | Viterbi: Yes / Training: No | Depends | Yes |
| OOV possible? | Yes (use byte fallback) | Yes (char fallback) | No (byte_fallback=True) | Never |
| Whitespace handling | Pre-tokenize first | Remove / add ▁ | Raw text, add ▁ | Pre-tokenize |
| Language agnostic? | Somewhat | Better | Yes | Yes |
| Used in | GPT-2/3/4 | LLaMA-2+, T5, mT5 | T5, ALBERT, LLaMA | GPT-2, Codex |

### Key Formulas
```
Fertility           = #tokens / #words           (lower is better)
CPT                 = #chars / #token            (higher is better)  
WFR                 = #fragmented_words / #words (lower is better)
Parity ratio        = lang_tokens / pivot_tokens (1.0 = perfectly fair)
Attention FLOPs     ≈ 4 · n² · d · L
BPE training cost   = O(N · V)
Unigram EM cost     = O(N · n · L_max) per iteration
```

### Unicode Normalization Decision Tree
```
Are inputs from multiple sources (web, user, files)?
  → YES: Always apply NFKC before tokenizer
  
Do you need to handle compatibility ligatures (ﬁ → fi)?
  → YES: Use NFKC or NFKD
  
Do you need morphological analysis?
  → YES: Use NFD (base chars + combining marks separately)
  
Default for LLM tokenizer training:
  → NFKC ✓
```

### Tokenizer Selection
```
English only:              Byte-Level BPE, 50k vocab
Multilingual:              SentencePiece Unigram, 128k-256k vocab
Code:                      BPE, 100k+ vocab, split_digits=True  
Zero OOV required:         Byte-Level BPE (GPT-2 style)
Morphologically rich:      Unigram LM or MorphBPE
```

---

## 📊 The Fairness Numbers (Approximate)

Normalized token counts for equivalent text (English = 1.0×):

```
English    1.0×  ████
French     1.2×  █████
German     1.3×  █████
Russian    1.4×  ██████
Chinese    1.5×  ██████
Arabic     1.8×  ████████
Hindi      2.0×  █████████
Tamil      3.5×  ████████████████
Yoruba     5.0×  ████████████████████████
```

Tamil users pay **3.5× more** in API costs and have **3.5× less** effective context window.

---

## 🗺️ Mental Model Map

```
                    TOKENIZATION
                         │
           ┌─────────────┼─────────────┐
           │             │             │
     BY SIZE          BY METHOD    BY LEVEL
           │             │             │
    ┌──────┤       ┌─────┤       ┌─────┤
    │      │       │     │       │     │
  Word  Subword  Char  BPE  Unigram Word Char Byte
           │       │
       SentencePiece
           │
     ┌─────┴──────┐
     │            │
   BPE mode  Unigram mode
```

---

## 📋 Evaluation Checklist

Before deploying a tokenizer, check:
- [ ] Fertility < 2.0 for all target languages
- [ ] CPT > 3.0 for all target languages  
- [ ] WFR < 30% for all target languages
- [ ] Parity ratio < 2.0× across all supported languages
- [ ] byte_fallback = True (no OOV possible)
- [ ] NFKC normalization enabled
- [ ] Tested on: emoji, rare chars, mixed scripts, code, numbers
- [ ] Consistent tokenization for NFC vs NFD inputs
- [ ] Special tokens properly defined ([BOS], [EOS], [PAD], [UNK])

---

## 🔗 Essential Papers (BibTeX)

```bibtex
@inproceedings{sennrich-2016-bpe,
  title = "Neural Machine Translation of Rare Words with Subword Units",
  author = "Sennrich, Rico and Haddow, Barry and Birch, Alexandra",
  booktitle = "Proceedings of ACL 2016",
  year = "2016",
}

@inproceedings{kudo-2018-unigram,
  title = "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates",
  author = "Kudo, Taku",
  booktitle = "Proceedings of ACL 2018",
  year = "2018",
}

@inproceedings{kudo-richardson-2018-sentencepiece,
  title = "{S}entence{P}iece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of EMNLP 2018",
  year = "2018",
}

@article{radford-2019-gpt2,
  title = "Language Models are Unsupervised Multitask Learners",
  author = "Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya",
  year = "2019",
}

@inproceedings{ahia-2023-fairness,
  title = "Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models",
  author = "Ahia, Orevaoghene and others",
  booktitle = "Proceedings of EMNLP 2023",
  year = "2023",
}

@article{petrov-2023-fairness,
  title = "Language Model Tokenizers Introduce Unfairness Between Languages",
  author = "Petrov, Aleksandar and others",
  journal = "arXiv:2305.15425",
  year = "2023",
}
```

---

*Generated with deep research into: Sennrich et al. 2016, Kudo 2018, Kudo & Richardson 2018, Radford et al. 2019, Rust et al. 2021, Ahia et al. 2023, Petrov et al. 2023, Zouhar et al. 2023, Blasi et al. 2024, and multilingual tokenization survey literature (2024-2025).*
