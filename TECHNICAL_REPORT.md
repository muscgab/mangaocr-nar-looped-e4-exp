# mangaocr-nar-looped-e4-exp — Technical Report

[English](TECHNICAL_REPORT.md) | [中文](TECHNICAL_REPORT_zh.md) · [README](README.md)

This report describes the model, its training, every evaluation we ran on it, the experiments that led to it and the checks
behind this release. Numbers come from the training logs, the evaluation outputs and the published benchmark files; the
scope of each number (set, normalization, precision) is given next to it.

**Contents**

1. [Overview](#1-overview)
2. [Model](#2-model)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Loops versus layers at equal training](#5-loops-versus-layers-at-equal-training)
6. [Earlier experiments that shaped the design](#6-earlier-experiments-that-shaped-the-design)
7. [Error analysis](#7-error-analysis)
8. [Release checks](#8-release-checks)
9. [Limitations](#9-limitations)
10. [Use of evaluation data during development](#10-use-of-evaluation-data-during-development)

## 1. Overview

e4 is a 54.2M-parameter non-autoregressive OCR model for cropped Japanese manga text boxes. Its decoder is four
transformer blocks that are run up to three times on the same hidden state ("loops"); each loop has its own read-out, so the
text can be read after loop 1, 2 or 3. It was our mainline checkpoint (internal run name `formal_v177`, finished
2026-10-10) before the released [MangaOCR-Nar-Small](https://github.com/muscgab/mangaocr-nar-small) ("N4"), which uses
the same encoder, memory and decoder blocks with a single loop.

| | e4 (this model) | MangaOCR-Nar-Small (N4) |
|---|---|---|
| Decoder | 4 shared blocks × 3 loops, read-out after each loop | 4 blocks, 1 pass |
| Parameters | 54,235,442 | 54.2M |
| Training images (unique) | 2,490,858 | 4,471,985 |
| Images seen | 10,000,000 (78,125 steps × 128) | 35,775,872 (279,499 steps × 128) |
| Read v1.3 EM (all 7,789) | L1 90.40 / L2 90.92 / L3 90.97 | 91.09 |

We release e4 as a reference for the looped decoder. For practical use N4 is the better choice: its accuracy is the same
or higher and it is faster. Section 5 explains why N4 replaced e4.

## 2. Model

### 2.1 Input

A crop is resized with its aspect ratio kept (PIL bicubic) to the largest size whose 14 × 14 patch grid has at most 324
cells, placed at the top-left of the grid canvas, and the rest of the canvas is filled with the median colour of the
resized image's border. Pixels are normalized with mean (0.48145466, 0.4578275, 0.40821073) and std
(0.26862954, 0.26130258, 0.27577711) and cut into patches in row-major order. The unused cells of the 324-cell budget
beyond the $h \times w$ grid are masked in attention; the filled canvas inside the grid is not.

### 2.2 Encoder

MonkeyOCRv2-S vision encoder ([MonkeyOCRv2](https://arxiv.org/abs/2607.11562)), initialized from
[zenosai/MonkeyOCRv2-S](https://huggingface.co/zenosai/MonkeyOCRv2-S): patch size 14, width 384, 12 blocks, 6 heads,
SwiGLU MLP 1536, 2D rotary position embedding over (row, column). We keep the patch embeddings $h_0$, the outputs of
blocks 1–11 ($h_1 \ldots h_{11}$) and the final post-normed output $h_{12}$.

### 2.3 Memory with gamma fusion

The decoder runs 12 block steps in total (3 loops × 4 blocks); step $j = 4(n-1) + i$ (loop $n = 1, 2, 3$, block
$i = 0, \ldots, 3$) reads its own memory

$$m_j = \big[\text{null}_1;\ \text{null}_2;\ b + \gamma_j \odot W_o f_j\big]$$

$$b = \mathrm{Bridge}(h_{12}) + \mathrm{Pos}(r, c)$$

$$f_j = \sum_{l=0}^{11} \mathrm{softmax}(w_j)_l \mathrm{RMSNorm}_l(h_l)$$

- $\mathrm{Bridge}$: RMSNorm → Linear 384→1024 → GELU → Linear 1024→512 → RMSNorm.
- $\mathrm{Pos}(r, c)$: a linear map of 32 Fourier features of the patch's relative row and column.
- $w_j \in \mathbb{R}^{12}$: static weights of step $j$ over the 12 encoder depths; $\gamma_j \in \mathbb{R}^{512}$: a per-channel
  scale, initialized to 0, so at the start of training every step reads only $b$.
- Two learned null tokens are prepended to every memory.

### 2.4 Looped decoder

65 output slots (64 characters + the end token), width 512, 8 heads. Each block has self-attention over the slots (QK-norm,
1D RoPE on half of each head's dimensions), cross-attention to $m_j$ (QK-norm on query and key) and a SwiGLU FFN with hidden
size 2048. The four blocks' attention and FFN weights are shared by the three loops; the pre-norm weights and the three
residual scales of every block are separate for each step $j$:

$$s \leftarrow s + \sigma^{(1)}_j \odot \mathrm{SelfAttn}(\mathrm{RMSNorm}^{(1)}_j(s))$$

$$s \leftarrow s + \sigma^{(2)}_j \odot \mathrm{CrossAttn}(\mathrm{RMSNorm}^{(2)}_j(s), m_j)$$

$$s \leftarrow s + \sigma^{(3)}_j \odot \mathrm{FFN}(\mathrm{RMSNorm}^{(3)}_j(s))$$

The input to loop 1 is $s_0 = P_{\text{slot}} + E_{\text{BLANK}}$, a learned position vector per slot plus the embedding of an input-only BLANK token (no image-dependent input). Between loops the normalized
state that feeds the read-out is passed on (the boundary used by [Ouro](https://arxiv.org/abs/2510.25741)), plus an input
injection and a loop embedding:

$$s^{(n)}_{\text{in}} = \mathrm{FinalNorm}_{n-1}\big(s^{(n-1)}_{\text{out}}\big) + \beta_n \odot s_0 + e_n,\qquad n = 2, 3$$

with $\beta_n, e_n \in \mathbb{R}^{512}$ initialized to 0. No tokens are passed between loops.

### 2.5 Read-out

After loop $n$, $\mathrm{logits} = \mathrm{FinalNorm}_n(s) E^{T} + \mathrm{bias}$, with the output head tied to the
character embedding $E$ (its rows for the 6,818 output classes). The 6,818 classes are the 6,817 characters of `nar_vocabulary_6817.json` and the end token; the
argmax of the 65 slots is cut at the first end token. Reading after loop $n$ computes only $n$ loops. During training an
extra occupancy head (one logit per slot) was trained; it is not used for inference and is not included in the weights.

### 2.6 Parameters

| Part | Parameters |
|---|---|
| Encoder (MonkeyOCRv2-S) | 28,547,712 |
| Memory (bridge, position, depth norms, $W_o$, $\gamma$, $w$, null) | 1,143,312 |
| Decoder blocks (4, shared by the loops) | 20,972,544 |
| Per-step norms and residual scales (12 steps) | 36,864 |
| Final norms (3), input injection (2), loop embedding (2) | 3,584 |
| Character embedding (tied head) and output bias | 3,498,146 |
| Slot positions | 33,280 |
| **Total** | **54,235,442** |

## 3. Training

### 3.1 Data

2,490,858 images, read as one stream with every source in its natural proportion:

| Source | Images | Share |
|---|---|---|
| Synthetic printed text | 995,156 | 40.0% |
| Synthetic handwriting | 800,000 | 32.1% |
| [AnimeText-OCR](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr) (real crops) | 695,702 | 27.9% |

- **Synthetic printed text**: Japanese lines from a private corpus of manga text, typeset vertically or horizontally with
  a pool of Japanese fonts and placed in speech-balloon assets or on text-free background crops cut from AnimeText pages.
- **Synthetic handwriting**: handwriting-style lettering that combines real kana glyphs cut from AnimeText images (upscaled
  4× with [Real-ESRGAN](https://arxiv.org/abs/2107.10833), RealESRGAN_x4plus_anime_6B) with handwriting-style fonts.
- Several distribution parameters of the synthetic data were set from aggregate statistics of the JMangaBench_Read
  candidate pool (published in `docs/pool_stats/` of the [benchmark repository](https://github.com/muscgab/JMangaBench_Read)).
  No image of the pool or the benchmark was used for training.
- **AnimeText-OCR**: Japanese crops with sequence confidence $0.90 \le s < 0.98$; crops from validation pages,
  crops with unusable text and duplicates removed (700,000 rows → 695,702). In our blind audits of this source, sentence-level
  label errors were about 7–8% at confidence 0.90–0.95 and 0 of 25 sampled at 0.95–0.98.
- **Labels**: 1–64 characters, all in the 6,817-character vocabulary, normalized with the project's label rules (for
  example full-width `？！（）` are written half-width). The model's outputs follow these conventions; the benchmark's RIN v1
  normalization treats such variants as equal.
- The training images, the text corpus, the fonts and the renderers are not released.

### 3.2 Objective

For every loop $n$: cross-entropy over the slots that hold a character or the first end token (mean over those slots),
plus 0.1 × binary cross-entropy of the occupancy head. The three loops are weighted equally, and a monotonicity term
penalizes slots whose loss gets worse from one loop to the next:

$$\mathcal{L} = \frac{1}{3}\sum_{n=1}^{3}\big(\mathrm{CE}_n + 0.1 \mathrm{BCE}^{\text{occ}}_n\big) + 0.1 \cdot \frac{1}{2}\sum_{n=2}^{3} \mathrm{mean}_{\text{slots}} \mathrm{ReLU}\big(\mathrm{CE}_{n,t} - \mathrm{sg}(\mathrm{CE}_{n-1,t})\big)$$

where $\mathrm{sg}$ stops the gradient.

### 3.3 Optimization

| Setting | Value |
|---|---|
| Initialization | encoder from MonkeyOCRv2-S; decoder random (dropout 0.1); per-loop norms and residual scales start identical; $\gamma$, $\beta$, $e$ start at 0 |
| Optimizer | AdamW, $\beta = (0.9, 0.98)$, weight decay 0.01, gradient clipping 1.0 |
| Learning rate | decoder 4.9e-4, encoder 4.9e-5; cosine decay over all 78,125 steps |
| Warm-up | decoder 188 steps; encoder frozen for the first 750 steps, then warmed up over 375 steps |
| Batch | 128 (4 GPUs × 32), 78,125 steps, 10,000,000 images seen (4.0 epochs) |
| Input | 324-patch budget only (no multi-resolution training) |
| Precision | fp16 mixed precision with GradScaler; `torch.compile` |
| Hardware and time | 4 × RTX 2080 Ti; 18,257 s (5.1 h) wall time including evaluation, about 550 images/s on average |
| Seed | 20260929 |

In the first 750 steps 25,688,243 parameters were trainable (encoder frozen). At the last step the per-loop losses were
CE₁ 0.510, CE₂ 0.456, CE₃ 0.451 (one logged step, batch 128).

### 3.4 Validation during training

Exact match (%) of loops 1 / 3, from the training log; the first 2,000 crops of each validation set (all 2,248 for Sol).
`at` = AnimeText-OCR validation pages; `oc` / `hw` = synthetic printed / handwriting validation (held-out texts and
background pages); `sol_hw` = an earlier, unreviewed version of the Sol handwriting development set (see section 10).

| Step | at L1 / L3 | oc L1 / L3 | hw L1 / L3 | sol_hw L1 / L3 |
|---|---|---|---|---|
| 2,000 | 54.80 / 61.65 | 57.05 / 63.10 | 37.70 / 40.80 | 31.76 / 34.52 |
| 10,000 | 78.55 / 86.45 | 79.80 / 89.40 | 69.40 / 74.65 | 48.58 / 53.11 |
| 20,000 | 85.10 / 90.60 | 88.85 / 92.45 | 75.95 / 81.60 | 53.60 / 58.01 |
| 30,000 | 91.30 / 92.95 | 90.55 / 94.70 | 87.00 / 90.05 | 59.70 / 61.70 |
| 40,000 | 94.20 / 94.90 | 95.10 / 96.15 | 90.65 / 92.90 | 63.57 / 64.63 |
| 50,000 | 95.75 / 96.50 | 96.70 / 97.20 | 93.65 / 94.55 | 64.15 / 65.53 |
| 60,000 | 95.95 / 96.65 | 97.15 / 97.25 | 95.45 / 96.05 | 67.13 / 68.51 |
| 70,000 | 96.95 / 97.30 | 97.65 / 97.65 | 96.50 / 96.80 | 67.26 / 68.33 |
| 78,125 | 96.80 / 97.10 | 97.50 / 97.55 | 96.45 / 96.60 | 67.44 / 68.37 |

Early in training the later loops are far ahead (oc at 10k: +9.6 points from L1 to L3); by the end the gap is under
1 point on every set. On `sol_hw` the scores were flat over the last 20k steps; the later N4 run with 3.6 × more images seen
and more data reached higher scores (section 5).

## 4. Evaluation

Unless stated otherwise: [JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read) (7,789 boxes), RIN v1
normalization, the benchmark's own batch-1 predictions (training code, RTX 2080 Ti, fp16 autocast). The fp32 outputs of the
released code differ from these on at most 6 boxes (section 8).

### 4.1 Read v1.3

| Model | Composite ↑ | CER % ↓ | EM % ↑ | EM NFKC % ↑ |
|---|---|---|---|---|
| e4 L1 | 97.14 | 1.96 | 90.40 | 79.66 |
| e4 L2 | 97.28 | 1.75 | 90.92 | 80.06 |
| e4 L3 | 97.29 | 1.73 | 90.97 | 80.07 |
| MangaOCR-Nar-Small (N4) | 97.26 | 1.73 | 91.09 | 80.34 |

The composite is the equal-weight mean of 1 − NED over printed and non-printed boxes (benchmark definition). Other models
are on the [benchmark leaderboard](https://github.com/muscgab/JMangaBench_Read).

EM % by subset:

| Subset (boxes) | e4 L1 | e4 L2 | e4 L3 | N4 |
|---|---|---|---|---|
| New in v1.3 (4,660) | 90.71 | 91.24 | 91.31 | 91.35 |
| v1.2 kept (3,129) | 89.93 | 90.44 | 90.48 | 90.70 |
| Hard (784) | 69.01 | 70.54 | 70.41 | 70.92 |
| Gold (1,266) | 78.75 | 79.78 | 79.78 | 80.17 |
| Printed (5,877) | 93.98 | 94.33 | 94.38 | 94.44 |
| Non-printed (1,912) | 79.39 | 80.44 | 80.49 | 80.81 |
| 1–4 characters (1,504) | 91.69 | 91.62 | 91.49 | 91.76 |
| 5–16 characters (4,675) | 90.93 | 91.40 | 91.53 | 91.32 |
| 17–32 characters (1,441) | 90.70 | 91.39 | 91.26 | 91.67 |
| > 32 characters (169) | 61.54 | 67.46 | 68.64 | 73.96 |
| Vertical (7,158) | 91.03 | 91.48 | 91.55 | 91.59 |
| Horizontal (439) | 81.78 | 83.83 | 83.37 | 84.51 |
| Dialogue (6,948) | 91.00 | 91.47 | 91.58 | 91.55 |
| Narration (504) | 90.48 | 91.47 | 91.27 | 92.26 |
| Other (331) | 77.95 | 78.85 | 77.95 | 80.06 |

CER % for e4 L1 / L2 / L3: printed 1.44 / 1.28 / 1.25, non-printed 4.46 / 4.04 / 4.02, > 32 characters 8.35 / 6.92 / 7.10
(N4: 1.24, 4.14, 6.42).

### 4.2 What the extra loops change

Paired box-by-box comparison on Read v1.3 (7,789 boxes, RIN v1, exact McNemar test):

| Comparison | Outputs differ | Correct only in the first | Correct only in the second | p |
|---|---|---|---|---|
| L2 vs L1 | 263 | 75 (L2) | 34 (L1) | < 0.001 |
| L3 vs L2 | 129 | 19 (L3) | 15 (L2) | 0.61 |
| L3 vs L1 | 301 | 87 (L3) | 42 (L1) | < 0.001 |
| L3 vs N4 | 628 | 179 (L3) | 188 (N4) | 0.68 |
| L1 vs N4 | 659 | 169 (L1) | 223 (N4) | 0.007 |

- The second loop helps clearly; the third changes 129 outputs but the gain is not significant.
- e4 L3 and N4 are statistically tied, but they get different boxes right: 367 boxes are read correctly by exactly one of the
  two.
- The gain from loops is largest on long boxes (> 32 characters: 61.54 → 68.64) and horizontal boxes (81.78 → 83.37).

### 4.3 SFX set

The benchmark's separate SFX set (380 sound-effect and mimetic boxes, 376 non-printed; SFX is not a training target).
Ranked by 1 − NED in the benchmark.

| Model | 1 − NED ↑ | CER % ↓ | EM % ↑ | EM 95% CI |
|---|---|---|---|---|
| e4 L1 | 91.96 | 8.90 | 78.68 | 74.3–82.5 |
| e4 L2 | 92.73 | 8.15 | 81.05 | 76.8–84.7 |
| e4 L3 | 93.09 | 7.57 | 81.84 | 77.7–85.4 |
| N4 | 92.56 | 8.15 | 80.79 | 76.5–84.4 |

e4 L3 has the highest 1 − NED of all models on this set; the confidence intervals of e4 L3 and N4 overlap.

### 4.4 Manga109-s

123,014 boxes from 87 volumes of Manga109-s (boxes over 64 characters, with characters outside the vocabulary or with
an isolated dakuten removed from 123,212), the project's r7 label normalization on prediction and label, fp16, 324-patch
budget. Manga109-s has scanned print, a different domain from the web manga of Read. Several public models were trained on
Manga109, so we only compare our own models here.

| Model | EM % | CER % | 1–4 | 5–8 | 9–16 | 17–32 | 33–64 |
|---|---|---|---|---|---|---|---|
| e4 L3 | 91.62 | 1.43 | 95.4 | 91.7 | 92.1 | 89.3 | 73.8 |
| N4 | 92.08 | 1.39 | 95.9 | 91.7 | 92.5 | 90.1 | 74.4 |

Columns 1–4 … 33–64: EM % by label length in characters.

### 4.5 Speed

From the benchmark (RTX 2080 Ti, fp16, training-code inference path, 8 pinned CPU cores, preprocessing included):

| Model | Per image, mean (ms) | Per page p50 (ms) | Max throughput (img/s) |
|---|---|---|---|
| e4 L1 | 28.3 | 58.1 | 282 (batch 24) |
| e4 L2 | 35.5 | 64.3 | 267 (batch 48) |
| e4 L3 | 42.8 | 71.9 | 227 (batch 32) |
| N4 (released code, CUDA graphs) | 9.1 | 35.5 | 299 (batch 48) |

The N4 row uses its optimized release code and the e4 rows a plain PyTorch path, so per-image latency is not a
like-for-like comparison of the architectures; each extra loop adds about 7 ms per image on this path. The released e4
code is plain eager PyTorch in fp32 without any optimization, and we did not time it.

### 4.6 Development sets

Final checkpoint, training code, fp16 autocast, project r7 normalization:

| Set | L1 | L2 | L3 |
|---|---|---|---|
| Read v1.2, all 3,142 (EM %) | 89.62 | 90.13 | 90.13 |
| Read v1.2, printed 2,328 | 93.86 | 94.20 | 94.20 |
| Read v1.2, non-printed 814 | 77.52 | 78.50 | 78.50 |
| Sol handwriting, old labels, 2,248 | 67.44 | 67.93 | 68.37 |

## 5. Loops versus layers at equal training

To decide whether the formal model should loop, we trained three variants with **the same data (the 2,490,858 images
above), the same recipe and the same 10,000,000 images seen**, differing only in the decoder:

- **N4**: 4 blocks, 1 loop (`formal_e4d_N4`, not the released N4, which was trained later for 35.8M images);
- **L4x2**: 4 shared blocks, 2 loops (`formal_e4d_L4x2`);
- **e4**: 4 shared blocks, 3 loops (this model).

EM % on the development sets (project r7 normalization; Sol v2 = 603-box subset of Sol handwriting):

| Variant | Read v1.2 all | printed | non-printed | Sol (old) | Sol v2 |
|---|---|---|---|---|---|
| N4 L1 | 89.97 | 93.86 | 78.87 | 67.08 | 55.22 |
| L4x2 L1 | 89.78 | 93.77 | 78.38 | 67.22 | 55.89 |
| L4x2 L2 | 90.39 | 94.20 | 79.48 | 67.84 | 56.55 |
| e4 L1 | 89.62 | 93.86 | 77.52 | 67.44 | 57.55 |
| e4 L2 | 90.13 | 94.20 | 78.50 | 67.93 | 57.38 |
| e4 L3 | 90.13 | 94.20 | 78.50 | 68.37 | 58.71 |

Paired comparisons (boxes correct only in A / only in B, exact McNemar p):

| A vs B | Read v1.2 all | Sol (old) | Sol v2 |
|---|---|---|---|
| e4 L1 vs N4 L1 | 49 / 60, p = 0.34 | 86 / 78, p = 0.59 | 26 / 12, p = 0.034 |
| L4x2 L1 vs N4 L1 | 49 / 55, p = 0.62 | 85 / 82, p = 0.88 | 21 / 17, p = 0.63 |
| e4 L1 vs L4x2 L1 | 47 / 52, p = 0.69 | 82 / 77, p = 0.75 | 24 / 14, p = 0.14 |
| L4x2 L2 vs L4x2 L1 | 22 / 3, p < 0.001 | 29 / 15, p = 0.049 | 9 / 5, p = 0.42 |
| e4 L3 vs L4x2 L2 | 44 / 52, p = 0.48 | 79 / 67, p = 0.36 | 24 / 11, p = 0.041 |

What we concluded:

1. **Loop training did not make the first loop better.** e4 L1 and L4x2 L1 are tied with N4 L1 on Read v1.2 (p = 0.34 and
   0.62). The one significant difference (e4 L1 vs N4 L1 on Sol v2, 603 boxes) is in favour of e4 but is a single small
   subset among several comparisons.
2. **The gain comes from running more loops at inference.** Within one model, the second loop fixes far more boxes than it
   breaks (L4x2: 22 vs 3 on Read v1.2).
3. A single-loop model is cheaper to run. Our rule for the formal run was to train the 3-loop model only if 3-loop
   training clearly improved loops 1–2. It did not, so the formal model became a single-loop N4 trained 3.6 × longer on
   more data. On Read v1.3 that N4 (91.09) matches e4 L3 (90.97, p = 0.68).

## 6. Earlier experiments that shaped the design

These experiments were run on earlier models, data and protocols (older label normalization r4, other input budgets),
mostly on JMangaBench-Mixed (1,286 real crops, the project's earlier evaluation set) with a single seed each. Their
numbers are not comparable with sections 4–5; they explain why the architecture looks the way it does.

**Gamma fusion helps (2026-09-29).** A two-round 8-layer model trained from scratch without the fusion path
(`fusion = none`) against the same model with gamma fusion, same data and schedule: Mixed EM 74.96% vs 77.53%
(−2.57 points; 39 vs 72 boxes correct only in one, p = 0.002). Zeroing $\gamma$ of the trained fused model at inference
cost 4.20 points, so retraining without fusion recovers only part of it. The extra errors were mostly dropped characters.

**Looping instead of more layers (2026-10-01).** Starting from a trained model "C0", we kept 4 of its decoder layers and
looped them 3 times (per-loop norms/scales copied, Ouro-style boundary), trained on 950k images, and compared it with C0
trained for the same extra 950k images (control). Mixed EM: looped L1 80.56, L2 82.82, L3 83.28; control 83.67 (L3 vs
control: 34 vs 39 boxes, p = 0.64); C0 83.20. Loop 2 fixed 43 boxes and broke 14 relative to loop 1; loop 3 fixed 9 and
broke 3. A 4-block looped decoder matched an 8-layer one at the third loop with half the decoder parameters, which led to
the looped design of e4.

**Second-pass refinement and block-autoregressive decoding did not pay off (2026-09-30).**
- A second decoder round that reads the model's own draft changed 78–126 predictions on Mixed but fixed only 6–12 of
  250–297 first-round errors.
- Training with random ground-truth prefixes (block-AR, "T1") against plain NAR continuation ("C0") with equal images
  seen: Mixed EM 82.35 vs 83.20 (17 vs 28 boxes correct only in one, p = 0.135); T1's best block-AR decoding (83.28) only
  tied C0's plain decoding. Both ideas were dropped; e4 is single-pass per loop with no tokens fed back.

**Many decoder errors are alignment errors rather than missing visual information (2026-10-06).** On a looped model
(Read v1.2 L3 EM 83.55, r7), a CTC probe trained on the frozen encoder output read synthetic vertical lines with CER 2.19%
when each step saw a 3 × 3 patch neighbourhood in reading order (columns right to left, top to bottom), against 64.46% in
row-major order. Content characters the decoder dropped in the middle of a line were read by the probe in 52 of 58 cases on
synthetic data (90%), 42 of 64 on Read printed (66%) and 19 of 59 on Read non-printed (32%). Forcing the true output length (oracle) raised EM from
83.55 to 85.04 but fixed only end-of-line drops. Averaging the three loops' logits (83.61) and biasing the end token gave
no gain.

## 7. Error analysis

**Read v1.3 (e4 L3, RIN v1).** 703 of 7,789 boxes are wrong. 334 of them (48%) have the wrong length: 288 too short and
46 too long. Outputs that are too short (mostly a dropped character) are the largest group, as in our earlier models.

**Read v1.2 (e4 L3, 3,129 boxes with v1.3 labels, project r7).** 299 errors (EM 90.44%; printed 134 errors of 2,327,
non-printed 165 of 802). We looked at every error crop (on contact sheets) and sorted them:

| Verdict | Boxes |
|---|---|
| Model wrong | 260 |
| Look-alike characters, both readings plausible | 15 |
| Ambiguous image | 14 |
| Convention difference (e.g. solid ♥ read as ♡; the labels use ♡ in 73 boxes and ♥ in none) | 9 |
| Label wrong | 1 |

Main causes among printed boxes: small kana (19), the long-vowel mark in vertical text (19), vertical punctuation (18),
rare kanji (18; for example all 4 boxes with 吟 were wrong, although 吟 is in the vocabulary) and sideways Latin text (8).
Among non-printed boxes: sloppy handwriting read as another character (53), handwritten punctuation (32), small kana (22)
and hand-drawn hearts (12).

**Sol handwriting (e4 L3, 2,138 boxes after label review).** EM 71.84% (L1 70.72%); 602 errors, classified by
LLM-assisted review (Claude Opus 5.5 reviewers, with our corrections): model wrong 479, look-alike 57, ambiguous 36, label
wrong 30. Recurring patterns: 77 errors differ from the label only by a
heart, 128 drop a heart, 144 drop only the sentence-final mark.

## 8. Release checks

- **Weights**: `model.safetensors` (fp32, 233 tensors, 54,235,442 parameters) was converted from the final checkpoint
  (step 78,125). The tensors are bit-identical to the model part of the training checkpoint; only the training-only
  occupancy head (2 tensors) was dropped. The file loads back bit-identically.
- **Code**: `mangaocr_nar_looped_e4_exp/model.py` is a plain inference rewrite with the same parameter names, and
  `preprocess.py` is the numpy/Pillow preprocessing of MangaOCR-Nar-Small (same rules as training).
- **Parity**: on all 7,789 Read v1.3 crops, in fp32 with the same batches of 16, the released code and the training code
  give **identical texts for every box after each of the 3 loops (7,789 / 7,789 each)**.
- **Against the benchmark predictions** (training code, fp16 autocast, batch 1): identical texts on 7,784 / 7,788 / 7,783
  boxes for L1 / L2 / L3. Scores of the released fp32 outputs:

| | Composite | CER % | EM % |
|---|---|---|---|
| L1 | 97.14 | 1.96 | 90.41 |
| L2 | 97.28 | 1.75 | 90.92 |
| L3 | 97.29 | 1.72 | 90.96 |

- The command-line tool and the Python API were run on CUDA and on CPU with the 8 example crops (all 8 read as labelled
  up to RIN v1 on every loop).
- Apple MPS and fp16 inference with the released code were not tested.

## 9. Limitations

- Experimental reference model; MangaOCR-Nar-Small is as accurate or better and faster.
- Japanese only; trained on dialogue, narration and side notes. Onomatopoeia is not a target, Korean and English are not
  supported.
- At most 64 characters per box (65 slots). Accuracy drops on long boxes (> 32 characters: EM 68.64% for L3), and boxes
  longer than 64 characters cannot be read in one piece; split them first.
- Non-printed text has about three times the CER of printed text (4.02% vs 1.25% for L3).
- Text detection is not included; inputs must be cropped text boxes.
- The released code is unoptimized fp32 PyTorch.

## 10. Use of evaluation data during development

- **Read v1.2** (3,142 boxes) was the development set of this model family: we used it for model selection, the controlled
  comparisons in section 5 and the error analysis in section 7.
- **Sol handwriting** (2,248 boxes in its earlier version) was another development set: it was evaluated during training
  (section 3.4), used in section 5 and its e4 errors were reviewed (section 7). It shares 1,021 boxes with the 4,660 boxes
  new in Read v1.3 (1,004 of the 1,110 new non-printed boxes).
- Read v1.3 scores on the 4,660 new boxes are therefore listed separately (section 4.1); they are close to the scores on all
  boxes.
- No image from the Read candidate pool, Read v1.2/v1.3, the SFX set, Sol or Manga109-s was used for training. Aggregate
  statistics of the candidate pool were used to set several distribution parameters of the synthetic data (section 3.1).
- Manga109-s statistics were used earlier in the project for error analysis that informed the design; treat the
  Manga109-s numbers as seen during development, not as a pristine test.
