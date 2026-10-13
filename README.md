# mangaocr-nar-looped-e4-exp

[English](README.md) | [中文](README_zh.md) · [Technical report](TECHNICAL_REPORT.md)

[GitHub](https://github.com/muscgab/mangaocr-nar-looped-e4-exp) (code and docs only, no weights) · [Hugging Face](https://huggingface.co/muscgab/mangaocr-nar-looped-e4-exp) · [ModelScope](https://modelscope.cn/models/muscgab/mangaocr-nar-looped-e4-exp)

Experimental OCR model for Japanese manga text boxes, released as a reference. Non-autoregressive, 54.2M parameters. Its
4-block decoder runs up to three times ("loops") and the text can be read after loop 1, 2 or 3; more loops are slower and
somewhat more accurate.

For everyday use we recommend [MangaOCR-Nar-Small](https://github.com/muscgab/mangaocr-nar-small): same size, equal or
better accuracy, faster.

## Accuracy

[JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read), 7,789 boxes, RIN v1 normalization, outputs of this
release (fp32).

| Read-out | Composite ↑ | CER % ↓ | EM % ↑ |
|---|---|---|---|
| Loop 1 | 97.14 | 1.96 | 90.41 |
| Loop 2 | 97.28 | 1.75 | 90.92 |
| Loop 3 | 97.29 | 1.72 | 90.96 |
| MangaOCR-Nar-Small (reference) | 97.26 | 1.73 | 91.09 |

Subsets, the SFX set, Manga109-s, speed, training and the experiments behind this model are in the
[technical report](TECHNICAL_REPORT.md).

## Usage

Download the full repository from Hugging Face or ModelScope (GitHub has no weights), then install it:

```bash
git lfs install
git clone https://huggingface.co/muscgab/mangaocr-nar-looped-e4-exp
# or: modelscope download --model muscgab/mangaocr-nar-looped-e4-exp --local_dir mangaocr-nar-looped-e4-exp
cd mangaocr-nar-looped-e4-exp
pip install -e .
```

```python
from mangaocr_nar_looped_e4_exp import MangaOcrNarLoopedE4Exp
ocr = MangaOcrNarLoopedE4Exp(loop=3)       # loop = 1, 2 or 3; device: CUDA if available, else CPU
ocr("crop.png")                            # one crop -> str
ocr(["crop1.png", "crop2.png"])            # several crops as one batch -> list[str]
ocr("crop.png", loop=1)                    # choose the loop per call
ocr("crop.png", loop="all")                # [loop 1, loop 2, loop 3] for the cost of 3 loops
```

PIL images and uint8 RGB arrays are accepted too. Command line:

```bash
mangaocr-nar-looped-e4-exp --loop 2 crop1.png crop2.png    # prints "path<TAB>text" per line
```

Inputs are cropped text boxes (no text detection). The code is plain PyTorch in fp32 without speed optimizations.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Weights, fp32 (217 MB) |
| `config.json` | Architecture |
| `nar_vocabulary_6817.json` | Vocabulary (6,817 characters) |
| `mangaocr_nar_looped_e4_exp/` | Inference code |
| `samples/` | Example crops from JMangaBench_Read |
| `TECHNICAL_REPORT.md` | Technical report |

## Limitations

- Japanese dialogue, narration and side notes; onomatopoeia is not a target.
- At most 64 characters per box; split longer boxes first.
- Handwritten and decorative text has clearly more errors than printed text.

## License

- Code: Apache License 2.0 (`LICENSE`).
- Model weights: CC BY-NC-SA 4.0 (`LICENSE_MODEL`). The training data AnimeText-OCR is CC BY-NC-SA 4.0.
- The encoder is initialized from MonkeyOCRv2-S (Apache License 2.0). See `NOTICE.md` for provenance and acknowledgements.
