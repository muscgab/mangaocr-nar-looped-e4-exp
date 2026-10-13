# mangaocr-nar-looped-e4-exp

[English](README.md) | [中文](README_zh.md) · [技术报告](TECHNICAL_REPORT_zh.md)

[GitHub](https://github.com/muscgab/mangaocr-nar-looped-e4-exp)（只有代码和文档，不含权重） · [Hugging Face](https://huggingface.co/muscgab/mangaocr-nar-looped-e4-exp) · [ModelScope](https://modelscope.cn/models/muscgab/mangaocr-nar-looped-e4-exp)

日文漫画文本框 OCR 的实验模型，作为参考发布。非自回归，5,420 万参数。4 层解码器最多循环运行 3 次（"圈"），第 1、2、3 圈后都可以读出文字；圈数越多越慢，准确率略高。

日常使用推荐 [MangaOCR-Nar-Small](https://github.com/muscgab/mangaocr-nar-small)：大小相同，准确率持平或更高，速度更快。

## 准确率

[JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read)，7,789 框，RIN v1 归一化，本发布代码的输出（fp32）。

| 读出 | 综合分 ↑ | CER % ↓ | EM % ↑ |
|---|---|---|---|
| 第 1 圈 | 97.14 | 1.96 | 90.41 |
| 第 2 圈 | 97.28 | 1.75 | 90.92 |
| 第 3 圈 | 97.29 | 1.72 | 90.96 |
| MangaOCR-Nar-Small（参照） | 97.26 | 1.73 | 91.09 |

分项成绩、SFX 集、Manga109-s、速度、训练细节以及这个模型背后的实验，见[技术报告](TECHNICAL_REPORT_zh.md)。

## 使用

从 Hugging Face 或 ModelScope 下载完整仓库（GitHub 不含权重），然后安装：

```bash
git lfs install
git clone https://huggingface.co/muscgab/mangaocr-nar-looped-e4-exp
# 或：modelscope download --model muscgab/mangaocr-nar-looped-e4-exp --local_dir mangaocr-nar-looped-e4-exp
cd mangaocr-nar-looped-e4-exp
pip install -e .
```

```python
from mangaocr_nar_looped_e4_exp import MangaOcrNarLoopedE4Exp
ocr = MangaOcrNarLoopedE4Exp(loop=3)       # loop = 1、2 或 3；有 CUDA 用 CUDA，否则用 CPU
ocr("crop.png")                            # 一张裁图 -> str
ocr(["crop1.png", "crop2.png"])            # 多张裁图一个批次 -> list[str]
ocr("crop.png", loop=1)                    # 每次调用可单独选圈数
ocr("crop.png", loop="all")                # [第 1 圈, 第 2 圈, 第 3 圈]，耗时等于跑 3 圈
```

也接受 PIL 图像和 uint8 RGB 数组。命令行：

```bash
mangaocr-nar-looped-e4-exp --loop 2 crop1.png crop2.png    # 每行输出 "路径<TAB>文字"
```

输入是裁好的文本框（不含文字检测）。代码是未做速度优化的 fp32 PyTorch。

## 文件

| 文件 | 内容 |
|---|---|
| `model.safetensors` | 权重，fp32（217 MB） |
| `config.json` | 结构参数 |
| `nar_vocabulary_6817.json` | 词表（6,817 字） |
| `mangaocr_nar_looped_e4_exp/` | 推理代码 |
| `samples/` | 取自 JMangaBench_Read 的示例裁图 |
| `TECHNICAL_REPORT_zh.md` | 技术报告 |

## 局限

- 面向日文漫画的对白、旁白和小字注释；拟声词不是目标。
- 每框最多 64 字，更长的框需要先切开。
- 手写和美术字的错误明显多于印刷体。

## 许可

- 代码：Apache License 2.0（`LICENSE`）。
- 模型权重：CC BY-NC-SA 4.0（`LICENSE_MODEL`）。训练数据 AnimeText-OCR 为 CC BY-NC-SA 4.0。
- 编码器由 MonkeyOCRv2-S 初始化（Apache License 2.0）。来源与致谢见 `NOTICE.md`。
