# English Entropy Rate

**语言:** [English](README.md) | 中文

这个项目用传统压缩、语言模型交叉熵，以及一个小型的 LLM 算术编码压缩器，来实验估计英文文本的熵率。

项目起点是一个很自然的问题：

> 如果英文有大约 26 个字母，为什么英文熵率可以接近
> `1.3 bit/符号`，而不是 `5 bit/符号`？

这个仓库从三个角度探索这个问题：

- `baseline`：比较普通无损压缩器和原始 UTF-8 文本。
- `llm`：用 LLM 的 next-token 概率估计理想算术编码长度。
- `compress` / `decompress`：使用 LLM 概率和算术编码，真正读写 `.my-llm` 压缩文件。

## 当前结果

在 Project Gutenberg 版 *Middlemarch* 的规范化文本上测试。规范化文本只保留小写英文字母 `a-z` 和空格：

| 方法/模型 | 范围 | bits/char |
| --- | --- | ---: |
| gzip | 全书 | 2.8062 |
| bz2 | 全书 | 2.0179 |
| lzma | 全书 | 2.2080 |
| `distilbert/distilgpt2` 理想 LLM 编码 | 全书 | 1.3948 |
| `distilbert/distilgpt2` 理想 LLM 编码 | 前 50k 字符 | 1.4126 |
| `Qwen/Qwen3-0.6B-Base` 理想 LLM 编码 | 前 50k 字符 | 1.1473 |

LLM 结果来自：

```text
sum_i -log2 P(token_i | previous context)
```

它表示理想算术编码器在共享同一个模型和 tokenizer 时可以接近的 payload 长度；这里不包含模型权重、tokenizer 文件、格式头和有限精度编码开销。

## 快速开始

运行传统压缩 baseline：

```bash
PYTHONPATH=src python -m english_entropy_rate baseline data/sample.txt
```

估计 LLM 理想算术编码长度：

```bash
PYTHONPATH=src python -m english_entropy_rate llm \
  data/middlemarch_lowercase_letters_spaces.txt \
  --model distilbert/distilgpt2 \
  --limit-chars 50000 \
  --max-length 1024 \
  --stride 512
```

在 Apple Silicon 上，CLI 会自动优先使用 MPS 后端。比较硬件时可以用 `--device cpu` 或 `--device mps` 手动指定。

把 UTF-8 文本压缩为 `.my-llm`：

```bash
PYTHONPATH=src python -m english_entropy_rate compress \
  data/sample.txt \
  --model distilbert/distilgpt2
```

默认输出路径会追加 `.my-llm`，例如 `data/sample.txt.my-llm`。使用同一个模型解压：

```bash
PYTHONPATH=src python -m english_entropy_rate decompress \
  data/sample.txt.my-llm \
  /tmp/sample.roundtrip.txt \
  --model distilbert/distilgpt2
```

处理较大文本时，建议先用小模型和小前缀。当前压缩器 MVP 会为每个 token 重新计算上下文，所以结果正确但速度较慢。

## 语料准备

仓库中包含 Project Gutenberg 版 *Middlemarch*，以及一个只保留小写字母和空格的清洗版本：

```bash
PYTHONPATH=src python -m english_entropy_rate clean \
  data/middlemarch_gutenberg_145.txt \
  data/middlemarch_lowercase_letters_spaces.txt
```

清洗规则是：字母转小写，把所有非 `a-z` 字符当作词边界，并把连续空格压成单个空格。

## 指标

对于一个包含 `N` 个 UTF-8 字节、`C` 个 Python 字符的文本文件：

```text
bits_per_byte = compressed_bits / N
bits_per_char = compressed_bits / C
```

对于 LLM 估计器：

```text
ideal_bits = sum_i -log2 P(token_i | previous context)
```

如果编码端和解码端共享完全相同的模型和 tokenizer，这就是理想算术编码器可以接近的压缩长度。

## `.my-llm` 格式

压缩器会先写入固定长度二进制头，然后写入算术编码 payload：

```text
8 bytes   magic/version: MYLLM001
32 bytes  sha256(model_name)
8 bytes   原始 UTF-8 字节长度，大端无符号整数
4 bytes   tokenizer token 数量，大端无符号整数
4 bytes   第一个 token id，大端无符号整数
...       arithmetic-coded predicted tokens
```

第一个 token 直接存在 header 中，因为当前 MVP 编码的是
`P(token_i | previous tokens)`。算术编码 payload 存储第 `1..N-1` 个 token。
字节长度和 token 数量告诉解码器何时停止；如果没有这些元数据，不同长度消息对应的算术编码区间会产生歧义。

## 备注

- 尽量使用模型不太可能记忆过的 held-out 文本。
- 报告结果时要说明符号单位：byte、character、token 或 letter。
- 预训练模型是双方共享的侧信息。如果把模型权重算进压缩 payload，短文件会非常昂贵。
- `.my-llm` 压缩器当前优先保证正确性。后续值得加入 KV-cache 解码和编码端批量 scoring。
