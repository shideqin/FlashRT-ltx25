# LTX-2.5 + FlashRT：环境复现与生产构建指南

> 本指南用于在一台全新机器上重建 LTX-2.5（22B distilled，audio+video）在 FlashRT
> 上的开发 / 基准 / 生产环境。参考运行环境（本机实测）：
> **NVIDIA RTX 6000D（sm_120a，85 GB），驱动 580.126.18，CUDA 13.0，
> PyTorch 2.12.1+cu130，transformers 5.14.1（<5.15）**。
> RTX 5090（sm_120，32 GB）同样支持，部分结果见 `docs/ltx25_usage.md`。

---

## 1. 目录与仓库对应关系

| 本地目录 | 仓库 | 说明 |
|---|---|---|
| `FlashRT/` | <https://github.com/shideqin/FlashRT-ltx25> | **主工作仓库**：LTX-2.5 structures 集成 + `serve/` RTF 基准/诊断 harness |
| `LTX-2/` | <https://github.com/shideqin/LTX-2-ltx-kernels> | Lightricks/LTX-2 的 fork，含 `ltx-kernels` 的 CUDA 13 / gcc-11 构建修复 |
| `ComfyUI/` | <https://github.com/Comfy-Org/ComfyUI> | 官方仓库，无本地改动 |
| `models/` | — | 模型权重（~113 GB），**不在 git 上**，需下载或从旧机复制 |
| `hf_cache/` | — | FlashRT 预编译 kernel wheel（`flashrt/kernels-*`，~2.6 GB） |

Git 身份与推送目标：

```bash
git config user.name  "shideqin"
git config user.email "238426+shideqin@users.noreply.github.com"
# 推送目标：shideqin/FlashRT-ltx25 与 shideqin/LTX-2-ltx-kernels 的 main
```

---

## 2. 从零复现（新机器）

### 2.1 克隆

```bash
git clone https://github.com/shideqin/FlashRT-ltx25.git
git clone https://github.com/shideqin/LTX-2-ltx-kernels.git
git clone https://github.com/Comfy-Org/ComfyUI.git
```

### 2.2 系统依赖

| 组件 | 最低要求 | 备注 |
|---|---|---|
| GPU | SM120（RTX 5090 / RTX 6000D） | sage2-fvk / NVFP4 路径以 SM120 为准 |
| NVIDIA 驱动 | 550+ | CUDA 13 需 545+ |
| CUDA Toolkit | 13.0（推荐） | sm_120 建议 CUDA 13 |
| Python | 3.10 / 3.11 / 3.12 | 本机 3.12 |
| GCC/G++ | 11+（C++17） | |
| CMake | 3.24+ | |

### 2.3 安装 LTX-2 包（`ltx-core` / `ltx-pipelines` / `ltx-kernels`）

```bash
cd LTX-2
uv sync --package ltx-core --extra natten
```

`natten` 提供最快的 video VAE 解码；没有它 VAE 解码回退到更慢的 Triton `na3d` 路径。

### 2.4 构建 FlashRT 内核

**关键**：执行 `cmake ..` 的 Python 解释器必须与之后 `import flash_rt` 的解释器是同一个
（构建步骤用 `python3 -m pybind11 --cmakedir` 定位 pybind11 头文件）。**无需
`pip install flash-attn`**——FA2 以源码形式 vendored，随构建打进 `flash_rt_fa2.so`。

先装与你 CUDA 匹配的 PyTorch wheel（参考环境为 `torch 2.12.1+cu130`；FlashRT 对 torch
版本保持中立，稳定索引命令见 `USAGE.md`）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 示例：CUDA 12.8
```

然后：

```bash
cd FlashRT
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

pip install -e ".[torch]" pybind11 cmake "numpy>=1.24" safetensors \
    "transformers<5.15"

mkdir build && cd build
cmake ..            # 自动从 nvidia-smi 探测 GPU 架构
make -j$(nproc)
cp flash_rt_kernels*.so flash_rt_fa2*.so ../flash_rt/ 2>/dev/null || \
    cp flash_rt_kernels*.so ../flash_rt/
cd ..
```

构建产物：`flash_rt/flash_rt_kernels.so`（~3 MB，手写 kernel）+ `flash_rt/flash_rt_fa2.so`
（~135 MB，vendored Flash-Attention 2 fwd）。

### 2.5 下载模型（HuggingFace `Lightricks/LTX-2.5`，gated）

需先接受模型条款并用 **read token** 登录（`hf auth login`）。国内网络可加
`HF_ENDPOINT=https://hf-mirror.com`。

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir models/ltx-2.5
```

| 文件 | 大小 | 用途 |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18.7 GB | 预量化 W4A4 块权重（FlashRT NVFP4 FFN 链，无需校准） |
| `ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42 GB | bf16 逐组件对比基线 |
| `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | ~26 GB | 文本编码器（每个 pipeline 必需） |
| `ltx-2.5-video-vae-bf16.safetensors` | — | 视频 VAE |
| `ltx-2.5-audio-vae-bf16.safetensors` | — | 音频 VAE |
| `ltx-2.5-duration-head-bf16.safetensors` | — | duration head patch（可选） |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | — | 空间超分 |

### 2.6 环境变量

```bash
export FLASH_RT_LTX2_ROOT=/path/to/LTX-2   # 指向 LTX-2 monorepo（默认 /workspace/data/LTX-2）
export HF_HOME=/path/to/hf_cache            # FlashRT kernel wheel 缓存（flashrt/kernels-*）
export HF_ENDPOINT=https://hf-mirror.com    # 需要镜像时
```

`hf_cache` 可整目录从旧机复制（最快）；若没有，留空即可，FlashRT 首次运行会从
HuggingFace 自动下载 `flashrt/kernels-*` wheel。

### 2.7 冒烟验证

```python
import flash_rt

pipe = flash_rt.load_model(
    checkpoint="/path/to/LTX-2.5",   # 上面的 --local-dir
    config="ltx25",
    attention="sage2-fvk",           # 可选，默认 auto
    fuse=True,                       # W4A4 NVFP4 FFN 链，默认开
    compile_mode="capture",          # 默认 eager；capture 为整环 CUDA Graph
)
pipe.set_prompt("A golden retriever running through a sunny meadow")
print(pipe.infer(seed=42, output_path="out.mp4"))
```

---

## 3. `serve/` RTF 基准与诊断 harness

`serve/` 是 RTF（real-time factor）基准与诊断脚本集，随主仓库分发。核心脚本：

| 脚本 | 作用 |
|---|---|
| `ltx25_rtf.py` | 成对 warmup+计时 E2E 基准。`--res 768x512x49f\|1536x1024x121f`，`--attention sage2-fvk\|sdpa`，`--fuse`，`--compile capture`（省略则 eager），`--tag NAME` |
| `ltx25_rtf_exact.py` | 逐位精确臂（prompt embedding 缓存 + 常驻 transformer，无 swap/compile/capture），cos = 1.000000 |
| `ltx25_parity.py` | 两个 mp4 解码帧之间的 cosine / max-abs 对比 |
| `ltx25_gpu_share.py` | GPU 占用 vs 墙钟时间（floor proof） |
| `attn_shape_probe.py` | 真实 eager 形状探针 |
| `qscale_probe.py` | q-scale 布局不匹配证明（非 128 对齐 lq） |
| `attn_fix_probe.py` | 修复后 padded-lq 与 SDPA 的逐位对齐验证 |

**已知问题（已修复）**：capture + sage2 在 768 分辨率黑屏 = q-scale 布局不匹配
（非 128 倍数 lq）；`_attn_swap.py` 将 Q 侧 buffer pad 到 128 倍数，
`_nvfp4_ffn_swap.py` 对未对齐 M 做 padding。结果见 `serve/COMPARISON_LTX25_RTF.md`：
768 capture+sage2+FFN RTF 3.9×（cos 0.937），1536 capture+sage2+FFN RTF 7.6×。

---

## 4. 生产构建选项

| 路径 | 命令 | 备注 |
|---|---|---|
| 预构建镜像（最快） | `docker pull ghcr.io/liangsu8899/flashrt:latest` | 已含 CUDA 13.0 + torch 2.9 + 预编译内核 |
| 本地构建镜像 | `docker build -t flashrt:dev -f docker/Dockerfile .` | 冷构建 ~25 min，可传 `GPU_ARCH` / `CUTLASS_REF` 等 build arg（见 `docker/README.md`） |
| 原生构建 | 见 §2.4 | 无 Docker 场景 |

Thor（SM110，Jetson ARM64）不走上述镜像，按 `README.md` Option C 原生构建。

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/ltx25_usage.md` | LTX-2.5 集成详解：attention 后端、NVFP4 FFN 链、`compile_mode`、capture 模式内存生命周期、structures 层 attach/detach、逐块耗时 |
| `docs/ltx25_structures_design.md` | structures 层设计 |
| `serve/README.md` | RTF harness 使用与已知问题 |
| `AGENTS.md` | **structures 层开发操作规范**（继续优化前必读）：覆盖表、通用性证明、校准纪律、红线、验收标准 |
| `USAGE.md` / `README.md` §Build & install | FlashRT 完整安装 / API 参考 |
| `README.md`（LTX-2） | 模型清单与 `ltx-pipelines` 用法 |
