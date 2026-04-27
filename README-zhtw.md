<p align="center">
  <img src="assets/banner.png" alt="Lucebox" width="85%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e8e8ed?style=for-the-badge&labelColor=090909" alt="MIT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>開放的大型語言模型推論，針對每一款特定晶片從頭手工打造。</strong><br/>
  核心（Kernels）、投機解碼（speculative decoding）與量化（quantization），皆針對目標平台量身定制。<br/>
  我們不等待更好的矽晶片。我們重寫軟體。
</p>

---

## 盒中之物

目前有兩個專案，更多即將推出。每個專案都是獨立的發行版本，附有各自的基準測試與論文風格的技術說明。

<p align="center">
  <a href="megakernel/"><img src="assets/svg/card-megakernel-dark.svg" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="dflash/"><img src="assets/svg/card-dflash-dark.svg" alt="DFlash 27B" width="46%"></a>
</p>

---

## 01 · Megakernel Qwen3.5 0.8B on RTX 3090

**史上第一個針對混合 DeltaNet/Attention LLM 的 megakernel。** Qwen 3.5-0.8B 的全部 24 層融合至單一 CUDA dispatch，在 2020 年的 GPU 上達到 1.87 tok/J，以 2 倍吞吐量媲美 Apple 最新矽晶片。

```bash
# 1. 複製並進入目錄
git clone https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/megakernel

# 2. 安裝（需要 Python 3.10+、CUDA 12+、PyTorch 2.0+）。首次執行時從 HF 串流下載模型權重。
python -m venv .venv && source .venv/bin/activate   # Ubuntu 24+ 系統 Python 需要此步驟（PEP 668）
pip install --upgrade pip
pip install torch                          # 請先安裝再進行下一步；setup.py 在建置時會引入 torch
pip install -e . --no-build-isolation      # --no-build-isolation 讓建置過程能看到你剛安裝的 torch

# 3. 執行基準測試（prefill pp520 + decode tg128，對比 llama.cpp BF16 + PyTorch HF）
python final_bench.py
```

| 方法 | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **37,800** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | 不適用 |

**運作原理：** 82 個 block、512 個執行緒、一個持久化 kernel。層與層之間無需返回 CPU。權重直接從 HuggingFace 串流載入。以協作式網格同步取代每個 token 約 100 次的 kernel 啟動。在達到計算上限之前先觸及功耗上限，因此 DVFS 將緊密的執行效率直接轉換為節省的電力。

[完整說明 →](megakernel/README.md) · [基準測試 →](megakernel/RESULTS.md) · [部落格文章 →](https://lucebox.com/blog/megakernel)

> **Blackwell（RTX 5090、DGX Spark / GB10）：** 由 setup 自動偵測；NVFP4 decode 路徑在 GB10 上可達約 194 tok/s tg128。詳見 [megakernel/README.md#blackwell-sm_120--sm_121a](megakernel/README.md)。

---

## 02 · DFlash DDtree Qwen3.5 27B GGUF on RTX 3090

**首個 DFlash 投機解碼的 GGUF 移植版本。** Qwen3.5-27B 在單張 RTX 3090 上運行，採用 Q4_K_M 目標模型 + BF16 草稿模型，DDTree budget=22。

- **示範運行最高達 207 tok/s**（207.6 tok/s DFlash vs 38.0 tok/s AR，5.46×）
- **HumanEval 10 個提示詞基準測試平均 129.5 tok/s**
- **比自迴歸快 3.43 倍**（比鏈式投機解碼多 15%）
- **比相同硬體上的 SGLang AWQ 快 2.8 倍**
- **透過 TurboQuant TQ3_0 KV 快取，在 24 GB 內支援最高 256K 上下文**（128K Q4_0 基準測試：ctx=131072 時 134.78 tok/s）

```bash
# 1. 含子模組一起複製（拉取固定版本的 Luce-Org/llama.cpp@luce-dflash fork）
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash

# 2. 建置 C++/CUDA 解碼器（需要 CUDA 12+、CMake 3.18+）
# 預設編譯 75/80/86/89（CUDA 12.8+ 加入 120，CUDA 12.9+ 加入 sm_121/DGX Spark，CUDA 13.0+ 加入 sm_110/Thor）
# 僅使用 3090 的用戶可加入 -DCMAKE_CUDA_ARCHITECTURES=86 跳過其他架構，加快建置速度（約 3 分鐘）。
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j

# 3. 下載模型權重：約 16 GB Q4_K_M 目標模型 + 3.46 GB BF16 草稿模型
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/

# 4a. 單次串流生成
python3 scripts/run.py --prompt "def fibonacci(n):"

# 4b. 或重現論文風格基準測試（HumanEval + GSM8K + Math500，約 15 分鐘）
python3 scripts/bench_llm.py
```

| 基準測試 | AR (tok/s) | DFlash+DDTree (tok/s) | 加速倍數 |
|-----------|:----------:|:---------------------:|:-------:|
| **HumanEval** | 37.8 | **129.5** | **3.43×** |
| Math500 | 37.7 | 110.5 | 2.93× |
| GSM8K | 37.7 | 96.2 | 2.55× |

**塑造本專案的限制條件。** Qwen3.5-27B 的 AWQ INT4 加上 BF16 草稿模型，在 24 GB 顯示卡上無法為 DDTree 驗證狀態留出空間。Q4_K_M GGUF（約 16 GB 目標模型）是能在 RTX 3090 的 24 GB 上容納目標模型 + 3.46 GB 草稿模型 + budget=22 樹狀態 + KV 快取的最大格式。選擇它迫使我們在 ggml 之上進行新的移植，因為目前沒有任何公開的 DFlash 執行時期支援 GGUF 目標模型。

**我們做了什麼，沒做什麼。** 以下演算法並非我們原創：
- [**DFlash**](https://arxiv.org/abs/2602.06036)（z-lab，2026）：以目標隱藏狀態為條件的區塊擴散草稿。
- [**DDTree**](https://arxiv.org/abs/2604.12989)（Ringel 等人，2026）：在相同計算預算下優於鏈式驗證的樹狀結構驗證。

我們移植並調整的部分：
- 基於 ggml 的 C++/CUDA 解碼引擎（無 libllama，無 Python 執行時期，Q4_K_M 目標路徑）。
- 三個用於樹狀感知 SSM 狀態回滾的自訂 CUDA kernel：`ggml_ssm_conv_tree`、`ggml_gated_delta_net_tree`、`ggml_gated_delta_net_tree_persist`。
- 針對 RTX 3090 + Q4_K_M 目標模型進行 DDTree budget 掃描：**budget=22** 是最佳點。
- TQ3_0 KV 快取（TurboQuant 3.5 bpv，預設）+ 滑動 `target_feat` 環形緩衝區，在 24 GB 內支援最高 256K 上下文（Q4_0 作為舊版選項，上限約 128K）。

### 在其他 GPU 上執行（4090、5090、DGX Spark / GB10、Jetson AGX Thor）

開箱即支援；建置時只需正確的 CUDA 工具鏈。`dflash/CMakeLists.txt` 在 nvcc 夠新時會自動加入 Blackwell 架構，因此以上快速入門指令在更新的顯示卡上同樣適用。

| GPU | 架構 | 最低 CUDA | 狀態 |
|-----|:----:|:--------:|--------|
| RTX 3090 Ampere | `sm_86` | 12.0 | **參考基準，所有數據以此為準** |
| RTX 4090 Ada | `sm_89` | 12.0 | 應可運作，未驗證，請加入 `-DCMAKE_CUDA_ARCHITECTURES=89` |
| RTX 5090 Blackwell 消費級 | `sm_120` | 12.8 | 支援，CMake 自動加入 |
| DGX Spark / GB10 | `sm_121`（計算能力 12.1） | 12.9 | 支援，CMake 自動加入 |
| Jetson AGX Thor | `sm_110` | 13.0 | 支援，CMake 自動加入 |

確認你的目標裝置：
```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, 'sm_%d%d'%(p.major,p.minor), p.multi_processor_count,'SMs', round(p.total_memory/1e9,1),'GB')"
nvcc --version
```

**DGX Spark / GB10 快速入門：**
```bash
# 需要 CUDA 12.9+ 以支援 sm_121
nvcc --version  # 必須顯示 >= 12.9
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake 自動加入 sm_121
cmake --build build --target test_dflash -j
```

**Jetson AGX Thor 快速入門：**
```bash
# 需要 CUDA 13.0+ 以支援 sm_110 / AGX Thor
nvcc --version
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake 自動加入你的 nvcc 所支援的 Thor 架構
cmake --build build --target test_dflash -j
```

**無法自動移植的部分：**
- **DDTree `budget=22`** 是針對 3090 + Q4_K_M + 24 GB 調整的。在 VRAM 更多的顯示卡（5090 32 GB、GB10 128 GB 統一記憶體）上，需要重新掃描，更大的樹能提升驗證吞吐量，直到記憶體頻寬飽和為止。`scripts/bench_llm.py` 已內建掃描鉤子。
- **TQ3_0 KV 快取 + 滑動 `target_feat` 環形緩衝區** 是根據 24 GB（在 3090 上可支援 256K 上下文）設計的。在 GB10（128 GB 統一記憶體）/ 5090（32 GB）上，你可以進一步擴展上下文，或完全跳過量化，保持 F16 KV。
- **效能數據**（207 tok/s 示範、129.5 HumanEval、2.8× vs SGLang AWQ）是 RTX 3090 @ 原廠設定下的結果。Blackwell/Ada 尚未掃描，歡迎提交附有 `RESULTS.md` 條目的 PR。

[完整說明 →](dflash/README.md) · [基準測試 →](dflash/RESULTS.md) · [部落格文章 →](https://lucebox.com/blog/dflash27b)

> **Qwen3.6-27B（支援，實驗性草稿）：** 架構與 `qwen35` 相同，因此 3.6 的 Q4_K_M GGUF 可直接作為目標模型替換載入。搭配 z-lab 匹配的 [Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) 草稿模型（仍在訓練中，2026-04-26 快照），HumanEval 約達 78 tok/s（AL 5.05）；3.5 草稿模型約達 74 tok/s。3.5↔3.5 參考值為 129.5 tok/s。隨著 z-lab 完成草稿模型訓練，AL 應會持續提升。詳情請見 [dflash/README.md](dflash/README.md#qwen36-27b-target-experimental)。

---

## 為何存在

本地端 AI 應成為預設選項，而非特權：私人資料、無按 token 計費、無廠商鎖定。能夠執行強大模型的硬體早已擺在桌上。讓這些晶片高效運作的軟體卻尚未出現。

通用框架主導了過去十年，因為針對每款晶片手工調整 kernel 的成本太高。一套框架，在所有平台上表現尚可，但在任何平台上都不是最好。大部分矽晶片的潛能都被白白浪費。

AI 輔助開發改變了這個算盤。以前需要一個季度的重寫工作，現在一個發布週期內就能完成。Lucebox 是我們發布成果的地方，每次針對一款晶片和一個模型系列。MIT 開源，完整說明，可重現的基準測試。

---

## 環境需求

本 repo 中所有實驗均在 NVIDIA RTX 3090（2020 年）上建置、調整與基準測試，作為參考目標。支援的 GPU 家族：

- **Ampere**（sm_86，RTX 3090 / A 系列）：參考基準，CUDA 12+。
- **Ada**（sm_89，RTX 40xx）：應可運作，未驗證，CUDA 12+。
- **Blackwell 消費級**（sm_120，RTX 50xx 含 5090）：支援，CUDA 12.8+。
- **DGX Spark / GB10**（sm_121，計算能力 12.1）：支援，CUDA 12.9+。
- **Jetson AGX Thor**（sm_110）：支援，CUDA 13+。

PyTorch 2.0+。`dflash/` 需要 CMake 3.18+ 以及 `--recurse-submodules` 以取得固定版本的 `Luce-Org/llama.cpp@luce-dflash` fork（三個樹狀模式的 ggml op）；多架構建置是自動的（詳見[在其他 GPU 上執行](#running-on-other-gpus-4090-5090-dgx-spark--gb10-jetson-agx-thor)）。

**Megakernel 移植注意事項。** 比 dflash 更嚴格：`megakernel/setup.py` 固定了 `-arch=sm_86 -DNUM_BLOCKS=82`（3090 的 SM 數量）。若要在不同顯示卡上執行，請修改這兩個定義並執行 `pip install -e . --force-reinstall --no-deps`。網格是持久化的，每個 SM 一個 block，因此 `NUM_BLOCKS` 必須完全一致。建議起始點：4090 `sm_89` + `128`，5090 `sm_120` + `170`，DGX Spark / GB10 `sm_121` + 執行 `torch.cuda.get_device_properties(0).multi_processor_count` 讀取 SM 數量，Jetson AGX Thor `sm_110` + 同樣的 SM 數量查詢。

**選用：找出你的 GPU 最佳點：** `sudo nvidia-smi -pl 220`（megakernel 在 3090 上 220 W 時達到最佳 tok/J；其他顯示卡需重新掃描）。

---

## 儲存庫架構

```
lucebox-hub/
├── megakernel/    · Qwen 3.5-0.8B 的融合前向傳播
├── dflash/        · Qwen 3.5-27B 在 RTX 3090 上的 DFlash 投機解碼移植
└── assets/        · 橫幅圖片、卡片、圖表
```

---

## 開發路線圖

```
  Q1 2026    ▮▮▮▮▮▮▮▮▮▮    RTX 3090 核心與優化
  Q2 2026    ▮▮▮▮▮▯▯▯▯▯    Ryzen AI MAX+ 395 優化
  Q2 2026    ▮▮▯▯▯▯▯▯▯▯    異質 CPU + GPU 延遲優化
  Q2 2026    ▮▯▯▯▯▯▯▯▯▯    本地端 AI 機器的 Lucebox OS
  Q3 2026    ▯▯▯▯▯▯▯▯▯▯    Lucebox 正式發布
```

---

## 引用

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub},
  year   = {2026}
}
```

各專案的引用資訊請見各子專案的 README。

---

## 致敬

- [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)：megakernel 概念與每瓦智能（intelligence-per-watt）方法論。
- [z-lab/DFlash](https://arxiv.org/abs/2602.06036)（Wang 等人，2026）：區塊擴散投機解碼演算法。我們直接使用其發布的 Qwen3.5-27B-DFlash 草稿模型權重。
- [DDTree](https://arxiv.org/abs/2604.12989)（Ringel & Romano，2026）：DFlash 27B 用於實現 3.5 倍加速（相較於鏈式投機解碼）的樹狀結構驗證。[liranringel/ddtree](https://github.com/liranringel/ddtree)。
- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)、[Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen)：融合 Qwen kernel 的先驅研究。

---

## 社群

- **Discord**：[discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **網站**：[lucebox.com](https://lucebox.com)
- **問題回報**：[github.com/Luce-Org/lucebox-hub/issues](https://github.com/Luce-Org/lucebox-hub/issues)
- **部落格**：[lucebox.com/blog](https://lucebox.com/blog)

---

<p align="center">
  <sub><a href="LICENSE">MIT</a> · <a href="https://lucebox.com">Lucebox.com</a></sub>
</p>
