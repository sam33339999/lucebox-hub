<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  <strong>史上第一個針對混合 DeltaNet/Attention LLM 的 megakernel。</strong><br/>
  Qwen 3.5-0.8B 的全部 24 層融合至單一 CUDA dispatch。<br/>
  在 2020 年的 GPU 上達到 1.87 tok/J，以 2 倍吞吐量媲美 Apple 最新矽晶片。<br/><br/>
  <a href="https://lucebox.com/blog/megakernel">部落格文章</a> · <a href="RESULTS.md">基準測試</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

```
                        Prefill      Decode      tok/J
Megakernel (RTX 3090)   37,800       413         1.87  @220W
llama.cpp  (RTX 3090)   11,247       267         0.76
Apple M5 Max               -         229         1.76
```

> NVIDIA 與 Apple 之間的效能差距並非源自矽晶片本身。這是在有能力的硬體上執行通用軟體所造成的人為差距。

## 為何存在

傳統觀點認為 NVIDIA GPU 速度快但耗電，Apple Silicon 較慢但節能。從數字來看確實如此：RTX 3090 上的 llama.cpp 在 350W 下可達 267 tok/s（0.76 tok/J），而 M5 Max 在約 130W 下可達 229 tok/s（1.76 tok/J）。NVIDIA 更快，但效率差 2.3 倍。

Qwen 3.5-0.8B 採用混合 DeltaNet + Attention 架構（線性注意力與標準注意力交錯）。在此之前，這種模式沒有任何融合 kernel 存在。這是第一個。

受到 [Hazy Research 針對 Llama-1B 的 megakernel 研究](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) 的啟發，我們提出疑問：同樣的想法能否在消費級 GPU 上套用於混合 DeltaNet/Attention 模型？

我們認為問題從來不在硬體。RTX 3090 擁有 936 GB/s 的記憶體頻寬和 142 TFLOPS FP16 運算能力。從中只擠出 267 tok/s 是軟體問題。

**罪魁禍首：每個 token 約 100 次 kernel 啟動。** 每個層邊界都會將控制權交還給 CPU、調度下一個 kernel、從全域記憶體重新載入權重，並同步執行緒。對於 24 層而言，這些微秒累積起來，每一個都在白白消耗電力。

因此我們將所有內容融合進單一 kernel。

## 結果

### Decode 與 Prefill（Qwen 3.5-0.8B）

| 方法 | Prefill pp520 (tok/s) | Decode tg128 (tok/s) |
|--------|:---------------------:|:--------------------:|
| **Megakernel** | **37,800** | **413** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HuggingFace | 7,578 | 108 |

Prefill 快 3.4 倍，decode 快 1.55 倍，比 PyTorch 快 3.8 倍。相同硬體，相同模型，相同權重。

### 能源效率（DVFS 功耗掃描）

| 功率限制 | 時脈 | 實際功耗 | tok/s | tok/J | 對比基準 |
|:-----------:|:-----:|:----:|:-----:|:-----:|:--------:|
| 420W（原廠） | 1980 MHz | 314W | 433 | 1.38 | 基準線 |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% 速度，節省 5% 電力 |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% 速度，節省 30% 電力** |
| 150W | 405 MHz | 150W | 194 | 1.29 | 過於激進 |

甜蜜點在 220W：95% 的速度，節省 30% 電力。曲線是非線性的，緊密的執行直接轉化為節省的電力，直到 GPU 被壓榨得太過激進。

### 本不該存在的比較

| 指標 | RTX 3090（llama.cpp） | M5 Max | RTX 3090（Megakernel @220W） |
|--------|:--------------------:|:------:|:---------------------------:|
| tok/s | 267 | 229 | **411** |
| 功耗 | 350W | ~130W | 220W |
| tok/J | 0.76 | 1.76 | **1.87** |
| GPU 售價 | ~$900 | $2,499+（整機） | ~$900 |

一張 2020 年、售價約 900 美元的 GPU，限制功耗至 220W，在效率上媲美 Apple 最新晶片，同時提供 1.8 倍的吞吐量。

## 運作原理

單一持久化 CUDA kernel 在單一 dispatch 中處理整個 Qwen 3.5-0.8B 前向傳播。層與層之間無需返回 CPU。

**架構：** Qwen 3.5-0.8B 是一個混合模型，18 個 DeltaNet 層（帶有學習遞迴的線性注意力）與 6 個完整注意力層，以 3:1 的比例交錯。DeltaNet 相對於標準注意力的二次方複雜度，只需線性複雜度即可擴展上下文長度。這是次世代模型（Qwen3-Next、Kimi Linear）中的新興模式，但沒有任何框架為其提供優化的 kernel。

**Kernel 規格：**
- 82 個 block、512 個執行緒，RTX 3090 上所有 SM 保持滿載
- BF16 權重與激活值，關鍵處使用 FP32 累加
- 透過 warp 協作狀態更新在 F32 暫存器中進行 DeltaNet 遞迴
- 完整注意力機制，帶有 online softmax（融合 QKV、RoPE、因果遮罩、輸出投影）
- 層間使用協作式網格同步取代 kernel 啟動（零層間開銷）
- 在 kernel 內更新 KV 快取
- 直接從 HuggingFace 載入權重

**傳統框架的做法：** 每個 token 啟動約 100 個獨立 kernel，每個都需要付出 CPU 調度、權重重新載入和執行緒同步的代價。megakernel 消除了所有這些開銷。

## 為何 DeltaNet 重要

標準 transformer 已有多年的 kernel 優化：FlashAttention、PagedAttention、連續批次處理。混合 DeltaNet/Attention 架構較新，kernel 生態系尚不成熟：

- **MLX：** 無原生 DeltaNet kernel
- **llama.cpp：** 通用 DeltaNet 支援，無融合
- **vLLM/SGLang：** 透過 flash-linear-attention 的 Triton kernel，但無 megakernel 融合

隨著越來越多的模型走向混合架構（它們會的，因為線性注意力擴展性更好），你在什麼硬體上執行它們，不如*如何*執行它們重要。當你寫出真正善用 GPU 所有功能的 kernel——tensor core、共享記憶體、協作式網格啟動、暫存器常駐狀態——一張五年前的 GPU 就能媲美 Apple 最新晶片。

## 建置過程中的教訓

**`grid.sync()` 在迴圈內會靜默死鎖。** 我們嘗試在每個 token 的 DeltaNet 遞迴迴圈內同步所有 block。沒有錯誤訊息，只是程式掛起。解決方法：在層之間同步，而不是在層內同步。

**暫存器壓力會悄悄殺死效能。** 我們嘗試 `S_TILE=16` 以獲得更多指令級並行。靜默崩潰，沒有 CUDA 錯誤，暫存器溢出到本地記憶體，效能崩潰。`S_TILE=8` 是最佳點。

**功耗曲線是非線性的。** 從 420W 降至 300W 幾乎沒有損失（99.8% 速度）。從 300W 降至 220W 損失很小（95%）。從 220W 降至 150W 直接崩潰至 45%。megakernel 的緊密執行意味著 GPU 在達到功耗上限之前就先達到計算上限，直到被壓榨得太過激進。

## 快速入門

```bash
git clone https://github.com/Luce-Org/luce-megakernel
cd luce-megakernel
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                          # 請先安裝再進行下一步；setup.py 在建置時會引入 torch
pip install -e . --no-build-isolation      # --no-build-isolation 讓建置過程能看到你剛安裝的 torch
python final_bench.py    # 執行 pp520 tg128（正確暖身），印出 tok/s
```

**環境需求：**
- 在 NVIDIA RTX 3090（2020 年）上建置與基準測試；可移植至其他 Ampere+（sm_86+）NVIDIA GPU，需少量調整
- CUDA 12+
- PyTorch 2.0+
- 約 1.5 GB VRAM 用於 BF16 權重

**Blackwell（sm_120 / sm_121a）：** 可在 Blackwell 消費級 GPU（RTX 5090）和 NVIDIA DGX Spark（GB10）上執行，透過 NVFP4 decode 路徑。建置時自動偵測你的 GPU，`final_bench.py` 會調度至正確的後端；使用 `--backend nvfp4` 可強制指定。首批 DGX Spark 數據請見 [RESULTS.md](RESULTS.md#nvidia-dgx-spark-gb10-sm_121a)。

**選用：** 設定功率限制以找到 GPU 的最佳點：
```bash
sudo nvidia-smi -pl 220    # 或你的目標瓦數
```

## 檔案說明

| 檔案 | 說明 |
|------|-------------|
| `kernel.cu` | Decode megakernel，全部 24 層融合於單一 dispatch |
| `prefill.cu` | Prefill（cuBLAS + 獨立 kernel） |
| `torch_bindings.cpp` | PyTorch C++ 綁定 |
| `model.py` | 權重載入 + 解碼器 |
| `setup.py` | 建置設定（架構自動偵測，Blackwell 控制） |
| `final_bench.py` | 基準測試（prefill + decode，10 次暖身 + 20 次計時取平均——發布數據請用此） |
| `bench_pp_tg.py` | 帶有正確性檢查的快速單次基準測試（未暖身，prefill 數據偏低） |
| `RESULTS.md` | 完整基準測試結果與 DVFS 掃描 |
| `kernel_gb10_nvfp4.cu` | 僅 Blackwell：NVFP4 持久化 decode megakernel |
| `prefill_megakernel.cu` | 僅 Blackwell：單一 dispatch prefill megakernel |
| `model_nvfp4.py` | 僅 Blackwell：驅動上述 op 的 NVFP4 解碼器 |
| `final_bench_nvfp4.py` / `bench_pp_tg_nvfp4.py` | 由 `--backend nvfp4` 調度的 Blackwell 基準測試兄弟版本 |

## 範圍與限制

這是一個**研究概念驗證**，而非生產推論伺服器。

- **僅支援 batch size 1。** 目標是單用戶本地推論（llama.cpp/Ollama 使用情境），而非多租戶服務。如果你需要批次吞吐量，請使用 vLLM 或 SGLang。
- **單一模型，單一架構。** kernel 是針對 Qwen 3.5-0.8B 特定層模式（18 個 DeltaNet + 6 個 Attention）手工編寫的。若不重寫則無法推廣至其他模型。
- **僅 BF16。** 不支援量化（GGUF/GPTQ/AWQ）。我們以 BF16 進行基準測試，以將 kernel 級效率與量化取捨隔離開來。
- **0.8B 參數。** 這是個小型模型。隨著模型規模增長，計算開始主導 kernel 啟動開銷，megakernel 融合的優勢會縮小。我們選擇 0.8B 是因為它是第一個可用的混合 DeltaNet 模型，而非因為它能代表所有工作負載。
- **功耗測量方法。** 效率數據只測量加速器功耗（NVIDIA 使用 NVML，Apple 使用 `powermetrics`），遵循 [Hazy Research 的每瓦智能](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) 方法論。兩個平台的整機功耗都更高。
- **正確性。** `bench_pp_tg.py` 包含端對端正確性檢查，對比 megakernel 輸出與參考 decode 路徑。效能數字請使用 `final_bench.py`（正確暖身，10 次暖身 + 20 次計時取平均）。

目標是在消費級硬體上示範架構特定的 kernel 融合能消除真實的效率差距，並以開放的方式進行，讓他人可以重現、批評和延伸這項工作。

## 檔案說明

有任何問題、想法，或想看看其他人在做什麼？加入 [Luce Discord](https://discord.gg/yHfswqZmJQ)。

## 引用

如果你在研究中使用了這項工作：

```bibtex
@software{luce_megakernel_2026,
  title  = {Luce Megakernel: Fused Forward Pass for Hybrid DeltaNet/Attention LLMs},
  author = {Luce},
  url    = {https://github.com/Luce-Org/luce-megakernel},
  year   = {2026}
}
```

## 為何選擇 llama.cpp 作為基準？

llama.cpp 是最廣泛使用的本地推論引擎。這是大多數人實際在消費級 GPU 上執行的軟體。我們也附上了 PyTorch HuggingFace 的數據（慢 3.8 倍）作為第二個參考點。這不是對 llama.cpp 的批評，它是個優秀的專案。這個比較展示了架構特定優化能在通用框架之上解鎖什麼。

## 為何選擇 RTX 3090？

刻意選擇作為 NVIDIA 的「最壞情況」：一張 2020 年的 GPU，被廣泛認為耗電，二手售價約 900 美元。如果軟體差距在舊硬體上確實存在，那麼在更新的顯示卡上差距只會更大。

## 社群

有任何問題、想法，或想看看其他人在做什麼？加入 [Luce Discord](https://discord.gg/yHfswqZmJQ)。

---

MIT · [Lucebox](https://lucebox.com)

以 [Claude](https://claude.ai) 建置

靈感來自 [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)、[Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen)
