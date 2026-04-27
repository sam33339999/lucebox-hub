<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>首個 DFlash 投機解碼的 GGUF 移植版本。</strong><br/>
  Qwen3.5-27B 在單張 RTX 3090 上最高達 207 tok/s<sup>*</sup>（HumanEval 10 個提示詞基準測試平均 129.5 tok/s，DDTree budget=22）。24 GB 內支援 128K 上下文。<br/>
  比自迴歸快 3.43 倍（比鏈式投機解碼多 15%），比 SGLang AWQ 快 2.8 倍。<br/>
  <sub><sup>*</sup>示範運行：207.6 tok/s DFlash vs 38.0 tok/s AR（5.46×）。</sub><br/><br/>
  <a href="https://lucebox.com/blog/dflash27b">部落格文章</a> · <a href="RESULTS.md">基準測試</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                   AR (tok/s)   DFlash (tok/s)   加速倍數
HumanEval             37.78        129.52          3.43x
Math500               37.71        110.51          2.93x
GSM8K                 37.65         96.15          2.55x
```

> 消費級 GPU 可以在不需要多 GPU、不需要批次處理、不需要量化妥協的情況下，以對話級速度運行 27B 模型。瓶頸從來不是硬體，而是解碼演算法。

## 我們填補的差距

在配備 Q4_K_M 權重的 24 GB RTX 3090 上，Qwen3.5-27B 的自迴歸解碼無論使用哪個框架都只能達到約 37.7 tok/s。每個 token 都需要從 VRAM 讀取完整的模型。

投機解碼打破了這個天花板：一個小型草稿模型每步提出多個 token，目標模型在單次前向傳播中驗證它們。[DFlash（z-lab，2026）](https://arxiv.org/abs/2602.06036) 更進一步，採用**區塊擴散草稿**：一個 5 層非因果去噪草稿模型，以捕獲的目標隱藏狀態為條件。每步接受約 8 個 token，相比鏈式 EAGLE 的約 3 個。官方草稿模型為 [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)。[DDTree（Ringel & Romano，2026）](https://arxiv.org/abs/2604.12989) 在此基礎上加入樹狀結構驗證，恢復了最後 30% 的加速效果。

**缺少的部分：** 沒有任何公開實作能在消費級硬體上執行上述任一方法。z-lab 的目標是在 B200 上的 BF16（60+ GB VRAM）。沒有 GGUF 路徑。沒有 DDTree 移植版本。AWQ INT4 目標模型加上 BF16 草稿模型在 24 GB 上無法為驗證樹留出空間。

Q4_K_M GGUF（約 16 GB）是能在單張 RTX 3090 上容納目標模型 + 3.46 GB 草稿模型 + budget=22 樹狀態 + KV 快取的最大量化格式。選擇它迫使我們在 ggml 之上進行移植——ggml 是唯一具備一流 Gated DeltaNet CUDA kernel 和 GGUF Q4_K_M 載入器的執行時期。本 repo 就是這個移植版本：

- 約 2000 行基於 ggml 的 C++/CUDA（無 libllama，無 Python 執行時期）
- 固定版本的 llama.cpp fork，位於 [`Luce-Org/llama.cpp@luce-dflash`](https://github.com/Luce-Org/llama.cpp/tree/luce-dflash)，新增了三個樹狀模式的 ggml op：`ggml_ssm_conv_tree`、`ggml_gated_delta_net_tree`、`ggml_gated_delta_net_tree_persist`
- 硬編碼針對單一模型對，在 HumanEval 上平均解碼速度為 129.52 tok/s

## 結果

Qwen3.5-27B Q4_K_M，concurrency=1，n_gen=256，每資料集 10 個提示詞：

| 任務      | AR tok/s | DFlash+DDTree tok/s | AL   | 加速倍數 |
|-----------|:--------:|:-------------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**          | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**          | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**           | 6.14 | **2.55×** |

AR = 自迴歸（`test_generate`）。DFlash+DDTree = 在 budget=22 下進行樹狀驗證並快速回滾（`test_dflash`）。AL = 接受長度，每次草稿/驗證步驟的平均提交 token 數。可透過 `python3 scripts/bench_llm.py` 重現。

**在 24 GB 內支援最高 256K 上下文**，透過 TQ3_0 KV 快取（3.5 bpv，預設；Q4_0 舊版路徑上限約 128K）+ 滑動 `target_feat` 環形緩衝區（4096 槽）。TQ3 ≈ 相比 F16 節省 9.7× 記憶體；Q4_0 = 8×。

| 提示詞長度 | KV     | Prefill 時間 | Decode tok/s |
|:-------------:|:------:|:------------:|:------------:|
| 520（HE）      | Q8_0   | 0.06 秒      | 130          |
| 32K           | Q8_0   | 38 秒        | 85           |
| 64K           | Q4_0   | 126 秒       | 18           |
| 128K          | Q4_0   | ~10 分鐘     | ~15-20（估計） |

Prefill 數據假設 `--max-ctx` 大小符合提示詞（`run.py` / `bench_llm.py` 中自動調整）。過大設定——例如對 32K 提示詞使用 `--max-ctx=131072`——會觸發 FA 在未使用的 KV 上的跨步遍歷，在該比例下將 prefill 速度降低約 27×。

128K 模式下的 HE 10 個提示詞基準測試平均（ctx=131072，ddtree-budget=16）：**134.78 tok/s**，AL 8.33。

設定 `DFLASH27B_KV_TQ3=1`（TQ3_0，3.5 bpv，預設）或 `DFLASH27B_KV_Q4=1`（Q4_0，4.5 bpv，舊版）以啟用。完整掃描結果請見 [RESULTS.md](RESULTS.md)。

## Qwen3.6-27B 目標模型（實驗性）

Qwen3.6-27B 使用相同的 `qwen35` 架構字串以及與 3.5 相同的層/頭維度，因此 `test_dflash` 無需修改程式碼即可載入：

```bash
# 1. 目標模型
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# 2. 匹配的 3.6 草稿模型（受限：先接受條款 + 設定 HF_TOKEN）
huggingface-cli download z-lab/Qwen3.6-27B-DFlash --local-dir models/draft/

# 3. 基準測試
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf python3 scripts/bench_he.py --n-gen 128
```

> 草稿路徑固定在 `models/draft/model.safetensors`；更換目標模型需同時更換草稿檔案（或讓 `DFLASH_DRAFT` 指向不同的 `model.safetensors`）。

**吞吐量低於 3.5。** z-lab 於 2026-04-26 發布了匹配的 [Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) 草稿模型（仍在訓練中）。隨著草稿模型成熟，AL 應會提升。在同一張 RTX 3090 上的測量結果：

| 目標模型 | 草稿模型 | 基準測試 | AL | 接受率 | 平均 tok/s |
|---|---|---|---:|---:|---:|
| Qwen3.5-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash | HumanEval（README 設定） | 8.33 | ~65% | 134.78 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash（不匹配） | HumanEval（10 個提示詞，n_gen=128） | 4.74 | 30.6% | 73.67 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.6-27B-DFlash（仍在訓練） | HumanEval（10 個提示詞，n_gen=128） | 5.05 | 32.3% | 77.77 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash（不匹配） | Math（10 個提示詞，n_gen=128） | 3.63 | 23.7% | 57.00 |

**Qwen3.6-27B UD-Q4_K_XL**（unsloth Dynamic 2.0，10 個提示詞，n_gen=256，RTX 3090 24 GB，自動調整 `--max-ctx`）的完整 `bench_llm.py` 套件：

| 基準測試 | AR tok/s | DFlash tok/s | AL | 加速倍數 |
|---|---:|---:|---:|---:|
| HumanEval | 34.90 | 78.16 | 5.94 | **2.24×** |
| GSM8K | 34.89 | 59.65 | 4.43 | **1.71×** |
| Math500 | 35.13 | 69.77 | 5.15 | **1.99×** |
| **平均** | 34.97 | 69.19 | 5.17 | **1.98×** |

對比相同測試框架下的 Qwen3.5-27B：2.87× / 2.21× / 2.56×——因草稿不匹配造成跨代均勻下降約 22%，但在零重新訓練的情況下仍達到清晰的約 2×。

等到匹配 Qwen3.6 的 DFlash 草稿模型發布後，透過 `DFLASH_DRAFT=...` 直接替換，無需重新建置。

## 快速入門

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash

# 建置（CUDA 12+、CMake 3.18+、sm_86 相容 GPU；Jetson AGX Thor sm_110 需要 CUDA 13+）
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j

# 下載模型：約 16 GB 目標模型 + 3.46 GB 草稿模型
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/

# 串流單次生成
python3 scripts/run.py --prompt "def fibonacci(n):"

# 多輪對話 REPL
python3 examples/chat.py

# OpenAI 相容 HTTP 伺服器（可直接用於 Open WebUI / LM Studio / Cline）
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn transformers jinja2
.venv/bin/python scripts/server.py --port 8000 --daemon

# 重現論文數據
python3 scripts/bench_llm.py                                 # HE + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # 最小化 HE 基準測試
```

**長上下文模式（最高 256K）：**
```bash
DFLASH27B_KV_TQ3=1 DFLASH27B_PREFILL_UBATCH=16 \
  build/test_dflash models/Qwen3.5-27B-Q4_K_M.gguf \
  models/draft/model.safetensors /tmp/long_prompt.bin 64 /tmp/out.bin \
  --fast-rollback --ddtree --ddtree-budget=16 --max-ctx=4096   # align_up(提示詞 + n_gen + 64, 256)；長提示詞可提高至 262144
```

**環境需求：** NVIDIA sm_86+ GPU（3090、A10、A40、4090）或 Jetson AGX Thor sm_110，CUDA 12+（Thor 需要 CUDA 13+），24 GB VRAM，約 80 GB 磁碟空間。

## 運作原理

**區塊擴散草稿。** 每一步，草稿模型看到 `[last_target_token, MASK×15]` 加上最後 5 個捕獲的目標隱藏狀態。它在單次前向傳播中對遮罩進行去噪，生成 16 個以真實目標特徵為條件的候選 token。比鏈式 EAGLE 結構上更強：每個位置都以相同的捕獲上下文為條件，而非自身的嘈雜預測。

**DDTree 樹狀驗證。** 不是 16 個候選的單一鏈，而是最多 22 個節點的最優先樹，跨越每個位置的 top-K 分支。目標模型透過從父節點指針衍生的因果遮罩，一次性驗證整棵樹。Budget=22 是草稿準確率趨於穩定的最佳點。鏈式預播種很重要：在量化目標上對下等前綴進行純最優先構建加上貪婪驗證可以補救劣質後綴；`build_ddtree` 中的 `chain_seed=true` 標誌將 AL 從約 4 恢復至約 9。

**每步回滾，無需額外 kernel。** 驗證前，目標的遞迴狀態（SSM 中間值、conv 視窗、KV 快取）被快照保存；接受後，恢復至已提交的前綴。三個自訂 CUDA kernel 將回滾保持在關鍵路徑之外：

| Kernel | 用途 |
|--------|---------|
| `ggml_gated_delta_net_tree_persist` | 直接將 SSM 中間值寫入持久化緩衝區，跳過每步 9 毫秒的 `ggml_cpy` |
| `ggml_ssm_conv_tree` | 樹狀感知的 conv 狀態聚集：每個兄弟節點沿 DDTree 父鏈讀取其 K-1 視窗，而非 DFS 順序 |
| 滑動 `target_feat` 環形緩衝區 | 透過 `(pos % cap)` 實現的 4096 槽環形緩衝區，支援 128K 而無需保存 6.6 GB 的捕獲特徵 |

Prefill 和 decode 共享一個圖形建構器；鏈式模式只是帶有 `budget=n_spec+1` 且無分支的 DDTree。

## 架構說明

Qwen3.5-27B **並非**密集型 transformer。llama.cpp 將此架構稱為 `qwen35`：

- 64 層。每 4 層一個完整的 softmax 注意力層，其餘為 **Gated DeltaNet**（帶有學習遞迴的線性注意力）
- M-RoPE，維度段 `[11, 11, 10, 0]`
- 24 個 Q 頭，4 個 KV 頭，key/value 長度 256
- SSM 狀態快取與 KV 快取並列

DeltaNet 原語已是一個一流的 ggml op（`ggml_gated_delta_net`）。我們的 llama.cpp fork 新增了三個樹狀模式變體（`ggml_ssm_conv_tree`、`ggml_gated_delta_net_tree`、`ggml_gated_delta_net_tree_persist`），使 DDTree 驗證可以就地回滾 SSM 狀態，無需重播前向傳播。完整引擎（圖形建構器 + 解碼迴圈 + 回滾 + kernel）約 2000 行。

## 為何不用 llama.cpp / vLLM / z-lab？

- **llama.cpp**：透過 GGUF 執行 Qwen3.5-27B，但沒有 DFlash 整合。鏈式 EAGLE 不夠用；區塊擴散 + DDTree 需要一個繞過 `llama_decode` 的自訂解碼迴圈。
- **vLLM / SGLang**：BF16 的 Qwen3.5-27B 為 54 GB，因此單張 24 GB 顯示卡強制使用量化路徑。截至 2026-04，此架構的 GGUF 在 SGLang 上已損壞，vLLM 正在放棄 GGUF 支援。AWQ 在 SGLang 上以純自迴歸方式執行，達到 46.6 tok/s，但無法在 24 GB 上同時容納 BF16 草稿模型 + DDTree 樹狀態。Q4_K_M GGUF 是唯一能容納完整投機解碼堆疊的格式，本 repo 在 HumanEval 上平均達到 129.5 tok/s，**比相同硬體上的 SGLang AWQ 自迴歸快 2.8 倍**。
- **z-lab 參考實作**：vLLM / SGLang 整合將 DFlash 作為投機解碼方法發布，但僅針對在 NVIDIA B200 上（54+ GB VRAM）的 BF16 權重。沒有 GGUF 路徑。

## 範圍與限制

研究概念驗證，非生產環境。

- **Batch size 1**，單用戶本地推論目標（Ollama / LM Studio 使用情境）
- **單一模型對**：Qwen3.5-27B Q4_K_M 目標模型 + z-lab DFlash BF16 草稿模型。若不重寫圖形建構器則無法推廣。
- **僅貪婪解碼**：OpenAI 伺服器接受 `temperature`/`top_p` 但忽略。驗證路徑中的拒絕採樣是週末大小的添加工作。
- **僅 CUDA sm_86+ / sm_110 Thor**。不支援 Metal、ROCm、多 GPU。
- **Q4_K_M 目標模型**相比論文的 BF16，每個位置接受率損失約 30 分。Q5_K_M / Q6_K 能恢復大部分，如果它們適合記憶體的話。

正確性：`test_vs_oracle` 以 cos sim 0.999812 對比 PyTorch 參考驗證草稿圖。目標圖語義上與 llama.cpp 的 `models/qwen35.cpp` 匹配，並在自迴歸模式下與 `test_generate` 產生位元完全相同的輸出。

## 貢獻

對 `Luce-Org/lucebox-hub` 開 issue 或 PR。好的入門項目：

- 驗證路徑中的**溫度 / top-k 採樣**
- **完整 llama.cpp 整合**：新架構，`llama-speculative-dflash.cpp`，`llama-cli` / `llama-server` 接線

## 引用

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: GGUF port of block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/dflash},
  year   = {2026}
}

@article{dflash2026,
  title   = {DFlash: Block-Diffusion Speculative Decoding},
  author  = {z-lab},
  journal = {arXiv:2602.06036},
  year    = {2026}
}

@article{ddtree2026,
  title   = {Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author  = {Ringel, Liran and Romano, Yaniv},
  journal = {arXiv:2604.12989},
  year    = {2026}
}
```

---

MIT · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

靈感來自 [z-lab/DFlash](https://arxiv.org/abs/2602.06036)、[liranringel/ddtree](https://github.com/liranringel/ddtree)、[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)。
