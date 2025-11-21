# LMCacheConnectorV1 アーキテクチャ詳細ドキュメント

## 目次
1. [概要](#概要)
2. [システムアーキテクチャ](#システムアーキテクチャ)
3. [GPU通信メカニズム](#gpu通信メカニズム)
4. [KV-Cacheのやり取り方法](#kv-cacheのやり取り方法)
5. [GDSバックエンド最適化](#gdsバックエンド最適化)
6. [実装例と詳細](#実装例と詳細)

---

## 概要

`LMCacheConnectorV1`は、vLLMとLMCacheを統合して、分散KV-キャッシュ管理と外部キャッシュシステムとの通信を効率的に処理するコンポーネントです。

### 主な特徴
- **役割分離**: Scheduler側（スケジュール管理）とWorker側（実行管理）の明確な分離
- **非同期操作**: 層ごとの非同期ロード/セーブに対応
- **パイプライン化**: レイヤーごとの逐次実行による高速化
- **GDS対応**: GPUDirectStorage（GPUDirect Storage）を活用した高速転送
- **マルチモーダル対応**: マルチモーダル入力のハッシュ管理

---

## システムアーキテクチャ

### ファイル構成

```
vllm/distributed/kv_transfer/kv_connector/v1/
├── lmcache_connector.py                      # ラッパーと主インターフェース
├── base.py                                   # 基底クラス定義
├── lmcache_integration/
│   ├── vllm_v1_adapter.py                    # メイン実装（約1400行）
│   └── utils.py                              # ユーティリティ関数
└── factory.py                                # コネクタファクトリ
```

### 役割分離

#### Scheduler側の責務
- `get_num_new_matched_tokens()`: 外部キャッシュから何トークン読み込めるか判定
- `update_state_after_alloc()`: ブロック割当後のコネクタ状態更新
- `build_connector_meta()`: スケジューラ出力からメタデータ生成
- `request_finished()`: リクエスト完了時の非同期セーブ管理

#### Worker側の責務
- `start_load_kv()`: 外部キャッシュからロード開始（非同期可能）
- `wait_for_layer_load()`: 特定層のロード完了待機
- `save_kv_layer()`: 層のセーブ開始（非同期可能）
- `wait_for_save()`: すべてのセーブ完了待機
- `get_finished()`: 非同期転送完了のリクエストID取得

---

## GPU通信メカニズム

### GPU Connectorの種類

LMCacheConnectorV1は3つのGPU Connectorをサポートしています：

```python
# vllm_v1_adapter.py より
from lmcache.v1.gpu_connector import (
    VLLMPagedMemGPUConnectorV2,           # デフォルト、高性能
    VLLMPagedMemLayerwiseGPUConnector,   # レイヤーワイズ処理
    VLLMBufferLayerwiseGPUConnector,     # 中間バッファ併用
)
```

#### 1. VLLMPagedMemGPUConnectorV2（デフォルト）
- **特徴**: ページング管理されたメモリを直接使用
- **最適化**: MLA（Multi-head Latent Attention）対応
- **用途**: 一般的な大規模モデル
- **処理フロー**:
  ```
  KV Cache (LMCache) -> GPU Memory -> vLLM Paged Buffer
  ```

#### 2. VLLMPagedMemLayerwiseGPUConnector
- **特徴**: レイヤーごとに逐次処理
- **最適化**: パイプライン実行による高速化
- **用途**: 深いモデルの層ごと処理
- **処理フロー**:
  ```
  層1: KV -> GPU -> vLLM (並行実行)
  層2: KV -> GPU -> vLLM
  ...
  ```

#### 3. VLLMBufferLayerwiseGPUConnector
- **特徴**: 中間バッファを経由
- **最適化**: ブレンディング処理での互換性
- **用途**: キャッシュの動的ブレンディング時
- **処理フロー**:
  ```
  KV -> Intermediate Buffer -> GPU -> vLLM
  ```

### GPU通信の詳細フロー

#### ロード（Load）フロー

```
1. start_load_kv(forward_context)
   ↓
2. forward_context からKV-cacheを抽出
   ├─ KV cache dict: {layer_name: tensor}
   ├─ slot_mapping: ブロック→物理メモリ位置マップ
   └─ token_masks: vLLMキャッシュヒットをスキップ
   ↓
3. GPU Connectorへ渡す
   ├─ retrieve(): 全層一括ロード
   └─ retrieve_layer(layer_name): レイヤーワイズロード
   ↓
4. wait_for_layer_load(layer_name)
   └─ ロード完了まで待機（Attention層で呼び出し）
```

**実装コード** (vllm_v1_adapter.py:600-700)：
```python
def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
    # 各リクエストのLoadSpecを確認
    load_specs = self._req_meta_dict.get('load_specs', {})

    for req_id, load_spec in load_specs.items():
        # KVをforwardContextから抽出
        kv_caches = forward_context.kv_caches

        # slot_mappingを作成
        slot_mapping = create_slot_mapping(...)

        # LMCacheのretrieve()呼び出し
        self._lmcache_engine.retrieve(
            token_ids=token_ids,
            mask=mask,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping.cuda(),
        )
```

#### セーブ（Save）フロー

```
1. save_kv_layer(layer_name, kv_layer, attn_metadata)
   ↓
2. セーブ判定
   ├─ SaveSpec確認:
   │  ├─ skip_leading_tokens: 既保存トークンをスキップ
   │  └─ can_save: スケジューラが保存許可？
   │
   └─ チャンク境界確認（256トークンごと）
      └─ 部分的チャンクは最後のprefill時のみ保存
   ↓
3. セーブ実行
   ├─ store(): 全層一括セーブ
   └─ store_layer(layer_name): レイヤーワイズセーブ
   ↓
4. wait_for_save()
   └─ すべてのセーブ完了待機
```

**実装コード** (vllm_v1_adapter.py:700-800)：
```python
def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                   attn_metadata: "AttentionMetadata", **kwargs: Any) -> None:
    # SaveSpec取得
    save_spec = self._req_meta_dict.get('save_spec', {})

    # チャンク単位でのセーブ（デフォルト256トークン）
    chunk_size = self._lmcache_config.chunk_size  # 256

    for req_id, spec in save_spec.items():
        if not spec.can_save:
            continue

        # 既保存トークンをスキップ
        skip_tokens = spec.skip_leading_tokens

        # セーブ対象のトークンを計算
        num_tokens_to_save = calculate_tokens_to_save(
            total_tokens=len(token_ids),
            chunk_size=chunk_size,
            skip_tokens=skip_tokens
        )

        # LMCacheにセーブ
        self._lmcache_engine.store(
            token_ids=token_ids[skip_tokens:],
            kv_caches=kv_layer,
            slot_mapping=slot_mapping,
        )
```

### Slot Mappingメカニズム

Slot MappingはvLLMの物理メモリ位置マッピングです：

```python
# ReqMeta.from_request_tracker() より (vllm_v1_adapter.py:358-365)
block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
block_offsets = torch.arange(0, block_size, dtype=torch.long)
slot_mapping = (
    block_offsets.reshape((1, block_size))
    + block_ids.reshape((num_blocks, 1)) * block_size
)
slot_mapping = slot_mapping.flatten()[:len(token_ids)]
```

**例**:
```
block_size = 16, allocated_block_ids = [10, 20, 30]

slot_mapping:
  Block 10: [160, 161, 162, ..., 175]  (10*16 から 10*16+15)
  Block 20: [320, 321, 322, ..., 335]  (20*16 から 20*16+15)
  Block 30: [480, 481, 482, ..., 495]  (30*16 から 30*16+15)
```

このマッピングにより、GPU側のlmcache engineは確実に正しいメモリ位置にKVを配置できます。

---

## KV-Cacheのやり取り方法

### データ構造

#### LoadSpec（読込み仕様）
```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int        # vLLMで既にキャッシュされているトークン数
    lmcache_cached_tokens: int     # LMCacheにキャッシュされているトークン数
    can_load: bool                 # スケジューラが読込を許可しているか
```

#### SaveSpec（保存仕様）
```python
@dataclass
class SaveSpec:
    skip_leading_tokens: int       # 既に保存済みのトークン数（スキップ）
    can_save: bool                 # スケジューラが保存を許可しているか
```

#### ReqMeta（リクエストメタデータ）
```python
@dataclass
class ReqMeta:
    req_id: str                    # リクエストID
    token_ids: list[int]           # トークンID
    slot_mapping: torch.Tensor     # GPU内のメモリマッピング
    is_last_prefill: bool          # 最後のプリフィル段階か
    save_spec: SaveSpec | None     # 保存仕様
    load_spec: LoadSpec | None     # 読込仕様
    disagg_spec: DisaggSpec | None # 非集約仕様（分散実行）
    request_configs: dict | None   # リクエスト固有の設定
```

### チャンキング戦略

LMCacheConnectorV1は**256トークンチャンク**を基本単位とします：

```python
# vllm_v1_adapter.py:295-328
lmcache_chunk_size = 256  # 設定可能

chunk_boundary = (
    cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
)

# 部分的チャンク（255トークンなど）の扱い:
# - 通常: スキップ（フラグメンテーション防止）
# - 最後のprefill: 保存

num_tokens_to_save = (
    (input_token_len // lmcache_chunk_size * lmcache_chunk_size)
    if not is_last_prefill or discard_partial_chunks
    else input_token_len
)
```

**メリット**:
- メモリフラグメンテーション削減
- メモリアライメント最適化
- 転送効率向上

### RequestTrackerライフサイクル

```
1. from_new_request()
   ├─ 新規リクエスト受け取り
   ├─ prompt_token_ids抽出
   └─ block_ids記録
   ↓
2. update()（スケジュール時に繰り返し）
   ├─ 新しいトークンID追加
   ├─ 新しいブロックID追加
   └─ decode phase判定（新トークン=1）
   ↓
3. ReqMeta.from_request_tracker()
   ├─ slot_mapping生成
   ├─ SaveSpec/LoadSpec統合
   └─ マルチモーダルハッシュ適用
   ↓
4. request_finished()
   └─ 非同期セーブ開始（必要時）
```

### マルチモーダル対応

マルチモーダル入力のハッシュトークン置き換え：

```python
# utils.py:83-99
def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
) -> torch.Tensor:
    """マルチモーダルプレースホルダーをハッシュで上書き"""
    n = token_ids.size(0)
    for hash_str, placeholder in zip(mm_hashes, mm_positions):
        start, length = placeholder.offset, placeholder.length
        if start >= n:
            continue
        end = min(start + length, n)
        # ハッシュ値をトークンIDに適用
        token_ids[start:end] = hex_hash_to_int16(hash_str)
    return token_ids
```

---

## GDSバックエンド最適化

### バックエンド種類と構成

```
LMCache Engine
├─ Backend: Local CPU
│  └─ 特徴: プロセス内キャッシュ、最短遅延
│
├─ Backend: Remote Server
│  ├─ Protocol: ZMQ / gRPC
│  ├─ Serialization: naive / msgpack / pickle
│  └─ 特徴: ネットワーク経由、スケーラブル
│
├─ Backend: GDS (GPUDirect Storage)
│  ├─ Hardware: NVMe SSD + GPU
│  ├─ Protocol: RDMA (Mellanox等)
│  └─ 特徴: GPU直結、DMA転送、最高速
│
└─ Backend: NixL (RDMA)
   ├─ Hardware: RDMA対応NIC + GPU
   ├─ Protocol: RDMA Verbs
   └─ 特徴: 完全非同期、ホストバイパス
```

### GDS使用時の最適化

#### 1. ホストバイパス転送

**従来の転送パス**:
```
GPU Memory -> Host Memory -> Storage
  ↓            ↓
  PCI-e        PCIe/Network
```

**GDS転送パス**:
```
GPU Memory -> Storage (Direct)
  ↓
  NVMe Controller
  (No Host CPU Involvement)
```

**実装** (vllm_v1_adapter.py:409-450):
```python
def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
) -> LMCacheEngine:

    # GDS バックエンド使用判定
    if lmcache_config.backend == "gds":
        # NixLベースのホストバイパス転送
        connector = create_gds_connector(
            enable_rdma=True,
            backend_config={
                'cache_dir': '/mnt/nvme',
                'max_concurrent_ops': 4,
                'block_size': 1048576,  # 1MB blocks
            }
        )
```

#### 2. ホストメモリバッファ排除

GDS使用時: ホストメモリを経由しない

```python
# GDSの場合
GPU Register -> GPU L2 Cache -> NVMe SSD

# 非GDS（従来）の場合
GPU Register -> Host Memory -> Disk
                    ↓
             CPU Side Effect
```

**メモリ効率**: GPU VRAM削減（ホストバッファ不要）

#### 3. 非同期ジェネレータベース実行

層ごとの非同期処理でGDS遅延をマスク：

```python
# vllm_v1_adapter.py (レイヤーワイズ)

def retrieve_layer() -> Generator:
    """レイヤーごとに非同期でロード"""
    for layer_idx in range(num_layers):
        # Layer N-1をロード中に
        # Layer Nを計算（パイプライン化）
        yield layer_idx

        # GDS読込み遅延をマスク
        async_wait(layer_idx)
```

#### 4. 圧縮とキャッシュコヒーレンス

GDS時の最適化:

```
設定項目:
├─ enable_compression: KV圧縮（int8等）
├─ compression_ratio: 圧縮率（デフォルト1.0=なし）
├─ enable_cache_coherence: マルチGPU間の一貫性保証
└─ prefetch_window: 先読みトークン数

効果:
├─ 転送データ量: 最大75%削減（圧縮時）
├─ 帯域幅: NVMe帯域フル利用可能
└─ レイテンシ: パイプライン化で隠蔽
```

### GDS非使用時の最適化

#### 1. CPU キャッシュ利用

GDS未使用時（Local/Remote backend）:

```python
# ホストメモリ バッファ経由
GPU <- Host L3 Cache <- Storage
           ↑
     CPU キャッシュ効果
```

**最適化ポイント**:
```python
- Host buffer size: GPU VRAM の20-30%推奨
- CPU prefetch: read-ahead有効化
- NUMA awareness: ローカルCPUノード優先
```

#### 2. ネットワーク最適化（Remote使用時）

```python
# vllm_v1_adapter.py より
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer

lmcache_config.remote_url = "tcp://cache-server:5000"
lmcache_config.remote_serde = "msgpack"  # バイナリ効率重視

# 環境変数での設定
os.environ['LMCACHE_REMOTE_URL'] = 'tcp://cache-server:5000'
os.environ['LMCACHE_REMOTE_SERDE'] = 'msgpack'
```

**シリアライゼーション選択**:
```
naive:   圧縮なし、最速（ローカルネットワーク向け）
msgpack: 適度な圧縮、バランス型
pickle:  Python固有、非推奨（セキュリティ）
```

#### 3. Local Backend （CPU内メモリ）

最小遅延、キャッシュ効果最大:

```python
lmcache_config.backend = "local"  # CPU内メモリキャッシュ
lmcache_config.cache_size_gb = 64  # ホストRAM内

# 最適:
#   - 単一マシン、複数GPU
#   - キャッシュサイズ < ホストRAM
#   - 低レイテンシ要求
```

---

## 実装例と詳細

### 例1: デフォルト設定での実行フロー

```python
# 初期化
config = VllmConfig(
    kv_transfer_config=KVTransferConfig(
        connector='lmcache',
        extra_config={
            'use_native': False,  # 最新版LMCache使用
        }
    )
)

connector = LMCacheConnectorV1(config, KVConnectorRole.WORKER)

# Scheduler側（build_connector_meta）
metadata = scheduler_connector.build_connector_meta(scheduler_output)
#  ↓ メタデータ送信
# Worker側で受け取り
worker_connector.bind_connector_metadata(metadata)

# Forward前のロード開始（非同期）
worker_connector.start_load_kv(forward_context)

# Forward実行（パイプライン化）
for layer_idx, layer in enumerate(model.layers):
    # このレイヤーのロード完了待機
    worker_connector.wait_for_layer_load(f"layer_{layer_idx}")

    # Forward実行
    output = layer(input, attention_metadata)

    # セーブ開始（非同期）
    worker_connector.save_kv_layer(f"layer_{layer_idx}", kv, attention_metadata)

# Forward完了後、すべてのセーブ待機
worker_connector.wait_for_save()

# リクエスト完了時の非同期セーブ管理
async_blocks = scheduler_connector.request_finished(request, block_ids)
```

### 例2: GDS バックエンド使用時

```python
# GDS設定ファイル (lmcache_config.yaml)
backend: gds
cache_dir: /mnt/nvme/llm_cache
gds_config:
  enable_rdma: true
  max_concurrent_ops: 4
  block_size: 1048576

# コネクタ初期化時にGDSが自動選択される
# 環境変数
export LMCACHE_CONFIG_FILE=/path/to/lmcache_config.yaml

# GPU直結転送、ホストバイパス
# Transfer: GPU -> NVMe (DMA, No CPU involved)
```

### 例3: 層ワイズパイプライン処理

```python
# 設定
lmcache_config.enable_layerwise = True
lmcache_config.layerwise_connector = \
    "VLLMPagedMemLayerwiseGPUConnector"

# Forward時のフロー
# 時刻 t:    Layer 0 計算中 | Layer 1 ロード中
# 時刻 t+1:  Layer 1 計算中 | Layer 2 ロード中
# 時刻 t+2:  Layer 2 計算中 | Layer 3 ロード中
#
# パイプライン効果: レイテンシをマスク
# Forward時間 ≈ max(compute_time, load_time)
```

---

## 設定とトレーニング

### 主要な設定パラメータ

```python
# vllm_v1_adapter.py より抽出

# チャンク設定
chunk_size = 256  # トークン単位

# 非同期設定
enable_async_load = True           # 非同期ロード
enable_async_save = True           # 非同期セーブ

# レイヤーワイズ設定
enable_layerwise = False           # レイヤーワイズ処理
layerwise_connector_type = "paged" # "paged" or "buffer"

# ブレンディング設定
enable_pd = False                  # Progressive Distillation
blending_ratio = 0.5               # キャッシュ活用率

# MLA対応
use_mla = False                    # Multi-head Latent Attention

# セーブ設定
save_decode_cache = False          # decodeフェーズでセーブ
discard_partial_chunks = True      # 部分チャンクをスキップ

# 分散実行設定
enable_disaggregation = False      # 分散prefill対応
```

### 環境変数での制御

```bash
# LMCache設定ファイル
export LMCACHE_CONFIG_FILE=/path/to/config.yaml

# リモートバックエンド
export LMCACHE_REMOTE_URL=tcp://cache-server:5000
export LMCACHE_REMOTE_SERDE=msgpack

# GDS設定
export LMCACHE_GDS_ENABLED=1
export LMCACHE_GDS_BLOCK_SIZE=1048576

# デバッグ
export LMCACHE_LOG_LEVEL=DEBUG
```

---

## パフォーマンス特性

### レイテンシ比較

```
ロード遅延:
├─ Local CPU:    < 1ms (GPU近傍メモリ)
├─ Remote TCP:   10-100ms (ネットワーク遅延)
└─ GDS NVMe:     5-50ms (DMA, ホストバイパス)

セーブ遅延:
├─ Local CPU:    < 1ms
├─ Remote TCP:   50-200ms
└─ GDS NVMe:     10-100ms
```

### スループット特性

```
スループット:
├─ Local CPU:    最大メモリBW (~100GB/s)
├─ Remote TCP:   最大ネットBW (~1-10GB/s)
└─ GDS NVMe:     NVMe BW (~3-7GB/s, GPU↔直結)
```

---

## ジェネレータベースの層ワイズ非同期処理

### 層ワイズレトリーバーの実装

vllm_v1_adapter.py では、層ワイズ処理にジェネレータを使用しています：

```python
# start_load_kv() より (vllm_v1_adapter.py:829-852)

# 非同期ジェネレータの作成と初期化
for idx, request in enumerate(metadata.requests):
    if request.load_spec is None:
        continue

    # retrieve_layer()がジェネレータを返す
    layerwise_retriever = self.lmcache_engine.retrieve_layer(
        tokens[:lmcache_cached_tokens],
        token_mask[:lmcache_cached_tokens],
        kvcaches=kvcaches,
        slot_mapping=slot_mapping[:lmcache_cached_tokens],
        sync=sync,  # 最後のリクエストはsync=True
    )

    # 最初の2層を先読み（パイプライン化開始）
    next(layerwise_retriever)
    next(layerwise_retriever)

    self.layerwise_retrievers.append(layerwise_retriever)
```

### 層ワイズ待機メカニズム

```python
# wait_for_layer_load() より (vllm_v1_adapter.py:880-901)

def wait_for_layer_load(self, layer_name: str) -> None:
    """各レイヤーでの呼び出し時にそのレイヤーのロード完了を待機"""

    # すべてのレイヤーワイズレトリーバーから次の層のロード完了を待つ
    for layerwise_retriever in self.layerwise_retrievers:
        ret_token_mask = next(layerwise_retriever)  # ブロック

        # 最後の層で統計情報を出力
        if self.current_layer == self.num_layers - 1:
            assert ret_token_mask is not None
            num_retrieved_tokens = ret_token_mask.sum().item()
            logger.info("Retrieved %s tokens", num_retrieved_tokens)

    return
```

### パイプライン化のタイムライン

```
時刻 t:     Layer 0計算中 | Layer 1ロード準備中
時刻 t+1:   Layer 1計算中 | Layer 2ロード準備中
時刻 t+2:   Layer 2計算中 | Layer 3ロード準備中
...
時刻 t+N:   Layer Nロード | 計算完了

利点:
- ロード遅延をマスク
- GPU計算と転送のオーバーラップ
- 全体スループット向上（compute time に接近）
```

### 層ワイズストア（セーブ）の実装

```python
# save_kv_layer() より (vllm_v1_adapter.py:939-1002)

# 最初の層でストアジェネレータを作成
if self.current_layer == 0:
    self.layerwise_storers = []

    for request in connector_metadata.requests:
        save_spec = request.save_spec
        if save_spec is None or not save_spec.can_save:
            continue

        # チャンク境界アライン（スキップトークンをチャンク倍数に）
        skip_leading_tokens = (
            skip_leading_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )

        # マスク作成（既保存トークンはスキップ）
        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        # ストアジェネレータ作成
        layerwise_storer = self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            sync=is_first,  # 最初のリクエストでsync
        )
        self.layerwise_storers.append(layerwise_storer)

# 各層でのストア進行
for layerwise_storer in self.layerwise_storers:
    next(layerwise_storer)  # 層ワイズストア実行

self.current_layer += 1

# すべてのレイヤー完了時に最終待機
# wait_for_save() 内で最後のnext()を実行
for layerwise_storer in self.layerwise_storers:
    next(layerwise_storer)  # 最終完了待機
```

---

## Token Mask と Store Mask

### Token Mask（ロード時）

```python
# start_load_kv() より (vllm_v1_adapter.py:820-826)

# vLLMですでにキャッシュされているトークンをマスク
token_mask = torch.ones(len(tokens), dtype=torch.bool)

# vLLMキャッシュヒットのトークン数（チャンク粒度）
masked_token_count = (
    request.load_spec.vllm_cached_tokens
    // self._lmcache_chunk_size
    * self._lmcache_chunk_size
)

# False: ロードスキップ, True: ロード対象
token_mask[:masked_token_count] = False
```

例：
```
vLLMキャッシュ: 128トークン (チャンク粒度で512トークン分)
LMCacheキャッシュ: 1000トークン

token_mask:
  [:512]  = False  (vLLMで既にキャッシュ → スキップ)
  [512:]  = True   (LMCacheから読込)

最終結果: トークン[512:1000]をLMCacheから読込
```

### Store Mask（セーブ時）

```python
# save_kv_layer() より (vllm_v1_adapter.py:973-974)

store_mask = torch.ones(len(token_ids), dtype=torch.bool)
store_mask[:skip_leading_tokens] = False  # 既保存トークンをスキップ
```

例：
```
総トークン: 1000
既セーブ: 768トークン

store_mask:
  [:768]  = False  (既セーブ → スキップ)
  [768:]  = True   (新規セーブ)

セーブ対象: トークン[768:1000] の 232トークン
```

---

## 実行ロール（kv_role）による動作の違い

### kv_producer ロール

KVキャッシュの生成を担当するWorker：

```python
# save_kv_layer() より (vllm_v1_adapter.py:959-962)

if self.kv_role == "kv_producer":
    skip_leading_tokens = 0  # 最初からセーブ
else:
    # 既セーブ分をスキップ
    skip_leading_tokens = save_spec.skip_leading_tokens
```

**特徴**:
- すべてのトークンをセーブ
- チャンク粒度アライン不要
- prefill時にフルセーブ

### kv_consumer ロール

KVキャッシュを消費のみするWorker：

```python
# save_kv_layer() より (vllm_v1_adapter.py:925-927)

if self.kv_role == "kv_consumer":
    # Don't do save if the role is kv_consumer
    return
```

**特徴**:
- セーブ操作スキップ
- キャッシュの読込のみ
- 分散prefill時の受信側

---

## エラーハンドリング

### ロードエラー検出

```python
# start_load_kv() より (vllm_v1_adapter.py:863-878)

# 期待値との比較
num_retrieved_tokens = ret_token_mask.sum().item()
num_expected_tokens = (
    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
)

if num_retrieved_tokens < num_expected_tokens:
    logger.error(
        "The number of retrieved tokens is less than the "
        "expected number of tokens! This should not happen!"
    )
    logger.error(
        "Num retrieved tokens: %d, num expected tokens: %d",
        num_retrieved_tokens,
        num_expected_tokens,
    )
```

**原因と対策**:
- ネットワーク接続エラー → Retry実装
- キャッシュ削除（期限切れ） → キャッシュ再生成
- ディスク容量不足 → キャッシュクリーンアップ

---

## パフォーマンス最適化チェックリスト

### GPU Connector選択
```
[ ] MLA対応モデルか？
    YES → VLLMPagedMemGPUConnectorV2
    NO  → 要確認
[ ] レイヤーワイズパイプライン必要か？
    YES → VLLMPagedMemLayerwiseGPUConnector
[ ] キャッシュ動的ブレンディング使用か？
    YES → VLLMBufferLayerwiseGPUConnector
```

### バックエンド選択
```
[ ] シングルマシン複数GPU
    → Local または GDS推奨
[ ] マルチマシン分散
    → Remote (TCP) or NixL (RDMA)
[ ] 遅延クリティカル
    → GDS + LayerwiseConnector
[ ] スループット重視
    → Local または NVMe GDS
```

### チャンク設定
```
[ ] デフォルト256トークン で十分か
[ ] メモリ制約が厳しい？
    → chunk_size削減（64/128）
[ ] ネットワーク遅延大きい？
    → chunk_size増加（512/1024）
```

### スレッドセーフティ
```
[ ] ジェネレータの同時アクセス？
    → ThreadLocal で管理
[ ] CUDAストリーム競合？
    → 明示的なストリーム指定
[ ] メタデータの競合？
    → ロック機構確認
```

---

## まとめ

LMCacheConnectorV1は以下の特徴で効率的なKV-キャッシュ管理を実現します：

1. **GPU通信**: 複数のGPU Connector選択肢で様々なワークロード対応
   - ページドメモリ管理
   - MLA対応
   - レイヤーワイズパイプライン

2. **チャンキング**: 256トークン単位でメモリ効率と転送効率を最適化
   - フラグメンテーション削減
   - アライメント最適化
   - 部分チャンク処理

3. **非同期処理**: ジェネレータベースの層ワイズパイプライン化でロード遅延をマスク
   - GPU計算と転送のオーバーラップ
   - 動的なロード/セーブ管理
   - 柔軟なスケジュール対応

4. **GDS対応**: ホストバイパス転送で最高速性能実現
   - GPU直結NVMeアクセス
   - RDMA統合
   - CPU干渉排除

5. **柔軟性**: Local/Remote/GDSバックエンド自由選択
   - 環境に応じた最適化選択
   - スケーラビリティ確保
   - 拡張性維持

### 選択すべきバックエンド
- **シングルマシン複数GPU**: Local または GDS
- **マルチマシン分散**: Remote (TCP) または NixL (RDMA)
- **超低遅延要求**: GDS + LayerwiseConnector
- **最高スループット**: Local + LayerwiseConnector
