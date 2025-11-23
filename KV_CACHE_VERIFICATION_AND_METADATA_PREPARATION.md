# KV Cache 存在確認とメタデータ準備メカニズム

---

## 概要

KV-Cacheの取得は単純な「ファイル読込」ではなく、複雑な多段階のメタデータ準備と検証プロセスです。以下のステークホルダーが関わります：

```
┌─────────────────────┐
│ Scheduler Process   │  ← キャッシュ命中判定、メタデータ生成
└──────────┬──────────┘
           │ (メタデータ送信)
           ↓
┌─────────────────────┐
│ Worker Process      │  ← GPU ロード実行
└──────────┬──────────┘
           │ (LMCache engineへ委譲)
           ↓
┌─────────────────────┐
│ LMCache Engine      │  ← 実際のKV検索・取得（外部ライブラリ）
│ (外部ライブラリ)    │
└─────────────────────┘
```

---

## フェーズ 1：Scheduler側の存在確認

### 1.1 Token ID 収集

**リクエスト作成時**:
```python
# vllm/v1/core/sched/scheduler.py:400-410
# または
# vllm/v1/request.py

request = Request(
    request_id="req_123",
    prompt_token_ids=[1, 2, 3, ..., 2048],  ← トークンID列
    sampling_params=SamplingParams(...),
)
```

**マルチモーダルハッシング**（画像などがある場合）:
```python
# vllm_v1_adapter.py:1140-1145 より

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils \
    import apply_mm_hashes_to_token_ids, extract_mm_features

# Step 1: マルチモーダル特徴を抽出
mm_hashes, mm_positions = extract_mm_features(request)
# mm_hashes: ["a1b2c3d4", "e5f6g7h8"]  (画像のハッシュ)
# mm_positions: [PlaceholderRange(offset=0, length=256), ...]

# Step 2: トークンIDの一部をハッシュで置き換え
if mm_hashes and mm_positions:
    token_ids_tensor = torch.tensor(request.prompt_token_ids)
    apply_mm_hashes_to_token_ids(
        token_ids_tensor,    # トークンID配列（GPU に移動）
        mm_hashes,           # 画像ハッシュ
        mm_positions         # 置き換え位置
    )
    # 結果: token_ids[0:256] が画像ハッシュに置き換わる
    token_ids = token_ids_tensor.tolist()

# Step 3: キャッシュ検索用のトークンID確定
# [image_hash, 5, 6, ..., 2048]  ← これが Lookup に渡される
```

### 1.2 Lookup Client への問い合わせ

**ファイル**: `vllm_v1_adapter.py:1113-1200`

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    """
    外部キャッシュから何トークン読み込めるか判定する

    Called from: Scheduler (vllm/v1/core/sched/scheduler.py:408-426)
    """

    # 1. トークンID取得（既にマルチモーダルハッシュ適用済み）
    token_ids = request.prompt_token_ids  # [image_hash, 5, 6, ..., 2048]

    # 2. Lookup ID 生成（非同期操作の追跡用）
    lookup_id = (
        request.request_id  # "req_123"
        if self.async_loading
        else str(uuid.uuid4())  # "uuid-..." (同期モード)
    )
    self._lookup_requests_in_step.append(lookup_id)

    # 3. 【重要】Lookup Client へ問い合わせ
    # ← この時点で LMCache バックエンドが検索実行
    num_external_hit_tokens = self.lookup_client.lookup(
        token_ids=token_ids,              # [image_hash, 5, 6, ...]
        lookup_id=lookup_id,              # "req_123" または UUID
        request_configs=extract_request_configs(request.sampling_params),
    )

    # 4. 戻り値の解釈
    # - num_external_hit_tokens: 外部キャッシュで見つかったトークン数
    #   例: 1024 (最初の1024トークンがキャッシュに存在)
    #   または: None (キャッシュ検索中、後で再問い合わせ)

    if num_external_hit_tokens is None:
        logger.debug(f"Lookup pending for {request.request_id}")
        return None, False  # Scheduler は この request をスキップ

    # 5. 割り当てトークン数を計算
    need_to_allocate = num_external_hit_tokens - num_computed_tokens
    # 例: num_computed_tokens=256 (vLLM ローカルキャッシュ)
    #     num_external_hit_tokens=1024 (LMCache)
    #     → need_to_allocate=768 (新たに割り当て必要)

    # 6. 最後のトークンは再計算（キャッシュ効果を活かすため）
    if num_external_hit_tokens == request.num_tokens:
        need_to_allocate -= 1

    # 7. LoadSpec を保存（Worker側で使用）
    self.load_specs[request.request_id] = LoadSpec(
        vllm_cached_tokens=num_computed_tokens,      # 256
        lmcache_cached_tokens=num_external_hit_tokens,  # 1024
        can_load=False,  # ← 後で allocate 後に True に
    )

    logger.info(
        f"Request {request.request_id}: "
        f"vLLM={num_computed_tokens}, LMCache={num_external_hit_tokens}, "
        f"allocate={need_to_allocate}"
    )

    return need_to_allocate, False
```

### 1.3 Lookup Client の実装

**外部ライブラリ** (`from lmcache.v1.lookup_client import LookupClientFactory`)

```python
# 概念的な実装（LMCache ライブラリ内）

class LookupClient:
    """キャッシュ検索クライアント"""

    def lookup(self, token_ids, lookup_id, request_configs=None):
        """
        Token hash を生成してキャッシュを検索
        """
        # Step 1: Token hash を生成
        # SHA256([1, 2, 3, ..., 2048]) → "a1b2c3d4e5f6g7h8..."
        token_hash = self.compute_token_hash(token_ids)

        # Step 2: バックエンドでキャッシュ検索
        # - ファイルが存在するか?
        # - メタデータは有効か?
        # - タイムスタンプは期限内か?
        cache_entry = self.backend.get_cache_entry(token_hash)

        if cache_entry is None:
            return 0  # キャッシュミス

        # Step 3: 部分マッチをチェック
        # キャッシュに 1024 トークン保存されているが、
        # 検索トークンが 2048 個の場合、最初の 1024 個だけマッチ
        matched_tokens = self._find_longest_prefix_match(
            token_ids, cache_entry.cached_token_ids
        )

        # Step 4: 結果を返す
        return matched_tokens  # 例: 1024

    def compute_token_hash(self, token_ids):
        """トークン列をハッシュ化（キャッシュキー生成）"""
        # Token ID リストを連結してハッシュ
        token_bytes = b''.join(
            token_id.to_bytes(4, 'big') for token_id in token_ids
        )
        return hashlib.sha256(token_bytes).hexdigest()
```

---

## フェーズ 2：ブロック割当と RequestTracker 作成

### 2.1 ブロック割当

**ファイル**: `vllm/v1/core/sched/scheduler.py:450-480`

```python
# Scheduler が KV キャッシュマネージャーに割当リクエスト

blocks = kv_cache_manager.allocate_blocks(
    request_id="req_123",
    num_blocks_to_allocate=768,  # need_to_allocate から計算
)
# blocks: [5, 10, 15, 20, ...]  (ブロックID)
```

### 2.2 RequestTracker 作成

**ファイル**: `vllm_v1_adapter.py:113-236`

```python
@dataclass
class RequestTracker:
    """リクエストのスケジュール状態を追跡"""

    req_id: str                          # "req_123"
    prompt_len: int                      # 2048
    token_ids: list[int]                 # [1, 2, 3, ..., 2048]
    allocated_block_ids: list[int]       # [5, 10, 15, 20, ...]
    num_saved_tokens: int = 0            # 初期値: 0
    mm_hashes: list[str] | None = None   # ["a1b2c3d4", ...]
    mm_positions: list[PlaceholderRange] | None = None
    skip_save: bool = False


# Scheduler で RequestTracker を作成
tracker = RequestTracker.from_new_request(
    lmcache_config,
    new_request_data,
    num_tokens_to_compute=1024,  # 256 (vLLM) + 768 (allocate)
    lmcache_cached_tokens=1024,  # Lookup 結果から
    skip_save=False,
)
# tracker.token_ids = [1, 2, ..., 2048]  ← 全トークン
# tracker.allocated_block_ids = [5, 10, 15, ..., 67]  ← 48ブロック（768トークン）
```

---

## フェーズ 3：ReqMeta 生成とメタデータ準備

### 3.1 ReqMeta の生成

**ファイル**: `vllm_v1_adapter.py:259-388`

```python
@dataclass
class ReqMeta:
    """Worker側で使用するリクエストメタデータ"""

    req_id: str
    token_ids: list[int]           # どのトークンを操作するか
    slot_mapping: torch.Tensor     # GPU メモリアドレス
    is_last_prefill: bool
    save_spec: SaveSpec | None     # セーブ仕様
    load_spec: LoadSpec | None     # ロード仕様
    disagg_spec: DisaggSpec | None # 分散実行仕様
    request_configs: dict | None   # リクエスト固有設定


# Scheduler で ReqMeta を生成
req_meta = ReqMeta.from_request_tracker(
    tracker,
    block_size=16,               # トークン/ブロック
    lmcache_chunk_size=256,      # チャンク単位
    load_spec=load_specs["req_123"],  # Lookup の結果
    discard_partial_chunks=True,
    save_decode_cache=False,
)

# 内部処理:
# 1. Token ID の絞り込み
req_meta.token_ids = [1, 2, ..., 2048]  # チャンク境界まで

# 2. Slot Mapping の生成
# Block ID [5, 10, 15, ..., 67] を GPU メモリアドレスに変換
#
# slot_mapping = (
#     [0, 1, 2, ..., 15]  (各ブロック内オフセット)
#     + [5, 10, 15, ..., 67] * 16  (ブロック ID * ブロックサイズ)
# ).flatten()
#
# → [80, 81, ..., 95,          # Block 5 (5*16 = 80)
#    160, 161, ..., 175,        # Block 10 (10*16 = 160)
#    240, 241, ..., 255,        # Block 15 (15*16 = 240)
#    ...
#    1072, 1073, ..., 1087]     # Block 67 (67*16 = 1072)

req_meta.slot_mapping = slot_mapping  # GPU アドレス

# 3. Save Spec の設定
req_meta.save_spec = SaveSpec(
    skip_leading_tokens=0,  # 新規リクエストなので 0
    can_save=True,          # Scheduler が許可
)

# 4. Load Spec の設定
req_meta.load_spec = LoadSpec(
    vllm_cached_tokens=256,       # ローカルで既にキャッシュ
    lmcache_cached_tokens=1024,   # LMCache から読込
    can_load=False,               # ← ここは False（後で True に）
)
```

### 3.2 Save Spec 決定ロジック

**セーブするかどうかの判定**:
```python
# vllm_v1_adapter.py:289-308 より

skip_leading_tokens = tracker.num_saved_tokens  # 初期: 0
chunk_boundary = (
    cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size)
    * lmcache_chunk_size
)

skip_save = tracker.disagg_spec is None and (
    tracker.skip_save  # リクエストレベルのフラグ
    or (
        tracker.num_saved_tokens > 0  # 既に何か保存済み?
        and input_token_len < chunk_boundary  # チャンク未満?
    )
    or (
        tracker.is_decode_phase  # デコード段階?
        and not save_decode_cache
    )
    or (tracker.request_configs or {}).get("lmcache.skip_save", False)
)

# 【例】
# - 新規リクエスト (num_saved_tokens=0):
#   → skip_save = False (セーブする)
#
# - デコード中 (is_decode_phase=True, save_decode_cache=False):
#   → skip_save = True (セーブしない)
#
# - チャンク未満の続きトークン (1000tokens, chunk=256):
#   → skip_save = True (256単位で待つ)
```

---

## フェーズ 4：スケジューラ側のメタデータ生成

### 4.1 build_connector_meta()

**ファイル**: `vllm_v1_adapter.py:1264-1375`

```python
def build_connector_meta(self, scheduler_output: SchedulerOutput):
    """
    Scheduler が Worker へ送信するメタデータを生成

    Called: Scheduler main loop (毎ステップ)
    """

    # Step 1: 完了リクエストの削除
    for finished_req_id in scheduler_output.finished_req_ids:
        self._request_trackers.pop(finished_req_id, None)
        self._unfinished_requests.pop(finished_req_id, None)

    # Step 2: メタデータ初期化
    metadata = LMCacheConnectorMetadata(
        requests=[],  # 追加していく
        lookup_requests_in_step=self._lookup_requests_in_step,
    )

    # Step 3: 新規リクエストの処理
    for new_request in scheduler_output.new_requests:
        # 3.1: Lookup して キャッシュ命中数を確認
        num_external_hit_tokens, _ = self.get_num_new_matched_tokens(
            new_request, num_computed_tokens=0
        )

        if num_external_hit_tokens is None:
            continue  # キャッシュ検索中、次のステップで再試行

        # 3.2: RequestTracker を作成
        tracker = RequestTracker.from_new_request(
            self.config,
            new_request,
            num_tokens_to_compute=num_external_hit_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            skip_save=...,
        )
        self._request_trackers[new_request.req_id] = tracker

        # 3.3: ReqMeta を生成
        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=self._block_size,
            lmcache_chunk_size=self._lmcache_chunk_size,
            load_spec=self.load_specs.get(new_request.req_id),
            discard_partial_chunks=self._discard_partial_chunks,
            save_decode_cache=self._save_decode_cache,
        )

        if req_meta is not None:
            metadata.add_request(req_meta)

    # Step 4: 既存リクエストの更新
    for request in scheduler_output.scheduled_requests:
        tracker = self._request_trackers[request.request_id]

        # New tokens が追加されたことを記録
        if request.new_token_ids:
            tracker.update(
                new_token_ids=request.new_token_ids,
                new_block_ids=request.block_ids,
            )

        # 新しい ReqMeta を生成
        req_meta = ReqMeta.from_request_tracker(
            tracker,
            ...,
            load_spec=self.load_specs.get(request.request_id),
        )

        if req_meta is not None:
            metadata.add_request(req_meta)

    # Step 5: Lookup リクエストのクリア
    self._lookup_requests_in_step = []

    return metadata  # Worker へ送信
```

---

## フェーズ 5：Worker側での メタデータ受け取り

### 5.1 メタデータバインディング

**ファイル**: `vllm/v1/worker/gpu_model_runner.py` + `kv_connector_model_runner_mixin.py`

```python
# Forward 実行前

def maybe_setup_kv_connector(scheduler_output):
    """Scheduler からのメタデータを受け取り"""

    if has_kv_transfer_group():
        kv_connector = get_kv_transfer_group()

        # Scheduler からのメタデータを バインド
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata  # LMCacheConnectorMetadata
        )

        # この時点で Worker は以下を知る：
        # - どのリクエストをロードするか
        # - どのトークンをロードするか
        # - どこに GPU メモリに配置するか (slot_mapping)
        # - セーブするか、しないか (save_spec)
```

### 5.2 Forward Context 準備

```python
# Forward 前

forward_context = ForwardContext(
    ...
    kv_caches={
        "layer_0.self_attn.k_cache": torch.Tensor(..., device="cuda:0"),
        "layer_0.self_attn.v_cache": torch.Tensor(...),
        "layer_1.self_attn.k_cache": torch.Tensor(...),
        ...
    }
)
```

---

## フェーズ 6：Worker側での start_load_kv()

### 6.1 メタデータ取得と GPU ロード開始

**ファイル**: `vllm_v1_adapter.py:769-878`

```python
def start_load_kv(self, forward_context: "ForwardContext", **kwargs):
    """
    Scheduler が準備したメタデータを使って、
    GPU バッファへのロードを開始（非同期）
    """

    # Step 1: Scheduler からのメタデータを取得
    metadata = self._parent._get_connector_metadata()
    # metadata.requests = [ReqMeta(...), ReqMeta(...), ...]
    # metadata.lookup_requests_in_step = ["req_123", ...]

    # Step 2: Forward context から GPU KV バッファを抽出
    self._init_kv_caches_from_forward_context(forward_context)
    kvcaches = list(self.kv_caches.values())
    # kvcaches = [
    #     torch.Tensor(num_slots, num_heads, head_size),
    #     torch.Tensor(...),
    #     ...
    # ]

    # Step 3: 各リクエストについてロード操作を設定
    for idx, request_meta in enumerate(metadata.requests):
        if request_meta.load_spec is None:
            continue  # ロード不要

        tokens = request_meta.token_ids  # [1, 2, 3, ..., 2048]

        # GPU へ移動
        slot_mapping = request_meta.slot_mapping.cuda()
        # slot_mapping = [80, 81, ..., 95, 160, ..., 1087] (GPU アドレス)

        # Token mask を作成：
        # vLLM で既にキャッシュされているトークンをスキップ
        token_mask = torch.ones(len(tokens), dtype=torch.bool)
        masked_token_count = (
            request_meta.load_spec.vllm_cached_tokens  # 256
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False
        # token_mask = [F, F, ..., F (×256), T, T, ..., T (×1792)]

        lmcache_cached_tokens = request_meta.load_spec.lmcache_cached_tokens  # 1024

        # Step 4: LMCache engine に retrieve を委譲
        # ← ここで LMCache がバックエンドを使ってデータ取得開始

        if self.use_layerwise:
            # レイヤーワイズ（非同期パイプライン）
            layerwise_retriever = self.lmcache_engine.retrieve_layer(
                tokens[:lmcache_cached_tokens],       # [1, 2, ..., 2048]
                token_mask[:lmcache_cached_tokens],   # [F×256, T×768]
                kvcaches=kvcaches,                    # GPU バッファ
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
                # slot_mapping = [80, 81, ..., GPU_ADDR]
                sync=(idx == last_idx),
            )

            # パイプライン化：最初の2層を先読み
            next(layerwise_retriever)  # Layer 0 の DMA 開始
            next(layerwise_retriever)  # Layer 1 の DMA 開始

            self.layerwise_retrievers.append(layerwise_retriever)
        else:
            # 全層一括
            ret_token_mask = self.lmcache_engine.retrieve(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
                req_id=request_meta.req_id,
            )
```

---

## フェーズ 7：LMCache Engine での 実際のKV検索・取得

### 7.1 バックエンド操作

**外部ライブラリ** (`lmcache.v1.cache_engine`)

```python
# 概念的な実装

class LMCacheEngine:
    def __init__(self, config, backend):
        self.backend = backend  # "gds", "local", "remote" など
        self.cache_dir = config.cache_dir

    def retrieve_layer(self, tokens, token_mask, kvcaches, slot_mapping):
        """
        Token hash に基づいてバックエンドからKVを検索・取得
        """
        # Step 1: Token hash をキャッシュキーに
        token_hash = self.compute_token_hash(tokens)
        # token_hash = "a1b2c3d4e5f6g7h8..."

        # Step 2: バックエンドでファイルパス・メタデータを確認
        cache_file_path = self.get_cache_file_path(token_hash)
        # cache_file_path = "/cache/req_123_chunk_0.kv"
        # または: "/mnt/nvme/a1b2c3d4e5f6/layer_0.kv" (GDS の場合)

        # Step 3: レイヤーごとに DMA 開始
        for layer_idx in range(num_layers):
            if not token_mask[layer_idx]:
                continue  # Skip

            # File offset を計算
            file_offset = self.compute_layer_offset(
                layer_idx,
                token_hash,
            )

            # 4. GPU バッファへ DMA 開始（バックエンド依存）
            if self.backend == "gds":
                # GDS (GPU ↔ NVMe 直結)
                cuFile.read(
                    fd=self.gds_fd,
                    gpu_ptr=kvcaches[layer_idx].data_ptr() + slot_mapping[0] * element_size,
                    offset=file_offset,
                    size=len(slot_mapping) * element_size,
                )
            elif self.backend == "local":
                # Host RAM
                host_data = self.host_cache_pool[token_hash][layer_idx]
                self.copy_host_to_gpu(
                    host_data,
                    kvcaches[layer_idx],
                    slot_mapping,
                )
            elif self.backend == "remote":
                # Network (TCP/gRPC)
                host_data = self.request_from_server(token_hash, layer_idx)
                self.copy_host_to_gpu(
                    host_data,
                    kvcaches[layer_idx],
                    slot_mapping,
                )

            # Yield: 次のレイヤーへ制御
            yield
```

---

## データ構造とメタデータマッピング

### 全体フロー図

```
【Scheduler Process】
┌──────────────────────────────────────────────────────┐
│ Request:                                             │
│ - prompt_token_ids: [1, 2, 3, ..., 2048]           │
│ - mm_hashes: ["image_hash_1", ...]                 │
└──────────┬───────────────────────────────────────────┘
           │ get_num_new_matched_tokens()
           ↓
┌──────────────────────────────────────────────────────┐
│ Lookup Client (外部 LMCache):                       │
│ - Token hash SHA256([1,2,3,...])                   │
│ - Cache entry 検索                                  │
│ - Return: 1024 (matched tokens)                    │
└──────────┬───────────────────────────────────────────┘
           │ LoadSpec 生成
           ↓
┌──────────────────────────────────────────────────────┐
│ RequestTracker:                                      │
│ - token_ids: [1, 2, ..., 2048]                     │
│ - allocated_block_ids: [5, 10, 15, ...]            │
│ - num_saved_tokens: 0                              │
└──────────┬───────────────────────────────────────────┘
           │ ReqMeta.from_request_tracker()
           ↓
┌──────────────────────────────────────────────────────┐
│ ReqMeta:                                             │
│ - token_ids: [1, 2, ..., 2048]                     │
│ - slot_mapping: [80, 81, ..., 1087]  (GPU addr)   │
│ - load_spec: LoadSpec(256, 1024, False)            │
│ - save_spec: SaveSpec(0, True)                     │
└──────────┬───────────────────────────────────────────┘
           │ build_connector_meta()
           ↓
┌──────────────────────────────────────────────────────┐
│ LMCacheConnectorMetadata:                            │
│ - requests: [ReqMeta(...), ReqMeta(...), ...]      │
│ - lookup_requests_in_step: ["req_123", ...]        │
└──────────┬───────────────────────────────────────────┘
           │ (Worker へ送信)
           ↓
【Worker Process】
┌──────────────────────────────────────────────────────┐
│ bind_connector_metadata(metadata)                    │
│ ↓                                                    │
│ start_load_kv(forward_context)                      │
│ - token_ids: [1, 2, ..., 2048]                     │
│ - token_mask: [F×256, T×768]  (Load対象)           │
│ - slot_mapping: [80, 81, ..., 1087] (GPU addr)     │
│ - load_spec: {vllm: 256, lmcache: 1024}            │
└──────────┬───────────────────────────────────────────┘
           │ lmcache_engine.retrieve_layer()
           ↓
┌──────────────────────────────────────────────────────┐
│ LMCache Engine:                                      │
│ - Token hash: SHA256([1,2,...])                    │
│ - Cache file: /cache/a1b2c3d4.kv  or  /mnt/nvme  │
│ - Layer offset: 0, 1024, 2048, ... (bytes)        │
│ - GPU offset: 80, 81, ..., 1087 (slot_mapping)    │
│ - Backend: "gds" / "local" / "remote"             │
│ ↓ cuFile.read() or memcpy or network_request()   │
│ → GPU バッファ に DMA 書込                         │
└──────────────────────────────────────────────────────┘
```

---

## キャッシュ命中条件

```python
# LMCache が マッチ判定する条件

class CacheEntry:
    """バックエンドに保存されるメタデータ"""

    token_hash: str                # SHA256([1,2,3,...,N])
    num_tokens: int                # N
    timestamp: float               # 保存時刻
    expiry_time: float             # 有効期限
    model_name: str                # "llama-2-7b"
    dtype: str                     # "float16"
    kv_shape: tuple                # (num_layers, 2, 256, num_heads, head_size)
    per_layer_offsets: list        # [0, 1024, 2048, ...] (bytes)


# キャッシュヒット条件：
# 1. Token hash が一致
# 2. Prefix match: 検索トークンの先頭が キャッシュトークンと一致
# 3. タイムスタンプが expiry_time 内
# 4. メタデータ (dtype, shape, model) が一致
```

---

## キャッシュミスの場合

```python
# num_external_hit_tokens = 0

# Scheduler:
if num_external_hit_tokens == 0:
    # キャッシュミス
    # → すべてのトークンを新規計算
    need_to_allocate = request.num_tokens - num_computed_tokens

    # LoadSpec には can_load=False
    load_spec = None

# Worker:
if load_spec is None:
    # ロード操作スキップ
    return  # start_load_kv() は何もしない

# Forward: 普通に attention 計算

# Save: すべてのトークンをキャッシュに保存
save_spec = SaveSpec(skip_leading_tokens=0, can_save=True)
```

---

## まとめ：誰が何をやるか

| 処理 | 担当 | 場所 | 詳細 |
|------|------|------|------|
| **Token 収集** | Scheduler | `request.prompt_token_ids` | ユーザー入力から |
| **MM Hash 適用** | Scheduler | `vllm_v1_adapter.py:1140` | 画像などのハッシュ置き換え |
| **Token Hash 生成** | Lookup Client | LMCache外部ライブラリ | SHA256(tokens) |
| **キャッシュ検索** | Lookup Client | LMCache外部ライブラリ | ファイル存在確認、メタデータ検証 |
| **LoadSpec 生成** | Scheduler | `vllm_v1_adapter.py:1191` | matched_tokens から |
| **RequestTracker 作成** | Scheduler | `vllm_v1_adapter.py:155` | トークン・ブロック追跡 |
| **Slot Mapping 計算** | Scheduler | `vllm_v1_adapter.py:358` | Block ID → GPU address |
| **ReqMeta 生成** | Scheduler | `vllm_v1_adapter.py:260` | すべてのメタデータ統合 |
| **LMCacheConnectorMetadata 構築** | Scheduler | `vllm_v1_adapter.py:1264` | Worker 送信用メタデータ |
| **メタデータ受け取り** | Worker | `gpu_model_runner.py` | bind_connector_metadata() |
| **GPU ロード開始** | Worker | `vllm_v1_adapter.py:769` | LMCache engine へ委譲 |
| **実際のKV取得** | LMCache Engine | 外部ライブラリ | バックエンド経由の読込 |
| **ファイルパス・オフセット** | LMCache Engine | 外部ライブラリ | Token hash からファイル確定 |

---

