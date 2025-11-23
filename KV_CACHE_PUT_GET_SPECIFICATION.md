# KV-Cache Put/Get 詳細仕様書
## GPU側・CPU側のメモリ転送メカニズムとフックポイント

---

## 目次
1. [概要](#概要)
2. [ディレクトリ構成](#ディレクトリ構成)
3. [GPU側：KV-CacheのGet（ロード）](#gpu側kv-cacheのgetロード)
4. [GPU側：KV-CachePut（セーブ）](#gpu側kv-cacheのputセーブ)
5. [CPU側：ストレージバックエンド](#cpu側ストレージバックエンド)
6. [メモリ転送パス](#メモリ転送パス)
7. [フックポイントと拡張方法](#フックポイントと拡張方法)
8. [実装例：カスタムコネクタ](#実装例カスタムコネクタ)

---

## 概要

vLLMのKV-Cacheシステムは以下の3層構造を持ちます：

```
┌─────────────────────────────────────────────────────┐
│  GPU層: vLLM Paged Buffer                           │
│  - Attention計算に直接使用                          │
│  - トークンごとのスロット管理                      │
└─────────────────┬───────────────────────────────────┘
                  │ <-- KVConnector --> │
                  │ Retrieve/Store      │
                  ▼
┌─────────────────────────────────────────────────────┐
│  CPU層: LMCache Engine + Backend                    │
│  - メモリ管理、キャッシュ検証                      │
│  - バックエンド選択 (Local/Remote/GDS)            │
└─────────────────┬───────────────────────────────────┘
                  │ データ読み書き
                  ▼
┌─────────────────────────────────────────────────────┐
│  ストレージ層: 永続化                               │
│  - ホストメモリ、ディスク、ネットワーク         │
│  - GDS (GPUDirect Storage) 接続                    │
└─────────────────────────────────────────────────────┘
```

---

## ディレクトリ構成

```
vllm/
├── distributed/kv_transfer/
│   ├── kv_connector/                    # コネクタメインディレクトリ
│   │   ├── factory.py                   # コネクタ登録・生成ファクトリ (169行)
│   │   ├── base.py                      # 基底クラス (v0用)
│   │   ├── utils.py                     # 共通ユーティリティ
│   │   │
│   │   └── v1/                          # V1 API (実装用)
│   │       ├── lmcache_connector.py     # LMCache ラッパー (192行)
│   │       ├── lmcache_integration/
│   │       │   ├── vllm_v1_adapter.py   # メイン実装 (1397行) ★重要
│   │       │   └── utils.py             # 設定・MM処理 (222行)
│   │       │
│   │       ├── base.py                  # KVConnectorBase_V1 (474行)
│   │       ├── shared_storage_connector.py  # ディスク保存例
│   │       ├── p2p/                     # P2P (マシン間)
│   │       │   ├── p2p_nccl_connector.py
│   │       │   └── p2p_nccl_engine.py
│   │       ├── nixl_connector.py        # RDMA対応
│   │       ├── multi_connector.py       # 複数コネクタ
│   │       └── offloading_connector.py  # オフロード専用
│   │
│   ├── kv_lookup_buffer/                # キャッシュ検索
│   │   ├── base.py                      # 基底クラス
│   │   ├── simple_buffer.py             # シンプル実装
│   │   └── mooncake_store.py            # Mooncake統合
│   │
│   └── kv_pipe/                         # パイプライン転送
│       ├── base.py
│       ├── pynccl_pipe.py
│       └── mooncake_pipe.py
│
├── v1/
│   └── worker/
│       ├── gpu_model_runner.py          # GPU KV初期化 (4600+ 行)
│       └── kv_connector_model_runner_mixin.py  # 統合
│
└── LMCACHE_CONNECTOR_V1_ARCHITECTURE.md # 設計説明書
```

---

## GPU側：KV-CacheのGET（ロード）

### 1. GPU バッファの初期化

**ファイル**: `vllm/v1/worker/gpu_model_runner.py:4272-4302`

```python
def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig):
    """GPU KV バッファの割り当て"""
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}

    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # GPU に int8 バッファを割り当て（コンパクト表現）
        tensor = torch.zeros(
            kv_cache_tensor.size,           # (total_blocks * block_size * hidden_dim)
            dtype=torch.int8,               # コンパクト型
            device=self.device              # GPU memory (VRAM)
        )
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    return kv_cache_raw_tensors  # Dict[str, Tensor]
```

**メモリレイアウト**: `vllm/v1/worker/gpu_model_runner.py:4423-4457`

```python
def _reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors):
    """GPU バッファを目的のレイアウトに変形"""
    kv_caches: dict[str, torch.Tensor] = {}

    for group in self._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        attn_backend = group.backend

        for layer_name in group.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]

            # Attention backend が要求するメモリレイアウト
            # HND (Hidden, Num_heads, Depth): NVIDIA GPUs
            # NHD (Num_heads, Hidden, Depth): AMD, custom
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()

            # 目的のシェイプを計算
            kv_cache_shape = tuple(
                kv_cache_shape[i] for i in kv_cache_stride_order
            )

            # 逆順列を計算（転置）
            inv_order = [
                kv_cache_stride_order.index(i)
                for i in range(len(kv_cache_stride_order))
            ]

            # 最終的な GPU テンサ（正しいメモリレイアウト）
            kv_caches[layer_name] = (
                raw_tensor
                .view(dtype)                  # 型変換
                .view(kv_cache_shape)        # 形状変更
                .permute(*inv_order)         # GPU最適レイアウト
            )

    return kv_caches
```

### 2. ロード：スケジューラ側の判定

**ファイル**: `vllm_v1_adapter.py:1113-1200`

```python
# Scheduler process
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    """
    外部キャッシュから何トークン読み込めるか判定

    Returns:
        (num_matched_tokens, is_async) - 読込みトークン数と非同期フラグ
    """

    # 1. リクエストのトークンを取得
    token_ids = request.prompt_token_ids

    # 2. マルチモーダルハッシュを適用（画像データなど）
    if mm_hashes and mm_positions:
        apply_mm_hashes_to_token_ids(
            torch.tensor(token_ids),
            mm_hashes,
            mm_positions
        )

    # 3. Lookup Client に問い合わせ
    # （LMCache engine が内部で検索）
    lookup_id = request.request_id if self.async_loading else str(uuid.uuid4())
    self._lookup_requests_in_step.append(lookup_id)

    num_external_hit_tokens = self.lookup_client.lookup(
        token_ids,
        lookup_id=lookup_id,
        request_configs=extract_request_configs(request.sampling_params)
    )

    # 4. 割り当てすべきトークン数を計算
    need_to_allocate = num_external_hit_tokens - num_computed_tokens
    if num_external_hit_tokens == request.num_tokens:
        need_to_allocate -= 1  # 最後のトークンは再計算

    # 5. LoadSpec（ロード仕様）を保存
    self.load_specs[request.request_id] = LoadSpec(
        vllm_cached_tokens=num_computed_tokens,
        lmcache_cached_tokens=num_external_hit_tokens,
        can_load=False  # 後で allocate 後に True に
    )

    return (num_external_hit_tokens, False)  # 非同期フラグ
```

### 3. ロード：Worker側の実行

**ファイル**: `vllm_v1_adapter.py:769-878`

```
時系列:
┌────────────────────────────────────────────────┐
│ Forward Pass 開始                              │
│                                                 │
│ 1. start_load_kv() が呼ばれる                 │
│    ↓                                            │
│ 2. KV キャッシュを extract                     │
│    ↓                                            │
│ 3. LMCache engine.retrieve() を非同期開始      │
│    ↓                                            │
│ 4. Attention layer loop:                       │
│    - wait_for_layer_load() でロード待機       │
│    - Attention 計算実行                        │
│    ↓                                            │
│ 5. save_kv_layer() で GPU から ストレージ へ │
│    ↓                                            │
│ 6. wait_for_save() で全セーブ完了待機         │
└────────────────────────────────────────────────┘
```

#### start_load_kv（ロード開始）

```python
def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any):
    """
    外部キャッシュから vLLM GPU バッファへロード開始（非同期）
    """
    # 1. Forward context から GPU KV バッファを抽出
    self._init_kv_caches_from_forward_context(forward_context)
    kvcaches = list(self.kv_caches.values())  # GPU 上のテンサリスト

    # メタデータ取得（Scheduler から送付）
    metadata = self._parent._get_connector_metadata()
    assert isinstance(metadata, LMCacheConnectorMetadata)

    # 2. 各リクエストについてロード操作を設定
    for idx, request in enumerate(metadata.requests):
        if request.load_spec is None:
            continue  # この request はロード不要

        tokens = request.token_ids

        # GPU へ移動（CPU -> GPU）
        slot_mapping = request.slot_mapping.cuda()
        assert len(tokens) == len(slot_mapping)

        # トークンマスク：vLLM キャッシュヒット部分をスキップ
        token_mask = torch.ones(len(tokens), dtype=torch.bool)
        masked_token_count = (
            request.load_spec.vllm_cached_tokens  # vLLM で既キャッシュ
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False  # ロード対象外

        lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens

        # 3. LMCache engine に retrieve リクエスト送信
        if self.use_layerwise:
            # レイヤーワイズ非同期（パイプライン化）
            if self.enable_blending:
                # ブレンディング用（キャッシュ動的ブレンディング）
                self.blender.blend(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,  # GPU バッファ
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                )
            else:
                # 標準的なレイヤーワイズ
                layerwise_retriever = self.lmcache_engine.retrieve_layer(
                    tokens[:lmcache_cached_tokens],  # トークンID
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,              # GPU バッファ
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],  # GPU アドレス
                    sync=(idx == last_idx),  # 最後のリクエストで同期
                )

                # パイプライン化：最初の2層を先読み
                next(layerwise_retriever)  # Layer 0
                next(layerwise_retriever)  # Layer 1

                self.layerwise_retrievers.append(layerwise_retriever)
        else:
            # 全層一括ロード
            ret_token_mask = self.lmcache_engine.retrieve(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,              # GPU バッファへ直接ロード
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
                request_configs=request.request_configs,
                req_id=request.req_id
            )

            # ロード検証
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = (
                lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
            )
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(f"Load mismatch: {num_retrieved_tokens} < {num_expected_tokens}")
```

#### wait_for_layer_load（ロード待機）

```python
def wait_for_layer_load(self, layer_name: str):
    """
    Attention layer から呼ばれ、そのレイヤーの GPU ロード完了を待機
    """
    if self.layerwise_retrievers:
        logger.debug(f"Waiting for layer {self.current_layer} to load")

    # すべての layerwise_retriever から次の層のロード完了を待つ
    for layerwise_retriever in self.layerwise_retrievers:
        ret_token_mask = next(layerwise_retriever)  # ブロック（同期点）

        # 最後の層で統計ログ
        if self.current_layer == self.num_layers - 1:
            assert ret_token_mask is not None
            num_retrieved_tokens = ret_token_mask.sum().item()
            logger.info(f"Retrieved {num_retrieved_tokens} tokens")

    return
```

### 4. Slot Mapping：GPU アドレッシング

**ファイル**: `vllm_v1_adapter.py:358-365`

```python
# Slot mapping の生成
block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
block_offsets = torch.arange(0, block_size, dtype=torch.long)

# Slot = block_id * block_size + offset
slot_mapping = (
    block_offsets.reshape((1, block_size))        # (1, block_size)
    + block_ids.reshape((num_blocks, 1)) * block_size  # (num_blocks, 1)
)

# Flatten: (num_blocks * block_size) -> 1D
slot_mapping = slot_mapping.flatten()[:len(token_ids)]  # 1D テンサ
assert slot_mapping.dtype == torch.long
```

**例：メモリマッピング**
```
Block ID: [5, 10, 15]        (3つのブロック割当)
Block size: 16               (ブロックあたり16トークン)

Slot Mapping:
  Token 0-15   -> Slots 80-95     (block 5)
  Token 16-31  -> Slots 160-175   (block 10)
  Token 32-47  -> Slots 240-255   (block 15)

GPU KV バッファ:
  [... | Block 5 | ... | Block 10 | ... | Block 15 | ...]
  80             160             240

Slot mapping により、ロードされたデータが正しい位置に配置される
```

### 5. GPU メモリのトークン配置

**GPU KV テンサ構造**:
```
KV Tensor Layout (GPU):
┌─────────────────────────────────────────────┐
│ Layer 0                                     │
│  ┌──────────────────────────────────────┐  │
│  │ K/V Cache (Num_slots, Num_heads, Head_size)
│  │ ┌─────────┬─────────┬─────────────┐  │  │
│  │ │ Block 0 │ Block 1 │ ... Block N │  │  │
│  │ ├─ Slot 0 ├─ Slot 16├─ Slot M*16 ┤  │  │
│  │ ├─ Slot 1 ├─ Slot 17├─ ...       ┤  │  │
│  │ │  ...    │  ...   │  ...       │  │  │
│  │ └─ Slot 15┴─ Slot 31┴─ Slot ?   ┘  │  │
│  └──────────────────────────────────────┘  │
│                                             │
│ Layer 1, Layer 2, ...                      │
└─────────────────────────────────────────────┘

Token-to-GPU-Memory Mapping:
Token 0 -> Slot 80  (Block 5, offset 0)
Token 1 -> Slot 81  (Block 5, offset 1)
...
Token 15 -> Slot 95 (Block 5, offset 15)
Token 16 -> Slot 160 (Block 10, offset 0)
```

---

## GPU側：KV-CachePUT（セーブ）

### 1. セーブ：スケジューラ側の判定

**ファイル**: `vllm_v1_adapter.py:260-388`

```python
@staticmethod
def from_request_tracker(
    tracker: RequestTracker,
    block_size: int,
    lmcache_chunk_size: int = 256,
    load_spec: LoadSpec | None = None,
    discard_partial_chunks: bool = True,
    save_decode_cache: bool = False,
) -> Optional["ReqMeta"]:
    """
    Request のセーブ仕様を判定
    """
    input_token_len = len(tracker.token_ids)

    # 最後のプリフィル段階か？
    is_last_prefill = (input_token_len == tracker.prompt_len)

    # セーブのスキップ判定
    skip_leading_tokens = tracker.num_saved_tokens  # 既保存分
    chunk_boundary = (
        cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
    )

    # セーブをスキップするケース
    skip_save = tracker.disagg_spec is None and (
        tracker.skip_save  # リクエストレベルのスキップ
        or (
            tracker.num_saved_tokens > 0
            and input_token_len < chunk_boundary  # 部分チャンク（デコード中）
        )
        or (
            tracker.is_decode_phase
            and not save_decode_cache  # デコード時のセーブ無効
        )
        or (tracker.request_configs or {}).get("lmcache.skip_save", False)
    )

    if skip_save and load_spec is None:
        return None  # セーブ不要

    # セーブするトークン数を計算
    num_tokens_to_save = (
        (input_token_len // lmcache_chunk_size * lmcache_chunk_size)
        if not is_last_prefill or discard_partial_chunks
        else input_token_len
    )

    # SaveSpec を構築
    save_spec = SaveSpec(skip_leading_tokens, not skip_save)

    # Slot mapping 生成
    token_ids = input_token_ids[:num_tokens_to_save]
    block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
    block_offsets = torch.arange(0, block_size, dtype=torch.long)

    slot_mapping = (
        block_offsets.reshape((1, block_size))
        + block_ids.reshape((num_blocks, 1)) * block_size
    )
    slot_mapping = slot_mapping.flatten()[:len(token_ids)]

    return ReqMeta(
        req_id=tracker.req_id,
        token_ids=token_ids,
        slot_mapping=slot_mapping,
        is_last_prefill=is_last_prefill,
        save_spec=save_spec,
        load_spec=load_spec,
        disagg_spec=tracker.disagg_spec,
        request_configs=tracker.request_configs,
    )
```

### 2. セーブ：Worker側の実行

**ファイル**: `vllm_v1_adapter.py:904-1002`

```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,      # GPU KV バッファ（shape: [num_tokens, num_heads, head_size]）
    attn_metadata: "AttentionMetadata",
    **kwargs,
):
    """
    GPU KV バッファをストレージへセーブ開始（非同期）
    """
    if not self.use_layerwise:
        return  # 非レイヤーワイズモードはwait_for_save()で処理

    if self.kv_role == "kv_consumer":
        return  # Consumer ロールはセーブしない

    metadata = self._parent._get_connector_metadata()
    assert isinstance(metadata, LMCacheConnectorMetadata)
    assert len(self.kv_caches) > 0

    kvcaches = list(self.kv_caches.values())  # GPU KV バッファ

    # Layer 0 の時のみセッター初期化
    if self.current_layer == 0:
        self.layerwise_storers = []
        is_first = True

        for idx, request in enumerate(metadata.requests):
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            slot_mapping = request.slot_mapping
            assert len(slot_mapping) == len(token_ids)

            # GPU へ移動（CPU slot_mapping -> GPU）
            slot_mapping = slot_mapping.cuda()

            # セーブするトークンのマスク
            skip_leading_tokens = save_spec.skip_leading_tokens

            if self.kv_role == "kv_consumer":
                continue  # Consumer はセーブしない
            elif self.kv_role == "kv_producer":
                skip_leading_tokens = 0  # Producer は最初からセーブ
            else:
                # チャンク境界にアラインメント
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

                if skip_leading_tokens == len(token_ids):
                    continue  # 全部既セーブ

            # Store mask：既セーブをスキップ
            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                f"Storing KV cache for {len(token_ids) - skip_leading_tokens} "
                f"out of {len(token_ids)} tokens "
                f"(skip_leading_tokens={skip_leading_tokens}) "
                f"for request {request.req_id}"
            )

            # LMCache engine へセーブリクエスト送信（ジェネレータ）
            layerwise_storer = self.lmcache_engine.store_layer(
                token_ids,
                mask=store_mask,             # どのトークンをセーブするか
                kvcaches=kvcaches,           # GPU バッファ
                slot_mapping=slot_mapping,   # GPU アドレス
                offset=skip_leading_tokens,
                sync=is_first,               # 最初のリクエストで同期
            )
            self.layerwise_storers.append(layerwise_storer)
            if is_first:
                is_first = False

    # 各層でセーブを進める（ジェネレータnext()）
    for layerwise_storer in self.layerwise_storers:
        next(layerwise_storer)  # 層のセーブを非同期実行

    self.current_layer += 1
```

### 3. セーブ待機

**ファイル**: `vllm_v1_adapter.py:1005-1101`

```python
def wait_for_save(self):
    """
    すべてのセーブ操作完了を待機
    """
    metadata = self._parent._get_connector_metadata()
    assert isinstance(metadata, LMCacheConnectorMetadata)

    # Lookup 用バッファのアンロック（ピン留め解除）
    self.lmcache_engine.lookup_unpin(
        metadata.lookup_requests_in_step
    )

    if self.kv_role == "kv_consumer":
        return  # Consumer はセーブしない

    # レイヤーワイズセーブの最終待機
    if self.use_layerwise:
        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)  # 最終完了待機
        return

    # 非レイヤーワイズ（全層一括）
    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    for request in metadata.requests:
        save_spec = request.save_spec
        if (save_spec is None or not save_spec.can_save) \
                and self.kv_role != "kv_producer":
            continue

        token_ids = request.token_ids
        slot_mapping = request.slot_mapping.cuda()
        assert len(slot_mapping) == len(token_ids)

        skip_leading_tokens = save_spec.skip_leading_tokens

        # Producer は全部セーブ
        if self.kv_role == "kv_producer":
            assert request.disagg_spec is not None
            skip_leading_tokens = min(skip_leading_tokens, request.disagg_spec.num_transferred_tokens)
        else:
            # チャンク境界アラインメント
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

        # Store mask
        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        # LMCache engine へセーブ（ブロック）
        self.lmcache_engine.store(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
        )

        # KV cache manager の更新
        self.kv_cache_manager.update_kv_cache_count(
            request.req_id,
            num_tokens_to_save,
        )
```

---

## CPU側：ストレージバックエンド

### 1. LMCache Engine 初期化

**ファイル**: `vllm_v1_adapter.py:409-528`

```python
def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
) -> LMCacheEngine:
    """LMCache engine をバックエンド設定で初期化"""

    # ビルダーから取得（singleton）
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    # 1. メタデータ構築
    kv_dtype = get_kv_cache_torch_dtype(...)
    use_mla = mla_enabled(model_config)

    num_layer = model_config.get_num_layers(parallel_config)
    kv_shape = (
        num_layer,
        1 if use_mla else 2,     # MLA: 1層, 通常: 2層(K,V)
        chunk_size,              # 256 トークン
        num_kv_head,
        head_size
    )

    metadata = LMCacheEngineMetadata(
        model=model_config.model,
        world_size=parallel_config.world_size,
        rank=parallel_config.rank,
        framework="vllm",
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
    )

    # 2. GPU Connector 選択
    if lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            vllm_gpu_connector = VLLMBufferLayerwiseGPUConnector(...)
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseGPUConnector(...)
    else:
        vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(...)  # デフォルト

    # 3. LMCache engine ビルド
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,          # バックエンド設定
        metadata,
        vllm_gpu_connector,      # GPU との通信用
        tpg.broadcast,           # Tensor Parallel Broadcast
        tpg.broadcast_object,
    )

    return engine
```

### 2. バックエンド設定

**ファイル**: `utils.py:32-73` (Configuration Loading)

```python
def lmcache_get_or_create_config() -> Config | V1Config:
    """
    LMCache 設定をロード（thread-safe singleton）
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                # 環境変数から設定モード判定
                use_experimental = not is_false(
                    os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")
                )

                if use_experimental:
                    LMCacheEngineConfig = V1Config  # v1.0+
                else:
                    LMCacheEngineConfig = Config    # legacy

                # 設定ファイルまたは環境変数から読み込み
                if "LMCACHE_CONFIG_FILE" in os.environ:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                else:
                    _config_instance = LMCacheEngineConfig.from_env()

                # 環境変数オーバーライド
                _config_instance.update_config_from_env()

    return _config_instance
```

**バックエンド選択肢**:
```yaml
# lmcache_config.yaml
backend: "gds"              # Local / Remote / GDS / NixL

# Local: ホストメモリ内（最速、単一マシン用）
local_config:
  cache_size_gb: 64         # ホスト RAM

# Remote: ネットワーク経由（マルチマシン用）
remote_config:
  remote_url: "tcp://cache-server:5000"
  remote_serde: "msgpack"   # naive / msgpack / pickle

# GDS: GPU-NVMe 直結（超高速）
gds_config:
  cache_dir: "/mnt/nvme"
  enable_rdma: true
  max_concurrent_ops: 4
  block_size: 1048576       # 1MB

# その他
chunk_size: 256             # トークン単位
enable_async_loading: true  # 非同期ロード
use_layerwise: true         # レイヤーワイズ処理
enable_blending: false      # キャッシュ動的ブレンディング
save_decode_cache: false    # デコード時のセーブ
save_unfull_chunk: true     # 部分チャンクのセーブ
```

### 3. Lookup Client（キャッシュ検索）

**ファイル**: `vllm_v1_adapter.py:1113-1200`

```python
# Scheduler 側
self.lookup_client = LookupClientFactory.create_lookup_client(
    vllm_config, config
)

# Worker 側
self.lookup_server = LookupClientFactory.create_lookup_server(
    self.lmcache_engine, vllm_config
)

if self.async_loading and isinstance(self.lookup_server, LMCacheAsyncLookupServer):
    self.lmcache_engine.post_init(async_lookup_server=self.lookup_server)
```

**Lookup フロー**:
```
Token Sequence: [id0, id1, id2, ...]
  ↓
Token Hash: SHA256(token_ids)
  ↓
Cache Lookup: キャッシュ内で同じハッシュを検索
  ↓
Hit Detection: 何トークンまで一致したか判定
  ↓
Return: (num_matched_tokens, cache_metadata)
```

### 4. ストレージ I/O パス

**Local Backend** (ホストメモリ):
```
GPU KV              CPU Memory Pool       Disk
[torch.Tensor]  ->  [buffer]          ->  (cache miss時)
    GPU VRAM        Host DRAM
    (コンパクト)     (圧縮/非圧縮)
```

**Remote Backend** (ネットワーク):
```
GPU KV              Serialize          Network       Cache Server
[torch.Tensor]  ->  [bytes]        ->  [socket]  ->  [storage]
    GPU VRAM        (msgpack)          TCP/gRPC      Host/Disk
    (コンパクト)     (圧縮オプション)
```

**GDS Backend** (GPU Direct Storage):
```
GPU KV              NVMe Controller      SSD
[torch.Tensor]  -> [DMA]            ->  [NVMe]
    GPU VRAM        (ホストバイパス)     (直結)
    (ホストメモリ不経由)
```

---

## メモリ転送パス

### Transfer Flow Diagram

```
        GPU Computation
              ↑
              │
        wait_for_layer_load()  ←── Blocks for load
              ↑
        ┌─────┴──────────────┐
        │ Attention Layer    │
        └─────────────────────┘
              ↑
        Layer-wise Retriever
        (ジェネレータ)
              ↑
     ┌────────┴────────┐
     │ LMCache Engine  │
     │  (retrieve_layer())
     └────────┬────────┘
              ↑
        Backend Interface
              ↑
        ┌─────┴──────────────┐
        │ Backend選択:        │
        │ - Local (メモリ)   │
        │ - Remote (TCP)     │
        │ - GDS (直結)      │
        │ - NixL (RDMA)     │
        └─────┬──────────────┘
              ↑
        ┌─────┴──────────────┐
        │ Storage/Cache      │
        │ - Host DRAM        │
        │ - Disk             │
        │ - Network          │
        │ - NVMe (GDS)       │
        └────────────────────┘
```

### Token-Level Data Flow

```
1. Token Creation (Generator)
   Request prompt_token_ids: [id0, id1, id2, ...]
        ↓

2. Token Hashing (Lookup)
   Hash: SHA256([id0, id1, id2, ...])
        ↓

3. Cache Lookup
   lookup_client.lookup(tokens) → (num_matched, meta)
        ↓

4. Allocation & Scheduling
   Scheduler: allocate blocks for unmatched tokens
        ↓

5. GPU Buffer Setup
   slot_mapping: token_index → gpu_slot
        ↓

6. Load from Cache
   for each matched token:
       retrieve(token_id, slot) → load to GPU buffer
        ↓

7. Forward Computation
   Attention(key=gpu_buffer, query=input)
        ↓

8. Save to Cache
   for each new token:
       store(token_id, slot) → save from GPU buffer
        ↓

9. Eviction (when full)
   lru_evict() or fifo_evict() → remove oldest
```

---

## フックポイントと拡張方法

### 1. 主要なフック点

vLLMのKV-Cacheシステムは複数の拡張ポイント（フック）を提供しています：

#### A. コネクタレベルのフック

**ファイル**: `factory.py:26-169`

```python
class KVConnectorFactory:
    """コネクタ登録・生成ファクトリ"""

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str):
        """カスタムコネクタを登録"""
        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, config: VllmConfig, role: KVConnectorRole):
        """コネクタを生成"""
        connector_cls = cls.get_connector_class(config.kv_transfer_config)
        return connector_cls(config, role)
```

**登録済みコネクタ**:
```python
# factory.py より
KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1",
)

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector",
)

# 他: P2pNcclConnector, NixlConnector, MultiConnector, etc.
```

#### B. Lookup Client のフック

```python
# vllm_v1_adapter.py:585-587
self.lookup_client = LookupClientFactory.create_lookup_client(
    vllm_config, config
)
```

Lookup client を差し替えることで、キャッシュ検索ロジックをカスタマイズできます。

#### C. バックエンドのフック

```python
# LMCache が内部でバックエンド選択
# 環境変数・設定ファイルで選択可能
```

#### D. GPU Connector のフック

```python
# vllm_v1_adapter.py:476-517
# 3つのGPU Connector から選択可能
- VLLMPagedMemGPUConnectorV2 (デフォルト)
- VLLMPagedMemLayerwiseGPUConnector
- VLLMBufferLayerwiseGPUConnector
```

### 2. 実装フローチャート

```
カスタム実装の手順:

1. KVConnectorBase_V1 を継承
   └─ 必須メソッド実装:
      ├─ start_load_kv()
      ├─ wait_for_layer_load()
      ├─ save_kv_layer()
      ├─ wait_for_save()
      ├─ get_num_new_matched_tokens()
      ├─ update_state_after_alloc()
      ├─ build_connector_meta()
      └─ request_finished()

2. factory.py に登録
   └─ KVConnectorFactory.register_connector(...)

3. 実行時に選択
   └─ --kv-connector=CustomConnector
```

---

## 実装例：カスタムコネクタ

### シンプルな例：SharedStorageConnector

**ファイル**: `shared_storage_connector.py`

```python
class SharedStorageConnector(KVConnectorBase_V1):
    """ディスク上の共有ストレージにKVを保存"""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info("Storage path: %s", self._storage_path)

    # --- Worker 側 ---

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any):
        """ディスクから GPU バッファへロード"""
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SharedStorageConnectorMetadata)

        # 各リクエストについてロード
        for request in connector_metadata.requests:
            if request.is_store:
                continue  # セーブのみの request

            # ファイルパスを生成
            file_path = os.path.join(
                self._storage_path,
                f"{request.token_ids.sum().item()}.safetensors"
            )

            if not os.path.exists(file_path):
                logger.warning(f"Cache file not found: {file_path}")
                continue

            # ディスク -> メモリへロード
            with safetensors.open_file(file_path) as f:
                for layer_idx, layer_name in enumerate(self.kv_caches.keys()):
                    key = f"{layer_name}.kv"
                    kv_data = f.get_tensor(key)  # ディスク読み込み

                    # GPU へコピー
                    gpu_tensor = kv_data.to(self.device)

                    # Slot mapping に従って配置
                    slot_mapping = request.slot_mapping
                    for i, slot in enumerate(slot_mapping):
                        self.kv_caches[layer_name][slot] = gpu_tensor[i]

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ):
        """GPU バッファをディスクへセーブ"""
        connector_metadata = self._get_connector_metadata()

        for request in connector_metadata.requests:
            if not request.is_store:
                continue

            # ファイルパスを生成
            file_path = os.path.join(
                self._storage_path,
                f"{request.token_ids.sum().item()}.safetensors"
            )

            # GPU -> CPU
            kv_cpu = kv_layer.cpu()

            # CPU -> ディスク (safetensors形式)
            save_file(
                {f"{layer_name}.kv": kv_cpu},
                file_path,
            )
            logger.info(f"Saved KV cache to {file_path}")

    def wait_for_layer_load(self, layer_name: str):
        """ロード完了待機（同期なので不要）"""
        pass

    def wait_for_save(self):
        """セーブ完了待機（同期なので不要）"""
        pass

    # --- Scheduler 側 ---

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """
        ディスク上に保存されたトークン数を確認
        """
        # 簡単な実装：常に全トークンをディスクから読み込む
        return (len(request.prompt_token_ids), False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """ブロック割当後の状態更新"""
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """メタデータ生成"""
        metadata = SharedStorageConnectorMetadata()

        for request in scheduler_output.scheduled_requests:
            metadata.add_request(
                token_ids=request.prompt_token_ids,
                block_ids=request.block_ids,
                block_size=self._block_size,
                is_store=False,  # または True
                mm_hashes=getattr(request, 'mm_hashes', []),
            )

        return metadata

    def request_finished(self, request, block_ids):
        """リクエスト完了時処理"""
        return False, None
```

### 使用方法

```bash
# カスタムコネクタを使用
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --kv-connector=SharedStorageConnector \
    --kv-connector-extra-config="shared_storage_path=/mnt/cache"
```

---

## 詳細リファレンス

### 必須メソッド一覧

| メソッド | ロール | 用途 | 同期 |
|---------|--------|------|------|
| `start_load_kv()` | Worker | ロード開始 | 非同期 |
| `wait_for_layer_load()` | Worker | ロード待機 | 同期 |
| `save_kv_layer()` | Worker | セーブ開始 | 非同期 |
| `wait_for_save()` | Worker | セーブ待機 | 同期 |
| `get_num_new_matched_tokens()` | Scheduler | キャッシュ命中判定 | 同期 |
| `update_state_after_alloc()` | Scheduler | 割当後処理 | 同期 |
| `build_connector_meta()` | Scheduler | メタデータ生成 | 同期 |
| `request_finished()` | Scheduler | リクエスト終了 | 同期 |

### デバッグ用フック

```python
# ロギングレベルを上げる
import logging
logging.getLogger("vllm.distributed.kv_transfer").setLevel(logging.DEBUG)

# 環境変数
export LMCACHE_LOG_LEVEL=DEBUG
export VLLM_DEBUG_KV_TRANSFER=1

# 独自ログを追加
def start_load_kv(self, forward_context, **kwargs):
    logger.debug(f"[START_LOAD] Requests: {len(metadata.requests)}")
    logger.debug(f"[START_LOAD] KV caches: {list(self.kv_caches.keys())}")
    ...
```

---

## 今後の拡張例

### マルチバックエンド対応

```python
class HybridConnector(KVConnectorBase_V1):
    """複数バックエンド対応（優先度ベース）"""

    def __init__(self, vllm_config, role):
        # Primary: GDS (超高速)
        # Secondary: Remote (フォールバック)
        # Tertiary: Local (最終手段)
        self.backends = [
            GDSBackend(...),
            RemoteBackend(...),
            LocalBackend(...),
        ]

    def start_load_kv(self, forward_context, **kwargs):
        for backend in self.backends:
            try:
                backend.retrieve(...)
                break  # 成功したら終了
            except BackendException:
                continue  # 次のバックエンド試行
```

### インテリジェント プリフェッチング

```python
class SmartPrefetchConnector(KVConnectorBase_V1):
    """将来のトークンを先読み"""

    def start_load_kv(self, forward_context, **kwargs):
        # 現在のトークン + 予測トークンを先読み
        future_tokens = self.predict_next_tokens(metadata.requests)
        self.prefetch_async(future_tokens)
```

---

## まとめ

### GPU側：
1. **初期化**: `torch.zeros(..., device=cuda)`で GPU バッファ割当
2. **ロード**: `start_load_kv()` で非同期転送開始
3. **待機**: `wait_for_layer_load()` で層ごと同期
4. **セーブ**: `save_kv_layer()` で GPU→ストレージ

### CPU側：
1. **判定**: Lookup client でキャッシュ命中判定
2. **抽出**: メタデータで LoadSpec/SaveSpec 生成
3. **転送**: LMCache engine で バックエンド経由データ移動

### フック：
- **コネクタレベル**: KVConnectorBase_V1 継承
- **Lookup**: LookupClientFactory カスタマイズ
- **バックエンド**: 設定ファイルで選択
- **GPU Connector**: 3つの選択肢から選択

---
