# Draft upstream issue: CacheBlend fails on vLLM 0.32 (packed K|V layout unsupported by layerwise/blend path)

*Draft for [LMCache/LMCache](https://github.com/LMCache/LMCache/issues). Not yet filed.*

---

**Title:** CacheBlend (`enable_blending`) crashes on vLLM 0.32: layerwise connector doesn't support packed `NL_X_NB_BS_NH_CS` KV layout

## Environment

- vLLM 0.32.0, torch 2.13.0+cu130
- LMCache 0.5.6 (pinned explicitly; see note 1)
- NVIDIA L40S, driver 595.91, Ubuntu 24.04, Python 3.12
- Model: Qwen/Qwen2.5-7B-Instruct (bf16), `LLM(..., enable_prefix_caching=False)`
- `kv_transfer_config=KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")`
- Env: `LMCACHE_ENABLE_BLENDING=True LMCACHE_USE_LAYERWISE=True LMCACHE_BLEND_SPECIAL_STR=" # # " LMCACHE_BLEND_CHECK_LAYERS=1 LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15 LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True`

Repro script: <https://github.com/sarvanithin/chunkkv/blob/master/scripts/week1_repro.py> (`--mode blend`).

## Root cause

vLLM 0.32 allocates each layer's KV cache as a single packed tensor:

```
shape  (23236, 4, 16, 256)   # [num_blocks, kv_heads, block_size, 2*head_dim]
stride (16384, 256, 1024, 1)
```

`normalize_kv_and_discover_format` detects this correctly as `EngineKVFormat.NL_X_NB_BS_NH_CS`
and normalizes it to `(23236, 16, 4, 256)`. But `VLLMBufferLayerwiseGPUConnector`, which
`enable_blending` requires, passes these tensors to `single_layer_kv_transfer`. That function
only implements `NL_X_TWO_NB_BS_NH_HS` (flash-attn), `NL_X_NB_TWO_BS_NH_HS` (flashinfer) and
MLA. The first `store_layer` crashes:

```
File ".../lmcache/v1/cache_engine.py", line 770, in store_layer
File ".../lmcache/v1/gpu_connector/gpu_connectors.py", line 1044, in batched_from_gpu
IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)
```

(`assert_is_vllm_flash_attn_or_flash_infer` doesn't catch it.) The engine then hangs
instead of exiting.

Related: #4160 (closed) fixed the packed fused-K/V layout `NL_X_NB_BS_NH_TWO_HS` for the CacheBlend V3 path, but the layerwise path that `LMCacheConnectorV1` + `enable_blending` uses still fails on the packed `NL_X_NB_BS_NH_CS` layout vLLM 0.32 produces. #3238 (open, non-prefix reuse broken with the V1 connector) may share the cause.

**Suggested fix:** support `NL_X_NB_BS_NH_CS` in the layerwise transfer by viewing the last
dim as `[2, head_dim]` (K = `[..., :head_dim]`, V = `[..., head_dim:]`), and in the blender's
KV read/write. Alternatively, fail fast at connector init with a clear message when
`enable_blending` meets an unsupported format.

## Other breakages hit on the way (same setup)

1. **Resolver picks a stale LMCache.** `uv pip install vllm lmcache` resolves `lmcache==0.4.3`
   (opentelemetry/protobuf pins conflict with newer releases). Its prebuilt `c_ops` fail
   against torch 2.13 with
   `undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`.
2. **Model never registered for blending.** `LMCBlenderBuilder.get_or_create` calls
   `VLLMModelTracker.get_model("vllm-instance")`, but nothing in vLLM 0.32 or LMCache calls
   `register_model`:
   `ValueError: vllm model for vllm-instance not found.` Workaround: wrap
   `vllm.v1.worker.gpu_worker.Worker.initialize_from_config` to register
   `self.model_runner.get_model()` (requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`).
3. **Moved module.** `lmcache/v1/compute/attention/flash_attn.py` imports `vllm.attention`.
   vLLM 0.32 moved it to `vllm.model_executor.layers.attention`.

Happy to send a PR for the layout support and the fail-fast check if that's the direction
you'd like.
