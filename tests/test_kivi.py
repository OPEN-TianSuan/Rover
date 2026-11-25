import pytest
import torch

from rover.cache.kivi import KIVIConfig, LayerKIVICache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_residual_compaction_preserves_tokens(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = KIVIConfig(
        bits=2,
        group_size=32,
        residual_length=128,
        prefill_chunk_size=64,
        device=torch.device(device),
        kv_heads=8,
        head_dim=128,
        max_len=256,
    )
    cache = LayerKIVICache(config)
    keys = torch.randn(1, 8, 177, 128, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    for start, end in [(0, 64), (64, 128), (128, 177)]:
        cache.update_range(start, end, keys[:, :, start:end], values[:, :, start:end])
    assert cache.prefix_len == 32
    assert cache.residual_len == 145
    torch.testing.assert_close(cache.key_residual, keys[0, :, 32:], rtol=0, atol=0)
    torch.testing.assert_close(cache.value_residual, values[0, :, 32:], rtol=0, atol=0)


def test_checkpoint_layers_do_not_overlap(tmp_path):
    from rover.cache.kivi import KIVIKVCache
    from rover.storage.files import CacheFile, create_cache_file

    config = KIVIConfig(
        bits=2,
        group_size=32,
        residual_length=128,
        prefill_chunk_size=17,
        device=torch.device("cpu"),
        kv_heads=2,
        head_dim=32,
        max_len=256,
    )
    cache = KIVIKVCache(num_layers=2, config=config)
    for layer in cache.layers:
        keys = torch.randn(1, 2, 177, 32, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        for start in range(0, 177, 17):
            end = min(start + 17, 177)
            layer.update_range(start, end, keys[:, :, start:end], values[:, :, start:end])
    key_path, value_path = tmp_path / "k.bin", tmp_path / "v.bin"
    create_cache_file(key_path, cache.prefill_key_file_size())
    create_cache_file(value_path, cache.prefill_value_file_size())
    restored = KIVIKVCache(num_layers=2, config=config, cache_metadata=cache.get_metadata())
    with CacheFile(key_path) as k_file, CacheFile(value_path) as v_file:
        for index in range(2):
            cache.save_prefill_layer(index, 0, 177, k_file, v_file)
        for index in range(2):
            restored.load_prefill_layer(index, 0, 177, k_file, v_file)
    for actual, expected in zip(restored.layers, cache.layers):
        torch.testing.assert_close(actual.key_residual, expected.key_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual.value_residual, expected.value_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual.key_prefix.packed, expected.key_prefix.packed, rtol=0, atol=0)
        torch.testing.assert_close(actual.value_prefix.packed, expected.value_prefix.packed, rtol=0, atol=0)
