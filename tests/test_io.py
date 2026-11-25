import torch

from rover.storage.files import CacheFile, create_cache_file


def test_multichunk_tensor_io(tmp_path):
    path = tmp_path / "tensor.bin"
    tensor = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    create_cache_file(path, tensor.nbytes)
    with CacheFile(path) as file:
        file._IO_CHUNK_BYTES = 32
        file.write_tensor(tensor, 0)
        restored = torch.empty_like(tensor)
        file.read_tensor(restored, tensor.nbytes, 0)
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
