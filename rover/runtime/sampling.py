"""Token selection and reproducible sampling."""

import torch

from rover.runtime.config import RuntimeConfig


class Sampling:
    def __init__(self, runtime_config: RuntimeConfig, generated_count: int = 0):
        self.runtime_config = runtime_config
        self.params = runtime_config.sampling
        self.generator_cpu = torch.Generator(device="cpu")
        self.sample_index = generated_count
        self._device = torch.device(runtime_config.device)

    def sample_next_token(self, logits: torch.Tensor) -> int:
        if not self.params.do_sample:
            return int(logits.argmax(dim=-1).item())
        logits_cpu = logits.detach().float().to(device="cpu")
        if self.params.temperature != 1.0:
            logits_cpu = logits_cpu / self.params.temperature
        vocab_size = int(logits_cpu.shape[-1])
        if self.params.top_k > 0 and self.params.top_k < vocab_size:
            k = int(self.params.top_k)
            values, indices = torch.topk(logits_cpu, k=k, dim=-1)
            filtered = torch.full_like(logits_cpu, float("-inf"))
            filtered.scatter_(dim=-1, index=indices, src=values)
            logits_cpu = filtered
        if self.params.top_p < 1.0:
            sorted_idx = torch.argsort(logits_cpu, dim=-1, descending=True, stable=True)
            sorted_logits = torch.gather(logits_cpu, dim=-1, index=sorted_idx)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative > self.params.top_p
            to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
            filtered = torch.full_like(logits_cpu, float("-inf"))
            filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
            logits_cpu = filtered
        probs = torch.softmax(logits_cpu, dim=-1)
        # A token-indexed random stream resumes exactly without extra RNG checkpoint files.
        self.generator_cpu.manual_seed((self.params.seed + self.sample_index) % (2**63))
        token = int(torch.multinomial(probs, num_samples=1, generator=self.generator_cpu).item())
        self.sample_index += 1
        return token
