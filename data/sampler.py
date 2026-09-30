import math
from typing import Iterator, Optional, Sequence
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    """
    Distributed sampler that performs weighted random sampling with replacement across multiple GPUs.
    Ensures that:
      1. Every rank generates the identical multinomial sequence using a synchronized generator.
      2. The sampled indices are cleanly partitioned across ranks (rank 0 gets [0::N], rank 1 gets [1::N]).
      3. All ranks get an identical number of samples per epoch to avoid DDP desynchronization or deadlocks.
      4. Epoch-based reshuffling is supported via `set_epoch(epoch)`.
    """
    def __init__(
        self,
        dataset,
        weights: Sequence[float],
        num_samples: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, need 0 <= rank < {num_replicas}")

        self.dataset = dataset
        if not isinstance(weights, torch.Tensor):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
        else:
            self.weights = weights.to(dtype=torch.double)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.replacement = replacement

        if num_samples is None:
            num_samples = len(self.weights)

        # Number of samples per replica
        self.num_samples_per_replica = math.ceil(num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Synchronously sample total_size indices across all replicas
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()

        # Deterministically slice for this rank
        sub_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(sub_indices) == self.num_samples_per_replica

        return iter(sub_indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
