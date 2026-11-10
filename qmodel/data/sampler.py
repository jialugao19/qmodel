from torch.utils.data import Sampler
import numpy as np

from qmodel.data.cpp.permute import Permute64

class CustomSampler(Sampler):
    def __init__(
        self,
        dataset,
        seed: int,
        shuffle: bool,
        infinite: bool,
        global_batch_size: int,
        rank: int,
        world_size: int,
        drop_last: bool,
    ):
        """Create a sampler that maps global batches to one rank section."""
        # Store dataset and ordering state.
        self.dataset = dataset
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.shuffler = Permute64(len(self.dataset), seed)

        # Store distributed section geometry.
        assert global_batch_size % world_size == 0
        self.global_batch_size = global_batch_size
        self.local_batch_size = global_batch_size // world_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last

        # Store one-shot iterator state.
        self.not_yet_started = True
        self.start = 0

    def set_start_iter(self, iteration: int):
        """Set the global sample offset used by checkpoint resume."""
        # Resume from a global batch boundary.
        assert iteration % self.global_batch_size == 0
        self.start = iteration

    def __iter__(self):
        """Yield this rank's indices from each global batch."""
        # Enforce the existing sampler contract that one instance is used once.
        assert self.not_yet_started
        self.not_yet_started = False

        # Iterate the first epoch from the requested global sample offset.
        start = self.start % (len(self.dataset) // self.global_batch_size * self.global_batch_size) if self.drop_last else self.start
        while True:
            yield from self._iter_epoch(start)
            if not self.infinite:
                break
            start = 0

    def _iter_epoch(self, start: int):
        """Yield rank-local sections for one dataset pass."""
        # Walk global batches in sample-index order.
        n = len(self.dataset)
        for global_start in range(start, n, self.global_batch_size):
            global_end = global_start + self.global_batch_size
            if self.drop_last and global_end > n:
                break

            # Build one global batch and slice this rank's contiguous section.
            global_positions = np.arange(global_start, min(global_end, n), dtype=np.int64)
            if self.shuffle:
                global_indices = self.shuffler.permute(global_positions)
            else:
                global_indices = global_positions

            local_start = self.rank * self.local_batch_size
            local_end = min(local_start + self.local_batch_size, len(global_indices))
            yield from global_indices[local_start:local_end].tolist()

    def __len__(self):
        """Return the number of samples yielded by this rank per dataset pass."""
        # Count full global batches first.
        n = len(self.dataset)
        full_steps = n // self.global_batch_size
        count = full_steps * self.local_batch_size

        # Add this rank's final partial section for eval/predict.
        if not self.drop_last:
            remainder = n % self.global_batch_size
            local_start = self.rank * self.local_batch_size
            local_end = min(local_start + self.local_batch_size, remainder)
            count += max(0, local_end - local_start)
        return count

    def state_dict(self, consumed_batches):
        # Record consumed positions independently of loader prefetch.
        return {
            "seed": self.seed, "shuffle": self.shuffle,
            "size": len(self.dataset), "global_batch_size": self.global_batch_size,
            "world_size": self.world_size, "consumed_batches": consumed_batches,
        }

    def load_state_dict(self, state):
        # Require unchanged ordering and distributed geometry.
        for key in ("seed", "shuffle", "size", "global_batch_size", "world_size"):
            actual = len(self.dataset) if key == "size" else getattr(self, key)
            if actual != state[key]:
                raise ValueError(f"Sampler mismatch: {key}")
        self.set_start_iter(state["consumed_batches"] * self.global_batch_size)
