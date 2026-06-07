from qmodel.data.sampler import CustomSampler


class RangeDataset:
    def __init__(self, n_items: int):
        """Store dataset length for sampler tests."""
        # Keep only the length because sampler never reads dataset values.
        self.n_items = n_items

    def __len__(self) -> int:
        """Return dataset length."""
        # Expose the synthetic map-style dataset size.
        return self.n_items


def _indices(
    n_items: int,
    global_batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool,
    start: int = 0,
) -> list[int]:
    """Collect sampler indices for one rank."""
    # Build the sampler with the requested distributed geometry.
    sampler = CustomSampler(
        RangeDataset(n_items),
        seed=17,
        shuffle=shuffle,
        infinite=False,
        global_batch_size=global_batch_size,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
    )

    # Apply checkpoint-style global offset before iteration.
    if start > 0:
        sampler.set_start_iter(start)
    return list(iter(sampler))


def test_contiguous_sections_without_shuffle():
    """Verify rank-local contiguous sections from global batches."""
    # Collect two rank sections from three complete global batches.
    rank0 = _indices(24, 8, 0, 2, shuffle=False, drop_last=True)
    rank1 = _indices(24, 8, 1, 2, shuffle=False, drop_last=True)

    # Check each rank owns one contiguous half of every global batch.
    assert rank0 == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    assert rank1 == [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]


def test_resume_from_global_sample_offset():
    """Verify resume starts from the requested global batch."""
    # Resume at the third global batch.
    rank0 = _indices(24, 8, 0, 2, shuffle=False, drop_last=True, start=16)
    rank1 = _indices(24, 8, 1, 2, shuffle=False, drop_last=True, start=16)

    # Check both ranks resumed from the same global offset.
    assert rank0 == [16, 17, 18, 19]
    assert rank1 == [20, 21, 22, 23]


def test_shuffle_reconstructs_single_rank_global_order():
    """Verify shuffled rank sections reconstruct single-rank order."""
    # Build single-rank global order and two distributed sections.
    single = _indices(24, 8, 0, 1, shuffle=True, drop_last=True)
    rank0 = _indices(24, 8, 0, 2, shuffle=True, drop_last=True)
    rank1 = _indices(24, 8, 1, 2, shuffle=True, drop_last=True)

    # Reconstruct global batch order by concatenating rank sections per step.
    merged = []
    for step_idx in range(3):
        merged.extend(rank0[step_idx * 4:(step_idx + 1) * 4])
        merged.extend(rank1[step_idx * 4:(step_idx + 1) * 4])

    assert merged == single


def test_eval_keeps_partial_global_batch():
    """Verify eval keeps final partial global batch sections."""
    # Collect eval sections with a two-sample final partial global batch.
    rank0 = _indices(18, 8, 0, 2, shuffle=False, drop_last=False)
    rank1 = _indices(18, 8, 1, 2, shuffle=False, drop_last=False)

    # Check rank0 owns the final two remaining rows and rank1 has no final section.
    assert rank0 == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17]
    assert rank1 == [4, 5, 6, 7, 12, 13, 14, 15]
