from typing import List, Tuple

import numpy as np
from torch import Tensor, from_numpy


def shuffle_sequence(sequence_length: int) -> np.ndarray:
    """Shuffle indexes of tokens."""
    return np.random.permutation(sequence_length)


def create_perm_msk(indexes: np.ndarray, length: int) -> np.ndarray:
    """Create permutation mask for XLNet."""
    perm_msk = np.ones((length, length), dtype=np.float32)
    for i, index in enumerate(np.nditer(indexes)):
        perm_msk[index, indexes[: i + 1]] = 0
    return perm_msk


def select_msk_tokens(sequence_length: int, length: int, portion: float) -> np.ndarray:
    """Select masked tokens and return binary mask for them."""
    masked_msk = np.zeros(length, dtype=np.int32)
    count_masked = int(sequence_length * portion)
    masked_indexes = np.random.choice(sequence_length, count_masked)
    masked_msk[masked_indexes] = 1
    return masked_msk


def transform_xlnet(
    sentence: List[int], length: int, portion: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transfrom to tensor input token ids and create permutation mask and binary mask for masked tokens.
    """
    sequence = np.array(sentence, dtype=np.int32)
    sequence_length_with_padding = sequence.shape[0]  # length with padding
    indexes = shuffle_sequence(sequence_length=length)
    perm_msk = create_perm_msk(indexes=indexes, length=sequence_length_with_padding)
    masked_msk = select_msk_tokens(
        sequence_length=length, length=sequence_length_with_padding, portion=portion
    )
    return from_numpy(sequence), from_numpy(perm_msk), from_numpy(masked_msk)
