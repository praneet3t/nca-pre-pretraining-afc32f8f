"""
Structured / unstructured synthetic data sources for pre-pre-training,
to test whether the transfer to language comes from STRUCTURE (sudoku, NCA)
vs any synthetic tokens (random) vs nothing (scratch).

Token alphabet (all within pt_vocab_size=64000 so embeddings match the OWT model):
  0      = pad (masked in loss)
  1..9   = sudoku digits (also the alphabet for the random control)
  10     = <grid> start
  11     = </grid> end
  vocab  = 12 distinct tokens
"""
import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 0
START = 10
END = 11
SYNTH_VOCAB = 12  # only tokens 0..11 are used


def make_sudoku_solution(rng):
    """Generate a uniformly-random valid solved 9x9 sudoku grid (values 1-9)
    via pattern + random band/row/stack/col/digit permutations."""
    base, side = 3, 9
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side
    rBase = range(base)
    rows = [g * base + r for g in rng.permutation(rBase) for r in rng.permutation(rBase)]
    cols = [g * base + c for g in rng.permutation(rBase) for c in rng.permutation(rBase)]
    nums = rng.permutation(np.arange(1, side + 1))
    return np.array([[nums[pattern(r, c)] for c in cols] for r in rows])


def _pack_grid_seq(make_digits, seq_len, rng):
    """Pack [START <81 digits> END]* into a length-seq_len sequence.
    make_digits() -> np.array of 81 ints in 1..9."""
    tokens = []
    while len(tokens) < seq_len - 1:
        tokens.append(START)
        tokens.extend(int(x) for x in make_digits())
        tokens.append(END)
    seq = tokens[:seq_len]
    seq = seq + [PAD] * (seq_len - len(seq))
    target = seq[1:] + [PAD]
    # mask positions whose target is PAD (no prediction over padding)
    target = [t if t != PAD else -100 for t in target]
    return seq, target


class SudokuDataset(Dataset):
    """Valid solved sudoku grids serialized as digit-token sequences."""
    def __init__(self, num_sequences, seq_len, seed=0):
        self.seq_len = seq_len
        self.n = num_sequences
        self.data = []
        for i in range(num_sequences):
            rng = np.random.default_rng(seed + i)
            def make_digits(r=rng):
                return make_sudoku_solution(r).flatten()
            self.data.append(_pack_grid_seq(make_digits, seq_len, rng))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s, t = self.data[i]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)


class RandomDigitDataset(Dataset):
    """Unstructured control: same alphabet/shape as SudokuDataset but digits are
    i.i.d. uniform over 1..9 (no sudoku constraints). Isolates structure."""
    def __init__(self, num_sequences, seq_len, seed=0):
        self.seq_len = seq_len
        self.n = num_sequences
        self.data = []
        for i in range(num_sequences):
            rng = np.random.default_rng(seed + 100000 + i)
            def make_digits(r=rng):
                return r.integers(1, 10, size=81)
            self.data.append(_pack_grid_seq(make_digits, seq_len, rng))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s, t = self.data[i]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)


def get_dataset(source, num_sequences, seq_len, seed=0):
    if source == "sudoku":
        return SudokuDataset(num_sequences, seq_len, seed)
    if source == "random":
        return RandomDigitDataset(num_sequences, seq_len, seed)
    raise ValueError(f"unknown source: {source}")
