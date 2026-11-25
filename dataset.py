import random
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader


class SymbolicDataset(Dataset):
    def __init__(self, examples, tok, config, augment: bool):
        self.examples = examples
        self.augment = augment
        self.max_transpose = config.max_transpose
        self.pitch_range = tuple(config.pitch_range)
        self.n_tries = 3
        if augment:
            self.table, self.inverse = pitch_table(tok)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        ids = ex["ids"]
        if self.augment:
            for _ in range(self.n_tries):
                k = random.randint(-self.max_transpose, self.max_transpose)
                if k == 0:
                    break

                novo = transpose_ids(ids, k, self.table, self.inverse, self.pitch_range)
                if novo is not None:
                    ids = novo
                    break

        return torch.tensor(ids, dtype=torch.long), ex["sep_idx"]


def worker_init_fn(worker_id):
    random.seed(torch.initial_seed() % 2 ** 32)  # semente por worker, para o k aleatório


def collate(batch, pad_id):
    B = len(batch)
    lengths = torch.tensor([len(ids) for ids, _ in batch])
    L = int(lengths.max())

    x = torch.full((B, L), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((B, L), dtype=torch.bool)
    sep_idx = torch.zeros(B, dtype=torch.long)

    for i, (ids, sep) in enumerate(batch):
        n = len(ids)
        x[i, :n] = ids
        loss_mask[i, sep + 2:n] = True  # piano + EOS
        sep_idx[i] = sep

    pad_mask = torch.arange(L)[None, :] >= lengths[:, None]
    return {"x": x, "loss_mask": loss_mask, "pad_mask": pad_mask, "sep_idx": sep_idx, "lengths": lengths}


def make_loader(examples, tok, config, batch_size, augment, shuffle, num_workers=0):
    ds = SymbolicDataset(examples, tok, config, augment)
    pad_id = special_id(tok, "PAD")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=partial(collate, pad_id=pad_id), worker_init_fn=worker_init_fn)


def check_batch(batch, tok, n=3):
    """Sanidade da máscara de perda (use sempre antes de treinar)."""
    x, mask = batch["x"], batch["loss_mask"]
    sep_id, out_id, eos_id = special_id(tok, "SEP"), special_id(tok, OUT_TOKEN), special_id(tok, "EOS")

    assert not (mask & batch["pad_mask"]).any(), "padding entrou na máscara de perda"
    for i in range(min(n, x.shape[0])):
        sep, L = int(batch["sep_idx"][i]), int(batch["lengths"][i])
        assert x[i, sep] == sep_id and x[i, sep + 1] == out_id, "SEP/OUT_Piano fora do lugar"

        alvo = x[i, 1:][mask[i, 1:]]  # o que o modelo é treinado a prever
        assert alvo[0] == x[i, sep + 2], "o alvo não começa no 1º token do piano"
        assert alvo[-1] == eos_id, "o alvo não termina em EOS"
        assert alvo.numel() == L - (sep + 2), "tamanho do alvo != len(piano)+1"

    print("check_batch OK")