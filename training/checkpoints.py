"""Atomic trusted training snapshots and complete RNG restoration."""

import random

import numpy as np
import torch


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
