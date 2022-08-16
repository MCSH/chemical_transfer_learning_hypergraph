import torch

ctx = {
    "device": "cuda",
    "mte_activation": lambda: torch.nn.Sigmoid(),
    "mtx_activation": lambda: torch.nn.Sigmoid(),
}
