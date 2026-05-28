from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Python과 torch seed를 설정한다."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
