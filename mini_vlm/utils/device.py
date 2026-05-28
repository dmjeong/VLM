from __future__ import annotations


def select_torch_device(preferred_device: str = "auto"):
    """학습/추론에 사용할 torch device를 고른다.

    의도: Mac 개발 환경에서는 CUDA가 없어도 MPS가 가능할 수 있다. CPU로 고정하면 full epoch 실험 시간이
    불필요하게 길어지므로, 설정이 `auto`일 때 CUDA, MPS, CPU 순서로 선택한다.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch device를 선택하려면 torch가 필요합니다.") from exc

    normalized = preferred_device.strip().lower()
    if normalized != "auto":
        return torch.device(normalized)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
