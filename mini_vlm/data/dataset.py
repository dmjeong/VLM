from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class MiniVlmSample:
    """JSONL annotation 한 줄을 표현한다."""

    sample_id: str
    image: str
    question: str
    answer: str
    task: str = "caption"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: int) -> "MiniVlmSample":
        image = str(payload.get("image") or "").strip()
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if not image:
            raise ValueError(f"{index}번째 샘플에 image가 없습니다.")
        if not question:
            raise ValueError(f"{index}번째 샘플에 question이 없습니다.")
        if not answer:
            raise ValueError(f"{index}번째 샘플에 answer가 없습니다.")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return cls(
            sample_id=str(payload.get("sample_id") or f"sample-{index:06d}"),
            image=image,
            question=question,
            answer=answer,
            task=str(payload.get("task") or "caption"),
            metadata=dict(metadata),
        )


class MiniVlmDataset(Sequence[MiniVlmSample]):
    """mini VLM JSONL 데이터셋.

    의도: 데이터 로딩 단계에서는 이미지 tensor 변환을 하지 않고 경로와 텍스트 계약만 검증한다.
    참고: 설계서 5장 데이터 설계.
    선택 이유: 이미지 processor는 DINOv3 모델에 종속적이므로 collator에서 처리하는 편이 교체와 테스트가 쉽다.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        require_images: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.require_images = require_images
        self._samples = load_jsonl_samples(self.annotation_path)
        if self.require_images:
            self.validate_image_paths()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> MiniVlmSample:
        return self._samples[index]

    def image_path_for(self, sample: MiniVlmSample) -> Path:
        path = Path(sample.image)
        if path.is_absolute():
            return path
        return self.image_root / path

    def validate_image_paths(self) -> None:
        missing = [str(self.image_path_for(sample)) for sample in self._samples if not self.image_path_for(sample).exists()]
        if missing:
            raise FileNotFoundError("존재하지 않는 이미지 파일이 있습니다: " + ", ".join(missing[:5]))


def load_jsonl_samples(path: str | Path) -> list[MiniVlmSample]:
    samples: list[MiniVlmSample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for index, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{index}번째 JSONL 항목은 객체여야 합니다.")
            samples.append(MiniVlmSample.from_dict(payload, index=index))
    return samples
