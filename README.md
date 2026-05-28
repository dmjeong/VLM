# DINOv3 Mini VLM

> 목표: DINOv3 vision encoder와 작은 LLM을 직접 연결해 mini Vision-Language Model을 만든다.

## 현재 방향

```text
이미지
  -> DINOv3 vision encoder
  -> patch features / CLS feature
  -> MLP visual adapter
  -> LLM embedding space
  -> LLM
  -> 답변 생성
```

이번 프로젝트는 Web/API/SaaS가 아니다. 모델 구조, tensor shape, 학습 루프, 추론 CLI를 먼저 만든다.

## 검증

현재 기본 테스트는 `torch`와 `transformers` 없이도 실행된다. 실제 DINOv3/LLM 통합은 optional 경로다.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall mini_vlm tests
```

실제 모델 실험을 하려면 의존성을 설치한다.

```bash
pip install '.[model]'
```

## 문서

- 계획: `docs/01-plan/features/dinov3-mini-vlm.plan.md`
- 설계: `docs/02-design/features/dinov3-mini-vlm.design.md`
- 구현 추적: `docs/02-design/features/dinov3-mini-vlm.do.md`
