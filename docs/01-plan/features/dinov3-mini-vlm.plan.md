# DINOv3 Mini VLM 계획서

> 버전: 1.0.0 | 날짜: 2026-05-28 | 상태: Plan 완료 후보
> 기능명: `dinov3-mini-vlm`
> 방향: Web/SaaS가 아니라 모델 자체를 이해하고 만드는 실험형 VLM 프로젝트

---

## 1. 개요

### 1.1 목적

이 프로젝트의 목적은 Reddit식 교육형 VLM 접근을 따라, DINOv3 vision encoder와 작은 LLM을 연결해 직접 동작하는 mini Vision-Language Model을 만드는 것이다.

목표 구조는 다음과 같다.

```text
이미지
  -> DINOv3 vision encoder
  -> patch features / CLS feature
  -> Q-Former 또는 Perceiver Resampler 또는 MLP projector
  -> LLM embedding space
  -> LLM
  -> 답변 생성
```

이번 계획에서 가장 중요한 원칙은 다음이다.

- 조잡한 Web UI를 만들지 않는다.
- SaaS, Admin, RAG, Agent 기능을 넣지 않는다.
- 먼저 모델 구조, 데이터, 학습 루프, 추론 루프를 이해 가능한 코드로 만든다.
- DINOv3와 LLM을 단순 호출하지 않고, 중간 정렬 모듈을 직접 구현하고 학습한다.

### 1.2 배경

이전 계획은 실제 모델보다 Web/API/SaaS 형태를 먼저 만들며 목표가 흐려졌다. 사용자는 Qwen3-VL을 단순 fine-tuning하는 것을 넘어, Reddit 글처럼 VLM 구성 원리를 직접 이해하고 만들고자 한다.

따라서 이번 계획은 다음 질문에 답하는 방향으로 재정의한다.

| 질문 | 이번 계획의 답 |
|------|----------------|
| DINOv3를 LLM과 붙여 VLM을 만들 수 있는가? | 가능하다. 단, feature alignment 모듈이 필요하다. |
| 무엇을 직접 구현하는가? | DINOv3 wrapper, visual token resampler/projector, LLM embedding 결합, 학습 루프 |
| 무엇을 가져다 쓰는가? | pretrained DINOv3, pretrained small LLM, tokenizer, image processor |
| 첫 성공 기준은 무엇인가? | 이미지 1장과 질문을 넣으면 짧은 caption 또는 답변을 생성한다. |

---

## 2. 목표

### 2.1 1차 목표

- [ ] DINOv3에서 이미지 patch feature와 CLS feature를 추출한다.
- [ ] DINOv3 feature를 LLM embedding 차원으로 변환하는 projector를 구현한다.
- [ ] MLP projector baseline을 먼저 구현한다.
- [ ] Q-Former 또는 Perceiver Resampler 중 하나를 2차 실험 모듈로 설계한다.
- [ ] LLM 입력 embedding 앞에 visual token을 붙여 답변을 생성한다.
- [ ] caption 데이터로 Stage 1 alignment 학습을 수행한다.
- [ ] 작은 VQA/instruction 데이터로 Stage 2 instruction tuning을 수행한다.
- [ ] 모델 구조, 학습 의도, 참고 근거, 실험 결과를 모두 한글 문서로 기록한다.

### 2.2 2차 목표

- [ ] LLM LoRA를 적용해 visual instruction following 성능을 높인다.
- [ ] patch feature 기반 방식과 CLS-only 방식의 차이를 비교한다.
- [ ] DINOv3 freeze와 partial fine-tuning의 차이를 비교한다.
- [ ] 최소 평가셋을 만들어 caption 품질과 VQA 정답률을 기록한다.
- [ ] 추론 CLI를 만들어 Web 없이 이미지+질문 테스트를 반복한다.

### 2.3 하지 않을 것

- Web UI, SaaS 관리 페이지, Admin dashboard를 만들지 않는다.
- RAG, vector DB, document upload 기능을 넣지 않는다.
- Qwen3-VL급 대규모 모델을 처음부터 재현한다고 주장하지 않는다.
- 대규모 pretraining을 목표로 하지 않는다.
- 모델 성능을 과장하지 않는다.

---

## 3. 범위

### 3.1 포함 범위

| 영역 | 포함 내용 |
|------|-----------|
| Vision Encoder | Hugging Face DINOv3 모델 로딩, patch/CLS feature 추출 |
| Visual Token Adapter | MLP projector baseline, Q-Former/Resampler 후보 설계 |
| LLM 연결 | visual embedding과 text embedding 결합 |
| 학습 | Stage 1 caption alignment, Stage 2 VQA/instruction tuning |
| 데이터 | 작은 로컬 JSONL 데이터셋, Conceptual Captions 스타일 샘플 지원 |
| 추론 | CLI 기반 이미지 질문 답변 |
| 테스트 | shape test, forward pass test, dataset parsing test, smoke training test |
| 기록 | Plan/Design/Do/Analysis/Problem-Solution log |

### 3.2 제외 범위

| 제외 항목 | 제외 이유 |
|----------|-----------|
| Web UI | 현재 핵심 목표가 모델 구조 이해와 학습이기 때문 |
| API 서버 | 모델 forward/inference가 안정된 뒤 붙일 수 있음 |
| SaaS 과금/계정/관리 | 제품화 단계가 아님 |
| RAG | mini VLM이 먼저 이미지와 질문을 이해해야 함 |
| 대규모 GPU 분산 학습 | 첫 실험은 단일 GPU 또는 작은 batch에서 검증 |
| 완전한 Qwen3-VL 재현 | 데이터/컴퓨팅 규모가 다름 |

---

## 4. 모델 아키텍처 계획

### 4.1 기본 구조

```text
Image
  -> DINOv3ImageProcessor
  -> DINOv3ViTModel
  -> patch_features: [batch, patches, vision_dim]
  -> VisualAdapter
  -> visual_tokens: [batch, visual_token_count, llm_dim]
  -> LLM token embeddings
  -> concat([visual_tokens, text_embeddings])
  -> LLM generate
```

### 4.2 모듈 후보

| 모듈 | 1차 선택 | 대안 | 이유 |
|------|----------|------|------|
| Vision Encoder | DINOv3 ViT small/base | DINOv3 ConvNeXT | patch token 구조가 VLM 연결에 직관적 |
| Visual Adapter | MLP projector | Q-Former, Perceiver Resampler | baseline을 먼저 성공시킨 뒤 복잡도 증가 |
| LLM | Qwen3 0.6B 또는 SmolLM 계열 | TinyLlama, Phi 계열 | 단일 GPU/로컬 실험 가능성 |
| 학습 방식 | projector 학습 후 LoRA | full fine-tuning | 안정성과 비용 우선 |
| 데이터 | caption JSONL | VQA JSONL, UI screenshot QA | 첫 목표는 caption 생성 |

### 4.3 학습 단계

#### Stage 0: Shape 검증

목표는 학습 전에 tensor shape가 맞는지 확인하는 것이다.

```text
image -> DINOv3 -> visual adapter -> LLM embedding dim
```

성공 기준:

- batch forward가 에러 없이 실행된다.
- visual token shape가 설정값과 일치한다.
- LLM embedding dim과 projector output dim이 일치한다.

#### Stage 1: Caption Alignment

목표는 visual token이 LLM에게 의미 있는 prefix가 되도록 학습하는 것이다.

학습 대상:

- MLP projector 또는 Q-Former
- 필요 시 LLM LoRA 일부

초기 freeze 정책:

| 컴포넌트 | Stage 1 |
|----------|---------|
| DINOv3 | freeze |
| Projector/Resampler | train |
| LLM | freeze 또는 LoRA off |

#### Stage 2: Visual Instruction Tuning

목표는 이미지 설명을 넘어 질문에 답하게 만드는 것이다.

학습 데이터 예:

```json
{
  "image": "images/sample.png",
  "question": "이 화면에서 사용자가 눌러야 할 버튼은?",
  "answer": "오른쪽 아래의 전송 버튼입니다."
}
```

학습 대상:

- Projector/Resampler
- LLM LoRA

---

## 5. 데이터 계획

### 5.1 데이터 형식

첫 데이터 형식은 JSONL로 고정한다.

```json
{
  "image": "images/example.jpg",
  "question": "Describe this image.",
  "answer": "A dog is sitting on the grass.",
  "task": "caption"
}
```

VQA 데이터도 같은 형식을 사용한다.

```json
{
  "image": "images/ui-001.png",
  "question": "이 화면에서 오류 메시지는 무엇인가요?",
  "answer": "모델 endpoint가 설정되지 않았다는 메시지입니다.",
  "task": "vqa"
}
```

### 5.2 데이터 단계

| 단계 | 데이터 | 목표 개수 | 목적 |
|------|--------|-----------|------|
| D0 | 수동 샘플 | 10~30개 | 파이프라인 검증 |
| D1 | caption 데이터 | 1천~5천개 | alignment 학습 |
| D2 | VQA 데이터 | 200~1천개 | 질문 답변 학습 |
| D3 | UI/document screenshot QA | 100~500개 | 사용자가 원하는 도메인 적응 |

### 5.3 데이터 품질 원칙

- 이미지 파일 경로가 실제 존재해야 한다.
- 질문에는 필요한 visual token placeholder 정책을 명확히 둔다.
- 답변은 짧고 검증 가능해야 한다.
- OCR이 필요한 문제는 별도 태그를 붙인다.
- train/validation split을 반드시 둔다.

---

## 6. 프로젝트 구조 계획

Web/SaaS 구조를 없애고 모델 연구 구조로 바꾼다.

```text
configs/
  dinov3-mini-vlm.yaml
data/
  raw/
  processed/
  samples/
mini_vlm/
  data/
    dataset.py
    collator.py
  models/
    vision_encoder.py
    visual_adapter.py
    qformer.py
    mini_vlm.py
  training/
    train_alignment.py
    train_instruction.py
  inference/
    infer_cli.py
  utils/
    checkpoints.py
    seed.py
tests/
  test_dataset.py
  test_visual_adapter.py
  test_forward.py
docs/
  00-history/
  01-plan/
  02-design/
  03-analysis/
```

---

## 7. 구현 순서

| 순서 | 작업 | 산출물 | 완료 기준 |
|------|------|--------|-----------|
| 1 | 기존 Web/API 중심 코드 제거 여부 결정 | 정리 로그 | 모델 프로젝트 구조만 남김 |
| 2 | 설정 파일 작성 | `configs/dinov3-mini-vlm.yaml` | 모델명, dim, token 수 정의 |
| 3 | Dataset/Collator 구현 | `mini_vlm/data/*` | JSONL 로딩 테스트 통과 |
| 4 | DINOv3 wrapper 구현 | `vision_encoder.py` | patch/CLS feature shape 확인 |
| 5 | MLP projector 구현 | `visual_adapter.py` | LLM dim projection 테스트 |
| 6 | MiniVLM forward 구현 | `mini_vlm.py` | image+text batch forward 통과 |
| 7 | Stage 1 학습 루프 | `train_alignment.py` | overfit tiny dataset 성공 |
| 8 | CLI 추론 | `infer_cli.py` | 이미지+질문 답변 출력 |
| 9 | Q-Former/Resampler 확장 | `qformer.py` | MLP baseline과 비교 가능 |
| 10 | Stage 2 instruction tuning | `train_instruction.py` | VQA 샘플 overfit 성공 |

---

## 8. 성공 기준

### 8.1 최소 성공 기준

- [ ] 이미지 1장을 DINOv3에 넣어 patch feature를 얻는다.
- [ ] visual adapter가 LLM embedding dimension으로 변환한다.
- [ ] MiniVLM forward pass가 에러 없이 동작한다.
- [ ] tiny dataset 10개를 overfit할 수 있다.
- [ ] CLI에서 이미지와 질문을 넣으면 답변 문자열이 생성된다.

### 8.2 1차 완료 기준

- [ ] `python -m unittest discover -s tests -v` 통과
- [ ] `python -m mini_vlm.inference.infer_cli --image ... --question ...` 실행 성공
- [ ] 학습 loss가 tiny dataset에서 감소하는 로그 확보
- [ ] checkpoint 저장과 재로드 성공
- [ ] 한글 문서에 문제와 해결 내역 기록

### 8.3 품질 기준

- 모델 코드는 모듈별로 분리한다.
- 모든 tensor shape 변환 지점에는 의도 주석을 남긴다.
- 학습 스크립트는 config 기반으로 실행한다.
- 외부 모델명과 경로는 코드에 하드코딩하지 않는다.
- 실험 실패도 문제-해결 로그에 기록한다.

---

## 9. 리스크와 대응

| 리스크 | 영향 | 가능성 | 대응 |
|--------|------|--------|------|
| DINOv3 feature가 LLM에 잘 정렬되지 않음 | 답변 품질 저하 | 높음 | tiny dataset overfit부터 확인 |
| pooled CLS만 사용해 세부 정보 손실 | 위치/세부 설명 실패 | 높음 | patch token baseline을 기본으로 사용 |
| GPU 메모리 부족 | 학습 불가 | 중간 | DINOv3 small/base, 작은 LLM, freeze, LoRA 사용 |
| OCR 성능 부족 | 문서/화면 이해 약함 | 높음 | OCR은 별도 task로 분리, 처음엔 caption/VQA 중심 |
| 학습 데이터 품질 부족 | 모델이 무의미한 답변 생성 | 높음 | 수동 검수 샘플부터 시작 |
| Web/API 유혹으로 범위가 커짐 | 프로젝트 재혼란 | 높음 | 이번 PDCA에서는 Web/API 제외 |

---

## 10. 참고 자료

| 자료 | 용도 |
|------|------|
| DINOv3 Hugging Face 문서 | vision encoder 로딩과 feature 추출 |
| Meta DINOv3 소개 | DINOv3의 성격과 장단점 이해 |
| Reddit VLM from scratch 글 | 전체 접근 방식 참고 |
| `avbiswas/vlm` GitHub | Q-Former + small LLM 학습 흐름 참고 |
| BLIP-2 논문/구조 | Q-Former 설계 참고 |
| Flamingo/Perceiver Resampler 계열 자료 | visual token 압축 대안 참고 |

---

## 11. 기존 계획 삭제 기록

사용자 요청에 따라 기존 Web/API 중심 계획은 삭제했다.

삭제 대상:

- `docs/01-plan/features/real-model-chat-platform.plan.md`
- `docs/02-design/features/real-model-chat-platform.design.md`

삭제 이유:

- 사용자의 현재 목표는 서비스 화면이 아니라 DINOv3와 LLM을 연결한 mini VLM 구현이다.
- 기존 계획은 실제 모델 구조 구현보다 API/Web 연결에 초점이 있었다.
- 새 계획은 모델 내부 구조, 학습 루프, 데이터 흐름을 우선한다.

---

## 12. 다음 단계

다음 PDCA 단계는 Design이다.

```text
$pdca design dinov3-mini-vlm
```

Design 단계에서 결정할 항목:

- DINOv3 모델 크기 선택
- LLM 후보 선택
- visual adapter 1차 구현 방식
- tensor shape 상세 설계
- 학습 loss와 masking 정책
- checkpoint 저장 형식
- 최소 테스트 세트
