# DINOv3 Mini VLM 설계서

> 버전: 1.0.0 | 날짜: 2026-05-28 | 상태: Design 완료 후보
> 기능명: `dinov3-mini-vlm`
> 계획 문서: [dinov3-mini-vlm.plan.md](../../01-plan/features/dinov3-mini-vlm.plan.md)

---

## 1. 설계 목표

이 설계서는 DINOv3 vision encoder와 작은 LLM을 직접 연결해 mini VLM을 만드는 구현 기준을 정의한다.

핵심 목표는 다음 구조를 실제 코드와 학습 루프로 검증하는 것이다.

```text
이미지
  -> DINOv3 vision encoder
  -> patch features / CLS feature
  -> visual adapter
  -> LLM embedding space
  -> LLM
  -> 답변 생성
```

이번 설계에서 Web, API, SaaS, RAG, Admin은 제외한다. 산출물은 모델 코드, 학습 스크립트, 추론 CLI, 테스트, 한글 기록 문서다.

---

## 2. 전체 아키텍처

### 2.1 런타임 흐름

```text
이미지 파일 + 질문
  -> MiniVlmDataset
  -> MiniVlmCollator
  -> VisionEncoder(DINOv3)
  -> VisualAdapter(MLP baseline)
  -> LlmBackbone(input embeddings)
  -> MiniVlmForConditionalGeneration
  -> loss 또는 generated answer
```

### 2.2 학습 흐름

```text
JSONL annotation
  -> image load
  -> question/answer tokenization
  -> DINOv3 patch tokens
  -> visual tokens
  -> concat visual tokens + text tokens
  -> causal LM loss
  -> adapter checkpoint 저장
```

### 2.3 주요 설계 원칙

| 원칙 | 설명 |
|------|------|
| 모델 우선 | Web UI보다 forward/loss/generation을 먼저 완성한다. |
| 작은 baseline 우선 | Q-Former보다 MLP projector를 먼저 성공시킨다. |
| freeze 우선 | DINOv3와 LLM을 먼저 freeze하고 adapter만 학습한다. |
| shape 명시 | 모든 모듈 입출력 tensor shape를 문서와 테스트로 검증한다. |
| 실패 기록 | 학습이 안 되는 실험도 문제-해결 로그에 기록한다. |

---

## 3. 프로젝트 구조

```text
configs/
  dinov3-mini-vlm.json
data/
  samples/
    train.jsonl
    validation.jsonl
    images/
mini_vlm/
  __init__.py
  config.py
  data/
    __init__.py
    dataset.py
    collator.py
  models/
    __init__.py
    vision_encoder.py
    visual_adapter.py
    qformer.py
    mini_vlm.py
  training/
    __init__.py
    train_alignment.py
    train_instruction.py
  inference/
    __init__.py
    infer_cli.py
  utils/
    __init__.py
    checkpoints.py
    seed.py
tests/
  test_dataset.py
  test_collator.py
  test_visual_adapter.py
  test_forward.py
docs/
  00-history/
  01-plan/
  02-design/
  03-analysis/
```

현재 남아 있는 `apps/`, `services/api/`, `services/inference/`는 이전 v2 Web/API 실험의 잔여물이다. Do 단계 첫 작업에서 삭제하거나 `archive`로 이동해 모델 프로젝트와 섞이지 않게 한다.

---

## 4. 설정 설계

### 4.1 설정 파일

`configs/dinov3-mini-vlm.json`

```json
{
  "experiment_name": "dinov3-mini-vlm-mlp-baseline",
  "vision_model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
  "llm_model_id": "Qwen/Qwen3-0.6B",
  "adapter_type": "mlp",
  "visual_token_count": 32,
  "max_text_length": 256,
  "train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.0001,
  "num_train_epochs": 3,
  "freeze_vision": true,
  "freeze_llm": true,
  "use_lora": false,
  "output_dir": "artifacts/dinov3-mini-vlm/mlp-baseline"
}
```

### 4.2 설정 객체

| 객체 | 필드 | 역할 |
|------|------|------|
| `MiniVlmConfig` | model ids, adapter type, token count | 전체 실험 설정 |
| `DataConfig` | train/validation path, image root | 데이터 경로 |
| `TrainingConfig` | batch, lr, epoch, precision | 학습 파라미터 |
| `CheckpointConfig` | output dir, save steps | 저장 정책 |

YAML 대신 JSON을 1차 선택으로 둔다. 이유는 표준 라이브러리만으로 파싱 가능하고, 초기 구현 복잡도가 낮기 때문이다.

---

## 5. 데이터 설계

### 5.1 JSONL annotation

```json
{
  "sample_id": "sample-0001",
  "image": "images/dog.jpg",
  "question": "Describe this image.",
  "answer": "A dog is sitting on the grass.",
  "task": "caption",
  "metadata": {
    "source": "manual"
  }
}
```

### 5.2 데이터 클래스

| 클래스 | 역할 |
|--------|------|
| `MiniVlmSample` | JSONL 한 줄을 표현 |
| `MiniVlmDataset` | 이미지 경로와 텍스트 샘플 로딩 |
| `MiniVlmBatch` | collator가 만든 batch |

### 5.3 Batch 구조

```text
pixel_values: FloatTensor [B, 3, H, W]
input_ids: LongTensor [B, T]
attention_mask: LongTensor [B, T]
labels: LongTensor [B, T]
sample_ids: list[str]
```

`labels`는 답변 token만 학습하도록 masking한다.

```text
질문 prompt token label = -100
답변 token label = token id
padding label = -100
```

visual token label은 MiniVLM forward 단계에서 앞쪽에 붙으며 모두 `-100`으로 확장한다.

---

## 6. 모델 모듈 설계

### 6.1 `VisionEncoder`

역할:

- DINOv3 image processor로 이미지를 전처리한다.
- DINOv3 model을 호출한다.
- CLS token과 patch token을 분리한다.

입출력:

```text
입력:
  pixel_values: [B, 3, H, W]

출력:
  cls_token: [B, vision_dim]
  patch_tokens: [B, patch_count, vision_dim]
  pooled_output: [B, vision_dim] 또는 None
```

초기 정책:

- `freeze_vision=true`가 기본값이다.
- Do 단계에서는 `torch.no_grad()`와 `requires_grad_(False)`를 모두 적용한다.
- DINOv3 integration test는 모델 다운로드가 필요하므로 optional로 분리한다.
- 일반 unit test는 fake vision output을 사용한다.

### 6.2 `VisualAdapter`

역할:

- DINOv3 feature를 LLM embedding dimension으로 변환한다.
- patch token 수를 고정된 visual token 수로 압축한다.

공통 인터페이스:

```text
forward(patch_tokens, cls_token) -> visual_tokens
```

입출력:

```text
patch_tokens: [B, P, vision_dim]
cls_token: [B, vision_dim]
visual_tokens: [B, V, llm_dim]
```

여기서:

- `P`: DINOv3 patch 개수
- `V`: `visual_token_count`, 초기값 32
- `vision_dim`: DINOv3 hidden size
- `llm_dim`: LLM embedding dimension

### 6.3 `MlpVisualAdapter`

1차 baseline이다.

처리 흐름:

```text
patch_tokens
  -> sequence pooling 또는 adaptive token pooling
  -> LayerNorm
  -> Linear(vision_dim -> hidden_dim)
  -> GELU
  -> Linear(hidden_dim -> llm_dim)
  -> visual_tokens
```

압축 방식:

- patch token sequence를 `visual_token_count` 구간으로 나눈다.
- 각 구간을 mean pooling한다.
- pooling 결과를 MLP로 변환한다.

선택 이유:

- Q-Former보다 구현과 디버깅이 쉽다.
- shape 오류를 빠르게 잡을 수 있다.
- tiny dataset overfit 성공 여부를 빠르게 확인할 수 있다.

### 6.4 `QFormerVisualAdapter`

2차 실험 모듈이다.

처리 흐름:

```text
learnable query tokens: [V, qformer_dim]
  -> cross attention to DINOv3 patch tokens
  -> query outputs
  -> Linear(qformer_dim -> llm_dim)
```

초기 Design에는 인터페이스와 테스트 계획만 포함하고, Do 1차에서는 MLP baseline을 우선한다.

### 6.5 `LlmBackbone`

역할:

- tokenizer 로딩
- causal LM 로딩
- input embedding 조회
- freeze 또는 LoRA 적용 준비

초기 정책:

- LLM은 `AutoModelForCausalLM` 기반으로 로딩한다.
- `get_input_embeddings()`를 사용해 text embedding을 얻는다.
- Stage 1에서는 `freeze_llm=true`를 기본값으로 둔다.
- Stage 2에서 LoRA를 켠다.

### 6.6 `MiniVlmForConditionalGeneration`

전체 모델 wrapper다.

Forward 흐름:

```text
pixel_values
  -> vision_encoder
  -> visual_adapter
input_ids
  -> llm input embedding
concat visual_tokens + text_embeddings
  -> llm(inputs_embeds=..., attention_mask=..., labels=...)
  -> loss/logits
```

label 확장:

```text
visual_labels = [-100] * visual_token_count
combined_labels = concat(visual_labels, text_labels)
```

attention mask 확장:

```text
visual_attention = [1] * visual_token_count
combined_attention = concat(visual_attention, text_attention_mask)
```

---

## 7. Tensor Shape 상세

| 단계 | Tensor | Shape |
|------|--------|-------|
| 이미지 입력 | `pixel_values` | `[B, 3, H, W]` |
| DINOv3 output | `last_hidden_state` | `[B, 1 + P, vision_dim]` |
| CLS 분리 | `cls_token` | `[B, vision_dim]` |
| Patch 분리 | `patch_tokens` | `[B, P, vision_dim]` |
| Adapter 출력 | `visual_tokens` | `[B, V, llm_dim]` |
| Text token | `input_ids` | `[B, T]` |
| Text embedding | `text_embeddings` | `[B, T, llm_dim]` |
| Combined embedding | `inputs_embeds` | `[B, V + T, llm_dim]` |
| Combined labels | `labels` | `[B, V + T]` |
| LLM logits | `logits` | `[B, V + T, vocab_size]` |

Do 단계의 모든 주요 모듈 테스트는 이 shape를 기준으로 작성한다.

---

## 8. 학습 설계

### 8.1 Stage 0: Shape Smoke Test

목적:

- 외부 모델을 다운로드하기 전에도 shape 계약을 검증한다.

방법:

- fake vision encoder
- fake LLM 또는 tiny random causal LM
- 임의 tensor batch

성공 기준:

- forward pass가 loss를 반환한다.
- `visual_tokens` shape가 `[B, V, llm_dim]`이다.
- combined attention/labels 길이가 `V + T`이다.

### 8.2 Stage 1: Caption Alignment

목적:

- DINOv3 feature가 LLM의 soft prompt처럼 동작하도록 adapter를 학습한다.

Freeze 정책:

| 컴포넌트 | 학습 여부 |
|----------|-----------|
| DINOv3 | freeze |
| MLP adapter | train |
| LLM | freeze |

Loss:

- causal LM cross entropy
- 질문/prompt token은 `-100`
- 답변 token만 loss 계산

초기 overfit 기준:

- 10개 이하 tiny dataset에서 loss가 명확히 감소한다.
- 같은 이미지와 질문에 대해 학습 답변과 비슷한 문자열이 나온다.

### 8.3 Stage 1B: LLM LoRA 보강

Stage 1 adapter-only가 수렴하지 않으면 LLM LoRA를 켠다.

정책:

- DINOv3 freeze 유지
- adapter train
- LLM linear layer LoRA train

### 8.4 Stage 2: Visual Instruction Tuning

목적:

- caption이 아니라 질문에 답하게 만든다.

데이터:

- VQA
- UI screenshot QA
- 간단한 OCR-like 질문

주의:

- DINOv3는 OCR 특화 모델이 아니므로, OCR 성능을 초기 성공 기준으로 삼지 않는다.
- 위치/객체 질문은 patch token 기반 adapter로 먼저 검증한다.

---

## 9. 추론 설계

### 9.1 CLI

명령 예시:

```bash
python -m mini_vlm.inference.infer_cli \
  --config configs/dinov3-mini-vlm.json \
  --checkpoint artifacts/dinov3-mini-vlm/mlp-baseline \
  --image data/samples/images/example.jpg \
  --question "Describe this image."
```

출력 예시:

```text
질문: Describe this image.
답변: A dog is sitting on the grass.
모델: dinov3-mini-vlm-mlp-baseline
```

### 9.2 Generation 방식

1차 구현:

- `inputs_embeds`를 LLM에 직접 전달한다.
- Hugging Face `generate(inputs_embeds=...)`가 사용하는 모델에서 동작하는지 확인한다.

fallback 구현:

- 수동 greedy decoding loop를 구현한다.
- 첫 token 이후부터는 generated token embedding을 이어 붙인다.

이 fallback은 모델별 `generate` 제약을 피하기 위한 안전장치다.

---

## 10. Checkpoint 설계

### 10.1 저장 대상

```text
artifacts/dinov3-mini-vlm/mlp-baseline/
  config.json
  visual_adapter.pt
  tokenizer/
  lora_adapter/        # Stage 1B 이후
  training_state.json
  metrics.jsonl
```

### 10.2 저장하지 않는 대상

- DINOv3 원본 weight
- LLM 원본 weight

대신 `config.json`에 `vision_model_id`, `llm_model_id`를 저장한다.

---

## 11. 테스트 설계

| 테스트 | 목적 | 외부 다운로드 필요 여부 |
|--------|------|------------------------|
| `test_dataset.py` | JSONL 파싱, 이미지 경로 검증 | 없음 |
| `test_collator.py` | token/label/padding/masking 검증 | 없음 또는 fake tokenizer |
| `test_visual_adapter.py` | MLP adapter shape 검증 | 없음 |
| `test_forward.py` | MiniVLM forward loss 검증 | 없음, fake modules |
| `test_generation.py` | greedy decoding smoke test | 없음, fake modules |
| `test_dinov3_integration.py` | 실제 DINOv3 feature 추출 | 있음, optional |
| `test_llm_integration.py` | 실제 LLM embedding 연결 | 있음, optional |

기본 단위 테스트는 네트워크와 모델 다운로드 없이 통과해야 한다. 실제 DINOv3/LLM integration test는 환경변수로 켠다.

```bash
RUN_MODEL_INTEGRATION_TESTS=1 python -m unittest discover -s tests -v
```

---

## 12. 구현 순서

1. 이전 Web/API 코드 정리
2. `pyproject.toml`을 mini VLM 의존성 기준으로 수정
3. `configs/dinov3-mini-vlm.json` 작성
4. `mini_vlm/config.py` 작성
5. `mini_vlm/data/dataset.py` 작성
6. `mini_vlm/data/collator.py` 작성
7. `mini_vlm/models/visual_adapter.py` 작성
8. `mini_vlm/models/vision_encoder.py` 작성
9. `mini_vlm/models/mini_vlm.py` 작성
10. fake module 기반 forward 테스트 작성
11. `train_alignment.py` 작성
12. tiny dataset overfit 실험
13. `infer_cli.py` 작성
14. Q-Former/Resampler 설계 확장
15. Stage 2 instruction tuning 작성

---

## 13. 의존성 설계

### 13.1 필수 의존성

| 패키지 | 용도 |
|--------|------|
| `torch` | tensor, 모델, 학습 |
| `transformers` | DINOv3, LLM, tokenizer |
| `pillow` | 이미지 로딩 |

### 13.2 선택 의존성

| 패키지 | 용도 |
|--------|------|
| `peft` | LLM LoRA |
| `accelerate` | GPU 학습 보조 |
| `safetensors` | checkpoint 저장 |

첫 Do에서는 optional dependency를 분리해, 순수 shape/unit test는 무거운 모델 다운로드 없이 실행되게 한다.

---

## 14. 검증 기준

### 14.1 Design 대비 Do 완료 기준

- [ ] 계획한 프로젝트 구조가 생성된다.
- [ ] Dataset과 Collator 테스트가 통과한다.
- [ ] MLP adapter shape 테스트가 통과한다.
- [ ] fake module 기반 MiniVLM forward 테스트가 통과한다.
- [ ] 실제 DINOv3 feature extraction optional test가 통과한다.
- [ ] tiny dataset overfit 실험 로그가 남는다.
- [ ] inference CLI가 답변을 출력한다.

### 14.2 실험 기록 기준

각 실험은 다음 항목을 기록한다.

| 항목 | 설명 |
|------|------|
| 실험명 | config의 `experiment_name` |
| 목적 | 무엇을 검증했는지 |
| 모델 | DINOv3 id, LLM id |
| 학습 대상 | adapter only, LoRA, partial vision 등 |
| 데이터 | train/validation sample 수 |
| 결과 | loss, 예시 답변 |
| 문제 | 실패 원인 |
| 다음 조치 | 다음 실험 방향 |

---

## 15. 리스크와 대응

| 리스크 | 원인 | 대응 |
|--------|------|------|
| adapter-only 학습이 의미 있는 답변을 만들지 못함 | frozen LLM이 visual soft prompt를 충분히 해석하지 못함 | Stage 1B에서 LLM LoRA 적용 |
| DINOv3 patch 수가 커서 메모리 사용량 증가 | 고해상도 이미지와 많은 patch token | image size 제한, visual token pooling |
| OCR 질문에 약함 | DINOv3가 OCR 특화 backbone이 아님 | OCR은 별도 task로 분리하고 초기 성공 기준에서 제외 |
| `generate(inputs_embeds=...)`가 모델별로 불안정 | HF model 구현 차이 | 수동 greedy decoding fallback |
| 테스트가 모델 다운로드에 의존 | CI/로컬 환경 차이 | fake module unit test와 optional integration test 분리 |

---

## 16. 참고 자료

| 자료 | 링크 | 사용 이유 |
|------|------|----------|
| DINOv3 Hugging Face 문서 | https://huggingface.co/docs/transformers/model_doc/dinov3 | DINOv3 feature extraction과 모델 클래스 확인 |
| Meta DINOv3 소개 | https://ai.meta.com/dinov3/ | DINOv3가 범용 vision backbone임을 확인 |
| Reddit VLM from scratch 글 | https://www.reddit.com/r/learnmachinelearning/comments/1qxksbw/how_to_write_vision_language_models_from_scratch/ | Reddit식 2-stage VLM 접근 참고 |
| `avbiswas/vlm` | https://github.com/avbiswas/vlm | Q-Former + small LLM 구조 참고 |
| Qwen3-VL 공식 repo | https://github.com/QwenLM/Qwen3-VL | VLM 학습/서빙과 visual-language 모델 구조 참고 |

---

## 17. 다음 단계

다음 PDCA 단계는 Do다.

```text
$pdca do dinov3-mini-vlm
```

Do 단계 첫 작업:

1. 이전 Web/API 코드 제거
2. mini VLM 프로젝트 디렉터리 생성
3. config, dataset, collator부터 구현
4. fake tensor 기반 adapter/forward 테스트 작성
