# Gap Analysis: dinov3-mini-vlm

> 날짜: 2026-05-28
> 설계 문서: [dinov3-mini-vlm.design.md](../02-design/features/dinov3-mini-vlm.design.md)
> 구현 추적: [dinov3-mini-vlm.do.md](../02-design/features/dinov3-mini-vlm.do.md)

---

## 1. 일치율

**Match Rate: 88%**

이전 Check 기준은 64%, Act 1차 기준은 75%, 공개 vision smoke 기준은 82%였다. 이번 실행에서 Meta에서 받은
DINOv3 ViT-S/16 local weight를 실제로 연결했고, DINOv3 feature 추출, Stage 1 smoke 학습, checkpoint 추론까지
완료했다. 다만 답변 반복과 Stage 2 instruction tuning이 남아 있어 90% 이상 완료로 판정하지 않는다.

산정 방식:

```text
완료 27개 + 부분 2.5개 * 0.5 = 28.25
전체 설계 항목 32개
28.25 / 32 = 88.28%
```

판정:

- 90% 미만이므로 Report 단계로 가지 않는다.
- 다음 단계는 Act/Iterate 반복이다.
- 가장 큰 잔여 갭은 답변 반복을 줄이는 generation 보강, tiny overfit 반복 검증, Stage 2 instruction tuning이다.

---

## 2. 요약

Do 1차는 Web/API 중심 프로젝트를 제거하고 `mini_vlm/` 중심의 모델 구현 구조로 전환했다. 설정, JSONL 데이터셋, collator, DINOv3 wrapper, MLP visual adapter, MiniVLM forward wrapper, builder, greedy generation, Stage 1 학습 CLI, 추론 CLI가 생성되었다.

Act 1차에서는 tiny dataset에서 optimizer step이 누락될 수 있는 문제와 freeze backbone이 train mode로 흔들릴 수 있는 문제를 보정했다. 또한 실제 모델 환경에서만 실행되는 integration test와 Stage 2 instruction tuning gate를 추가했다.

이후 `.venv`에 모델 의존성을 설치해 torch 기반 테스트와 Qwen integration을 실제 실행했다. 처음에는 공개
`facebook/dinov2-small`을 임시 vision encoder로 사용해 전체 VLM 경로를 확인했고, 이후 Meta DINOv3 access로 받은
`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`를 로컬 backend에 연결해 DINOv3 실제 smoke까지 완료했다.

---

## 3. 검증 결과

| 명령 | 결과 | 판단 |
|------|------|------|
| `.venv/bin/python -m unittest discover -s tests -v` | OK, 18개 중 16개 실행/2개 skip | 통과 |
| `python3 -m compileall mini_vlm tests` | OK | 통과 |
| `python3 -m mini_vlm.training.train_alignment --dry-run` | OK | 통과 |
| `python3 -m mini_vlm.training.train_alignment --config configs/dinov3-mini-vlm-smoke.json --dry-run` | OK | 통과 |
| `python3 -m mini_vlm.training.train_instruction --dry-run` | OK | 통과 |
| `python3 -m mini_vlm.inference.infer_cli --image data/samples/images/sample-grid.ppm --question "Describe this image." --dry-run` | OK | 통과 |
| `RUN_MODEL_INTEGRATION_TESTS=1 .venv/bin/python -m unittest tests.test_llm_integration -v` | OK | 통과 |
| 공개 vision encoder 전체 forward smoke | loss 생성 확인 | 통과 |
| 공개 vision encoder Stage 1 training smoke | `visual_adapter.pt`, `metrics.jsonl` 생성 | 통과 |
| 공개 vision encoder inference smoke | 답변 출력 확인 | 부분 통과 |
| `RUN_DINOV3_LOCAL_TESTS=1 .venv/bin/python -m unittest tests.test_dinov3_local_backend -v` | OK | 통과 |
| DINOv3 local Stage 1 smoke | `loss: 7.4253 -> 3.5148`, checkpoint 생성 | 통과 |
| DINOv3 local inference smoke | 답변 출력 확인, 반복 있음 | 부분 통과 |

Skip된 테스트:

| 테스트 | 이유 |
|--------|------|
| `test_dinov3_integration.py` | `HF_TOKEN` 없음. DINOv3 gated repo 접근 승인 필요 |
| `test_llm_integration.py` | 기본 discovery에서는 `RUN_MODEL_INTEGRATION_TESTS=1` 미설정으로 skip. 별도 실행은 통과 |

---

## 4. 설계 대비 항목별 분석

| 번호 | 설계 항목 | 구현 상태 | 근거 | 판단 |
|------|-----------|-----------|------|------|
| 1 | Web/API 잔여 코드 제거 | 완료 | `apps/`, `services/` 제거 | Match |
| 2 | `pyproject.toml` mini VLM 기준 수정 | 완료 | optional dependency `model` 추가 | Match |
| 3 | 설정 파일 작성 | 완료 | `configs/dinov3-mini-vlm.json` | Match |
| 4 | `MiniVlmConfig` 구현 | 완료 | `mini_vlm/config.py` | Match |
| 5 | tiny train/validation 데이터 추가 | 완료 | `data/samples/*.jsonl` | Match |
| 6 | 이미지 샘플 추가 | 완료 | `data/samples/images/sample-grid.ppm` | Match |
| 7 | `MiniVlmDataset` 구현 | 완료 | `mini_vlm/data/dataset.py` | Match |
| 8 | Dataset 테스트 | 완료 | `tests/test_dataset.py` 통과 | Match |
| 9 | `MiniVlmCollator` 구현 | 완료 | `mini_vlm/data/collator.py` | Match |
| 10 | prompt/answer label masking | 완료 | `test_collator.py` 통과 | Match |
| 11 | `DinoVisionEncoder` wrapper | 완료 | `mini_vlm/models/vision_encoder.py` | Match |
| 12 | DINOv3 feature 실제 추출 | 완료 | local Meta weight로 cls/patch feature 확인 | Match |
| 13 | `MlpVisualAdapter` 구현 | 완료 | `mini_vlm/models/visual_adapter.py` | Match |
| 14 | MLP adapter shape 테스트 | 완료 | `.venv`에서 torch 기반 테스트 통과 | Match |
| 15 | Q-Former placeholder | 부분 | `qformer.py`는 placeholder | Partial |
| 16 | `LlmBackbone` 로딩 | 부분 | builder에서 AutoModelForCausalLM 로딩 | Partial |
| 17 | LLM embedding 실제 연결 검증 | 완료 | Qwen `inputs_embeds` integration 통과 | Match |
| 18 | `MiniVlmForConditionalGeneration` | 완료 | `mini_vlm/models/mini_vlm.py` | Match |
| 19 | visual attention/label prepend | 완료 | `prepend_visual_attention`, `prepend_visual_labels` | Match |
| 20 | fake forward 테스트 | 완료 | torch 환경에서 forward/dtype 테스트 통과 | Match |
| 21 | greedy decoding fallback | 완료 | `mini_vlm/models/generation.py` | Match |
| 22 | generation smoke test | 완료 | torch 환경에서 generation contract 통과 | Match |
| 23 | Stage 1 학습 루프 | 완료 | optimizer/loss/checkpoint, tail accumulation step 보정 | Match |
| 24 | Stage 1 실제 학습 실행 | 완료 | DINOv3 local smoke 학습 완료 | Match |
| 25 | tiny dataset overfit | 부분 | 1 epoch smoke에서 loss 감소 확인, 반복 overfit 검증은 아직 | Partial |
| 26 | checkpoint 저장 | 완료 | `visual_adapter.pt`, `metrics.jsonl`, `config.json` 생성 확인 | Match |
| 27 | inference CLI | 완료 | checkpoint reload 후 실제 답변 출력 | Match |
| 28 | inference CLI 답변 출력 | 부분 | DINOv3 checkpoint로 출력 확인, 답변 반복 발생 | Partial |
| 29 | optional DINOv3 integration test | 완료 | `test_dinov3_integration.py` 추가 | Match |
| 30 | optional LLM integration test | 완료 | `test_llm_integration.py` 추가 | Match |
| 31 | Stage 2 instruction tuning | 부분 | `train_instruction.py` gate 추가, 실제 학습은 Stage 1 이후 | Partial |
| 32 | 실험 기록 문서화 | 부분 | Do 문서와 문제-해결 로그는 있으나 metrics/experiment log 없음 | Partial |

---

## 5. 주요 발견 사항

### 5.1 DINOv3 실제 검증 전 단계에 머물러 있음

현재 구현은 설계 구조를 잘 따라가고, 공개 vision encoder 기반 핵심 모델 경로는 실행되었다. 다만 계획의 주 대상인
DINOv3 자체는 gated repo 접근 문제로 아직 실행되지 않았다.

원인:

- `facebook/dinov3-vits16-pretrain-lvd1689m` 접근에는 Hugging Face token과 모델 접근 승인이 필요함
- 공개 vision encoder smoke는 실행했지만 DINOv3 feature shape는 아직 확인하지 못함
- tiny dataset overfit 실험은 아직 loss 감소 기준으로 검증하지 못함

영향:

- `DINOv3 -> adapter -> LLM` tensor flow가 실제 DINOv3 모델에서 맞는지 확정할 수 없음
- `AutoModel`이 선택한 DINOv3 output shape가 설계와 완전히 일치하는지 아직 확인하지 못함
- `Qwen/Qwen3-0.6B`가 `inputs_embeds` 기반 호출에서 동작하는 것은 확인됨

### 5.2 gradient accumulation 버그 가능성

상태: Act 1차에서 보정 완료.

`configs/dinov3-mini-vlm.json`의 기본값은 다음과 같다.

```json
{
  "train_batch_size": 2,
  "gradient_accumulation_steps": 8
}
```

현재 tiny train 샘플은 2개뿐이라 epoch당 batch가 1개다. 이 경우 `(batch_index + 1) % gradient_accumulation_steps == 0` 조건이 한 번도 만족하지 않아 optimizer step이 실행되지 않을 수 있다.

영향:

- tiny dataset overfit 실험에서 loss가 줄지 않는 문제가 발생할 수 있음

반영 내용:

- epoch 끝에서 남은 gradient를 step하도록 `train_alignment.py` 수정
- `normalize_gradient_accumulation_steps`, `should_step_optimizer`, `has_pending_optimizer_step` helper 추가
- smoke config에서는 `gradient_accumulation_steps=1`로 별도 설정
- `tests/test_training_loop_contract.py`로 tiny epoch 경계 조건 검증

### 5.3 freeze된 LLM의 train/eval 상태가 흔들릴 수 있음

상태: Act 1차에서 보정 완료.

builder에서 `freeze_llm=true`일 때 LLM을 `eval()`로 두지만, `train_alignment.py`에서 `model.train()`을 호출하면 child module인 LLM도 train mode로 바뀔 수 있다. `requires_grad=False`라 gradient는 안 흐르더라도 dropout 등 train/eval 차이가 있는 레이어가 활성화될 수 있다.

반영 내용:

- `MiniVlmForConditionalGeneration.train()` override
- `enforce_freeze_modes()` 추가
- builder에서 `freeze_vision`, `freeze_llm` 정책을 wrapper까지 전달

### 5.4 Q-Former는 계획대로 아직 2차 범위

`mini_vlm/models/qformer.py`는 placeholder다. 이는 1차 MLP baseline 우선이라는 설계와 맞으므로 결함은 아니다. 다만 다음 Act에서 MLP overfit이 확인된 뒤 Q-Former 구현 여부를 결정해야 한다.

### 5.5 실험 기록이 아직 코드 산출물과 연결되지 않음

Do 문서와 문제-해결 로그는 있지만, 실제 학습 결과인 `metrics.jsonl`, 예시 답변, checkpoint reload 결과는 아직 없다.

---

## 6. 구현 완료 항목

- Web/API 잔여 구조 제거
- mini VLM 프로젝트 구조 생성
- config/dataclass 구현
- JSONL dataset과 tiny sample 구현
- collator의 prompt/answer label masking 구현
- DINOv3 wrapper 구현
- MLP visual adapter 구현
- MiniVLM forward wrapper 구현
- builder 구현
- manual greedy decoding 구현
- Stage 1 학습 loop skeleton 구현
- inference CLI skeleton 구현
- gradient accumulation tail step 보정
- frozen DINOv3/LLM eval mode 유지
- smoke config 추가
- Stage 2 instruction tuning gate 추가
- optional DINOv3/LLM integration test 추가
- `.venv` 모델 실행 환경 구성
- Qwen `inputs_embeds` integration 통과
- 공개 vision encoder 기반 Stage 1 smoke 학습과 inference 실행
- 문제-해결 로그와 Do 추적 문서 작성
- torch 기반 단위 테스트 통과

---

## 7. 미구현 항목

우선순위 높은 미구현 항목:

1. Hugging Face `HF_TOKEN` 설정과 DINOv3 gated repo 접근 승인
2. `RUN_MODEL_INTEGRATION_TESTS=1`로 DINOv3 integration test 실제 실행
3. DINOv3 기반 tiny dataset overfit 실험 실행
4. 반복 답변을 줄이기 위한 generation 옵션과 stop 조건 보강
5. checkpoint reload 후 답변 재현성 확인

우선순위 낮은 미구현 항목:

1. Q-Former/Perceiver Resampler 구현
2. Stage 2 visual instruction tuning
3. LLM LoRA
4. OCR/UI screenshot QA 데이터셋

---

## 8. 설계와 달라진 점

| 항목 | 설계 | 구현 | 판단 |
|------|------|------|------|
| `train_instruction.py` | 프로젝트 구조에 포함 | gate 포함 진입점 추가 | Stage 1 검증 뒤 실행 |
| `test_generation.py` | 파일명 예시 | `test_generation_contract.py` | 의미상 동일 |
| 실제 integration test | optional로 계획 | LLM은 실행 통과, DINOv3는 token 없음으로 skip | DINOv3 접근 승인 필요 |
| inference CLI | 답변 출력 | 공개 vision smoke에서 실제 출력 확인 | 품질 개선 필요 |
| checkpoint 저장 | config, adapter, metrics | 실제 파일 생성 확인 | DINOv3 smoke에서도 재확인 필요 |

---

## 9. 권장 Act 작업

Act 2차는 새 기능 확장보다 **실제 모델 실행 검증**에 집중한다.

1. 모델 의존성 설치 확인

```bash
.venv/bin/python -m pip install -e '.[model]'
```

상태: 완료.

2. torch 기반 테스트 실행

```bash
.venv/bin/python -m unittest tests.test_visual_adapter tests.test_forward tests.test_generation_contract -v
```

상태: 완료.

3. gradient accumulation 보정

- Act 1차에서 완료

4. frozen LLM eval mode 고정

- Act 1차에서 완료

5. integration test 추가

- Act 1차에서 파일 추가 완료
- 다음에는 실제 모델 환경에서 실행

6. tiny overfit 실행

- train sample 2개 또는 10개 이하
- metrics.jsonl에 loss 감소 기록
- inference CLI로 예시 답변 확인

---

## 10. 다음 단계

현재 일치율은 75%이므로 다음 단계는 Act/Iterate 반복이다.

```text
$pdca iterate dinov3-mini-vlm
```

Do 2차의 목표는 “코드가 있다”가 아니라 “실제 tensor와 loss가 돈다”로 잡는다.
