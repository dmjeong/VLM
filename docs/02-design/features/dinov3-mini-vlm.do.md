# DINOv3 Mini VLM 구현 추적 문서

> 버전: 0.1.1 | 날짜: 2026-05-28 | 상태: Act 1차 반영
> 기능명: `dinov3-mini-vlm`
> 계획 문서: [dinov3-mini-vlm.plan.md](../../01-plan/features/dinov3-mini-vlm.plan.md)
> 설계 문서: [dinov3-mini-vlm.design.md](./dinov3-mini-vlm.design.md)

---

## 1. 이번 Do 범위

이번 Do는 모델 프로젝트로 방향을 고정하고, Web/API 잔여 코드를 제거한 뒤 DINOv3 Mini VLM의 1차 skeleton을 구현한다.

포함 범위:

| 영역 | 구현 내용 |
|------|-----------|
| 프로젝트 정리 | `apps/`, `services/` 기반 Web/API 코드 제거 |
| 설정 | `configs/dinov3-mini-vlm.json`, `MiniVlmConfig` |
| 데이터 | JSONL 샘플, `MiniVlmDataset`, `MiniVlmCollator` |
| 모델 | `DinoVisionEncoder`, `MlpVisualAdapter`, `MiniVlmForConditionalGeneration` |
| 학습 | Stage 1 alignment dry-run CLI |
| 추론 | inference dry-run CLI |
| 테스트 | config, dataset, collator, visual adapter, forward 테스트 |

제외 범위:

- 실제 DINOv3 모델 다운로드
- 실제 LLM 다운로드
- 실제 GPU 학습
- Q-Former 구현
- Web/API 재도입

---

## 2. 구현 파일

| 파일 | 역할 |
|------|------|
| `configs/dinov3-mini-vlm.json` | 실험 설정 |
| `data/samples/train.jsonl` | tiny train 샘플 |
| `data/samples/validation.jsonl` | tiny validation 샘플 |
| `mini_vlm/config.py` | 설정 dataclass와 loader |
| `mini_vlm/data/dataset.py` | JSONL 데이터셋 |
| `mini_vlm/data/collator.py` | prompt/answer token, label masking, image path 처리 |
| `mini_vlm/models/vision_encoder.py` | DINOv3 wrapper |
| `mini_vlm/models/visual_adapter.py` | MLP visual adapter baseline |
| `mini_vlm/models/mini_vlm.py` | visual token + text embedding 결합 wrapper |
| `mini_vlm/training/train_alignment.py` | Stage 1 학습 dry-run 진입점 |
| `mini_vlm/inference/infer_cli.py` | 추론 dry-run 진입점 |
| `tests/test_config.py` | 설정 로드 테스트 |
| `tests/test_dataset.py` | JSONL/이미지 경로 테스트 |
| `tests/test_collator.py` | label masking 테스트 |
| `tests/test_visual_adapter.py` | torch 설치 시 adapter shape 테스트 |
| `tests/test_forward.py` | torch 설치 시 MiniVLM forward 테스트 |

---

## 3. 구현 의도

### 3.1 Web/API 제거

이전 v2는 실제 모델 endpoint 연결을 목표로 했지만, 현재 목표는 모델 구조를 직접 만드는 것이다. 따라서 Web/API 코드는 삭제했다.

### 3.2 MLP baseline 우선

Q-Former는 더 VLM다운 구조지만, 첫 구현에서는 shape와 loss 계약을 빠르게 확인해야 한다. 그래서 patch token을 고정 개수 visual token으로 mean pooling하고 MLP로 LLM dimension에 맞추는 baseline을 먼저 구현했다.

### 3.3 torch/transformers optional

현재 로컬 환경에는 `torch`, `transformers`가 없다. 기본 테스트는 설치 없이 통과해야 하므로 config/dataset/collator 테스트는 순수 Python으로 구성했다. 모델 shape 테스트는 torch가 설치된 환경에서만 실행된다.

---

## 4. 현재 제한

| 제한 | 이유 | 다음 조치 |
|------|------|-----------|
| 실제 forward 테스트는 skip | torch 미설치 | `pip install '.[model]'` 후 재실행 |
| DINOv3 integration 미검증 | transformers와 모델 다운로드 필요 | optional integration test 추가 |
| 학습 루프는 dry-run | GPU/torch 미설치 | Stage 1 training loop 구현 |
| Q-Former 미구현 | baseline 우선 | MLP overfit 후 구현 |

---

## 5. 검증 명령

| 명령 | 결과 |
|------|------|
| `python3 -m unittest discover -s tests -v` | 통과, 11개 중 8개 실행/3개 torch 미설치로 skip |
| `python3 -m compileall mini_vlm tests` | 통과 |
| `python3 -m mini_vlm.training.train_alignment --dry-run` | 통과 |
| `python3 -m mini_vlm.inference.infer_cli --image data/samples/images/sample-grid.ppm --question "Describe this image." --dry-run` | 통과 |

---

## 6. 다음 구현 후보

1. torch 설치 환경에서 adapter/forward 테스트 실제 통과 확인
2. fake module 기반 테스트를 torch 설치 환경에서 더 촘촘히 작성
3. Stage 1 학습 루프 구현
4. tiny dataset overfit 실험
5. DINOv3 integration test 추가
6. 실제 LLM embedding 연결

---

## 7. 추가 구현 기록

| 항목 | 내용 |
|------|------|
| 실제 builder | `mini_vlm/models/builder.py`에서 DINOv3, LLM, MLP adapter를 설정 기반으로 생성 |
| greedy decoding | `mini_vlm/models/generation.py`에 `inputs_embeds` 기반 수동 greedy decoding 추가 |
| 학습 루프 | `train_alignment.py`에 adapter 학습 loop, metrics 기록, adapter checkpoint 저장 경로 추가 |
| 추론 루프 | `infer_cli.py`에 checkpoint adapter 로드와 이미지+질문 추론 경로 추가 |
| dry-run 보정 | torch 미설치 환경에서도 dry-run이 import error 없이 실행되도록 실제 모델 import를 지연 |

---

## 8. Act 1차 반영 기록

Check 단계에서 확인된 갭 중, 실제 모델 의존성 없이 고칠 수 있는 항목을 먼저 반영했다. 이번 Act의 의도는
“모델을 만들었다”가 아니라 “실제 학습으로 들어갔을 때 조용히 실패할 가능성을 줄이는 것”이다.

| 항목 | 변경 내용 | 의도 |
|------|-----------|------|
| gradient accumulation 보정 | `train_alignment.py`에 epoch 끝 잔여 gradient step 처리 추가 | tiny dataset에서 batch 수가 accumulation step보다 작아도 optimizer가 반드시 한 번은 step하도록 함 |
| freeze mode 고정 | `MiniVlmForConditionalGeneration.train()` override와 `enforce_freeze_modes()` 추가 | `model.train()` 호출 후에도 freeze된 DINOv3/LLM은 eval mode를 유지 |
| 학습 가능 파라미터 검증 | optimizer 생성 전 trainable parameter가 비어 있으면 명시적 오류 발생 | freeze 설정 실수로 학습이 진행되지 않는 상황을 즉시 드러냄 |
| smoke config 추가 | `configs/dinov3-mini-vlm-smoke.json` 추가 | 실제 모델 설치 후 1 epoch smoke/overfit 실험을 더 작게 실행하기 위한 설정 |
| Stage 2 진입점 추가 | `mini_vlm/training/train_instruction.py` 추가 | instruction tuning은 Stage 1 검증 이후 실행되도록 gate를 둠 |
| integration test 추가 | `test_dinov3_integration.py`, `test_llm_integration.py` 추가 | `RUN_MODEL_INTEGRATION_TESTS=1`일 때 실제 DINOv3/LLM 연결을 검증 |
| 계약 테스트 추가 | `test_training_loop_contract.py` 추가 | torch 없이도 accumulation 경계 조건과 Stage 2 gate를 검증 |

### 8.1 이번 Act에서 추가한 코드 주석 의도

- `MiniVlmForConditionalGeneration.train()` 주석은 왜 freeze module을 eval로 되돌리는지 설명한다.
- `train_alignment.py`의 accumulation helper 주석은 tiny dataset에서 optimizer step이 누락되는 실패 모드를 기록한다.
- `train_instruction.py` 주석은 Stage 2를 바로 실행하지 않는 이유와 Stage 1 gate 기준을 남긴다.

### 8.2 Act 1 검증 결과

| 명령 | 결과 |
|------|------|
| `python3 -m unittest discover -s tests -v` | 통과, 17개 중 12개 실행/5개 skip |
| `python3 -m compileall mini_vlm tests` | 통과 |
| `python3 -m mini_vlm.training.train_alignment --dry-run` | 통과 |
| `python3 -m mini_vlm.training.train_alignment --config configs/dinov3-mini-vlm-smoke.json --dry-run` | 통과 |
| `python3 -m mini_vlm.training.train_instruction --dry-run` | 통과 |
| `python3 -m mini_vlm.inference.infer_cli --image data/samples/images/sample-grid.ppm --question "Describe this image." --dry-run` | 통과 |

### 8.3 남은 제한

Act 1 작성 시점에는 로컬 Python 3.14 환경에 `torch`, `transformers`가 없었다. 이후 9장에서 `.venv` 모델 실행
환경을 구성했고, Qwen forward와 공개 vision encoder smoke는 실제 실행했다. DINOv3 자체는 gated repo 접근 문제로
아직 남아 있다.

---

## 9. 모델 실행 환경 및 Smoke 실행 기록

날짜: 2026-05-28

### 9.1 환경 구성

| 항목 | 결과 |
|------|------|
| 가상환경 | `.venv` 생성 |
| Python | 3.14.0 |
| torch | 2.12.0 |
| torchvision | 0.27.0 |
| transformers | 5.9.0 |
| MPS | 사용 가능 |

설치 중 발견한 문제:

| 문제 | 원인 | 조치 |
|------|------|------|
| `pip install -e '.[model]'` 실패 | `setuptools`가 `data`, `configs`, `mini_vlm`을 모두 top-level package로 오인 | `pyproject.toml`에 `mini_vlm*`만 package로 포함하도록 설정 |
| `AutoImageProcessor` 실패 | `torchvision` 누락 | model optional dependency에 `torchvision` 추가 |
| DINOv3 접근 실패 | `facebook/dinov3-vits16-pretrain-lvd1689m`가 Hugging Face gated repo | `HF_TOKEN`과 모델 접근 승인 필요. 테스트는 token 없으면 skip |
| Qwen forward dtype 실패 | Qwen embedding은 `bfloat16`, adapter 출력은 `float32` | visual token을 LLM embedding dtype으로 캐스팅 |

### 9.2 실제 실행 결과

기본 테스트:

```text
.venv/bin/python -m unittest discover -s tests -v
결과: OK, 18개 중 16개 실행/2개 skip
```

LLM integration:

```text
RUN_MODEL_INTEGRATION_TESTS=1 .venv/bin/python -m unittest tests.test_llm_integration -v
결과: OK
```

DINOv3 integration:

```text
RUN_MODEL_INTEGRATION_TESTS=1 .venv/bin/python -m unittest tests.test_dinov3_integration -v
결과: HF_TOKEN 없음으로 skip
```

공개 vision encoder smoke:

```text
vision_model_id: facebook/dinov2-small
pixel_values: (1, 3, 224, 224)
cls_token: (1, 384)
patch_tokens: (1, 256, 384)
```

전체 VLM forward smoke:

```text
vision_model_id: facebook/dinov2-small
llm_model_id: Qwen/Qwen3-0.6B
visual_tokens: (1, 4, 1024), torch.bfloat16
inputs_embeds: (1, 19, 1024), torch.bfloat16
logits: (1, 19, 151936), torch.bfloat16
loss: 16.403688430786133
```

Stage 1 training smoke:

```text
config: configs/open-vision-qwen-smoke.json
출력: artifacts/dinov3-mini-vlm/open-vision-qwen-smoke
생성 파일: config.json, metrics.jsonl, visual_adapter.pt
metrics:
{"epoch": 0, "step": 1, "optimizer_step": 1, "loss": 12.901323318481445}
{"epoch": 0, "step": 2, "optimizer_step": 2, "loss": 5.451408386230469}
```

Inference smoke:

```text
질문: Describe this image.
답변: Welcome.
Answer: Welcome.
Answer: Welcome.
...
```

판단:

- 실제 `DINO 계열 vision encoder -> MLP adapter -> Qwen inputs_embeds -> loss -> checkpoint -> inference` 경로는 동작했다.
- 다만 DINOv3 자체는 gated repo 접근 문제로 아직 실행하지 못했다.
- 1 epoch tiny smoke의 답변은 반복이 심해 품질 검증으로 볼 수 없고, 실행 경로 검증으로만 해석한다.
- 같은 output dir로 재실행할 때 metrics가 섞이지 않도록 `train_alignment.py`에서 기존 `metrics.jsonl`을 새 실험 시작 전에 삭제한다.

---

## 10. Meta DINOv3 로컬 Weight 실행 기록

날짜: 2026-05-28

사용자가 Meta DINOv3 access 승인 후 제공한 weight 목록에서 가장 작은 ViT-S/16 LVD-1689M backbone을 받았다.
signed download URL은 문서에 남기지 않고, 로컬 파일명만 기록한다.

| 항목 | 값 |
|------|----|
| DINOv3 repo | `external/dinov3` |
| 로컬 weight | `models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| weight 크기 | 약 83MB |
| config | `configs/dinov3-local-vits16-qwen-smoke.json` |
| vision backend | `torchhub` 직접 import 방식 |
| backbone | `dinov3_vits16` |
| LLM | `Qwen/Qwen3-0.6B` |

### 10.1 로딩 방식 변경

`torch.hub.load("external/dinov3", ...)`는 `hubconf.py` 전체를 읽으며 segmentation 부가 의존성까지 요구했다.
우리는 backbone만 필요하므로 `external/dinov3`를 `sys.path`에 추가한 뒤 `dinov3.hub.backbones.dinov3_vits16`을 직접
import하는 방식으로 구현했다.

### 10.2 DINOv3 feature 확인

```text
model: DinoVisionTransformer
embed_dim: 384
cls_token: (1, 384)
patch_tokens: (1, 196, 384)
```

### 10.3 Stage 1 학습 smoke

```text
명령: .venv/bin/python -m mini_vlm.training.train_alignment --config configs/dinov3-local-vits16-qwen-smoke.json
출력: artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-smoke
생성 파일: config.json, metrics.jsonl, visual_adapter.pt
metrics:
{"epoch": 0, "step": 1, "optimizer_step": 1, "loss": 7.425304889678955}
{"epoch": 0, "step": 2, "optimizer_step": 2, "loss": 3.5148324966430664}
```

### 10.4 Inference smoke

```text
질문: Describe this image.
답변: The image is a 3D object that is a cube with a square base ...
```

판단:

- 계획의 핵심 경로인 `DINOv3 -> MLP adapter -> Qwen embedding -> Qwen -> 답변`이 실제 weight로 실행되었다.
- 아직 답변 반복이 있어 품질 검증은 아니다.
- 다음 Act는 generation stop 조건과 tiny overfit 반복 실험을 보강한다.

---

## 11. 학습 진행률 추적 보강

날짜: 2026-05-28

기존 학습 루프는 `metrics.jsonl`에는 batch별 loss를 남겼지만, 콘솔에는 epoch 진행률과 loss 변화가 보이지 않았다.
학습을 직접 실행하는 동안 상태를 판단할 수 있도록 다음 출력을 추가했다.

```text
학습 시작: experiment=..., samples=17, epochs=10, batch_size=1, batches/epoch=17, device=...
[epoch 1/10] 시작
[epoch 1/10 batch 1/17   5.9%] step=1 opt_step=1 loss=... avg=...
[epoch 1/10 batch 2/17  11.8%] step=2 opt_step=2 loss=... avg=...
[epoch 1/10] 완료 avg_loss=... first=... last=... delta=... min=... max=...
```

추가 산출물:

| 파일 | 내용 |
|------|------|
| `metrics.jsonl` | batch 이벤트와 epoch summary 이벤트 |
| `training_summary.json` | epoch별 평균 loss, 첫 loss, 마지막 loss, min/max loss |

---

## 12. Act 2차: 성능 개선 실험 경로 추가

날짜: 2026-05-28

epoch를 늘려도 성능 향상이 크지 않은 문제를 보고, 단순 MLP/frozen LLM baseline에서 한 단계 더 실험할 수 있게
아래 4가지를 구현했다.

| 항목 | 변경 내용 | 의도 |
|------|-----------|------|
| generation 반복 방지 | `max_new_tokens`, `repetition_penalty`, `no_repeat_ngram_size`, `stop_strings`를 config에 추가 | 추론이 `Answer:` 패턴이나 같은 n-gram을 반복하는 현상을 줄임 |
| validation loss | 매 epoch 종료 후 validation dataloader를 돌려 `val_loss`를 콘솔, `metrics.jsonl`, `training_summary.json`에 기록 | train loss만으로 overfit/일반화 상태를 착각하지 않도록 함 |
| Qwen LoRA | `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`를 config에 추가하고 PEFT로 Qwen에 LoRA 적용 | Qwen 전체를 학습하지 않고 attention projection 일부만 저비용으로 적응 |
| Perceiver adapter | `PerceiverResamplerAdapter` 추가 | mean pooling MLP 대신 learnable query가 DINOv3 patch/CLS token을 읽어 visual token을 생성 |

### 12.1 active config 변경

`configs/dinov3-local-vits16-qwen-smoke.json`은 다음 실험으로 전환했다.

| 설정 | 값 |
|------|----|
| `experiment_name` | `dinov3-local-vits16-qwen-lora-perceiver` |
| `adapter_type` | `perceiver` |
| `visual_token_count` | `8` |
| `adapter_hidden_dim` | `256` |
| `use_lora` | `true` |
| `device` | `auto` |
| `lora_target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `repetition_penalty` | `1.15` |
| `no_repeat_ngram_size` | `3` |
| `output_dir` | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-lora-perceiver` |

### 12.2 구현 의도

- `PerceiverResamplerAdapter`는 DINOv3의 모든 patch token을 평균으로 뭉개지 않고, 학습 가능한 visual query가
  cross-attention으로 필요한 patch 정보를 선택하게 한다.
- LoRA는 Qwen 본체 weight를 직접 크게 바꾸지 않고, attention projection 옆에 작은 rank adapter만 추가한다.
  이 프로젝트에서는 `freeze_llm=true`를 유지하되 `use_lora=true`일 때 LoRA parameter만 학습 가능하게 했다.
- inference CLI는 `visual_adapter.pt`뿐 아니라 `llm_lora/` 폴더도 로드한다. 이 경로가 없으면 LoRA 학습 결과가
  추론에 반영되지 않는 문제가 생긴다.
- `device=auto`는 CUDA, MPS, CPU 순서로 학습/추론 device를 선택한다. 현재 Mac 환경에서는 MPS forward/backward가
  통과했으므로 full epoch 실행 시 CPU보다 빠른 경로를 사용할 수 있다.

### 12.3 검증 결과

| 명령 | 결과 |
|------|------|
| `.venv/bin/python -m compileall mini_vlm tests` | 통과 |
| `.venv/bin/python -m unittest discover -s tests -v` | 통과, 29개 중 26개 실행/3개 skip |
| `RUN_DINOV3_LOCAL_TESTS=1 .venv/bin/python -m unittest tests.test_dinov3_local_backend -v` | 통과 |
| LoRA/Perceiver build smoke | `PerceiverResamplerAdapter`, `PeftModelForCausalLM`, trainable parameter `4,274,432` 확인 |
| MPS 1샘플 학습 smoke | device `mps`, train loss `10.8925`, validation loss `6.3908`, `visual_adapter.pt`, `llm_lora/adapter_model.safetensors` 생성 |
| checkpoint inference smoke | `visual_adapter.pt`와 `llm_lora/`를 로드해 MPS 추론 CLI 실행 성공 |

### 12.4 현재 판단

이번 변경은 품질 개선 완료가 아니라, 품질을 개선할 수 있는 학습 경로를 연 것이다. 1샘플 smoke의 답변은 여전히
이미지와 맞지 않으므로 성능 검증으로 해석하면 안 된다. 다음 판단은 full 17샘플/10 epoch 실행 후
`training_summary.json`에서 train loss와 validation loss가 함께 내려가는지 보는 방식으로 진행한다.

---

## 13. NaN 안정화 패치

날짜: 2026-05-28

full 10 epoch 실행에서 epoch 1의 4번째 batch부터 loss가 `NaN`으로 전파되는 문제가 발생했다. 원인은 데이터 파일
파손이 아니라, LoRA/Perceiver 학습 중 일부 batch의 gradient norm이 매우 커졌는데도 gradient clipping과 non-finite
guard가 없어 optimizer step이 그대로 진행된 것이었다.

### 13.1 변경 내용

| 항목 | 변경 내용 | 의도 |
|------|-----------|------|
| learning rate 하향 | `1e-4`에서 `2e-5`로 변경 | tiny dataset + LoRA 학습에서 한 번의 update가 너무 크게 튀지 않게 함 |
| seed 고정 | `seed=42` 추가 | adapter 초기화와 DataLoader shuffle을 재현 가능하게 함 |
| gradient clipping | `max_grad_norm=1.0` 추가 | 큰 gradient norm이 나와도 실제 update 크기를 제한 |
| non-finite guard | loss/gradient/parameter가 NaN 또는 Inf이면 즉시 중단 | 깨진 상태로 다음 batch와 checkpoint를 오염시키지 않음 |
| JSON strict 저장 | `allow_nan=False` 적용 | 결과 파일에 비표준 JSON `NaN`이 조용히 저장되지 않게 함 |

### 13.2 재실행 결과

같은 active config로 10 epoch 재학습을 완료했다.

| 지표 | 값 |
|------|----|
| train sample | `17` |
| validation sample | `6` |
| device | `mps` |
| final train avg loss | `2.7920` |
| final validation loss | `3.1667` |
| metrics finite 검사 | `metrics.jsonl` 190줄 전체 통과 |
| summary finite 검사 | `training_summary.json` 통과 |

판단:

- NaN 문제는 해결되었다.
- validation loss는 epoch 9에서 `2.8078`까지 내려갔다가 epoch 10에서 `3.1667`로 올랐다. 현재 데이터가 매우 작아서
  overfit/진동이 쉽게 발생한다.
- checkpoint 추론은 실행되지만 답변 품질은 아직 좋지 않다. 다음 개선은 NaN 안정화가 아니라 데이터셋 확장,
  prompt/answer 포맷 정리, instruction tuning 분리 쪽으로 봐야 한다.

---

## 14. Wikimedia Commons 1K Bootstrap 데이터셋

날짜: 2026-05-28

기존 데이터셋은 train 17개/validation 6개뿐이라, loss가 내려가도 모델이 실제 시각 정보를 배운다고 보기 어려웠다.
먼저 1000개 규모의 bootstrap JSONL 데이터를 만들기 위해 Wikimedia Commons API 기반 수집 스크립트를 추가했다.

### 14.1 수집 방식

| 항목 | 내용 |
|------|------|
| 수집 API | Wikimedia Commons MediaWiki API `query + imageinfo` |
| metadata | `url`, `size`, `mime`, `extmetadata` |
| 라이선스 필터 | Public domain, CC0, CC BY, CC BY-SA 계열 |
| 이미지 필터 | JPEG/PNG, 256px 이상 |
| 관련성 필터 | 제목/설명/카테고리에 class label token이 포함된 결과만 사용 |
| 제외어 | 사고/무기/스포츠 cup/오탐 가능성이 큰 제목 일부 제외 |
| 라벨 생성 | class label 기반 caption/VQA template |

### 14.2 생성 결과

| 파일/폴더 | 내용 |
|------|------|
| `data/wikimedia_commons_1k/images/` | 다운로드 이미지 51장 |
| `data/wikimedia_commons_1k/all.jsonl` | 전체 샘플 1000개 |
| `data/wikimedia_commons_1k/train.jsonl` | train 샘플 800개, 이미지 40장 |
| `data/wikimedia_commons_1k/validation.jsonl` | validation 샘플 100개, 이미지 5장 |
| `data/wikimedia_commons_1k/test.jsonl` | test 샘플 100개, 이미지 5장 |
| `data/wikimedia_commons_1k/split_manifest.json` | 이미지 단위 split manifest |
| `data/wikimedia_commons_1k/sources.jsonl` | 이미지별 원본 URL, 라이선스, 저작자, 크기 |
| `data/wikimedia_commons_1k/README.md` | 수집 조건과 주의사항 |

검증:

| 명령 | 결과 |
|------|------|
| `MiniVlmDataset(...train...)` | 800개 로드 성공 |
| `MiniVlmDataset(...validation...)` | 100개 로드 성공 |
| `MiniVlmDataset(...test...)` | 100개 로드 성공 |
| image split leakage | train/validation/test 이미지 교집합 0개 |
| config dry-run | `dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1` 통과 |
| 1 batch forward | device `mps`, loss `6.5034`, visual token shape `(1, 16, 1024)` |
| unit test | 31개 중 28개 실행/3개 skip, 통과 |

### 14.3 추가 config

`configs/dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1.json`을 추가했다.

의도:

- LoRA를 바로 켜지 않고 `use_lora=false`로 adapter-only Stage 1을 먼저 확인한다.
- `visual_token_count=16`, `adapter_hidden_dim=512`로 기존 smoke보다 시각 token capacity를 늘린다.
- 1000개 template 데이터는 완성 데이터셋이 아니라 bootstrap 데이터이므로, 먼저 adapter가 이미지-label 연결을
  학습하는지 보는 용도로 쓴다.

### 14.4 한계

- Commons 검색 기반이라 일부 이미지에는 여러 객체가 같이 들어있다.
- class label template로 생성한 답변이므로 사람이 작성한 caption/VQA보다 다양성이 낮다.
- validation/test는 이미지 누수는 없지만 각각 5장 이미지 기반이라 평가 폭이 좁다.
- 이 데이터만으로 좋은 VLM이 되기는 어렵다. 다음 단계는 사람이 검수한 high-quality validation set과
  best validation checkpoint 저장이다.
