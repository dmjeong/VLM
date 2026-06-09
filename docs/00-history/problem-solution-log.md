# 문제-해결 로그

> 프로젝트: DINOv3 Mini VLM
> 목적: 모델 구조 구현 중 발생한 문제, 원인, 해결책, 결과를 기록한다.

---

## PSL-20260528-001

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | `dinov3-mini-vlm` Do 1차 |
| 증상 | 이전 프로젝트 루트에 Web/API 중심 코드가 남아 있어 모델 프로젝트 방향과 충돌 |
| 원인 | 직전 v2가 실제 endpoint 연결을 목표로 했기 때문에 `apps/`, `services/api`, `services/inference`가 남아 있었음 |
| 검토한 대안 | 그대로 유지 / archive 폴더로 이동 / 삭제 |
| 결정 | 현재 PDCA 범위에서 Web/API는 제외이므로 파일을 삭제하고 모델 프로젝트 구조만 남김 |
| 결과 | `mini_vlm/`, `configs/`, `data/samples/`, `tests/` 중심 구조로 전환 |
| 관련 파일 | `README.md`, `mini_vlm/*`, `docs/02-design/features/dinov3-mini-vlm.do.md` |

## PSL-20260528-002

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 테스트 가능성 확인 |
| 증상 | 로컬 환경에 `torch`, `transformers`가 설치되어 있지 않음 |
| 원인 | v2 리셋 이후 최소 Python 환경만 존재 |
| 검토한 대안 | 즉시 대용량 의존성 설치 / 모든 테스트 skip / 순수 Python 계약 테스트와 torch optional 테스트 분리 |
| 결정 | config/dataset/collator는 순수 Python으로 검증하고, visual adapter/forward는 torch 설치 시 실행되는 optional 테스트로 둠 |
| 결과 | 무거운 모델 의존성 없이 기본 계약을 검증할 수 있는 Do 1차 구조 확보 |
| 관련 파일 | `tests/test_config.py`, `tests/test_dataset.py`, `tests/test_collator.py`, `tests/test_visual_adapter.py`, `tests/test_forward.py`, `pyproject.toml` |

## PSL-20260528-003

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Stage 1 학습/추론 dry-run 검증 |
| 증상 | `train_alignment --dry-run`, `infer_cli --dry-run`이 torch 미설치 환경에서 `ModuleNotFoundError`로 실패 |
| 원인 | dry-run 이전 top-level import에서 `mini_vlm.models.builder`를 로딩했고, builder가 torch 의존 모듈을 import함 |
| 검토한 대안 | torch 설치 강제 / dry-run 제거 / 실제 모델 import를 학습/추론 실행 함수 내부로 지연 |
| 결정 | dry-run은 순수 설정 확인만 하도록 유지하고, torch/transformers import는 실제 학습/추론 함수 내부로 이동 |
| 결과 | torch 미설치 환경에서도 dry-run과 기본 테스트가 통과 |
| 관련 파일 | `mini_vlm/training/train_alignment.py`, `mini_vlm/inference/infer_cli.py` |

## PSL-20260528-004

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | `dinov3-mini-vlm` Act 1차 |
| 증상 | tiny dataset에서는 `train_batch_size=2`, `gradient_accumulation_steps=8` 조합 때문에 epoch당 batch가 1개뿐이고 optimizer step이 한 번도 실행되지 않을 수 있음 |
| 원인 | 기존 학습 루프가 accumulation 경계 batch에서만 `optimizer.step()`을 호출하고 epoch 끝 잔여 gradient를 처리하지 않음 |
| 검토한 대안 | 기본 config의 accumulation 값을 1로 변경 / 학습 루프에서 epoch tail step 처리 / smoke config만 별도 추가 |
| 결정 | 기본 config는 유지하되 학습 루프에 epoch tail step을 추가하고, 실제 smoke용 `dinov3-mini-vlm-smoke.json`은 accumulation 1로 분리 |
| 결과 | tiny dataset에서도 backward 이후 optimizer step이 누락되지 않도록 보정됨 |
| 관련 파일 | `mini_vlm/training/train_alignment.py`, `tests/test_training_loop_contract.py`, `configs/dinov3-mini-vlm-smoke.json` |

## PSL-20260528-005

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | freeze backbone 안정화 |
| 증상 | `model.train()` 호출 시 freeze된 DINOv3/LLM child module도 train mode로 바뀔 수 있음 |
| 원인 | PyTorch의 train/eval 전환은 child module 전체에 전파되며, `requires_grad=False`와 eval mode는 별개의 상태임 |
| 검토한 대안 | training loop에서 매번 `llm.eval()` 호출 / wrapper에서 `train()` override / freeze하지 않음 |
| 결정 | `MiniVlmForConditionalGeneration.train()`을 override하고 `enforce_freeze_modes()`에서 freeze module eval 상태를 보장 |
| 결과 | Stage 1에서 adapter만 학습하고 backbone의 dropout 등 train-mode 동작이 켜지는 리스크를 줄임 |
| 관련 파일 | `mini_vlm/models/mini_vlm.py`, `mini_vlm/models/builder.py` |

## PSL-20260528-006

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 실제 모델 검증 준비 |
| 증상 | 실제 DINOv3/LLM 연결을 검증할 test entry가 없어, torch 설치 후 무엇을 돌려야 하는지 불명확함 |
| 원인 | Do 1차는 기본 구조와 dry-run에 집중했고, 모델 다운로드가 필요한 테스트를 분리하지 못했음 |
| 검토한 대안 | 기본 테스트에 모델 다운로드 포함 / integration test를 항상 skip / 환경변수로 명시 실행 |
| 결정 | `RUN_MODEL_INTEGRATION_TESTS=1`일 때만 실제 모델 integration test가 실행되도록 분리 |
| 결과 | 기본 테스트는 가볍게 유지하면서, 모델 환경에서는 DINOv3 patch feature와 LLM `inputs_embeds` 계약을 확인할 수 있음 |
| 관련 파일 | `tests/test_dinov3_integration.py`, `tests/test_llm_integration.py` |

## PSL-20260528-007

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 모델 실행 환경 구성 |
| 증상 | `.venv`에서 `pip install -e '.[model]'` 실행 시 `Multiple top-level packages discovered` 오류 발생 |
| 원인 | flat-layout에서 `data`, `configs`, `mini_vlm`이 모두 package 후보로 발견됨 |
| 검토한 대안 | editable install 포기 / `src` layout 전환 / package discovery를 명시 |
| 결정 | 현재 구조를 유지하고 `pyproject.toml`에서 `mini_vlm*`만 package로 포함 |
| 결과 | editable install 성공, `.venv`에 `torch`, `transformers`, `peft`, `accelerate` 설치 |
| 관련 파일 | `pyproject.toml` |

## PSL-20260528-008

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | DINOv3 integration 실행 |
| 증상 | `AutoImageProcessor` 실행 시 `torchvision` 미설치 오류 발생 |
| 원인 | Transformers 5.9의 image processor 경로가 `torchvision` backend를 요구 |
| 검토한 대안 | 이미지 전처리 직접 구현 / `torchvision` optional dependency 추가 |
| 결정 | model extra에 `torchvision`을 추가 |
| 결과 | image processor backend 문제는 해결 |
| 관련 파일 | `pyproject.toml` |

## PSL-20260528-009

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | DINOv3 모델 다운로드 |
| 증상 | `facebook/dinov3-vits16-pretrain-lvd1689m` 접근이 401 Unauthorized로 실패 |
| 원인 | 해당 DINOv3 모델이 Hugging Face gated repo이며 현재 환경에 `HF_TOKEN`이 없음 |
| 검토한 대안 | DINOv3 접근 승인 후 token 사용 / 공개 DINO 계열 모델로 배선 smoke 실행 |
| 결정 | DINOv3 test는 token 없으면 skip하도록 하고, 공개 `facebook/dinov2-small`로 전체 VLM 배선 smoke를 실행 |
| 결과 | DINOv3 자체 검증은 보류. 공개 vision encoder로 `vision -> adapter -> Qwen -> loss -> checkpoint -> inference` 경로는 확인 |
| 관련 파일 | `tests/test_dinov3_integration.py`, `configs/open-vision-qwen-smoke.json` |

## PSL-20260528-010

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Qwen forward smoke |
| 증상 | adapter 출력 `float32`와 Qwen embedding `bfloat16`이 섞여 LLM linear layer에서 dtype mismatch 발생 |
| 원인 | `torch.cat` 이후 `inputs_embeds` dtype이 Qwen weight dtype과 맞지 않음 |
| 검토한 대안 | Qwen을 float32로 로딩 / adapter 전체를 bfloat16으로 변환 / visual token만 text embedding dtype으로 캐스팅 |
| 결정 | `MiniVlmForConditionalGeneration`와 generation 경로에서 visual token을 LLM embedding dtype으로 캐스팅 |
| 결과 | 전체 forward smoke 성공. `visual_tokens`, `inputs_embeds`, `logits`가 `bfloat16`으로 정렬됨 |
| 관련 파일 | `mini_vlm/models/mini_vlm.py`, `mini_vlm/models/generation.py`, `tests/test_forward.py` |

## PSL-20260528-011

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Stage 1 smoke 재실행 |
| 증상 | 같은 `output_dir`로 학습을 다시 실행하면 기존 `metrics.jsonl` 뒤에 새 로그가 append되어 실험 결과가 섞일 수 있음 |
| 원인 | 학습 루프가 metrics 파일을 항상 append 모드로만 열었음 |
| 검토한 대안 | output directory를 매번 timestamp로 새로 생성 / 기존 metrics 삭제 / JSONL에 run id 추가 |
| 결정 | 같은 config/output dir 재실행 시 이전 `metrics.jsonl`을 삭제하고 새 로그를 기록 |
| 결과 | smoke 실험을 반복해도 metrics가 이전 실행과 섞이지 않음 |
| 관련 파일 | `mini_vlm/training/train_alignment.py` |

## PSL-20260528-012

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Meta DINOv3 local weight 적용 |
| 증상 | Hugging Face gated repo 대신 Meta에서 받은 `.pth` weight를 프로젝트에 연결해야 함 |
| 원인 | 기존 `DinoVisionEncoder`는 Hugging Face `AutoModel` backend만 지원했음 |
| 검토한 대안 | Hugging Face token만 사용 / `torch.hub.load` 사용 / DINOv3 backbone 직접 import |
| 결정 | `torch.hub.load`는 부가 의존성을 많이 요구하므로 `dinov3.hub.backbones`에서 backbone factory를 직접 import |
| 결과 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`로 cls/patch feature, Stage 1 smoke 학습, checkpoint 추론까지 성공 |
| 관련 파일 | `mini_vlm/config.py`, `mini_vlm/models/vision_encoder.py`, `mini_vlm/models/builder.py`, `configs/dinov3-local-vits16-qwen-smoke.json` |

## PSL-20260528-013

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 학습 진행 상황 추적 |
| 증상 | Stage 1 학습 중 콘솔에 진행률과 epoch별 loss 변화가 보이지 않아 학습이 진행 중인지 판단하기 어려움 |
| 원인 | 기존 학습 루프가 batch별 loss를 `metrics.jsonl`에만 기록하고 콘솔 출력과 epoch summary를 제공하지 않음 |
| 검토한 대안 | tqdm 도입 / 단순 print progress / 외부 experiment tracker 연동 |
| 결정 | 의존성 추가 없이 batch 진행률, running average, epoch summary를 콘솔과 JSONL에 기록 |
| 결과 | 학습 중 `epoch/batch/percent/loss/avg`가 출력되고, `training_summary.json`이 생성됨 |
| 관련 파일 | `mini_vlm/training/train_alignment.py`, `tests/test_training_loop_contract.py` |

## PSL-20260528-014

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | epoch 5 이후 성능 정체 분석 후 Act 2차 |
| 증상 | tiny dataset으로 여러 epoch를 학습해도 답변 품질이 거의 좋아지지 않고, 추론 답변이 반복되거나 이미지와 무관한 문장을 생성 |
| 원인 | 기존 구조가 `DINOv3 -> mean pooling MLP adapter -> frozen Qwen`이라 이미지 정보 병목이 크고, LLM 본체는 시각 조건에 맞춰 거의 적응하지 못함. 또한 generation 단계에 반복 억제/stop 조건이 부족했음 |
| 검토한 대안 | 데이터만 늘리기 / LLM 전체 fine-tuning / LoRA만 추가 / adapter를 Perceiver 방식으로 개선 / generation guard 추가 |
| 결정 | 비용이 큰 전체 fine-tuning 대신 `PerceiverResamplerAdapter`로 patch token을 learnable query가 읽게 하고, Qwen에는 PEFT LoRA를 붙임. 동시에 validation loss, generation 반복 억제, `device=auto` 옵션을 config로 노출 |
| 결과 | active config가 `adapter_type=perceiver`, `use_lora=true`, `device=auto` 실험으로 전환됨. MPS 1샘플 smoke에서 train loss 10.8925, validation loss 6.3908이 기록되고 `visual_adapter.pt`, `llm_lora/adapter_model.safetensors`가 생성됨 |
| 관련 파일 | `configs/dinov3-local-vits16-qwen-smoke.json`, `mini_vlm/config.py`, `mini_vlm/models/visual_adapter.py`, `mini_vlm/models/builder.py`, `mini_vlm/models/generation.py`, `mini_vlm/training/train_alignment.py`, `mini_vlm/inference/infer_cli.py`, `mini_vlm/utils/device.py`, `tests/test_config.py`, `tests/test_device.py`, `tests/test_visual_adapter.py`, `tests/test_generation_contract.py` |

## PSL-20260528-015

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | full 10 epoch 학습 NaN 디버깅 |
| 증상 | `dinov3-local-vits16-qwen-lora-perceiver` 학습 결과에서 epoch 1의 4번째 batch부터 `loss`, `running_loss`, `validation_loss`가 `NaN`으로 전파됨 |
| 원인 | `learning_rate=1e-4`와 unclipped gradient 조합에서 일부 batch의 gradient norm이 매우 커졌고, 한 번 비정상 update가 발생한 뒤 이후 step 전체가 오염됨. 기존 학습 루프는 non-finite 값을 감지하지 않고 계속 checkpoint를 저장했음 |
| 검토한 대안 | CPU 강제 / LoRA 비활성화 / learning rate 하향 / gradient clipping / NaN 즉시 중단 |
| 결정 | MPS는 유지하되 `learning_rate=2e-5`, `max_grad_norm=1.0`, `seed=42`를 config에 추가하고, non-finite loss/gradient/parameter를 감지하면 즉시 `FloatingPointError`를 발생시키도록 수정 |
| 결과 | 같은 active config로 10 epoch 재학습 완료. 모든 metrics/summary 값이 finite이며 final train avg loss는 `2.7920`, validation loss는 `3.1667` |
| 관련 파일 | `configs/dinov3-local-vits16-qwen-smoke.json`, `mini_vlm/config.py`, `mini_vlm/training/train_alignment.py`, `mini_vlm/utils/checkpoints.py`, `tests/test_config.py`, `tests/test_training_loop_contract.py` |

## PSL-20260528-016

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 데이터 부족으로 인한 VLM 품질 정체 |
| 증상 | 기존 train 17개/validation 6개만으로는 loss가 내려가도 실제 이미지 질문 답변 품질이 좋아지지 않음 |
| 원인 | 데이터 수가 너무 적고, template/prompt 다양성이 부족해 모델이 시각 정보를 안정적으로 연결하기 어려움 |
| 검토한 대안 | 임의 웹 이미지 크롤링 / Hugging Face 공개 데이터셋 사용 / Wikimedia Commons 라이선스 메타데이터 기반 수집 |
| 결정 | 출처와 라이선스를 추적할 수 있는 Wikimedia Commons API를 사용하고, `imageinfo.extmetadata`에서 라이선스/저작자/원본 페이지를 기록 |
| 결과 | `data/wikimedia_commons_1k`에 이미지 51장, train 900개, validation 100개, 총 1000개 JSONL 샘플 생성. adapter-only Stage 1용 config 추가 |
| 관련 파일 | `scripts/data/collect_wikimedia_commons_dataset.py`, `data/wikimedia_commons_1k/*`, `configs/dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1.json`, `tests/test_config.py` |

## PSL-20260528-017

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | train/validation/test 데이터셋 정리 |
| 증상 | Wikimedia 1K 데이터가 train 900개/validation 100개만 있고 test split이 없었음. 또한 최초 split은 sample 단위라 같은 이미지가 train/validation에 섞일 수 있었음 |
| 원인 | 수집 스크립트가 빠른 bootstrap 생성을 위해 샘플 단위 shuffle split만 수행 |
| 검토한 대안 | 기존 split 유지 / sample 단위 test 추가 / image 단위 train-validation-test 재분리 |
| 결정 | 동일 이미지 누수를 막기 위해 `image` 경로 기준으로 group split을 수행하고, `all.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `split_manifest.json`을 생성 |
| 결과 | `all=1000`, `train=800`, `validation=100`, `test=100`. 이미지 기준으로 train/validation/test 교집합 0개 |
| 관련 파일 | `scripts/data/split_vlm_jsonl_by_image.py`, `data/wikimedia_commons_1k/*`, `mini_vlm/config.py`, `configs/dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1.json`, `tests/test_config.py` |

## PSL-20260528-018

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 학습 결과를 분석 가능한 산출물로 남기기 |
| 증상 | `metrics.jsonl`과 `training_summary.json`은 있었지만, 실험 설명에 바로 쓸 수 있는 그래프/리포트와 test 평가 결과 저장 흐름이 부족했음 |
| 원인 | 학습 루프는 checkpoint 저장 중심이고, 별도 reporting/evaluation 계층이 없었음 |
| 검토한 대안 | 외부 experiment tracker 도입 / matplotlib 의존성 추가 / SVG와 Markdown 직접 생성 |
| 결정 | 의존성 추가 없이 학습 종료 시 `loss_curves.csv`, `loss_curve.svg`, `training_report.md`를 생성하고, 별도 evaluation CLI로 `evaluation/test_summary.json`, `test_predictions.jsonl`, `test_report.md`를 생성 |
| 결과 | 학습 산출물 폴더만 보면 config, checkpoint, 원본 로그, 그래프, 리포트, 테스트 결과를 함께 확인할 수 있는 구조로 정리 |
| 관련 파일 | `mini_vlm/reporting/training_report.py`, `mini_vlm/evaluation/evaluate_cli.py`, `mini_vlm/utils/model_loading.py`, `mini_vlm/training/train_alignment.py`, `mini_vlm/inference/infer_cli.py`, `tests/test_training_report.py`, `tests/test_evaluation_helpers.py` |

## PSL-20260528-019

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 평가 결과를 웹페이지로 한눈에 보기 |
| 증상 | `test_summary.json`과 `test_predictions.jsonl`만으로는 어떤 이미지가 실패했는지, 어떤 객체가 취약한지 빠르게 보기 어려움 |
| 원인 | 평가는 sample 중심 JSONL이고, 사람이 리뷰할 때 필요한 이미지 썸네일/객체별 집계/문제 요약 계층이 없었음 |
| 검토한 대안 | React 앱 생성 / 외부 BI 도구 사용 / 정적 HTML 대시보드 생성 |
| 결정 | 별도 서버 없이 열 수 있는 `evaluation/dashboard.html`을 생성하고, 이미지별 exact/contains/object hit/token overlap과 질문별 정답-생성 답변을 표시 |
| 결과 | 평가 CLI 실행 후 checkpoint의 `evaluation/` 폴더에 HTML 대시보드가 함께 저장됨 |
| 관련 파일 | `mini_vlm/evaluation/dashboard.py`, `mini_vlm/evaluation/evaluate_cli.py`, `tests/test_evaluation_dashboard.py` |

## PSL-20260528-020

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 평가 대시보드 이미지 로딩 문제 |
| 증상 | `dashboard.html`을 열었을 때 이미지가 보이지 않을 수 있음 |
| 원인 | 초기 대시보드는 이미지 `src`에 `file:///Users/...` 절대경로를 사용했고, IDE preview/일부 브라우저 보안 정책에서는 로컬 절대 파일 참조가 차단될 수 있음 |
| 검토한 대안 | base64 data URI 내장 / 원본 `file://` 유지 / 대시보드 옆 asset 폴더로 이미지 복사 |
| 결정 | `evaluation/dashboard_assets/images/`에 평가 이미지를 복사하고 HTML에서는 상대경로로 참조 |
| 결과 | `dashboard.html`과 asset 폴더만 함께 있으면 브라우저/IDE preview에서 이미지가 안정적으로 로드되는 구조로 변경 |
| 관련 파일 | `mini_vlm/evaluation/dashboard.py`, `tests/test_evaluation_dashboard.py` |

## PSL-20260528-021

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 데이터셋을 1K에서 10K로 확장 |
| 증상 | Wikimedia 1K는 이미지 51장 기반이라 loss가 내려가도 실제 객체 인식 성능이 낮고, validation/test 폭도 좁음 |
| 원인 | 학습 이미지 다양성, 질문 유형, 상세 caption/추론형 답변이 부족함 |
| 검토한 대안 | Wikimedia만 계속 확장 / ShareGPT4V만 사용 / LVIS-Instruct4V만 사용 / 평가 벤치마크를 학습에 섞기 |
| 결정 | ShareGPT4V와 LVIS-Instruct4V는 train 중심으로 사용하고, MMBench DEV EN은 평가 성격을 유지해 validation/test에만 섞음. MME는 별도 공식 다운로드/라이선스 확인 후 importer 후보로 남김 |
| 결과 | `data/external_vlm_10k`를 생성하는 데이터 빌더와 10K Stage 1 config 추가 |
| 관련 파일 | `scripts/data/build_external_vlm_10k_dataset.py`, `configs/dinov3-local-vits16-qwen-external-10k-adapter-stage1.json`, `tests/test_external_dataset_builder.py` |

## PSL-20260528-022

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | External VLM 10K 실제 학습/테스트 실행 |
| 증상 | 10K 데이터로 3 epoch 학습을 시작했을 때 train epoch 1은 정상 완료됐지만 validation loss 계산 중 `NaN`으로 중단됨 |
| 원인 | MMBench validation/test의 긴 question/options prompt가 `max_text_length=192`를 초과했고, 기존 collator가 `prompt + answer` 전체를 단순 앞쪽 truncation하여 일부 샘플의 answer label을 모두 제거함. label이 전부 `-100`이면 supervised token이 없어 causal LM loss가 NaN이 될 수 있음 |
| 검토한 대안 | `max_text_length`만 증가 / MMBench 제거 / NaN batch skip / collator truncation 정책 수정 |
| 결정 | prompt 일부를 줄이더라도 answer token은 반드시 남기도록 collator를 수정. epoch별 adapter checkpoint를 먼저 저장하고 validation non-finite 에러에는 sample id를 포함 |
| 결과 | train/validation/test 전체 스캔에서 `bad_all_ignored=0` 확인. 10K 1 epoch 학습 완료, validation loss `2.4231`, test avg loss `2.4345`, generation 100 sample token overlap `0.4998` 기록 |
| 관련 파일 | `mini_vlm/data/collator.py`, `mini_vlm/training/train_alignment.py`, `tests/test_collator.py`, `configs/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1.json`, `History.md` |

## PSL-20260528-023

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Perceiver adapter에서 Q-Former 구조로 전환 실험 |
| 증상 | Perceiver adapter-only 학습은 loss가 내려가지만 이미지 grounding이 약하고, BLIP-2식 Q-Former/ITC/ITM/ITG 흐름과 차이가 큼 |
| 원인 | 기존 `adapter_type=qformer`는 config에만 존재하고 builder에서는 `NotImplementedError`로 막혀 있었음. 또한 기존 Perceiver에는 query self-attention 단계가 없어 query token 간 정보 교환이 제한적임 |
| 검토한 대안 | Perceiver 유지 / DistilBERT 기반 Q-Former를 바로 도입 / lightweight Q-Former adapter 먼저 구현 |
| 결정 | DistilBERT 초기화와 ITC pretraining은 다음 단계로 분리하고, 우선 learnable query token + self-attention + image cross-attention + FFN 형태의 `QFormerVisualAdapter`를 구현 |
| 결과 | Q-Former smoke 학습과 평가 완료. trainable parameter `9,142,528`, smoke train avg loss `5.4124`, validation/test loss `5.2324`, generation token overlap `0.3889` |
| 관련 파일 | `mini_vlm/models/visual_adapter.py`, `mini_vlm/models/builder.py`, `tests/test_visual_adapter.py`, `tests/test_config.py`, `configs/dinov3-local-vits16-qwen-qformer-smoke.json`, `configs/dinov3-local-vits16-qwen-external-10k-qformer-stage1-epoch1.json`, `History.md` |

## PSL-20260528-024

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Q-Former를 BLIP-2에 더 가깝게 완전한 학습 파이프라인으로 확장 |
| 증상 | Q-Former 구조만 구현하면 Perceiver와 비교 가능한 adapter는 되지만, 사용자가 기대한 “DistilBERT에서 시작한 Q-Former + ITC 사전정렬 + LLM 미세조정” 흐름과는 아직 차이가 있었음 |
| 원인 | 기존 구현은 lightweight Q-Former block을 random init으로 만들고 바로 LLM answer loss에 연결했다. 이미지-텍스트 contrastive 사전정렬과 DistilBERT self-attention/FFN weight 이식, 그리고 사전학습 adapter를 Stage 1 초기값으로 로드하는 경로가 없었음 |
| 검토한 대안 | Q-Former 구조만 유지 / ITC만 추가 / DistilBERT 초기화까지 추가 / ITM/ITG까지 모두 한 번에 구현 |
| 결정 | 먼저 실제 실행 가능한 완성 경로인 `DistilBERT 초기화 -> ITC 사전정렬 -> visual_adapter 초기값 로드 -> LLM answer loss 학습`을 구현. ITM/ITG는 별도 objective로 확장 예정 |
| 결과 | DistilBERT 초기화 Q-Former ITC smoke 완료: ITC train avg loss `0.7077`, validation loss `0.6253`. 이어 LLM smoke 완료: train avg loss `9.8322`, validation/test loss `7.9130`. 전체 단위 테스트 `72개 통과, 3개 skip` |
| 관련 파일 | `mini_vlm/models/qformer.py`, `mini_vlm/training/pretrain_qformer_itc.py`, `mini_vlm/training/train_alignment.py`, `mini_vlm/config.py`, `tests/test_qformer_itc.py`, `tests/test_training_loop_contract.py`, `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json`, `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke.json`, `History.md` |

## PSL-20260528-025

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Git 추적 누락 문제 |
| 증상 | `mini_vlm/models/visual_adapter.py`, `builder.py`, `qformer.py` 같은 모델 코드 변경이 `git status`에 보이지 않음 |
| 원인 | `.gitignore`의 `models/` 패턴이 루트의 모델 weight 폴더뿐 아니라 `mini_vlm/models/` 코드 폴더까지 무시함 |
| 검토한 대안 | 강제 add / `.gitignore` 유지 / 루트 폴더만 ignore하도록 수정 |
| 결정 | weight 저장용 루트 폴더만 무시하도록 `models/`를 `/models/`로 변경 |
| 결과 | `mini_vlm/models/` 코드 파일들이 Git status에 정상적으로 표시됨 |
| 관련 파일 | `.gitignore` |

## PSL-20260528-026

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Q-Former hidden dimension 정책 정리 |
| 증상 | 512 lightweight Q-Former와 768 DistilBERT 초기화 Q-Former config가 공존해 어떤 경로가 표준인지 혼란이 생김 |
| 원인 | 512는 빠른 구조 smoke용으로 만든 값이고, DistilBERT weight는 hidden size 768이므로 BLIP-2 스타일 초기화를 하려면 Q-Former hidden dim도 768이어야 함 |
| 검토한 대안 | 512/768 병행 유지 / 512를 projection으로 변환 / 768 DistilBERT 경로만 유지 |
| 결정 | Q-Former 실험은 768 DistilBERT 초기화 경로로 단일화. 512 Q-Former config는 삭제하고, `configs/*qformer*.json`는 `adapter_hidden_dim=768`만 통과하도록 테스트 추가 |
| 결과 | Q-Former active config는 `qformer-distilbert-itc-*` 4개만 남음. `MiniVlmConfig`는 DistilBERT 초기화인데 hidden dim이 768이 아니면 `ValueError`를 발생시킴 |
| 관련 파일 | `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json`, `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke.json`, `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json`, `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json`, `mini_vlm/config.py`, `tests/test_config.py`, `History.md` |

## PSL-20260528-027

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 768 DistilBERT Q-Former External 10K 실제 학습과 평가 |
| 증상 | “학습하면 되나?” 단계에서 실제 Stage 0 ITC, Stage 1 LLM answer loss, test generation을 모두 돌려 학습 파이프라인에 문제가 있는지 확인해야 했음 |
| 원인 | 이전 Perceiver adapter는 loss는 내려갔지만 image grounding이 약했고, Q-Former 768 경로는 smoke만 완료된 상태라 10K 규모에서 안정성/품질을 확인하지 못했음 |
| 검토한 대안 | Perceiver 3 epoch 재학습 / Q-Former Stage 1만 바로 학습 / DistilBERT Q-Former ITC 후 Stage 1 학습 / LoRA까지 즉시 활성화 |
| 결정 | BLIP-2 스타일에 맞춰 `DistilBERT 초기화 Q-Former -> ITC 사전정렬 -> Qwen answer loss Stage 1` 순서로 실행. LLM과 DINOv3는 freeze하고 Q-Former adapter만 먼저 안정화 |
| 결과 | Stage 0 ITC는 NaN 없이 완료됐고 train avg loss `0.5107`, validation loss `1.2883`을 기록. Stage 1도 NaN 없이 완료됐고 train avg loss `2.4225`, validation loss `1.5950`, test avg loss `1.5686`을 기록. 기존 Perceiver 1 epoch test loss `2.4345`보다 낮아짐 |
| 추가 발견 | 최초 generation 평가는 test 앞 100개만 사용해 이미지 4장에 몰렸음. `--generation-sampling even` 옵션을 추가해 100개 샘플이 69개 이미지에 퍼지도록 수정하고 재평가함 |
| 남은 문제 | 생성 품질은 아직 낮음. exact match `0.0100`, contains answer `0.0900`, avg token overlap `0.4545`, MMBench 생성 샘플 letter accuracy `0.3200`. bathroom을 사람 장면으로 설명하거나 multiple-choice letter를 틀리는 사례가 있음 |
| 관련 파일 | `mini_vlm/evaluation/evaluate_cli.py`, `tests/test_evaluation_helpers.py`, `History.md`, `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0/*`, `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/*` |

## PSL-20260528-028

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 평가 대시보드가 5개 이미지만 보여 보이는 문제 |
| 증상 | `dashboard.html`을 열었을 때 테스트 이미지가 5개만 표시되는 것처럼 보임 |
| 원인 | HTML에는 전체 이미지 카드 69개가 있었지만, 상단의 `가장 취약한 이미지` 요약 섹션이 5개 카드만 크게 보여서 전체 목록처럼 오해되기 쉬웠음. 또한 각 이미지의 질문 상세가 기본 펼침 상태라 전체 이미지 목록을 빠르게 훑기 어려웠음 |
| 검토한 대안 | 모델 평가 재실행 / HTML만 재생성 / 취약 이미지 섹션 제거 / 전체 이미지 섹션을 명확히 분리 |
| 결정 | 평가 재실행 없이 기존 `test_summary.json`과 `test_predictions.jsonl`로 HTML만 재생성. 상단 제목을 `가장 취약한 이미지 Top 5`로 변경하고, 별도 `전체 테스트 이미지` 섹션과 `69 / 69 images` 표시를 추가. 질문별 상세는 기본 접힘으로 변경 |
| 결과 | 재생성된 HTML에서 `image-card=69`, `weak-card=5`, asset image `69`개 확인. 대시보드가 Top 5 요약과 전체 이미지 목록을 명확히 구분함 |
| 관련 파일 | `mini_vlm/evaluation/dashboard.py`, `tests/test_evaluation_dashboard.py`, `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/dashboard.html` |

## PSL-20260528-029

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 대시보드의 모든 이미지가 `unknown`으로 보이는 문제 |
| 증상 | 평가 대시보드에서 이미지 분류명이 전부 `unknown`으로 보이고, object hit가 모두 0%로 표시됨 |
| 원인 | External 10K test split의 LVIS/MMBench 샘플에는 `metadata.object`가 없음. 기존 대시보드는 Wikimedia 객체 라벨 데이터셋 기준으로 만들어져 `metadata.object`가 없으면 무조건 `unknown`과 object hit 0%를 표시했음 |
| 검토한 대안 | 데이터셋에 임의 object 라벨 생성 / object hit 지표 제거 / 데이터셋별 fallback label과 MMBench 선택지 지표 추가 |
| 결정 | object 라벨이 없으면 LVIS는 `task`, MMBench는 `metadata.category`로 표시. object hit는 라벨이 있을 때만 표시하고, MMBench는 `choice` letter 정답률을 별도 표시 |
| 결과 | 재생성된 HTML에서 `unknown_count=0`, `image-card=69`, `choice` metric이 MMBench 50개 이미지에 표시됨. 현재 생성 평가 100개 중 성공 행은 22개, MMBench letter 기준은 16/50으로 아직 낮음 |
| 관련 파일 | `mini_vlm/evaluation/dashboard.py`, `tests/test_evaluation_dashboard.py`, `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/dashboard.html` |

## PSL-20260528-030

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | DINOv3를 계속 유지해야 하는지 확인하기 위한 vision encoder ablation |
| 증상 | DINOv3는 self-supervised representation이 매력적이지만, VQA/chat 서비스에는 CLIP/SigLIP처럼 image-text alignment가 이미 된 백본이 더 적합할 수 있다는 의문이 생김 |
| 원인 | 기존 구현은 DINOv3만 지원해 비전 백본의 pretraining objective 차이를 같은 Q-Former/LLM 조건에서 비교할 수 없었음 |
| 검토한 대안 | DINO만 epoch 확장 / CLIP만 추가 / SigLIP만 추가 / DINO, CLIP, SigLIP 세 후보를 같은 데이터와 Q-Former 구조로 비교 |
| 결정 | `CLIPVisionModel`, `SiglipVisionModel` backend를 추가하고 CLS token 유무 차이를 처리했다. CLIP은 CLS+patch 구조, SigLIP는 patch-only 구조라 pooled output 또는 mean feature를 CLS 역할로 사용하도록 분기했다 |
| 결과 | CLIP/SigLIP forward smoke, Stage 0 dry-run, Stage 1 dry-run이 통과했다. 이후 실제 Stage 0/Stage 1/evaluation까지 실행 가능한 config 4개와 비교 CLI를 추가했다 |
| 관련 파일 | `mini_vlm/models/vision_encoder.py`, `mini_vlm/evaluation/compare_vision_ablation.py`, `tests/test_vision_encoder.py`, `configs/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0.json`, `configs/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |

## PSL-20260528-031

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | DINO/CLIP/SigLIP 비교 실험 실행 |
| 증상 | DINO 3 epoch 시도 중 epoch3 첫 batch에서 gradient norm이 NaN이 되어 학습이 중단됨 |
| 원인 | epoch2까지 loss는 안정적으로 내려갔지만 특정 LVIS detailed-caption 샘플에서 gradient가 비정상화됐다. 현재 원인은 learning rate/gradient clipping 한계, 긴 caption 샘플, MPS 수치 안정성 중 하나로 추정된다 |
| 검토한 대안 | DINO 결과 폐기 / epoch1만 비교 / epoch2 checkpoint fallback으로 평가 / CLIP/SigLIP만 비교 |
| 결정 | `visual_adapter_epoch_2.pt` fallback 로더를 추가해 DINO epoch2 checkpoint를 평가하고, 비교 리포트에 최종 adapter가 아니라 epoch checkpoint로 평가했다는 메모를 남겼다 |
| 결과 | DINO epoch2 checkpoint는 test loss `1.4157`, MMBench choice accuracy `36.0%`, generation success `22.0%`. CLIP epoch1은 test loss `1.8066`, choice `34.0%`, success `20.0%`. SigLIP epoch1은 test loss `1.6055`, choice `38.0%`, success `8.0%`. 현재 수치만으로 DINO를 버릴 근거는 없지만, 서비스 기본 백본은 DINO/SigLIP를 병행 비교하는 방향으로 결정했다 |
| 관련 파일 | `mini_vlm/utils/model_loading.py`, `tests/test_model_loading.py`, `mini_vlm/evaluation/compare_vision_ablation.py`, `History.md`, `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md` |

## PSL-20260528-032

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 비교 리포트와 대시보드 자동 갱신 |
| 증상 | 개별 평가 대시보드는 각 checkpoint 폴더에 생성되지만, `vision-ablation/comparison.md`에서 바로 열 수 없고 비교 CLI가 대시보드를 최신 상태로 다시 생성하지 않았음 |
| 원인 | 기존 비교 CLI는 summary/predictions를 읽어 Markdown/JSON 비교표만 만들었고, config의 `image_root`를 알지 못해 dashboard 재생성을 담당하지 않았음 |
| 검토한 대안 | 수동으로 각 대시보드 열기 / 평가 CLI만 다시 실행 / 비교 CLI에서 config까지 연결해 dashboard 재생성 / 통합 HTML 대시보드 추가 |
| 결정 | 비교 대상마다 config 경로를 명시하고, `compare_vision_ablation` 실행 시 기존 `test_summary.json`과 `test_predictions.jsonl`로 개별 `evaluation/dashboard.html`을 재생성한다. 또한 `comparison.md`에 개별 dashboard 링크를 붙이고, `vision-ablation/dashboard.html` 통합 비교 대시보드를 생성한다 |
| 결과 | DINO/CLIP/SigLIP 4개 실험의 대시보드 갱신 상태가 모두 `업데이트`로 기록됐고, `comparison.md`에서 각 실험 대시보드를 바로 열 수 있게 됐다. 통합 비교 대시보드도 생성됨 |
| 관련 파일 | `mini_vlm/evaluation/compare_vision_ablation.py`, `tests/test_compare_vision_ablation.py`, `History.md`, `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md`, `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |

## PSL-20260528-033

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | Artifact 폴더 수가 너무 많아 현재 비교 결과를 찾기 어려움 |
| 증상 | `artifacts/dinov3-mini-vlm` 아래에 smoke, 폐기 실험, 초기 baseline, active 비교 실험이 모두 같은 레벨에 있어 사용자가 어떤 폴더를 봐야 하는지 혼란스러움 |
| 원인 | PDCA 반복마다 checkpoint와 dashboard를 보존했지만, active/legacy 구분 없이 루트 artifact 폴더에 계속 쌓였음 |
| 검토한 대안 | 전체 삭제 / active만 남기고 삭제 / legacy를 archive로 이동 / 문서만 정리 |
| 결정 | 삭제하지 않고 현재 비교에 쓰지 않는 15개 legacy artifact를 `artifacts/dinov3-mini-vlm/_archive/legacy-20260528`로 이동했다. 이동 목록과 복구 방법은 `manifest.json`에 기록했다 |
| 결과 | 루트 artifact는 DINO/CLIP/SigLIP 비교에 필요한 Stage 0/Stage 1/evaluation 폴더와 `vision-ablation` 중심으로 줄었다. 약 698MB가 archive 폴더로 이동했고, `compare_vision_ablation` 재실행으로 비교표/통합 대시보드가 정상 갱신됨을 확인했다 |
| 관련 파일 | `scripts/maintenance/cleanup_artifacts.py`, `History.md`, `artifacts/dinov3-mini-vlm/_archive/legacy-20260528/manifest.json`, `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md`, `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |

## PSL-20260528-034

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | 통합 비교 대시보드의 개별 대시보드 `열기` 링크 오류 |
| 증상 | `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html`에서 각 실험의 `열기` 링크를 클릭해도 개별 대시보드가 열리지 않음 |
| 원인 | 최초 링크가 `../.../evaluation/dashboard.html` 부모 상대경로였고, 일부 IDE/HTML preview에서는 부모 폴더 이동 링크를 잘못 해석하거나 차단할 수 있음 |
| 검토한 대안 | 절대 `file://` 링크 사용 / 원본 evaluation dashboard만 유지 / 통합 대시보드 하위에 개별 대시보드와 asset을 미러링 |
| 결정 | preview 호환성을 위해 각 실험 대시보드를 `vision-ablation/experiments/<experiment>/dashboard.html`로 복사하고, 해당 폴더에 `dashboard_assets`도 함께 복사한다. 통합 대시보드와 Markdown 링크는 같은 폴더 하위 상대경로만 사용한다 |
| 결과 | `열기` 링크가 모두 `experiments/.../dashboard.html`로 바뀌었고, 4개 링크 대상 파일 존재를 확인했다. 전체 테스트도 통과했다 |
| 관련 파일 | `mini_vlm/evaluation/compare_vision_ablation.py`, `tests/test_compare_vision_ablation.py`, `History.md`, `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html`, `artifacts/dinov3-mini-vlm/vision-ablation/experiments/*/dashboard.html` |

## PSL-20260528-035

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-28 |
| 맥락 | adapter-only 성능이 낮아 Qwen LoRA Stage 2 진행 |
| 증상 | DINO/CLIP/SigLIP 비교 후에도 generation success가 22% 수준이고, bathroom/bus 이미지를 사람 장면으로 설명하는 등 시각 grounding이 낮음 |
| 원인 | Stage 1은 Qwen 본체를 freeze하고 visual adapter만 학습했기 때문에, Qwen이 visual prefix 조건에 맞춰 답변 형식과 선택지 출력을 충분히 적응하지 못함 |
| 검토한 대안 | adapter만 추가 epoch / Qwen 전체 fine-tuning / Qwen LoRA / SigLIP로 전환 후 재학습 |
| 결정 | 가장 test loss가 낮았던 DINO Stage1 epoch2 adapter를 초기값으로 사용하고, Qwen에는 LoRA를 붙여 Stage 2를 1 epoch 학습했다. 수치 안정성을 위해 `learning_rate=5e-6`, `max_grad_norm=0.5`로 보수적으로 설정했다 |
| 결과 | LoRA 학습은 NaN 없이 완료됐다. validation loss는 `1.4499 -> 1.3150`, test loss는 `1.4157 -> 1.2866`, exact match는 `1% -> 6%`, contains answer는 `6% -> 10%`로 개선됐다. 다만 MMBench choice accuracy는 36%로 유지되고 generation success는 22%에서 21%로 소폭 하락해, 사용자 체감 품질은 아직 부족하다 |
| 관련 파일 | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1.json`, `mini_vlm/evaluation/compare_vision_ablation.py`, `scripts/maintenance/cleanup_artifacts.py`, `tests/test_config.py`, `History.md`, `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/*` |

## PSL-20260529-036

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-29 |
| 맥락 | 평가 대시보드에서 정답으로 보이는 답변도 `주의` 또는 실패처럼 표시되는 문제 |
| 증상 | “No, there are no humans or animals...”처럼 정답 의미를 말한 생성 답변이 exact/contains 불일치와 낮은 token overlap 때문에 실패로 표시됨. 반대로 객관식은 letter만 맞으면 생성 설명이 다른 답을 말해도 양호로 보일 수 있었음 |
| 원인 | 기존 dashboard success는 문장 완전 일치, 정답 문장 포함, object hit, choice letter, 높은 token overlap을 단순 혼합했다. yes/no 극성, count 숫자 일치, 짧은 사실형 핵심 token, choice 설명 충돌을 구분하지 않았음 |
| 검토한 대안 | LLM judge 도입 / BLEU·ROUGE 추가 / 기존 overlap threshold만 조정 / deterministic answer match 규칙 추가 |
| 결정 | 비용이 없고 재현 가능한 deterministic 규칙을 먼저 추가했다. `answer_match`를 별도 필드로 만들고 yes/no, counting, short fact, choice warning을 분리했다. LLM judge는 이후 사람이 검수할 gold set이 생긴 뒤 보조 평가로 검토한다 |
| 결과 | LoRA Stage2 대시보드의 `Answer match`가 28.0%로 재산정됐다. 문제 샘플은 `answer yes · reason yes-no`로 표시되고, 숫자 불일치와 Volvo/Tesla 같은 짧은 사실형 오답은 overlap이 높아도 실패한다. DINO/CLIP/SigLIP 개별 대시보드와 통합 비교 대시보드를 모두 재생성했다 |
| 관련 파일 | `mini_vlm/evaluation/dashboard.py`, `mini_vlm/evaluation/compare_vision_ablation.py`, `tests/test_evaluation_dashboard.py`, `History.md`, `artifacts/dinov3-mini-vlm/vision-ablation/comparison.json`, `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |

## PSL-20260529-037

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-29 |
| 맥락 | Playwright 설치 후 대시보드 자동 렌더 검증 |
| 증상 | 단위 테스트와 HTML 문자열 검증만으로는 실제 브라우저에서 이미지가 깨지는지, 모바일에서 가로 스크롤이 생기는지, 링크가 열리는지 확신하기 어려웠음 |
| 원인 | Playwright와 Chromium이 로컬 환경에 설치되어 있지 않았고, dashboard 렌더링을 반복 검증하는 스크립트가 없었음 |
| 검토한 대안 | 수동 브라우저 확인 / Node Playwright 사용 / Python Playwright 사용 / 테스트 suite에 강제 통합 |
| 결정 | Python 프로젝트 구조에 맞춰 `.venv`에 Playwright를 설치하고 Chromium browser binary를 받았다. 별도 maintenance script로 통합 대시보드와 LoRA 개별 대시보드를 desktop/mobile viewport에서 검증하도록 했다 |
| 결과 | 최초 검증에서 LoRA 개별 대시보드 mobile overflow `59px`를 발견했고, 긴 category 텍스트가 card 폭을 밀어내는 CSS 문제를 수정했다. 재검증 결과 4개 렌더 케이스 모두 PASS, 이미지 69개 broken image 0개, console/page error 0개 |
| 관련 파일 | `pyproject.toml`, `scripts/maintenance/verify_dashboard_render.py`, `mini_vlm/evaluation/dashboard.py`, `History.md`, `artifacts/dinov3-mini-vlm/vision-ablation/render-check/render_check.json`, `artifacts/dinov3-mini-vlm/vision-ablation/render-check/*.png` |

## PSL-20260529-038

| 항목 | 내용 |
|------|------|
| 날짜 | 2026-05-29 |
| 맥락 | `$pdca iterate`로 Eval80 80개 중 90% 목표까지 성능 반복 |
| 증상 | 기존 DINO + Qwen LoRA mini VLM은 Eval80에서 `25/80`, 31.2%에 그쳤고, 화면/대시보드상 성능 변화도 여러 실험별로 한눈에 비교하기 어려웠음 |
| 원인 | 현재 mini VLM은 DINO visual feature를 Qwen3-0.6B embedding으로 넘기는 adapter 학습 단계라, pretrained VLM 수준의 시각-언어 instruction alignment와 counting 능력이 부족함. 또한 Eval80 v1에는 불완전하거나 모호한 정답 라벨 4개가 섞여 있었음 |
| 검토한 대안 | mini VLM 추가 epoch / Qwen3 LoRA 추가 조정 / 강한 pretrained VLM baseline / MLX 4bit VLM 도입 / Eval80 라벨 품질 감사 / oracle union 상한 계산 |
| 결정 | 먼저 고정 Eval80 기준으로 강한 teacher/baseline을 같은 scoring pipeline에 태웠다. PyTorch MPS로 SmolVLM2-256M, Qwen2.5-VL-3B를 평가하고, Mac 16GB 환경에서 가능한 MLX 4bit Qwen2.5-VL-7B와 InternVL3-8B를 추가했다 |
| 결과 | 단일 모델 최고는 `mlx-internvl3-8b-4bit-strict`의 `64/80`, 80.0%. MLX Qwen2.5-VL-7B는 `63/80`, Qwen2.5-VL-3B는 `58/80`. 여러 모델 중 하나라도 맞힌 oracle union은 `72/80`, 90.0%였지만, 이는 사후 선택 상한선이라 실제 서비스 성능으로 간주하지 않음 |
| 추가 발견 | Eval80 품질 감사에서 `incomplete-answer` 2개, `count-answer-without-number` 1개, `vague-count-answer` 1개를 발견. 품질 이슈를 제외한 oracle union은 `71/76`, 93.4% |
| 남은 문제 | 자동 selector 없이 단일 모델은 아직 90% 미달. counting 전문 보조 모델, Eval80 v2 라벨 교체, teacher distillation 또는 stronger VLM 기반 구조 전환이 필요함 |
| 관련 파일 | `scripts/data/build_eval80_dataset.py`, `scripts/data/audit_eval_dataset.py`, `scripts/evaluation/evaluate_pretrained_vlm.py`, `scripts/evaluation/evaluate_mlx_vlm.py`, `scripts/evaluation/compare_eval80_runs.py`, `mini_vlm/evaluation/benchmark.py`, `mini_vlm/evaluation/dashboard.py`, `data/eval80/quality_report.json`, `artifacts/dinov3-mini-vlm/eval80/leaderboard.md`, `History.md` |
