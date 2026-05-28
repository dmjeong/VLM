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
