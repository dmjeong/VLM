# DINOv3 Mini VLM 실험 히스토리

> 목적: 모델 구조, 학습 데이터 수량, 성능 지표, 문제점, 개선 필요 항목을 시기별로 한눈에 추적한다.
> 기준일: 2026-05-28
> 샘플 정의: `sample = image + question + answer` 한 세트. 같은 이미지라도 질문/답변이 다르면 다른 sample이다.

---

## 전체 요약

| 시기 | 핵심 목표 | 상태 | 대표 산출물 |
|------|-----------|------|-------------|
| 1차 시기 | DINOv3와 Qwen을 실제로 연결하고 MLP adapter baseline 검증 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-smoke` |
| 2차 시기 | Perceiver adapter와 Qwen LoRA를 붙이고 NaN 안정화 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-lora-perceiver` |
| 3차 시기 | Wikimedia Commons 1K bootstrap 데이터로 adapter-only Stage 1 학습 | 진행 중 | `configs/dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1.json` |

---

## 1차 시기: DINOv3 + Qwen MLP Baseline

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Vision 입력 | 224 x 224 RGB |
| Vision feature | CLS token + patch tokens |
| Adapter | MLP visual adapter |
| Visual token 수 | 4 |
| Adapter hidden dim | 128 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen freeze, adapter만 학습 |
| LoRA | 사용 안 함 |
| 목적 | 전체 forward/loss/checkpoint/inference 경로 검증 |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 17 |
| validation samples | 6 |
| test samples | 없음 |
| epoch | 10 |
| batch size | 1 |
| 데이터 출처 | manual sample + Wikimedia 소량 샘플 |

### 3. 성능

| 지표 | 값 |
|------|----|
| final train avg loss | 3.7559 |
| validation loss | 기록 없음 |
| test metric | 없음 |
| 정성 평가 | checkpoint inference는 실행됐지만 답변 반복과 이미지 무관 답변이 발생 |

### 4. 문제점

- 데이터 수가 너무 적어 모델 품질을 판단하기 어렵다.
- MLP adapter가 patch 정보를 단순 압축하므로 이미지 정보 병목이 크다.
- validation loss가 초기에는 기록되지 않아 일반화 판단이 어렵다.
- test split이 없다.

### 5. 개선필요 항목

- validation loss 기록 추가.
- 반복 억제 generation 옵션 추가.
- MLP보다 표현력이 큰 adapter 검토.
- 데이터셋 확장.
- test split과 평가 스크립트 추가.

---

## 2차 시기: Perceiver Adapter + Qwen LoRA

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Adapter | PerceiverResamplerAdapter |
| Visual token 수 | 8 |
| Adapter hidden dim | 256 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen base freeze |
| LoRA | 사용 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 |
| generation guard | repetition penalty, no-repeat ngram, stop strings |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 17 |
| validation samples | 6 |
| test samples | 없음 |
| epoch | 10 |
| batch size | 1 |
| device | MPS |

### 3. 성능

| 지표 | 값 |
|------|----|
| final train avg loss | 2.7920 |
| final validation loss | 3.1667 |
| best validation loss | 2.8078, epoch 9 |
| test metric | 없음 |
| 정성 평가 | NaN은 해결됐지만 실제 답변 품질은 여전히 낮음 |

### 4. 문제점

- 초기 학습에서 `learning_rate=1e-4`와 unclipped gradient 때문에 NaN이 발생했다.
- 작은 데이터에서 LoRA까지 같이 학습하니 답변 문장 패턴이 쉽게 망가졌다.
- best validation checkpoint를 저장하지 않아 마지막 epoch가 best보다 나쁠 수 있다.
- test split이 없어 최종 성능 판단이 불가능하다.

### 5. 개선필요 항목

- `learning_rate=2e-5`, `max_grad_norm=1.0`, `seed=42` 유지.
- best validation checkpoint 저장 추가.
- LoRA는 Stage 1 adapter-only가 안정화된 뒤 Stage 2에서 켜기.
- test 평가 스크립트 추가.
- 더 큰 데이터셋으로 adapter가 시각 정보를 학습하는지 먼저 확인.

---

## 3차 시기: Wikimedia Commons 1K Adapter-only Stage 1

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Adapter | PerceiverResamplerAdapter |
| Visual token 수 | 16 |
| Adapter hidden dim | 512 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen freeze, adapter만 학습 |
| LoRA | 사용 안 함 |
| 목적 | LoRA 전에 adapter가 image-label 연결을 학습하는지 확인 |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| all samples | 1000 |
| train samples | 800 |
| validation samples | 100 |
| test samples | 100 |
| 전체 이미지 수 | 51 |
| train 이미지 수 | 40 |
| validation 이미지 수 | 5 |
| test 이미지 수 | 5 |
| split 단위 | image 기준 |
| image leakage | train/validation/test 교집합 0개 |
| epoch | 3 |
| gradient accumulation | 4 |

### 3. 성능

| 지표 | 값 |
|------|----|
| epoch 1 train avg loss | 5.3676 |
| epoch 1 validation loss | 3.1585 |
| epoch 2 train avg loss | 2.2347 |
| epoch 2 validation loss | 2.3101 |
| epoch 3 train avg loss | 1.7866 |
| epoch 3 validation loss | 1.9519 |
| best validation loss | 1.9519, epoch 3 |
| test avg loss | 1.8150 |
| test min/max loss | 0.4924 / 4.7624 |
| test exact match rate | 0.0000 |
| test contains answer rate | 0.0000 |
| test avg token overlap | 0.3874 |
| test generation coverage | 100/100 samples |
| 이미지별 상태 | test 이미지 5장 모두 `심각` |
| 분석 산출물 | `training_report.md`, `loss_curve.svg`, `evaluation/test_report.md`, `evaluation/test_predictions.jsonl`, `evaluation/dashboard.html` |
| 정성 평가 | 일부 yes/no 질문은 객체 단어를 포함하지만, airplane을 cat/bird/person 등으로 오답 생성하는 사례가 있어 실제 답변 품질은 아직 낮음 |

### 4. 문제점

- 1000 samples이지만 실제 이미지는 51장이라 이미지 다양성이 부족하다.
- template 기반 QA라 사람이 작성한 VQA보다 문장 다양성과 품질이 낮다.
- validation/test는 이미지 누수는 없지만 각각 5장 이미지 기반이라 평가 폭이 좁다.
- 일부 Commons 검색 결과는 라벨과 관련은 있으나 여러 객체가 같이 있을 수 있다.
- test loss는 낮아졌지만 생성 답변이 정답 문장을 그대로 따르지 못하고 객체를 헷갈리는 사례가 있다.

### 5. 개선필요 항목

- best validation checkpoint 저장 구현.
- test set 평가 스크립트를 학습 종료 후 자동 호출할지 결정.
- epoch별 고정 샘플 inference report 저장.
- 이미지 수를 51장에서 최소 수백 장 단위로 확장.
- 사람이 검수한 high-quality validation/test set 별도 구축.
- adapter-only Stage 1 성능 확인 후 4차 시기에서 LoRA Stage 2 진행.

---

## 4차 시기: External VLM 10K 데이터 확장

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Adapter | PerceiverResamplerAdapter |
| Visual token 수 | 16 |
| Adapter hidden dim | 512 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen freeze, adapter-only Stage 1 예정 |
| LoRA | 사용 안 함 |
| 목적 | Wikimedia 1K의 이미지/질문 다양성 부족을 줄이고, 10K 규모에서 adapter alignment 재학습 |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| all samples | 10000 |
| train samples | 8000 |
| validation samples | 1000 |
| test samples | 1000 |
| 전체 이미지 수 | 1795 |
| train 이미지 수 | 754 |
| validation 이미지 수 | 521 |
| test 이미지 수 | 520 |
| image leakage | train/validation/test 교집합 0개 |
| train source | LVIS-Instruct4V 6500, ShareGPT4V 1500 |
| validation source | LVIS-Instruct4V 500, MMBench DEV EN 500 |
| test source | LVIS-Instruct4V 500, MMBench DEV EN 500 |
| 데이터 경로 | `data/external_vlm_10k` |
| config | `configs/dinov3-local-vits16-qwen-external-10k-adapter-stage1.json` |

### 3. 성능

| 지표 | 값 |
|------|----|
| train loss | 미측정 |
| validation loss | 미측정 |
| test loss | 미측정 |
| exact/object accuracy | 미측정 |
| dry-run | config 로드 성공 |
| 정성 평가 | 학습 전 |

### 4. 문제점

- ShareGPT4V/LVIS 원본 라이선스와 사용 조건을 확인해야 하며, 재배포 범위를 제한해야 한다.
- MMBench는 평가용 벤치마크라 train에는 넣지 않았지만, validation/test에 포함되므로 지표 해석 시 benchmark 성격을 분리해서 봐야 한다.
- MME는 이번 자동 빌더에 아직 포함하지 못했고, 공식 다운로드/라이선스 확인 후 별도 importer가 필요하다.
- 질문 prompt variant로 sample 수를 늘렸기 때문에 sample 수 증가가 이미지 수 증가와 1:1로 대응하지 않는다.

### 5. 개선필요 항목

- best checkpoint 저장.
- External 10K Stage 1 학습 실행.
- test evaluation CLI로 `evaluation/dashboard.html` 생성.
- MMBench 전용 multiple-choice accuracy 계산.
- MME importer 추가.
- label/object accuracy 계산.
- inference sample report 자동 저장.
- 데이터 품질 필터 강화.
