# DINOv3 Mini VLM

> 목표: DINOv3 vision encoder와 작은 LLM을 직접 연결해 mini Vision-Language Model을 만드는 것이 목표

## 현재 방향

```text
이미지
  -> DINOv3 vision encoder
  -> Q-former
  -> LLM embedding space
  -> LLM
  -> 답변 생성
```

## 검증

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
| 4차 시기 | External VLM 10K 데이터셋 구축 | 완료 | `data/external_vlm_10k` |
| 5차 시기 | External VLM 10K 1 epoch 학습과 test 평가 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1` |
| 6차 시기 | 512 lightweight Q-Former adapter 구조 도입과 smoke 검증 | 폐기 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-smoke` |
| 7차 시기 | DistilBERT 초기화 Q-Former + ITC 사전정렬 + LLM 연결 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke` |
| 8차 시기 | Q-Former 실험을 BLIP-2 스타일 768 경로로 단일화 | 완료 | `configs/*qformer-distilbert*.json` |
| 9차 시기 | External 10K 768 Q-Former ITC + LLM Stage 1 학습/평가 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1` |
| 10차 시기 | DINOv3 유지 필요성 검증을 위한 CLIP/SigLIP 비전 백본 비교 | 완료 | `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md` |
| 11차 시기 | 오래된 smoke/폐기 실험 artifact 아카이브 정리 | 완료 | `artifacts/dinov3-mini-vlm/_archive/legacy-20260528/manifest.json` |
| 12차 시기 | DINOv3 best adapter에서 Qwen LoRA Stage 2 학습/평가 | 완료 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1` |

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

---

## 5차 시기: External VLM 10K 1 Epoch 학습과 Test 평가

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Adapter | PerceiverResamplerAdapter |
| Visual token 수 | 16 |
| Adapter hidden dim | 512 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen freeze, adapter-only Stage 1 |
| LoRA | 사용 안 함 |
| device | MPS |
| config | `configs/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1.json` |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 8000 |
| validation samples | 1000 |
| test samples | 1000 |
| epoch | 1 |
| batch size | 1 |
| gradient accumulation | 8 |
| optimizer step | 1000 |
| generation 평가 샘플 | test 100 samples |

### 3. 성능

| 지표 | 값 |
|------|----|
| epoch 1 first train loss | 8.1433 |
| epoch 1 last train loss | 2.2436 |
| epoch 1 train avg loss | 2.7176 |
| epoch 1 train loss decrease | 5.8998 |
| epoch 1 min/max train loss | 0.4729 / 14.3447 |
| validation loss | 2.4231 |
| test avg loss | 2.4345 |
| test min/max loss | 0.2444 / 5.9233 |
| test exact match rate | 0.0000 |
| test contains answer rate | 0.0000 |
| test avg token overlap | 0.4998 |
| 생성 평가 coverage | 100/1000 test samples |
| 단위 테스트 | 53개 통과, 3개 skip |

### 4. 문제점

- 최초 3 epoch 설정으로 학습했을 때 train epoch 1은 정상 종료했으나 validation에서 `loss=nan`이 발생했다.
- 원인은 MMBench 계열 긴 question/options 샘플이었다. 기존 collator가 `prompt_ids + answer_ids`를 앞에서부터 `max_text_length=192`로 잘라서, 일부 validation/test 샘플의 answer label이 전부 잘렸다.
- label이 전부 `-100`인 batch는 supervised token이 0개라 causal LM loss가 NaN이 될 수 있다.
- 수정 후 train/validation/test 전체를 다시 스캔했고 `bad_all_ignored=0`으로 확인했다.
- 1 epoch 학습 후 loss는 안정적으로 내려갔지만 생성 답변은 아직 이미지 내용을 안정적으로 맞히지 못한다. 예: bathroom 이미지를 umbrella/person scene으로 설명하는 오답이 있었다.
- exact match와 contains answer는 0.0이라, 현재 모델은 정답 문장 형식을 그대로 재현하지 못한다.

### 5. 개선필요 항목

- 평가 스크립트에 loss/generation 진행률 로그를 추가한다.
- generation prompt를 더 짧고 엄격하게 만들고, stop rule을 보강한다.
- `max_text_length`를 256 또는 384로 늘린 실험을 비교한다.
- 3 epoch 전체 학습을 다시 실행해 validation/test loss 하강이 계속되는지 확인한다.
- 생성 품질이 낮으므로 adapter-only Stage 1을 더 학습한 뒤 LoRA Stage 2를 진행한다.
- MMBench는 multiple-choice accuracy를 별도로 계산하고, LVIS/ShareGPT4V caption QA와 지표를 분리한다.
- best validation checkpoint 저장과 epoch별 sample inference report를 추가한다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| 최종 visual adapter | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/visual_adapter.pt` |
| epoch checkpoint | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/visual_adapter_epoch_1.pt` |
| 학습 metrics | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/metrics.jsonl` |
| 학습 summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/training_summary.json` |
| 학습 리포트 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/training_report.md` |
| loss 그래프 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/loss_curve.svg` |
| test summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/evaluation/test_summary.json` |
| test predictions | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/evaluation/test_predictions.jsonl` |
| test 리포트 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/evaluation/test_report.md` |
| 평가 대시보드 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1/evaluation/dashboard.html` |

---

## 6차 시기: Q-Former Adapter 구조 도입과 Smoke 검증

> 상태: 폐기. 이 시기는 512 hidden lightweight Q-Former가 실제로 동작하는지 확인한 기록으로만 유지한다.
> 이후 Q-Former 실험은 DistilBERT hidden size와 맞춘 768 BLIP-2 스타일 경로만 사용한다.

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Adapter | QFormerVisualAdapter |
| Q-Former 내부 구조 | learnable query token + query self-attention + image cross-attention + FFN |
| Visual token 수 | 16 |
| Q-Former hidden dim | 512 |
| LLM | Qwen/Qwen3-0.6B |
| 학습 정책 | DINOv3 freeze, Qwen freeze, Q-Former adapter만 학습 |
| LoRA | 사용 안 함 |
| 구현 범위 | DistilBERT 초기화 전 단계의 lightweight Q-Former block |
| smoke config | `configs/dinov3-local-vits16-qwen-qformer-smoke.json` |
| 10K 실행 config | 삭제됨. 768 DistilBERT ITC 경로로 대체 |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| smoke train samples | 17 |
| smoke validation/test samples | 6 |
| epoch | 1 |
| batch size | 1 |
| optimizer step | 17 |
| trainable parameters | 9,142,528 |
| total parameters | 626,793,600 |

### 3. 성능

| 지표 | 값 |
|------|----|
| smoke first train loss | 6.6263 |
| smoke last train loss | 4.5765 |
| smoke train avg loss | 5.4124 |
| smoke validation loss | 5.2324 |
| smoke test avg loss | 5.2324 |
| smoke generation samples | 3 |
| smoke exact match rate | 0.0000 |
| smoke contains answer rate | 0.0000 |
| smoke avg token overlap | 0.3889 |
| 단위 테스트 | 56개 통과, 3개 skip |

### 4. 문제점

- Q-Former 구조는 정상 동작하지만 smoke 데이터 17개/1 epoch만으로는 생성 품질이 좋아지지 않는다.
- Q-Former가 DistilBERT weight로 초기화된 것이 아니라 random query와 random Q-Former block에서 시작한다.
- BLIP-2식 ITC/ITM/ITG pretraining이 아직 없어, Q-Former가 이미지-텍스트 정렬을 충분히 배우지 못한 상태다.
- Q-Former는 Perceiver보다 trainable parameter가 많아 10K 학습 시간과 overfit 가능성을 같이 봐야 한다.
- 512 hidden 경로는 DistilBERT weight를 직접 복사할 수 없어 BLIP-2 스타일 목표와 맞지 않는다.

### 5. 개선필요 항목

- 512 lightweight Q-Former config는 사용하지 않는다.
- External 10K Q-Former Stage 1은 768 DistilBERT ITC 경로로 실행한다.
- Q-Former output을 LLM에 넣기 전에 MLP projector depth와 normalization 조합을 비교한다.
- Q-Former Stage 1이 안정화되면 Qwen LoRA Stage 2를 켠다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| Q-Former adapter 구현 | `mini_vlm/models/visual_adapter.py` |
| builder 연결 | `mini_vlm/models/builder.py` |
| smoke config | 삭제됨. `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json`로 대체 |
| 10K Q-Former config | 삭제됨. `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json`로 대체 |
| smoke checkpoint | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-smoke/visual_adapter.pt` |
| smoke 학습 리포트 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-smoke/training_report.md` |
| smoke 평가 리포트 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-smoke/evaluation/test_report.md` |
| smoke 평가 대시보드 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-smoke/evaluation/dashboard.html` |

---

## 7차 시기: DistilBERT 초기화 Q-Former + ITC 사전정렬 + LLM 연결

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Q-Former | `adapter_hidden_dim=768`, `adapter_layer_count=2` |
| 초기화 | DistilBERT self-attention/FFN weight를 Q-Former query block에 복사 |
| 새로 학습되는 부분 | learnable query token, vision projection, image cross-attention, LLM projector |
| 사전정렬 objective | ITC, image-text contrastive loss |
| Text encoder | `distilbert-base-uncased`, frozen |
| LLM 연결 | ITC로 학습한 `visual_adapter.pt`를 `init_visual_adapter`로 로드 후 Qwen answer loss 학습 |
| LLM | Qwen/Qwen3-0.6B, frozen |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| ITC smoke train samples | 17 |
| ITC smoke validation samples | 6 |
| ITC batch size | 2 |
| ITC 학습 batch | 8 |
| ITC skipped singleton batch | 1 |
| LLM smoke train samples | 17 |
| LLM smoke validation/test samples | 6 |
| LLM batch size | 1 |

### 3. 성능

| 지표 | 값 |
|------|----|
| ITC train avg loss | 0.7077 |
| ITC validation loss | 0.6253 |
| ITC first/last loss | 0.7221 / 0.7074 |
| LLM train avg loss | 9.8322 |
| LLM validation loss | 7.9130 |
| LLM test avg loss | 7.9130 |
| LLM generation samples | 3 |
| LLM exact match rate | 0.0000 |
| LLM contains answer rate | 0.0000 |
| LLM avg token overlap | 0.1667 |
| 단위 테스트 | 72개 통과, 3개 skip |

### 4. 문제점

- 17개 smoke 데이터에서는 DistilBERT 초기화 + ITC가 LLM answer loss를 바로 낮추지는 못했다.
- ITC batch size가 2라 negative 수가 너무 적고, 마지막 singleton batch는 contrastive 학습에 쓸 수 없어 skip된다.
- DistilBERT 초기화 Q-Former는 512 hidden lightweight Q-Former보다 파라미터가 크지만, BLIP-2 스타일을 목표로 하므로 768 경로로 단일화한다.
- `.gitignore`의 `models/` 패턴이 `mini_vlm/models/` 코드 폴더까지 무시하던 문제가 있었다. 루트 weight 폴더만 무시하도록 `/models/`로 수정했다.
- 아직 ITM/ITG objective는 구현하지 않았다. 현재 완성된 경로는 DistilBERT 초기화 + ITC + LLM answer tuning까지다.

### 5. 개선필요 항목

- External 10K에서 `qformer-distilbert-itc-stage0`를 먼저 돌리고, 이어 `qformer-distilbert-itc-to-llm-stage1`을 실행한다.
- ITC batch size를 가능한 크게 잡아 negative 수를 늘린다.
- ITC 이후 Q-Former의 retrieval accuracy, top-k matching accuracy를 추가한다.
- ITM objective를 추가해 hard negative image-text matching을 학습한다.
- ITG objective 또는 Qwen LoRA Stage 2를 추가해 생성 품질을 개선한다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| ITC 학습 코드 | `mini_vlm/training/pretrain_qformer_itc.py` |
| Q-Former ITC 모델 | `mini_vlm/models/qformer.py` |
| DistilBERT ITC smoke config | `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json` |
| DistilBERT ITC -> LLM smoke config | `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke.json` |
| 10K DistilBERT ITC config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |
| 10K DistilBERT ITC -> LLM config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json` |
| ITC checkpoint | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke/qformer_itc.pt` |
| ITC visual adapter | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke/visual_adapter.pt` |
| LLM 연결 checkpoint | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke/visual_adapter.pt` |

---

## 8차 시기: Q-Former 실험을 BLIP-2 스타일 768 경로로 단일화

### 1. 결정

| 항목 | 내용 |
|------|------|
| 폐기 대상 | 512 hidden lightweight Q-Former config |
| 유지 대상 | DistilBERT hidden size 768과 맞춘 Q-Former config |
| 이유 | DistilBERT attention/FFN weight는 768 기준이라 Q-Former hidden dim이 512이면 직접 초기화할 수 없음 |
| 기준 구조 | DINOv3 384 -> Q-Former 768 -> Qwen embedding projector |
| 테스트 정책 | `configs/*qformer*.json`는 모두 `adapter_hidden_dim=768`이어야 함 |

### 2. 삭제한 config

| 삭제 파일 |
|-----------|
| `configs/dinov3-local-vits16-qwen-qformer-smoke.json` |
| `configs/dinov3-local-vits16-qwen-external-10k-qformer-stage1-epoch1.json` |
| `configs/dinov3-local-vits16-qwen-qformer-itc-smoke.json` |
| `configs/dinov3-local-vits16-qwen-qformer-itc-to-llm-smoke.json` |
| `configs/dinov3-local-vits16-qwen-external-10k-qformer-itc-stage0.json` |
| `configs/dinov3-local-vits16-qwen-external-10k-qformer-itc-to-llm-stage1-epoch1.json` |

### 3. 유지한 active config

| 목적 | 경로 |
|------|------|
| smoke ITC | `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json` |
| smoke ITC -> LLM | `configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke.json` |
| 10K ITC | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |
| 10K ITC -> LLM | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json` |

### 4. 개선필요 항목

- 10K ITC를 768 경로로 실행한다.
- 이어서 10K ITC -> LLM Stage 1을 실행한다.
- 768 Q-Former에서 메모리/속도 병목이 생기면 batch size, gradient accumulation, sequence length를 조정한다.

---

## 9차 시기: External 10K 768 Q-Former ITC + LLM Stage 1 학습/평가

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight |
| Q-Former | `adapter_hidden_dim=768`, `adapter_layer_count=2` |
| 초기화 | DistilBERT self-attention/FFN weight를 Q-Former query block에 복사 |
| Stage 0 objective | ITC, image-text contrastive loss |
| Stage 0 text encoder | `distilbert-base-uncased`, frozen |
| Stage 1 objective | Qwen answer token causal LM loss |
| LLM | `Qwen/Qwen3-0.6B`, frozen |
| Visual token 수 | 16 |
| LoRA | 사용 안 함 |
| device | MPS |
| Stage 0 config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |
| Stage 1 config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json` |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 8000 |
| validation samples | 1000 |
| test samples | 1000 |
| Stage 0 epoch | 1 |
| Stage 0 batch size | 4 |
| Stage 0 batch count | 2000 |
| Stage 0 trainable parameters | 20,463,617 |
| Stage 1 epoch | 1 |
| Stage 1 batch size | 1 |
| Stage 1 gradient accumulation | 8 |
| Stage 1 optimizer step | 1000 |
| Stage 1 trainable parameters | 20,004,352 / 637,655,424 |
| generation 평가 샘플 | test 100 samples |
| generation sampling | `even` |
| generation 고유 이미지 수 | 69 |

### 3. 성능

| 지표 | 값 |
|------|----|
| Stage 0 ITC first train loss | 1.4785 |
| Stage 0 ITC last train loss | 0.1229 |
| Stage 0 ITC train avg loss | 0.5107 |
| Stage 0 ITC validation loss | 1.2883 |
| Stage 1 first train loss | 3.3832 |
| Stage 1 last train loss | 1.9864 |
| Stage 1 train avg loss | 2.4225 |
| Stage 1 validation loss | 1.5950 |
| test avg loss | 1.5686 |
| test min/max loss | 0.1637 / 5.6749 |
| test exact match rate | 0.0100 |
| test contains answer rate | 0.0900 |
| test avg token overlap | 0.4545 |
| MMBench generation letter accuracy | 0.3200, 50개 생성 샘플 기준 |
| 평가 helper 테스트 | `unittest` 9개 통과 |

### 4. 문제점

- 학습 자체는 정상이다. Stage 0과 Stage 1 모두 NaN 없이 완료됐고, Stage 1 validation/test loss는 기존 Perceiver 1 epoch 결과보다 낮다.
- Stage 0 ITC train loss는 크게 내려갔지만 validation loss가 1.2883으로 높다. validation/test에는 LVIS 외 MMBench가 섞여 있어 train보다 분포가 어렵다.
- 생성 품질은 아직 서비스 가능 수준이 아니다. bathroom 이미지를 사람 장면으로 설명하거나, multiple-choice에서 정답 letter를 틀리는 사례가 있다.
- `exact_match`는 매우 엄격하고, `contains_answer`는 정답 문장 전체 substring 기준이라 의미상 가까운 답도 낮게 잡힌다. 예: `A. spring` 정답에 대해 `A ... spring`을 생성해도 contains가 false가 될 수 있다.
- 최초 평가 생성 샘플은 test 앞 100개만 사용해 이미지 4장에 몰렸다. 평가 대표성이 낮아 `generation_sampling=even` 옵션을 추가하고 재평가했다.
- 대시보드 상단의 `가장 취약한 이미지` 요약이 5개만 보여 전체 테스트 이미지가 5개처럼 오해될 수 있었다. 제목을 `가장 취약한 이미지 Top 5`로 바꾸고, `전체 테스트 이미지` 섹션과 `69 / 69 images` 표시를 추가했다.
- External 10K test split에는 `metadata.object`가 없어 기존 대시보드가 전부 `unknown`과 object hit 0%로 표시했다. LVIS는 task, MMBench는 category로 표시하고 MMBench는 choice letter 정답률을 별도로 표시하도록 수정했다.

### 5. 개선필요 항목

- 768 Q-Former Stage 1을 3 epoch 이상으로 늘려 validation/test loss 추이를 확인한다.
- Stage 1이 안정화되면 Qwen LoRA Stage 2를 켜고, language response style을 함께 맞춘다.
- ITM objective를 추가해 image-text matching을 직접 학습한다.
- ITG objective 또는 caption generation pretraining을 추가해 Q-Former가 생성형 답변에 더 잘 연결되게 한다.
- MMBench는 exact/contains 대신 option letter accuracy를 공식 지표로 분리한다.
- 생성 평가는 `even` 100개와 별도로, 최종 후보 모델에서는 1000개 전체 generation 평가를 실행한다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| Stage 0 ITC visual adapter | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0/visual_adapter.pt` |
| Stage 0 ITC summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0/itc_summary.json` |
| Stage 0 ITC metrics | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0/itc_metrics.jsonl` |
| Stage 1 visual adapter | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/visual_adapter.pt` |
| Stage 1 training summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/training_summary.json` |
| Stage 1 loss graph | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/loss_curve.svg` |
| test summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/test_summary.json` |
| test predictions | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/test_predictions.jsonl` |
| test report | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/test_report.md` |
| 평가 대시보드 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/dashboard.html`, 전체 이미지 카드 69개, `unknown_count=0`, MMBench choice metric 표시 |

---

## 10차 시기: DINOv3/CLIP/SigLIP Vision Encoder Ablation

> 범위: 사용자가 요청한 “3번까지만” 실행. 즉, DINO epoch 확장 확인, CLIP/SigLIP 비전 백본 비교, test/dashbord/비교표 생성까지 수행했다.

### 1. 모델 구조

| 항목 | DINOv3 경로 | CLIP 경로 | SigLIP 경로 |
|------|-------------|-----------|-------------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight | `openai/clip-vit-base-patch16` | `google/siglip-base-patch16-224` |
| Vision pretraining 성격 | self-supervised image representation | image-text contrastive | sigmoid image-text contrastive |
| Q-Former | DistilBERT 초기화, hidden 768, 2 layers | 동일 | 동일 |
| Stage 0 objective | ITC | ITC | ITC |
| Stage 1 objective | Qwen answer token causal LM loss | 동일 | 동일 |
| LLM | `Qwen/Qwen3-0.6B`, frozen | 동일 | 동일 |
| LoRA | 사용 안 함 | 사용 안 함 | 사용 안 함 |
| 학습 정책 | vision/LLM freeze, Q-Former adapter만 학습 | 동일 | 동일 |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 8000 |
| validation samples | 1000 |
| test samples | 1000 |
| Stage 0 epoch | 1 |
| Stage 0 batch size | 4 |
| Stage 1 epoch | DINO는 epoch3 시도 후 epoch2 checkpoint 평가, CLIP/SigLIP는 epoch1 |
| Stage 1 batch size | 1 |
| Stage 1 gradient accumulation | 8 |
| generation 평가 샘플 | test 100 samples |
| generation sampling | `even` |
| generation 고유 이미지 수 | 69 |

### 3. 성능

| 실험 | ITC val loss | Stage 1 train loss | Stage 1 val loss | test loss | MMBench choice acc | generation success | overlap |
|------|--------------|--------------------|------------------|-----------|--------------------|--------------------|---------|
| DINOv3 ViT-S/16 1 epoch | 1.2883 | 2.4225 | 1.5950 | 1.5686 | 32.0%, 50개 기준 | 22.0%, 100개 기준 | 45.4% |
| DINOv3 ViT-S/16 epoch2 checkpoint | 1.2883 | 2.0133 | 1.4499 | 1.4157 | 36.0%, 50개 기준 | 22.0%, 100개 기준 | 42.0% |
| CLIP ViT-B/16 1 epoch | 1.2847 | 2.3083 | 1.8313 | 1.8066 | 34.0%, 50개 기준 | 20.0%, 100개 기준 | 43.6% |
| SigLIP ViT-B/16 1 epoch | 1.2727 | 2.3045 | 1.6249 | 1.6055 | 38.0%, 50개 기준 | 8.0%, 100개 기준 | 30.5% |

### 4. 문제점

- DINO epoch3 전체 완료는 실패했다. epoch2까지는 정상 하강했지만 epoch3 첫 batch에서 gradient norm이 NaN이 되어 중단됐다.
- DINO epoch3 config의 test 평가는 최종 `visual_adapter.pt`가 아니라 `visual_adapter_epoch_2.pt` fallback checkpoint로 수행했다.
- CLIP/SigLIP는 Stage 1 한 epoch 동안 NaN 없이 완료됐지만, test loss와 generation success 기준으로 DINO epoch2를 넘지는 못했다.
- SigLIP는 MMBench choice accuracy가 38.0%로 가장 높았지만, exact/contains/overlap 기반 generation success는 8.0%로 가장 낮았다.
- CLIP은 exact/contains rate가 DINO/SigLIP보다 높지만 test loss와 validation loss가 더 높아, 현재 구조에서는 확실한 기본값으로 보기 어렵다.
- 모든 후보가 아직 서비스 가능한 VLM 답변 품질은 아니다. Q-Former adapter-only + frozen Qwen만으로는 이미지 grounding과 답변 형식 정렬이 부족하다.

### 5. DINOv3 유지 판단

- DINOv3를 “반드시 기본 백본으로 유지해야 하는” 수치적 근거는 아직 약하다.
- 다만 이번 비교에서는 DINO epoch2 checkpoint가 test loss 1.4157로 가장 낮았고 generation success도 22.0%로 공동 최고라, DINO를 바로 버릴 근거도 없다.
- DINO의 장점은 self-supervised image representation이다. 라벨/캡션이 부족한 도메인 이미지에서 추가 self-supervised pretraining, dense feature, segmentation/retrieval/검출 보조 특징을 활용하기 좋다.
- CLIP/SigLIP의 장점은 이미 image-text alignment가 되어 있다는 점이다. 일반 VQA/chat 서비스의 빠른 품질 확보에는 보통 더 실용적인 출발점이다.
- 현재 결정: 서비스 기본 후보는 DINO와 SigLIP를 둘 다 유지하고, 다음 실험에서 LoRA/ITM/ITG를 붙인 뒤 최종 선택한다. DINO는 self-supervised 도메인 적응 연구 브랜치로 계속 보존한다.

### 6. 개선필요 항목

- DINO epoch3 NaN 원인 샘플과 gradient 폭주 지점을 분리해 learning rate, gradient clipping, batch 순서 재현성을 재점검한다.
- CLIP/SigLIP에도 epoch2 이상을 적용해, DINO epoch2와 같은 학습량 조건으로 비교한다.
- Qwen LoRA Stage 2를 추가해 frozen LLM 한계를 줄인다.
- ITM objective를 추가해 image-text matching을 직접 학습한다.
- ITG 또는 caption generation pretraining을 추가해 Q-Former output이 생성형 답변에 더 잘 연결되게 한다.
- 최종 후보는 100개 generation sample이 아니라 test 1000개 전체 generation 평가로 비교한다.

### 7. 산출물

| 산출물 | 경로 |
|--------|------|
| DINO epoch3 시도 config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch3.json` |
| CLIP Stage 0 config | `configs/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |
| CLIP Stage 1 config | `configs/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json` |
| SigLIP Stage 0 config | `configs/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0.json` |
| SigLIP Stage 1 config | `configs/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json` |
| 비교 JSON | `artifacts/dinov3-mini-vlm/vision-ablation/comparison.json` |
| 비교 리포트 | `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md` |
| 통합 비교 대시보드 | `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |
| CLIP 평가 대시보드 | `artifacts/dinov3-mini-vlm/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/dashboard.html` |
| SigLIP 평가 대시보드 | `artifacts/dinov3-mini-vlm/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1/evaluation/dashboard.html` |

### 8. 후속 자동화

- `compare_vision_ablation` 실행 시 각 실험의 기존 `test_summary.json`과 `test_predictions.jsonl`을 기준으로 개별 `evaluation/dashboard.html`을 다시 생성한다.
- `comparison.md`에는 통합 대시보드 링크와 각 실험별 대시보드 링크를 붙인다.
- `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html`에는 전체 비교표와 best test loss, best choice accuracy, best generation success 요약을 표시한다.
- 대시보드 갱신 상태는 `comparison.md`와 `comparison.json`에 함께 기록한다.

---

## 11차 시기: Artifact 폴더 정리

### 1. 정리 기준

| 분류 | 기준 |
|------|------|
| 유지 | 현재 DINO/CLIP/SigLIP 비교에 직접 쓰는 Stage 0, Stage 1, evaluation dashboard |
| 유지 | `vision-ablation` 비교 리포트와 통합 대시보드 |
| 아카이브 | smoke, one-sample, progress 출력, 폐기된 512 Q-Former, 초기 Perceiver/Wikimedia 1K 산출물 |
| 삭제 | 수행하지 않음 |

### 2. 유지한 루트 artifact

| 경로 |
|------|
| `artifacts/dinov3-mini-vlm/vision-ablation` |
| `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0` |
| `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1` |
| `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch3` |
| `artifacts/dinov3-mini-vlm/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0` |
| `artifacts/dinov3-mini-vlm/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1` |
| `artifacts/dinov3-mini-vlm/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0` |
| `artifacts/dinov3-mini-vlm/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1` |

### 3. 아카이브 결과

| 항목 | 값 |
|------|----|
| archive path | `artifacts/dinov3-mini-vlm/_archive/legacy-20260528` |
| manifest | `artifacts/dinov3-mini-vlm/_archive/legacy-20260528/manifest.json` |
| 이동한 폴더 수 | 15 |
| 이동한 총량 | 약 698MB |
| 복구 방법 | manifest의 `destination` 폴더를 `source` 경로로 다시 이동 |

### 4. 검증

- `compare_vision_ablation` 재실행 완료.
- `vision-ablation/comparison.md`, `vision-ablation/dashboard.html` 재생성 완료.
- 현재 비교 대시보드에 필요한 active artifact는 이동하지 않았다.

### 5. 링크 보정

- 최초 통합 대시보드는 개별 대시보드로 이동할 때 `../.../evaluation/dashboard.html` 상대경로를 사용했다.
- 일부 IDE/HTML preview에서는 부모 폴더로 이동하는 링크가 제대로 열리지 않을 수 있어, 각 실험 대시보드를 `artifacts/dinov3-mini-vlm/vision-ablation/experiments/*/dashboard.html` 아래로 미러링했다.
- `comparison.md`와 `vision-ablation/dashboard.html`의 `열기` 링크는 이제 `experiments/.../dashboard.html` 내부 경로를 사용한다.
- 미러링된 대시보드의 `dashboard_assets`도 함께 복사하므로 이미지 로딩도 유지된다.

---

## 12차 시기: DINOv3 + Qwen LoRA Stage 2

### 1. 모델 구조

| 항목 | 내용 |
|------|------|
| Vision encoder | Meta DINOv3 ViT-S/16 local weight, frozen |
| Visual adapter | DistilBERT 초기화 Q-Former, hidden 768, 2 layers |
| Adapter 초기값 | DINO Stage1 epoch2 checkpoint `visual_adapter_epoch_2.pt` |
| LLM | `Qwen/Qwen3-0.6B` |
| LoRA | 사용 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 |
| 학습 대상 | Q-Former visual adapter + Qwen LoRA |
| config | `configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1.json` |

### 2. 학습 수량

| 항목 | 수량 |
|------|------|
| train samples | 8000 |
| validation samples | 1000 |
| test samples | 1000 |
| epoch | 1 |
| batch size | 1 |
| gradient accumulation | 8 |
| optimizer step | 1000 |
| trainable parameters | 22,298,112 / 639,949,184 |
| learning rate | 0.000005 |
| max grad norm | 0.5 |

### 3. 성능

| 지표 | DINO epoch2 checkpoint | DINO + Qwen LoRA Stage2 |
|------|------------------------|--------------------------|
| train avg loss | 2.0133 | 1.9029 |
| validation loss | 1.4499 | 1.3150 |
| test avg loss | 1.4157 | 1.2866 |
| exact match rate | 0.0100 | 0.0600 |
| contains answer rate | 0.0600 | 0.1000 |
| avg token overlap | 0.4203 | 0.4341 |
| MMBench choice accuracy | 36.0%, 50개 기준 | 36.0%, 50개 기준 |
| generation success | 기존 기준 22.0%, 13차 answer match 기준 24.0% | 기존 기준 21.0%, 13차 answer match 기준 28.0% |

### 4. 문제점

- Loss와 exact/contains 지표는 좋아졌지만, 생성 답변의 시각 grounding은 아직 낮다.
- 예시: bathroom 이미지를 사람과 강아지 장면으로 설명하거나, bus 이미지를 사람 장면으로 설명하는 사례가 남아 있다.
- MMBench choice accuracy는 36.0%로 DINO epoch2와 동일해서, LoRA 1 epoch만으로 선택지 추론 능력이 개선되지는 않았다.
- 12차 당시 generation success는 dashboard의 기존 복합 기준상 22.0%에서 21.0%로 소폭 하락했다. 이후 13차에서 평가 기준을 고쳐 재산정하니 DINO epoch2는 24.0%, LoRA Stage2는 28.0%로 계산됐다.

### 5. 개선필요 항목

- LoRA learning rate와 학습 범위를 분리한다. 다음 실험은 Q-Former를 freeze하고 Qwen LoRA만 학습하는 조건을 비교한다.
- SigLIP 백본에도 같은 Qwen LoRA Stage 2를 적용해, image-text pretrained vision encoder와 LoRA 조합을 비교한다.
- ITM/ITG 또는 caption generation pretraining을 추가해 Q-Former가 생성 답변에 필요한 시각 정보를 더 잘 넘기게 한다.
- 현재 100개 generation sample 대신 최종 후보는 test 1000개 전체 generation 평가를 실행한다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| Stage2 visual adapter | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/visual_adapter.pt` |
| Stage2 Qwen LoRA | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/llm_lora/adapter_model.safetensors` |
| training summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/training_summary.json` |
| test summary | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/evaluation/test_summary.json` |
| test predictions | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/evaluation/test_predictions.jsonl` |
| 평가 대시보드 | `artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1/evaluation/dashboard.html` |
| 통합 비교 대시보드 | `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |

---

## 13차 시기: 평가 대시보드 answer match 기준 보정

### 1. 변경 배경

- 사용자가 대시보드에서 정답으로 보이는 답변도 `주의` 또는 실패처럼 보인다고 지적했다.
- 예시: 정답이 “사람이나 동물이 없다”이고 생성이 “No, there are no humans or animals...”인 경우, 기존 대시보드는 exact/contains/token overlap 중심이라 실패로 표시했다.
- 반대로 객관식은 letter만 맞으면 생성 설명이 정답 텍스트와 충돌해도 양호로 보일 수 있었다.

### 2. 변경한 평가 구조

| 항목 | 기존 | 변경 |
|------|------|------|
| 의미 정답 기준 | exact, contains, object, choice, 높은 overlap 혼합 | `answer_match` 별도 필드 추가 |
| yes/no | 문장 일치나 overlap에 의존 | 질문 극성과 답변 극성을 비교 |
| counting | overlap에 의존 | 숫자 추출 후 expected/generated 숫자 일치 확인 |
| 짧은 사실형 답변 | overlap만으로 통과 가능 | 핵심 token 누락 시 실패 |
| 객관식 | letter 정답이면 양호 | letter 정답은 인정하되 설명 충돌 시 `choice text warning` |
| 그룹 상태 | contains/overlap 중심 | `answer_match_rate` 중심 |

### 3. 성능지표 재산정

| 실험 | 기존 success | 보정 후 answer match |
|------|--------------|----------------------|
| DINOv3 ViT-S/16 1 epoch | 22.0% | 25.0% |
| DINOv3 ViT-S/16 epoch2 checkpoint | 22.0% | 24.0% |
| DINOv3 ViT-S/16 + Qwen LoRA Stage2 | 21.0% | 28.0% |
| CLIP ViT-B/16 1 epoch | 20.0% | 25.0% |
| SigLIP ViT-B/16 1 epoch | 8.0% | 25.0% |

### 4. 검증

- “No, there are no humans or animals...”는 `answer yes`, `reason yes-no`로 표시된다.
- “four magnets” 정답에 “two magnets”로 답한 경우는 overlap 75%여도 count mismatch로 실패한다.
- “Volvo” 정답에 “Tesla Model 3”로 답한 경우는 overlap 80%여도 short-fact mismatch로 실패한다.
- MMBench에서 letter는 맞지만 설명이 다른 선택지 텍스트를 말하면 `choice text warning`으로 표시한다.
- 전체 단위 테스트 `88개 통과, 3개 skip`.

### 5. 산출물

| 산출물 | 경로 |
|--------|------|
| 개별 대시보드 생성 코드 | `mini_vlm/evaluation/dashboard.py` |
| 비교 대시보드 생성 코드 | `mini_vlm/evaluation/compare_vision_ablation.py` |
| 대시보드 테스트 | `tests/test_evaluation_dashboard.py` |
| 통합 비교 JSON | `artifacts/dinov3-mini-vlm/vision-ablation/comparison.json` |
| 통합 비교 Markdown | `artifacts/dinov3-mini-vlm/vision-ablation/comparison.md` |
| 통합 비교 HTML | `artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html` |

---

## 14차 시기: Playwright 자동 렌더 검증

### 1. 변경 배경

- 대시보드 HTML은 정적 파일이라 단위 테스트만으로는 실제 브라우저 렌더링, 이미지 로딩, 모바일 overflow, 링크 동작을 확인하기 어렵다.
- 직전 검증에서 Playwright가 설치되어 있지 않아 자동 렌더 확인을 못 했으므로, 로컬 `.venv`에 Playwright와 Chromium을 설치하고 반복 실행 가능한 검증 스크립트를 추가했다.

### 2. 검증 항목

| 항목 | 확인 내용 |
|------|-----------|
| desktop/mobile 렌더 | 1440x1000, 390x844 viewport에서 각각 HTML 열기 |
| 문서 구조 | title, h1, 핵심 dashboard content 존재 |
| 이미지 로딩 | 모든 `<img>`가 `naturalWidth > 0`인지 확인 |
| 링크 | 통합 대시보드의 `열기` 링크가 실제 파일로 resolve되는지 확인 |
| 검색 필터 | `humans or animals` 검색 시 카드 수가 줄고 0개가 되지 않는지 확인 |
| answer match 표시 | 접힌 상세까지 포함해 `answer yes`, `reason yes-no` 텍스트 존재 확인 |
| 레이아웃 | 전체 문서 horizontal overflow가 4px 이하인지 확인 |
| 산출물 | desktop/mobile screenshot과 `render_check.json` 생성 |

### 3. 발견 및 수정

- 최초 Playwright 검증에서 LoRA 개별 대시보드의 모바일 viewport가 59px 가로 overflow를 냈다.
- 원인은 `structuralized_imagetext_understanding`처럼 긴 category 텍스트가 card grid의 min-content 폭을 키운 것이었다.
- `image-card-header > div { min-width: 0; }`, `h3 { overflow-wrap: anywhere; }`, `img { max-width: 100%; }`를 추가해 모바일 overflow를 제거했다.

### 4. 검증 결과

| 대상 | viewport | 결과 |
|------|----------|------|
| 통합 비교 대시보드 | desktop | PASS |
| 통합 비교 대시보드 | mobile | PASS |
| LoRA Stage2 개별 대시보드 | desktop | PASS |
| LoRA Stage2 개별 대시보드 | mobile | PASS |

### 5. 산출물

| 산출물 | 경로 |
|--------|------|
| 검증 스크립트 | `scripts/maintenance/verify_dashboard_render.py` |
| 검증 리포트 | `artifacts/dinov3-mini-vlm/vision-ablation/render-check/render_check.json` |
| desktop/mobile screenshots | `artifacts/dinov3-mini-vlm/vision-ablation/render-check/*.png` |

---

## 15차 시기: Eval80 90% 목표 반복과 강한 VLM baseline 도입

### 1. 모델 구조

| 실험 | 구조 |
|------|------|
| `dino-qwen-lora-stage2` | DINOv3 ViT-S/16 frozen + DistilBERT Q-Former 768 + Qwen3-0.6B LoRA |
| `smolvlm2-256m` | Hugging Face pretrained `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` |
| `qwen2_5-vl-3b` | Hugging Face pretrained `Qwen/Qwen2.5-VL-3B-Instruct`, standard prompt |
| `qwen2_5-vl-3b-strict` | 같은 3B 모델, task별 strict prompt |
| `mlx-qwen2_5-vl-7b-4bit-strict` | MLX 4bit `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| `mlx-internvl3-8b-4bit-strict` | MLX 4bit `mlx-community/InternVL3-8B-MLX-4bit` |

### 2. 학습/평가 수량

| 항목 | 수량 |
|------|------|
| 고정 평가셋 | Eval80 v1, 80 samples |
| 구성 | LVIS-Instruct-4V 40개 + MMBench 40개 |
| 목표 | 72/80 이상, 90% |
| 실제 추가 학습 | 이번 반복에서는 없음. 강한 pretrained/teacher baseline 평가로 구조 한계 진단 |
| 로컬 실행 환경 | PyTorch MPS + MLX 4bit |

### 3. 성능

| run | 정답 수 | 정확도 | 목표 |
|-----|--------:|-------:|------|
| `mlx-internvl3-8b-4bit-strict` | 64/80 | 80.0% | 미달 |
| `mlx-qwen2_5-vl-7b-4bit-strict` | 63/80 | 78.8% | 미달 |
| `qwen2_5-vl-3b` | 58/80 | 72.5% | 미달 |
| `qwen2_5-vl-3b-strict` | 57/80 | 71.2% | 미달 |
| `smolvlm2-256m` | 41/80 | 51.2% | 미달 |
| `dino-qwen-lora-stage2` | 25/80 | 31.2% | 미달 |
| oracle union | 72/80 | 90.0% | 상한선 기준 달성 |
| 품질 이슈 제외 oracle union | 71/76 | 93.4% | 상한선 기준 달성 |

주의: oracle union은 여러 모델 중 정답을 맞힌 모델을 사후적으로 고른 상한선이다. 자동 selector가 없으므로 실제 서비스 성능으로 보지 않는다.

### 4. 문제점

- 현재 직접 학습한 DINO + Qwen LoRA mini VLM은 Eval80에서 31.2%로, SaaS형 실제 서비스 기준과 거리가 크다.
- 강한 pretrained VLM도 counting에서 약하다. 최고 모델 기준 counting은 10개 중 4개만 맞췄다.
- Eval80 v1에는 라벨 품질 이슈 4개가 있다. 정답이 콜론으로 끝나거나, counting 질문인데 정답에 숫자가 없거나, 모호한 수량 표현이 포함된다.
- MLX InternVL3-8B는 성능은 최고였지만 이미지 prefill이 길어 MMBench 객관식에서 지연이 크다.

### 5. 개선필요 항목

- Eval80 v2를 만들어 라벨 품질 이슈를 교체하고, target 72/80을 다시 고정한다.
- 단일 모델 최고 80%에서 90%로 가려면 모델 selector, 재질문 전략, 또는 더 강한 teacher 모델이 필요하다.
- counting은 별도 object/counting 전문 보조 모델 또는 segmentation/detection 기반 후처리를 붙인다.
- mini VLM 자체는 DINO-only adapter 학습보다 pretrained VLM distillation 또는 SigLIP/CLIP image-text alignment 기반 구조로 재설계해야 한다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| Eval80 데이터 | `data/eval80/test.jsonl` |
| Eval80 manifest | `data/eval80/manifest.json` |
| 품질 감사 리포트 | `data/eval80/quality_report.json` |
| Eval80 리더보드 | `artifacts/dinov3-mini-vlm/eval80/leaderboard.md` |
| Eval80 리더보드 JSON | `artifacts/dinov3-mini-vlm/eval80/leaderboard.json` |
| Qwen2.5-VL-3B 대시보드 | `artifacts/dinov3-mini-vlm/eval80/qwen2_5-vl-3b/dashboard.html` |
| MLX Qwen2.5-VL-7B 대시보드 | `artifacts/dinov3-mini-vlm/eval80/mlx-qwen2_5-vl-7b-4bit-strict/dashboard.html` |
| MLX InternVL3-8B 대시보드 | `artifacts/dinov3-mini-vlm/eval80/mlx-internvl3-8b-4bit-strict/dashboard.html` |
| pretrained VLM 평가 CLI | `scripts/evaluation/evaluate_pretrained_vlm.py` |
| MLX VLM 평가 CLI | `scripts/evaluation/evaluate_mlx_vlm.py` |
| Eval80 비교 CLI | `scripts/evaluation/compare_eval80_runs.py` |

---

## 16차 시기: 데이터셋 정리와 Eval80 v2 생성

### 1. 정리 목적

- 기존 데이터셋이 `samples`, `wikimedia_commons_1k`, `external_vlm_10k`, `eval80`로 나뉘어 있었지만, 어떤 데이터가 학습/검증/최종평가용인지 한눈에 보이지 않았다.
- Eval80 v1에는 15차에서 발견한 blocking 라벨 이슈 4개가 있어 신규 성능 목표용으로 그대로 쓰기 어렵다.
- 기존 실험 재현성은 유지해야 하므로 Eval80 v1은 보존하고, 깨끗한 Eval80 v2를 별도 생성하기로 했다.

### 2. 데이터셋 현황

| 데이터셋 | 용도 | split | 수량 |
|----------|------|-------|------|
| `data/samples` | smoke/test fixture | train / validation | 17 / 6 |
| `data/wikimedia_commons_1k` | 초기 object/caption 실험 | train / validation / test | 800 / 100 / 100 |
| `data/external_vlm_10k` | 주 학습·검증·테스트 | train / validation / test | 8000 / 1000 / 1000 |
| `data/eval80` | 과거 90% 목표용 holdout v1 | test | 80 |
| `data/eval80_v2` | 신규 clean holdout | test | 80 |

### 3. Eval80 v2 생성 결과

| 항목 | Eval80 v1 | Eval80 v2 |
|------|-----------|-----------|
| sample 수 | 80 | 80 |
| source 구성 | LVIS 40 + MMBench 40 | LVIS 40 + MMBench 40 |
| task 구성 | counting 10, VQA 28, spatial 2, MC 40 | counting 8, VQA 30, spatial 2, MC 40 |
| unique image 수 | 60 | 59 |
| 품질 감사 finding | 4 | 0 |

### 4. 변경한 구조

| 변경 | 내용 |
|------|------|
| 품질 감사 모듈화 | `mini_vlm/data/quality.py`에 `audit_sample`, `has_blocking_quality_issue` 추가 |
| Eval80 생성 옵션 | `build_eval80_dataset.py`에 `--name`, `--exclude-quality-report`, `--require-clean-labels` 추가 |
| 인벤토리 생성 | `scripts/data/summarize_datasets.py`로 `data/README.md`, `data/dataset_inventory.json` 자동 생성 |
| 테스트 | `tests/test_dataset_quality.py` 추가 |

### 5. 운영 기준

- 과거 결과 재현은 `data/eval80/test.jsonl`을 사용한다.
- 신규 성능 목표와 재평가는 `data/eval80_v2/test.jsonl`을 우선 사용한다.
- 학습 데이터는 `data/external_vlm_10k/train.jsonl`, 검증은 `validation.jsonl`, 최종 고정 평가는 `eval80_v2/test.jsonl`로 분리한다.
- 대용량 이미지와 external 원본은 `.gitignore` 대상이므로 git에는 작은 JSONL/manifest/품질 리포트만 남긴다.

### 6. 산출물

| 산출물 | 경로 |
|--------|------|
| 데이터셋 인벤토리 문서 | `data/README.md` |
| 데이터셋 인벤토리 JSON | `data/dataset_inventory.json` |
| Eval80 v2 데이터 | `data/eval80_v2/test.jsonl` |
| Eval80 v2 manifest | `data/eval80_v2/manifest.json` |
| Eval80 v2 품질 리포트 | `data/eval80_v2/quality_report.json` |
| 품질 감사 모듈 | `mini_vlm/data/quality.py` |
| 품질 테스트 | `tests/test_dataset_quality.py` |


## 문서

- 계획: `docs/01-plan/features/dinov3-mini-vlm.plan.md`
- 설계: `docs/02-design/features/dinov3-mini-vlm.design.md`
- 구현 추적: `docs/02-design/features/dinov3-mini-vlm.do.md`
