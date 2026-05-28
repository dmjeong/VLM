# 리셋 로그

## 2026-05-27: v1 폐기와 v2 재시작

| 항목 | 내용 |
|------|------|
| 결정 | 기존 Agent/RAG/VLM 튜닝 프로토타입을 폐기하고 v2를 처음부터 작성 |
| 이유 | SaaS UI, Agent, RAG, 튜닝, Admin을 한 번에 진행해 핵심인 실제 모델 연결이 흐려짐 |
| 백업 | `/Users/jeongdongmin/VLM/_archive/VLM-v1-reset-20260527-201209` |
| v2 원칙 | 실제 모델 endpoint가 없으면 답변하지 않음 |
| 첫 목표 | `Web -> API -> Hugging Face 호환 endpoint -> 답변` |

## 2026-05-28: Web/API 계획 폐기와 DINOv3 Mini VLM 전환

| 항목 | 내용 |
|------|------|
| 결정 | `real-model-chat-platform` 계획/설계를 삭제하고 `dinov3-mini-vlm` 계획으로 전환 |
| 이유 | 사용자의 현재 목표가 Web/API 서비스가 아니라 DINOv3 vision encoder와 LLM을 직접 연결하는 VLM 구현으로 바뀜 |
| 삭제 | `docs/01-plan/features/real-model-chat-platform.plan.md`, `docs/02-design/features/real-model-chat-platform.design.md` |
| 새 목표 | 이미지 -> DINOv3 -> visual adapter -> LLM embedding -> LLM -> 답변 생성 |
| 범위 제한 | Web, API, SaaS, RAG, Admin 제외 |

## 2026-05-28: DINOv3 Mini VLM 설계 작성

| 항목 | 내용 |
|------|------|
| 결정 | `dinov3-mini-vlm` 설계 문서 작성 |
| 핵심 설계 | DINOv3 patch feature를 MLP adapter로 LLM embedding space에 투영하고, text embedding 앞에 visual token을 붙여 causal LM loss로 학습 |
| 1차 구현 | MLP visual adapter baseline, fake module 기반 shape/forward test, tiny dataset overfit |
| 2차 구현 | Q-Former 또는 Perceiver Resampler, LLM LoRA, visual instruction tuning |
| 제외 | Web, API, RAG, Admin |
