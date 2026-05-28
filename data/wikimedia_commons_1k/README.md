# Wikimedia Commons 1K Mini VLM Dataset

- 생성 시각: 2026-05-28T11:02:29.906988+00:00
- 이미지 수: 51
- 샘플 수: 1000
- split 방식: 이미지 단위 split. 같은 이미지가 train/validation/test에 중복되지 않음
- all: 1000 samples
- train: 800 samples, 40 images
- validation: 100 samples, 5 images
- test: 100 samples, 5 images
- 수집 방식: Wikimedia Commons MediaWiki API `query+imageinfo`
- 필터: JPEG/PNG, 256px 이상, Public domain/CC0/CC BY/CC BY-SA 계열 라이선스 메타데이터
- 라벨 방식: 검색 class label 기반 template caption/VQA
- split manifest: `split_manifest.json`

주의: 이 데이터셋은 실험용 bootstrap 데이터다. validation/test는 이미지 누수는 없지만 이미지 수가 작아 평가 폭이 좁다. 실제 VLM 품질 개선용으로는 사람이 검수한 caption/VQA가 더 필요하다.
