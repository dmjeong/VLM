# Web Image Sources

> 작성일: 2026-05-28
> 목적: `data/samples/images/web/`에 추가한 웹 이미지의 출처와 라이선스를 기록한다.

학습 데이터로 재배포될 수 있으므로 Public Domain 또는 CC0 이미지만 선택했다.

| 로컬 파일 | 원본 파일 | 라이선스 | 출처 |
|-----------|-----------|----------|------|
| `images/web/apple.jpg` | `File:Apple (1).jpg` | Public Domain | https://commons.wikimedia.org/wiki/File:Apple_(1).jpg |
| `images/web/banana.jpg` | `File:Banana-close-up.jpg` | Public Domain | https://commons.wikimedia.org/wiki/File:Banana-close-up.jpg |
| `images/web/coffee_mug.jpg` | `File:Coffee in a mug.jpg` | CC0 | https://commons.wikimedia.org/wiki/File:Coffee_in_a_mug.jpg |
| `images/web/pencil.jpg` | `File:Number-2-pencil.jpg` | Public Domain | https://commons.wikimedia.org/wiki/File:Number-2-pencil.jpg |
| `images/web/mouse.jpg` | `File:Apple magic mouse.jpg` | Public Domain | https://commons.wikimedia.org/wiki/File:Apple_magic_mouse.jpg |

## 샘플 생성 규칙

- 이미지 1개당 caption 질문 1개와 VQA 질문 2개를 train set에 추가했다.
- validation set에는 이미지 1개당 확인 질문 1개를 추가했다.
- 답변은 adapter overfit 확인을 쉽게 하기 위해 짧고 명확한 문장으로 작성했다.
