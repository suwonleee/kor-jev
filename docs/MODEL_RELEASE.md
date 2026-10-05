# 한국어 가중치 공개 절차

GitHub는 실행 코드, Hugging Face는 가중치·토크나이저·모델 설정·모델 카드를 배포합니다. Hub를 추론 API로 사용하는 것이 아니라 최초 다운로드 저장소로 사용하며, 이후 추론은 로컬에서 실행합니다.

## 현재 상태

실행 코드와 설치·빌드·모델 없는 API 검증은 준비됐습니다. `src/kor_jev/release.json`의 `published`가 false인 동안 한국어 모델의 공개 다운로드는 제공되지 않습니다. 이는 다른 모델로 대체하지 않고 명시적으로 실패합니다.

## 게시

체크포인트와 Hugging Face 인증에 접근할 수 있는 터미널에서 실행합니다. 인증이 없다면 `uv run hf auth login`으로 로그인하며, 토큰을 문서나 대화에 붙여 넣지 않습니다.

```bash
uv run python scripts/publish_model.py \
  --checkpoint /path/to/ko-v7/selected \
  --dry-run

uv run python scripts/publish_model.py \
  --checkpoint /path/to/ko-v7/selected
```

별도 `--repo-id`가 없으면 현재 인증 계정의 `kor-jev-ko-v7` 모델 저장소를 사용합니다. 게시 명령은 필수 파일·ko-v7 온도·파일 해시를 확인하고, 모델 저장소를 비공개로 생성한 뒤 파일과 라이선스를 한 커밋으로 올립니다. 파일 존재를 확인한 후 공개하고 익명 접근을 검증합니다. 기존 실험용 가중치의 게시이며 품질 채택 판정을 변경하지 않습니다.

성공 시 `src/kor_jev/release.json`에 Hub 저장소·고정 커밋·파일 크기·SHA256을 기록합니다. 원본 selected 가중치는 변경하지 않습니다. 학습 데이터와 optimizer·resume 상태는 업로드하지 않습니다.

## 새 설치 검증

게시 후 별도 새 체크아웃과 빈 체크포인트 디렉터리에서 검증합니다.

```bash
uv sync --extra mac
uv run kor-jev fetch
uv run kor-jev serve --backend mlx
```

CPU 환경은 `uv sync`, `uv run kor-jev serve --backend pytorch --device cpu`를 사용합니다. 실제 요청 예제로 응답을 확인한 뒤 모델 manifest와 README 상태를 GitHub에 반영해야 합니다. 설치·mock API 테스트 통과만으로 실제 가중치 다운로드와 추론까지 검증했다고 기록하지 않습니다.
