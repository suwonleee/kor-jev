# GitHub 가중치 공개 절차

GitHub 저장소는 실행 코드, 같은 저장소의 GitHub Releases는 한국어 가중치·토크나이저·모델 설정·모델 카드를 배포합니다. Hugging Face 로그인은 필요하지 않습니다. 사용자는 저장소를 클론하고 `kor-jev fetch`로 가중치를 받은 뒤 로컬에서 실행합니다.

## 현재 상태

실행 코드와 설치·빌드·모델 없는 API 검증은 준비됐습니다. `src/kor_jev/release.json`의 `published`가 false인 동안 한국어 모델의 공개 다운로드는 제공되지 않습니다. 다른 모델로 대체하지 않고 명시적으로 실패합니다.

## 게시

체크포인트를 읽을 수 있는 터미널과 로그인된 GitHub CLI(`gh`)가 필요합니다. 가중치는 Git 이력에 넣지 않습니다.

```bash
uv run python scripts/publish_model.py \
  --checkpoint /path/to/ko-v7/selected \
  --dry-run

uv run python scripts/publish_model.py \
  --checkpoint /path/to/ko-v7/selected
```

기본 대상은 `suwonleee/kor-jev`, 태그는 `ko-v7-preview`입니다. 필수 파일·ko-v7 온도·파일 해시와 GitHub의 파일당 2GiB 미만 조건을 확인합니다. 초안 릴리스를 생성한 뒤 파일을 올리고 GitHub가 계산한 크기·SHA256과 대조합니다. 검증 후 실험용 사전 릴리스로 공개합니다.

이미 공개된 태그는 덮어쓰지 않습니다. 동일 커밋의 초안 릴리스는 기존 파일의 해시가 같을 때만 이어 올릴 수 있습니다. 다른 내용은 새 `--tag`를 지정합니다. 원본 가중치는 수정하지 않으며 학습 데이터와 optimizer·resume 상태는 게시하지 않습니다.

성공 시 `src/kor_jev/release.json`에 GitHub 저장소·버전 태그·소스 커밋·파일 URL·크기·SHA256을 기록합니다. 게시 후 실제 익명 다운로드와 로컬 추론을 검증하고, manifest와 README 공개 상태를 커밋·푸시해야 사용자 설치가 완성됩니다.

## 사용자 실행

```bash
git clone https://github.com/suwonleee/kor-jev.git
cd kor-jev
uv sync --extra mac
uv run kor-jev fetch
uv run kor-jev serve --backend mlx
```

CPU 환경은 `uv sync`, `uv run kor-jev serve --backend pytorch --device cpu`를 사용합니다. 설치와 mock API 테스트 통과만으로 실제 가중치 다운로드·추론까지 검증했다고 기록하지 않습니다.
