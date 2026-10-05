# kor-jev

**노트북에서 실행하는 한국어 판정 모델.**

한국어 문장과 판단 기준을 입력하면 선택지별 확률, 참일 확률, 등급 점수를 반환합니다. 약 0.8B 규모의 Jeff/Qwen 기반 모델을 한국어 과제로 추가 학습했으며, Apple Silicon에서는 MLX로 실행할 수 있습니다. 모델과 실행에 필요한 파일을 받은 뒤에는 외부 API 없이 로컬에서 추론합니다.

문의 분류, 작업 경로 선택, 문서 조건 확인처럼 **정해진 후보 사이에서 판단하는 기능**에 사용할 수 있습니다. 자유로운 대화나 긴 답변 생성은 이 프로젝트의 평가 대상이 아닙니다.

> 현재 상태: 실험용 한국어 체크포인트 `ko-v7`. 이 공개 저장소는 소개와 평가 자료를 제공합니다. 실행 소스와 한국어 가중치의 공개 다운로드는 아직 제공하지 않습니다. 아래 실행 예제는 실행 소스와 해당 체크포인트를 보유한 경우를 기준으로 합니다.

[실행 방법](#실행-방법) · [요청 예제](#요청-예제) · [평가 결과](#평가-결과) · [학습 요약](#학습-요약) · [라이선스](#라이선스와-출처)

## 어떤 판단을 할 수 있나요?

상황(`state`)과 질문(`questions`)을 함께 보내며, 선택지와 기준은 한국어로 작성할 수 있습니다. 여러 질문을 한 요청에 담을 수도 있습니다.

| 유형 | 반환하는 값 | 사용 예 |
|---|---|---|
| `choice` | 선택한 후보, 후보별 확률, 확신도 | 문의를 결제·기술·계정 담당으로 분류 |
| `noul` | 조건이 참일 확률, 0~1 | 문서에 요청한 조건이 명시되어 있는지 확인 |
| `score` | 등급별 확률과 등급 번호의 기댓값 | 명시한 기준에 따라 답변 충실도를 0~4로 평가 |

답변 문장을 생성해 파싱하는 대신, 한 번의 모델 순전파로 판정 값을 계산합니다. 선택지 수와 입력 길이에 따라 처리 시간은 달라집니다.

## 실행 방법

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/getting-started/installation/)가 필요합니다. 체크포인트 디렉터리에는 가중치와 `decision_config.json` 등 저장된 모델 파일이 있어야 합니다. 원본 Jeff 가중치를 사용하면 원본 Jeff가 실행되며, 이 문서의 한국어 추가 학습 성능과는 구분해야 합니다.

### Apple Silicon · MLX

`pyproject.toml`이 있는 실행 소스 디렉터리에서 실행합니다. 이 소개 저장소만 내려받으면 서버를 실행할 수 없습니다. `KORJEV_CHECKPOINT`를 실제 한국어 체크포인트 경로로 바꾸세요.

```bash
uv sync --extra mac
export KORJEV_CHECKPOINT="/path/to/ko-v7/selected"

JEFF_BACKEND=mlx \
JEFF_CHECKPOINT="$KORJEV_CHECKPOINT" \
JEFF_ANY_MODEL=1 \
JEFF_HOST=127.0.0.1 PORT=8765 \
uv run jeff-serve
```

서버가 준비되면 [로컬 플레이그라운드](http://127.0.0.1:8765)에서 입력을 바꿔 볼 수 있습니다. 현재 MLX 경로는 텍스트 입력용입니다.

### PyTorch · CPU 또는 CUDA

```bash
uv sync
export KORJEV_CHECKPOINT="/path/to/ko-v7/selected"

JEFF_BACKEND=pytorch JEFF_DEVICE=cpu \
JEFF_CHECKPOINT="$KORJEV_CHECKPOINT" \
JEFF_ANY_MODEL=1 \
JEFF_HOST=127.0.0.1 PORT=8765 \
uv run jeff-serve
```

CUDA 환경에서는 `JEFF_DEVICE=cuda`로 지정합니다. 아래 속도 측정은 Apple Silicon MLX 결과이며, 다른 하드웨어의 속도를 보장하지 않습니다.

실행 명령과 환경변수의 `jeff` 이름은 기반 프로젝트의 인터페이스를 유지한 것입니다. `JEFF_ANY_MODEL=1`은 요청의 `model` 이름을 허용하며, 실제 가중치는 `JEFF_CHECKPOINT`로 결정됩니다. 응답의 모델 이름은 서버가 로드한 기반 모델을 표시합니다.

## 요청 예제

서버를 실행한 상태에서 다른 터미널에 붙여 넣습니다. 하나의 문의에 대해 담당 팀, 중복 결제 여부, 요청의 명확성을 함께 확인합니다.

```bash
curl --fail-with-body http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "kor-jev",
  "state": "어제 주문 금액이 두 번 결제됐습니다. 중복 결제된 한 건을 취소해 주세요.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "이 문의를 처리할 담당 팀을 선택하세요.",
      "criteria": {
        "billing": "결제·결제 취소·환불",
        "technical": "서비스 장애·기술 문제",
        "account": "계정·로그인"
      }
    },
    "duplicate_payment": {
      "type": "noul",
      "instructions": "고객이 동일 주문의 중복 결제를 명시했나요?"
    },
    "clarity": {
      "type": "score",
      "instructions": "요청 내용을 얼마나 명확하게 파악할 수 있나요?",
      "criteria": [
        "0: 문제와 원하는 조치를 모두 알 수 없음",
        "1: 문제 또는 원하는 조치 중 하나만 알 수 있음",
        "2: 문제와 원하는 조치가 모두 명시됨"
      ]
    }
  }
}
JSON
```

응답의 `answers.route`에는 `choice`, `probabilities`, `confidence`가 들어갑니다. `answers.duplicate_payment.noul`은 참일 확률이며, `answers.clarity.score`는 이 예제에서 0~2 범위의 기댓값입니다.

Choice의 `confidence`는 선택지 수를 고려해 보정한 값으로, 최대 확률과 다릅니다. 자동 처리 기준은 실제 업무 표본에서 정하고, 확신이 낮거나 판단이 중요한 경우 검토 단계로 보내는 방식으로 사용하세요.

Jev/System One 형식의 `choice`·`noul`·`score` 요청을 지원합니다. 공식 Python SDK 연결 예제는 [examples/jev_sdk_korean.py](examples/jev_sdk_korean.py)에 있습니다. 예제의 임계값은 사용법을 설명하기 위한 값입니다.

## 평가 결과

2026-10-05 측정 결과입니다. 비교 대상은 학습 시작 시 고정한 **원본 Jeff-Qwen3.5-0.8B 체크포인트**이며, 최신 upstream 릴리스나 상용 Jev와의 비교가 아닙니다.

### 공개 벤치와 개발용 평가 · MLX

| 지표 | 원본 Jeff | kor-jev · ko-v7 |
|---|---:|---:|
| 한국어 분류 11개 과제 평균 정확도 ↑ | 65.28% | 77.33% |
| 영어 분류 3개 과제 평균 정확도 ↑ | 86.89% | 87.44% |
| 개발용 골든셋 통과율 · 184건 ↑ | 82.61% · 152/184 | 90.22% · 166/184 |
| 한국어 STS 점수 MAE · 300건 ↓ | 0.790 | 0.603 |
| 한국어 분류 평균 보정 오차 · ECE ↓ | 0.130 | 0.063 |

공개 벤치는 KLUE, KoBEST, NSMC, Korean Hate Speech, MASSIVE와 영어 3개 과제를 포함해 총 **15개 과제·4,700건**입니다. 분류 평균은 과제별 정확도의 단순 평균이며, STS 점수 과제는 분류 평균에서 제외합니다. 골든셋 통과율은 유형별 판정 조건을 적용한 개발 지표로, 일반적인 분류 정확도와 다릅니다. STS MAE는 원래 0~5 점수 범위의 평균 절대 오차입니다.

### 새 한국어 시나리오 · PyTorch CPU

학습에 넣지 않은 16개 영역의 합성 시나리오 240건으로 별도 평가했습니다. Choice 120건, Noul 60건, Score 60건이며, 선택지 변형을 포함한 실제 요청 수는 480건입니다.

| 지표 | 원본 Jeff | kor-jev · ko-v7 |
|---|---:|---:|
| Choice·Noul 분류 정확도 · 180건 ↑ | 90.00% · 162/180 | 89.44% · 161/180 |
| Score 정규화 MAE · 60건 ↓ | 0.1757 | 0.1800 |

공개 한국어 벤치 평균은 개선됐지만, 이 새 평가에서는 원본보다 나은 일반화 성능을 확인하지 못했습니다. 실제 운영 데이터의 인간 정답 평가도 아직 없습니다. 일부 문장 유사도 분류와 부정 표현 과제의 성능 저하가 남아 있어, 현재 체크포인트는 범용 배포 채택 기준을 모두 충족하지 못한 실험용 모델입니다.

[평가 수치와 원자료 SHA256](docs/benchmarks/ko-v7-summary.json) · [새 평가 문항](tests/fixtures/generalization_ko_v7_final.jsonl) · [데이터 출처](docs/ko-data-sources.md)

### 노트북에서의 응답 시간

Apple M4 Pro · 메모리 48GB · macOS 26.3.1 · MLX에서 측정했습니다. 요청당 입력 하나, 사전 준비 요청 5회 이후의 로컬 HTTP 응답 시간으로, 토큰화·추론·응답 전송을 포함하고 서버 시작 시간은 제외합니다.

| ko-v7 측정 표본 | 중앙값 · p50 | p95 |
|---|---:|---:|
| 골든셋 · 184요청 | 38.9ms | 43.0ms |
| 개발 시나리오 · 200요청 | 43.0ms | 149.3ms |

긴 입력에서는 지연이 커질 수 있습니다. 해당 실행의 샘플링된 모델 프로세스 최대 RSS는 약 2.25GB였으며, GPU 메모리를 포함한 전체 필요 메모리나 최소 노트북 사양을 뜻하지 않습니다.

## 학습 요약

| 항목 | 구성 |
|---|---|
| 기반 모델 | Jeff-Qwen3.5-0.8B · Qwen3.5 계열 |
| 학습 방식 | 전체 가중치 지도학습 · 29,251행 · 1 epoch |
| 데이터 | 공개 한국어 데이터, 로컬 Qwen3.8-27B 생성 데이터, 원본 Jeff 기반 영어 리플레이 |
| 목적 함수 | 분류 교차 엔트로피 + Score 순서·기댓값 보조 손실 |
| 모델 선택·확률 보정 | 개발셋 NLL로 선택, 별도 보정셋에서 전역 온도 학습 |

학습·개발·보정·최종 평가를 분리했습니다. 폐쇄형 모델의 출력은 평가 문항 작성·검토에만 사용했고 학습 데이터에는 넣지 않았습니다. 회사·고객 데이터는 사용하지 않았습니다. 학습 데이터 자체는 이 저장소에서 배포하지 않습니다.

## 직접 확인해 볼 항목

자신의 사용 환경에서 다음을 확인하면 적용 범위를 정하기 쉽습니다.

1. **업무 분류:** 실제 문의와 정답 라벨로 정확도를 측정합니다.
2. **부정·예외 조건:** “가능하지 않음”, “승인 전”, “일부만 허용”이 포함된 문장을 확인합니다.
3. **선택지 안정성:** 순서와 표현을 바꾸거나 “기타”를 추가했을 때 판단을 비교합니다.
4. **점수 기준:** 사람이 매긴 점수와 모델의 Score 오차를 비교합니다.
5. **처리량:** 자신의 확신도 기준에서 자동 처리 비율, 오류율, p95 지연을 함께 측정합니다.

## 라이선스와 출처

코드는 [MIT License](LICENSE)입니다. 기반 프로젝트인 **Jeff의 Mathias Strasser**, **AutoJev의 Denis Yarats** 저작권 및 허가 고지를 유지합니다.

- [Jeff](https://github.com/firelex/jeff): 기반 학습·판정 API·MLX 실행 코드.
- [AutoJev](https://github.com/denis-pplx/autojev): Jeff 학습 코드의 기반 레시피.
- [Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B)와 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): 기반 가중치의 공개 라이선스는 Apache 2.0입니다. 한국어 추가 학습 가중치의 배포 조건은 실제 가중치 공개 시 별도로 명시합니다.
- [한국어 학습·평가 데이터 출처](docs/ko-data-sources.md): 데이터마다 별도 라이선스가 적용됩니다. 코드의 MIT 라이선스가 데이터까지 포괄하지 않습니다.

kor-jev는 Jeff를 바탕으로 한국어 판정 과제를 추가 학습한 독립 프로젝트입니다. TypeSafe/Jev의 공식 모델이거나 해당 제작자의 승인·보증을 받은 프로젝트가 아닙니다.
