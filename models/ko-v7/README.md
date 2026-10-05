---
license: apache-2.0
language:
- ko
- en
base_model:
- mstrasser/Jeff-Qwen3.5-0.8B
- Qwen/Qwen3.5-0.8B
library_name: transformers
pipeline_tag: zero-shot-classification
tags:
- decision-model
- jev-compatible
- local
- experimental
---

# kor-jev ko-v7

한국어 선택지 판정용 약 0.8B 모델입니다. 원본 Jeff 전체 가중치를 공개 한국어 데이터와 로컬 오픈 교사 데이터 29,251행으로 1 epoch 추가 학습했습니다. 분류 교차 엔트로피와 Score 보조 손실을 사용하고, 별도 보정셋에서 전역 온도를 맞췄습니다.

**실험용 체크포인트입니다.** 한국어 공개 벤치 평균은 개선됐으나, 새 합성 시나리오에서 원본보다 나은 일반화 성능을 입증하지 못했고 범용 채택 기준을 모두 충족하지 못했습니다.

| 지표 | 원본 Jeff | ko-v7 |
|---|---:|---:|
| 한국어 분류 11과제 평균 정확도, MLX | 65.28% | 77.33% |
| 개발용 골든셋 통과율 184건, MLX | 82.61% | 90.22% |
| STS 점수 MAE 300건, MLX | 0.790 | 0.603 |
| 새 시나리오 분류 180건, CPU | 90.00% | 89.44% |
| 새 시나리오 Score 정규화 MAE 60건, CPU | 0.1757 | 0.1800 |

새 평가는 16영역의 합성 240건이며, 인간 정답 운영 데이터 평가가 아닙니다. 공개 벤치 평균은 과제별 단순 평균입니다. 골든셋은 개발용이며 유형별 통과 조건을 사용합니다. 폐쇄형 모델 출력은 평가 작성·검토에만 사용했고 추가 학습 데이터에는 넣지 않았습니다. 회사·고객 데이터는 사용하지 않았습니다.

## 실행

이 체크포인트는 표준 대화 생성 모델과 달리 별도의 `readout.safetensors`, 답변 코드와 온도가 필요합니다. 일반 Transformers pipeline이나 Ollama에 그대로 넣는 사용법은 지원하지 않습니다. [kor-jev 실행 코드](https://github.com/suwonleee/kor-jev)의 `kor-jev fetch`와 `kor-jev serve`를 사용하세요.

Choice, Noul, Score를 Jev/System One 형식으로 제공합니다. MLX는 Apple Silicon 텍스트 추론용이며, PyTorch는 CPU/CUDA 경로를 제공합니다. 확신도 기준과 점수 rubric은 업무 표본에서 검증해야 합니다.

## 출처와 라이선스

기반 가중치는 Jeff-Qwen3.5-0.8B 및 Qwen3.5-0.8B입니다. 가중치는 Apache 2.0이며 코드의 MIT 라이선스와 구분합니다. NOTICE와 LICENSE를 함께 보존하세요. 학습 데이터는 배포하지 않으며, 각 데이터의 별도 라이선스는 [데이터 출처](https://github.com/suwonleee/kor-jev/blob/main/docs/ko-data-sources.md)에 정리했습니다.

독립 프로젝트로 TypeSafe/Jev의 공식 모델이거나 승인·보증을 받은 모델이 아닙니다. 전체 지표 정의와 원자료 해시는 [평가 요약](https://github.com/suwonleee/kor-jev/blob/main/docs/benchmarks/ko-v7-summary.json)을 참고하세요.
