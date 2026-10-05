# 한국어판 학습·평가 데이터 출처

학습 데이터는 배포하지 않는다. 각 출처는 자체 라이선스를 유지하며, 일부는 동일조건변경허락(CC BY-SA)이다.
학습에는 train 분할만, 평가(eval/bench)에는 validation/test 분할만 쓴다. 두 쪽 사이의 동일·근사 중복은 `korev.build` 가 제거한다.

| 데이터 | 출처 | 라이선스 | 학습 | 평가 |
|---|---|---|---|---|
| KLUE YNAT·NLI·STS | [klue/klue](https://huggingface.co/datasets/klue/klue) | CC BY-SA 4.0 | train | validation |
| KoBEST BoolQ·COPA·HellaSwag·SentiNeg·WiC | [skt/kobest_v1](https://huggingface.co/datasets/skt/kobest_v1) | CC BY-SA 4.0 | train | test |
| NSMC | [e9t/nsmc](https://github.com/e9t/nsmc) | CC0 1.0 | train | test |
| Korean Hate Speech | [kocohub/korean-hate-speech](https://github.com/kocohub/korean-hate-speech) | CC BY-SA 4.0 | train | dev |
| MASSIVE (ko-KR) | [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent), [scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario) (원본 Amazon MASSIVE, CC BY 4.0) | CC BY 4.0 | train | test |
| SST-2 | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2) | 출처 확인 필요 | train(리플레이) | validation |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | 출처 확인 필요 | train(리플레이) | test |
| BoolQ | [google/boolq](https://huggingface.co/datasets/google/boolq) | CC BY-SA 3.0 | train(리플레이) | validation |

## 생성 데이터

- 질문 문구·선택지 설명·합성 과제: 로컬 오픈 가중치 모델 `mlx-community/Qwen3.8-27B-4bit` 가 작성 (`korev.teach`).
  합성 예시는 같은 모델의 블라인드 재라벨과 일치한 것만 남긴다.
- 영어 리플레이의 목표 확률: 원본 Jeff-Qwen3.5-0.8B 의 출력 (`korev.distill`).
- 폐쇄형 모델 출력은 학습 데이터에 없다. 골든셋(eval/golden)은 평가 전용이며 학습에 쓰지 않는다.

## 쓰지 않은 것

- KorQuAD (CC BY-ND: 변경 금지 조건), AI Hub (재배포 제한 약관), Smilegate UnSmile (CC BY-NC-ND), 회사·고객 데이터.
