"""korev with the official System One Python SDK (typesafe-sdk, MIT): code written for the hosted API runs locally.

    # terminal 1: local server on the Korean checkpoint
    JEFF_BACKEND=mlx JEFF_CHECKPOINT=checkpoints/ko-v1/selected JEFF_ANY_MODEL=1 PORT=8765 uv run jeff-serve
    # terminal 2
    uv run --with typesafe-sdk python examples/jev_sdk_korean.py

Only two environment variables differ from the hosted setup: TYPESAFE_BASE_URL points at the local server, and
TYPESAFE_API_KEY is any non-empty string (the local server ignores it unless JEFF_API_KEY is set).
The patterns below are the documented ones: speculative fan-out (several questions on one state in one request),
confidence-gated routing with per-action thresholds, a catch-all option, and a score rubric with described levels."""

import os

os.environ.setdefault("TYPESAFE_BASE_URL", "http://127.0.0.1:8765")
os.environ.setdefault("TYPESAFE_API_KEY", "local")

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # noqa: E402

THRESHOLDS = {"auto_refund": 0.85, "suggest_label": 0.5}  # per action, not one global bar

ticket = {
    "문의": "지난달 요금이 두 번 결제됐어요. 금요일까지 환불 안 되면 해지하겠습니다.",
    "고객 등급": "business",
    "재문의 횟수": 2,
}

with TypeSafeClient() as client:
    response = client.system_one(
        state=ticket,
        questions={
            "route": Choice(
                instructions="어느 팀이 처리해야 하나요?",
                criteria={"billing": "결제·환불", "technical": "장애·버그", "account": "계정·로그인", "other": "기타"},
            ),
            "urgent": Noul(
                instructions="오늘 안에 답해야 하나요?",
                criteria={"true": "기한이 임박했거나 해지를 예고함", "false": "기다려도 됨"},
            ),
            "frustration": Score(
                instructions="고객의 감정 상태는?",
                criteria=["차분함: 사실만 전달", "불만이 있지만 정중함", "매우 화남: 항의·위협"],
            ),
        },
    )

route = response.answers["route"]
print("model:", response.model)
print("route:", route.choice, round(route.confidence, 2), route.probabilities)
print("urgent:", round(response.answers["urgent"].noul, 2))
print("frustration:", round(response.answers["frustration"].score, 2), "of 0-2")

if route.choice == "billing" and route.confidence >= THRESHOLDS["auto_refund"]:
    print("-> 환불 자동 처리 큐")
elif route.confidence >= THRESHOLDS["suggest_label"]:
    print("-> 상담원에게 추천 라벨과 함께 전달")
else:
    print("-> 사람 검토")
