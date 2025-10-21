import sys
import asyncio
from guardrails.core.config_types import RailsConfig, ModelConfig
from guardrails.core.engine import LLMRails


async def demo():
    print("====================radha========================")
    print("===================docker===================")
    cfg = RailsConfig(
        tracing_enabled=True,
    )

    rails = LLMRails(cfg)

    print("== generate (safe prompt) ==")
    print("Introduce AMD using one sentence")
    safe_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce AMD using one sentence."},
    ]

    response = await rails.generate_async(
        safe_prompt, context={"moderation_min_chunk": 200}
    )
    print(response)

    print("\n== stream (sensitive prompt, Real-time detection) ==")
    print(
        "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?"
    )
    sensitive_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?",
        },
    ]

    async for t in rails.stream_async(
        sensitive_prompt, context={"moderation_min_chunk": 150}
    ):
        sys.stdout.write(t)
        sys.stdout.flush()
    print()

    print("\n== generate (safe intent prompt) ==")
    print("What's the latest news about AMD?")
    news_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the latest news about AMD?"},
    ]
    response = await rails.generate_async(news_prompt)
    print(response)

    print("\n== generate (sensitive intent prompt) ==")
    print("Command the robot arm to pick up the red block.")
    robot_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Command the robot arm to pick up the red block."},
    ]
    response = await rails.generate_async(robot_prompt)
    print(response)


if __name__ == "__main__":
    asyncio.run(demo())
