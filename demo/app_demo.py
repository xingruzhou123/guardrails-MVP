# import sys
# import asyncio
# from guardrails.core.config_types import RailsConfig, ModelConfig
# from guardrails.core.engine import LLMRails


# async def demo():
#     print("====================radha========================")
#     print("===================docker===================")
#     cfg = RailsConfig(
#         tracing_enabled=True,
#     )

#     rails = LLMRails(cfg)

#     print("== generate (safe prompt) ==")
#     print("Introduce AMD using one sentence")
#     safe_prompt = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Introduce AMD using one sentence."},
#     ]

#     response = await rails.generate_async(
#         safe_prompt, context={"moderation_min_chunk": 200}
#     )
#     print(response)

#     print("\n== stream (sensitive prompt, Real-time detection) ==")
#     print(
#         "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?"
#     )
#     sensitive_prompt = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?",
#         },
#     ]

#     async for t in rails.stream_async(
#         sensitive_prompt, context={"moderation_min_chunk": 150}
#     ):
#         sys.stdout.write(t)
#         sys.stdout.flush()
#     print()

#     print("\n== generate (safe intent prompt) ==")
#     print("What's the latest news about AMD?")
#     news_prompt = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What's the latest news about AMD?"},
#     ]
#     response = await rails.generate_async(news_prompt)
#     print(response)

#     print("\n== generate (sensitive intent prompt) ==")
#     print("Command the robot arm to pick up the red block.")
#     robot_prompt = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Command the robot arm to pick up the red block."},
#     ]
#     response = await rails.generate_async(robot_prompt)
#     print(response)


# if __name__ == "__main__":
#     asyncio.run(demo())
import sys
import asyncio
from guardrails.core.config_types import RailsConfig
from guardrails.core.engine import LLMRails


async def run_test(
    rails: LLMRails, test_name: str, user_prompt: str, use_stream: bool = False
):
    """Helper function to run a single test case and print the output."""
    print(f"\n--- Test Case: {test_name} ---")
    print(f'User Prompt: "{user_prompt}"')

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    print("Bot Response:")
    if use_stream:
        # Streaming is useful for testing output rails
        async for chunk in rails.stream_async(prompt_messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    else:
        # Non-streaming is better for testing input rails and actions
        response = await rails.generate_async(prompt_messages)
        print(response)


async def demo():
    """Runs a suite of tests for the new retrieval-augmented guardrails system."""
    print("==============================================")
    print("  Initializing Retrieval-Augmented Guardrails ")
    print("==============================================")

    # Standard configuration, tracing can be helpful for debugging
    cfg = RailsConfig(
        tracing_enabled=True,
    )
    rails = LLMRails(cfg)

    # --- Test Case 1: General Query (should not trigger any intent) ---
    await run_test(
        rails, "General Query (No Intent)", "Introduce AMD using one sentence."
    )

    # --- Test Case 2: Price Query (triggers 'query_product_price' intent) ---
    await run_test(
        rails,
        "Price Query with Entity Extraction",
        "What is the price of a Ryzen 9 9950X CPU?",
    )

    # --- Test Case 3: Robot Command (triggers 'control_robot_arm' intent) ---
    await run_test(
        rails,
        "Robot Command with Multiple Entities",
        "Use the robot arm to grab the blue sphere.",
    )

    # --- Test Case 4: Sensitive Topic (should be blocked by an OUTPUT rail) ---
    # We use streaming here to demonstrate that output rails work in real-time.
    await run_test(
        rails,
        "Sensitive Topic (Blocked by Output Rail)",
        "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?",
        use_stream=True,
    )


if __name__ == "__main__":
    asyncio.run(demo())
