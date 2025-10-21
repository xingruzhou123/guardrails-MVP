# src/guardrails/actions/action_handler.py
from typing import Dict, Any

# This file contains the implementation of the actions that can be triggered by
# the LLM classifier. Each function should accept a dictionary of entities and
# return a string response.


def handle_price_query(entities: Dict[str, Any]) -> str:
    """
    Handles user queries about product prices.
    In a real application, this would query a database.
    """
    product_name = entities.get("product_name", "an unspecified product")

    # In a real-world scenario, you would perform a database lookup here.
    # For this demo, we'll use a simple hardcoded dictionary.
    print(f"--- [Action] Querying price for '{product_name}'... ---")

    price_db = {
        "ryzen 9 9950x cpu": "$599",
        "ryzen 7 9700x": "$399",
    }

    # Simple lookup, ignoring case
    price = price_db.get(product_name.lower())

    if price:
        return f"The price for {product_name} is {price}."
    else:
        return f"Sorry, I couldn't find a price for {product_name}."


def handle_robot_command(entities: Dict[str, Any]) -> str:
    """
    Handles user commands for a robot arm.
    In a real application, this would interface with robotics hardware/software.
    """
    action = entities.get("action", "do something")
    target = entities.get("target_object", "something")

    print(f"--- [Action] Executing robot command: {action} {target}... ---")

    # Simulate executing the command
    return f"Executing command: The robot arm will now `{action}` the `{target}`."


# You can add more action handler functions here as you define more intents.
