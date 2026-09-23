import logging
import os
import ollama
from config import SPREADSHEET_MODEL_NAME

# Suppress HTTP request logs from httpx, httpcore, and urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

MODEL_NAME = "llama3.2"


def ask_llama(prompt, system_instruction=None, model=None):
    """
    Sends a prompt to Ollama with temperature=0.0 for strictly grounded generation.
    Supports optional custom model override.
    """
    target_model = model or MODEL_NAME

    messages = []

    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })
    else:
        messages.append({
            "role": "system",
            "content": (
                "You are a strictly grounded Personal Second Brain Assistant. "
                "Answer ONLY using facts explicitly present in the retrieved context. "
                "Do NOT invent concepts, dates, numbers, inventory, or technical accounting terms."
            )
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        response = ollama.chat(
            model=target_model,
            messages=messages,
            options={
                "temperature": 0.0
            }
        )
        return response["message"]["content"]
    except Exception as err:
        # Fallback to default MODEL_NAME if custom model fails to load
        if target_model != MODEL_NAME:
            logging.warning(f"Failed calling model '{target_model}': {err}. Falling back to '{MODEL_NAME}'.")
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "temperature": 0.0
                }
            )
            return response["message"]["content"]
        raise err


def ask_spreadsheet_llm(prompt, system_instruction=None):
    """
    Executes an LLM call using the dedicated SPREADSHEET_MODEL_NAME.
    """
    return ask_llama(prompt, system_instruction=system_instruction, model=SPREADSHEET_MODEL_NAME)


if __name__ == "__main__":
    print("=" * 100)
    print(f"Local AI Assistant (Default: {MODEL_NAME}, Spreadsheet: {SPREADSHEET_MODEL_NAME})")
    print("=" * 100)
    print(ask_llama("Hello!"))

