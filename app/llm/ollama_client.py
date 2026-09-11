import logging
import ollama

# Suppress HTTP request logs from httpx, httpcore, and urllib3
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

MODEL_NAME = "llama3.2"

def ask_llama(prompt, system_instruction=None):
    """
    Sends a prompt to Ollama with temperature=0.0 for strictly grounded generation.
    """
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

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.0
        }
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print("=" * 100)
    print("Local AI Assistant (Llama 3.2)")
    print("=" * 100)
    print(ask_llama("Hello!"))
