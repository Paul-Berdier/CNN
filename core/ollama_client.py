from __future__ import annotations

from typing import Optional

from openai import OpenAI
from core.config import settings

def get_llm_client() -> OpenAI:
    return OpenAI(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key)


def chat(
    messages: list[dict],
    model: str = settings.llm_model,
    temperature: float = settings.default_temperature,
    **kwargs,
) -> dict:

    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    usage = response.usage
    return {
        "texte": response.choices[0].message.content,
        "model": response.model,
        "usage": {
            "input": getattr(usage, "prompt_tokens", 0),
            "output": getattr(usage, "completion_tokens", 0),
            "total": getattr(usage, "total_tokens", 0),
        },
        "raw": response,
    }


def generate(
    prompt: str,
    system: Optional[str] = None,
    model: str = settings.llm_model,
    temperature: float = settings.default_temperature,
) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, model=model, temperature=temperature)["texte"]


def is_available() -> bool:
    try:
        get_llm_client().models.list()
        return True
    except Exception:
        return False

if __name__ == "__main__":
    if not is_available():
        print(f"[!] Ollama ne répond pas sur {settings.ollama_base_url}")
        print(f"    Lance le serveur (ollama serve) et vérifie : ollama pull {settings.llm_model}")
        raise SystemExit(1)

    print(f"[OK] Ollama disponible. Modèle : {settings.llm_model}\n")
    out = chat(
        [
            {"role": "system", "content": "Tu es un assistant médical concis."},
            {"role": "user", "content": "En une phrase, quel est le principe du traitement d'un ulcère veineux ?"},
        ]
    )
    print("Réponse :", out["texte"])
    print("Tokens  :", out["usage"])