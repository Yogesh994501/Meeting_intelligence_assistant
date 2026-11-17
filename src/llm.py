"""Step 8 (part 1): LLM provider abstraction - Google Gemini (via langchain-google-genai) or local Ollama."""
from __future__ import annotations

import os

PROVIDERS = {
    "gemini": {"label": "Google Gemini", "model": "gemini-3.5-flash", "env": "GOOGLE_API_KEY"},
    "ollama": {"label": "Ollama (local, free)", "model": "llama3.1", "env": ""},
}


# Tried in order if the chosen model is retired / unavailable (Google retires models regularly).
GEMINI_FALLBACKS = ["gemini-3.5-flash", "gemini-flash-latest"]
_MODEL_ERRORS = ("404", "503", "not found", "no longer available", "deprecated", "not supported", "shut down")


def _to_text(content) -> str:
    """Gemini may return a str or a list of content parts."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        parts.append(p if isinstance(p, str) else p.get("text", "") if isinstance(p, dict) else "")
    return "".join(parts)


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = ""):
        self.provider = provider
        self.model = model
        env = PROVIDERS[provider]["env"]
        self.api_key = (api_key or os.getenv(env, "") or os.getenv("GEMINI_API_KEY", "")) if env else ""
        self.base_url = base_url or ("http://localhost:11434" if provider == "ollama" else "")

    @property
    def is_configured(self) -> bool:
        return self.provider == "ollama" or bool(self.api_key)

    def _gemini_call(self, model, system, user, max_tokens, temperature) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=self.api_key,
                                     temperature=temperature, max_output_tokens=max_tokens)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return _to_text(resp.content).strip()

    def test(self) -> tuple[bool, str]:
        """Quick connectivity check used by the sidebar 'Test LLM' button."""
        try:
            out = self.generate("Reply with the single word: OK", "ping", max_tokens=20, temperature=0)
            return True, f"{self.provider} / {self.model} replied: {out[:40] or '(empty)'}"
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    def generate(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.1) -> str:
        if self.provider == "gemini":
            candidates = [self.model] + [m for m in GEMINI_FALLBACKS if m != self.model]
            first_error = None
            for m in candidates:
                try:
                    out = self._gemini_call(m, system, user, max_tokens, temperature)
                    self.model = m  # remember the one that worked
                    return out
                except Exception as e:  # noqa: BLE001
                    first_error = first_error or e
                    if not any(t in str(e).lower() for t in _MODEL_ERRORS):
                        raise  # e.g. bad API key / quota: retrying other models won't help
            raise RuntimeError(
                f"Gemini model '{candidates[0]}' is unavailable (it may have been retired). "
                f"Type a current model name in the sidebar, e.g. gemini-3.5-flash. Original error: {first_error}"
            ) from first_error

        if self.provider == "ollama":
            import requests

            try:
                r = requests.post(
                    f"{self.base_url.rstrip('/')}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "options": {"temperature": temperature, "num_predict": max_tokens},
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    },
                    timeout=300,
                )
                r.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` "
                    f"and pull the model: `ollama pull {self.model}`."
                ) from e
            return r.json()["message"]["content"].strip()

        raise ValueError(f"Unknown provider: {self.provider}")
