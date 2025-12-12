# rag.py
# LLM wrapper for RAG supporting OpenAI, Google Gemini (Generative AI), and Anthropic Claude.
# Behavior:
# - Use provider defined by env var LLM_PROVIDER (values: "gemini", "claude")
# - Provide deterministic grounded fallback when no API key is configured.
#
# Note: you must install the appropriate provider SDKs (see README below).

import os
from typing import List, Dict

# Prompt template used for RAG
DEFAULT_PROMPT_TEMPLATE = """You are a regulatory compliance assistant. Use ONLY the provided context below to answer the user's question.
Cite sources inline in the form [title | chunk_id]. If the information is not present in the context, respond: "No supporting text found in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def build_prompt_from_chunks(chunks: List[Dict], question: str) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        title = c.get('title','unknown')
        cid = c.get('chunk_id','-')
        txt = c.get('text','')
        parts.append(f"[{i}] {title} | {cid}\n{txt}\n")
    context = "\n---\n".join(parts)
    return DEFAULT_PROMPT_TEMPLATE.format(context=context, question=question)

class LLMClient:
    def __init__(self, provider: str = None, temperature: float = 0.0, max_tokens: int = 512):
        """
        provider: 'openai' | 'gemini' | 'claude' or None (auto from env)
        """
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", max_tokens))

        # Flags for provider readiness
        self._openai_ready = False
        self._gemini_ready = False
        self._claude_ready = False

        # lazy imports to avoid hard dependency unless used

        if self.provider == "gemini":
            # Google Generative AI (gemini) client
            # Typical install: pip install google-generative-ai
            try:
                import google.generativeai as genai
                key = os.environ.get("GEMINI_API_KEY", "")
                if key:
                    genai.configure(api_key=key)
                    self.genai = genai
                    self._gemini_ready = True
            except Exception:
                self._gemini_ready = False

        elif self.provider == "claude":
            # Anthropic client
            # Typical install: pip install anthropic
            try:
                from anthropic import Anthropic  # type: ignore
                key = os.environ.get("ANTHROPIC_API_KEY", "")
                if key:
                    self.claude = Anthropic(api_key=key)
                    self._claude_ready = True
            except Exception:
                self._claude_ready = False

        else:
            # unknown provider: nothing configured
            pass

    def generate(self, prompt: str) -> str:
        """
        Generate text according to chosen provider. Falls back to deterministic grounded output (concatenated context)
        if the selected provider isn't configured.
        """
        # OPENAI branch
        if self.provider == "openai" and self._openai_ready:
            try:
                # Use ChatCompletion (chat model) if available
                resp = self.openai.ChatCompletion.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[{"role":"user","content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return f"[OpenAI call failed: {e}]\n{self._fallback_answer(prompt)}"

        # GEMINI (Google Generative AI) branch
        if self.provider == "gemini" and self._gemini_ready:
            try:
                # Using google.generativeai (genai) - simple text generation example
                # API shape may vary; this is a common pattern:
                resp = self.genai.generate_text(model=os.environ.get("GEMINI_MODEL","models/text-bison-001"),
                                                prompt=prompt,
                                                temperature=self.temperature,
                                                max_output_tokens=self.max_tokens)
                # resp may be a dict-like object; attempt to extract text robustly
                text = getattr(resp, "text", None) or (resp.get("candidates",[{}])[0].get("output","") if isinstance(resp, dict) else None)
                if not text:
                    # try string cast
                    text = str(resp)
                return text.strip()
            except Exception as e:
                return f"[Gemini call failed: {e}]\n{self._fallback_answer(prompt)}"

        # CLAUDE (Anthropic) branch
        if self.provider == "claude" and self._claude_ready:
            try:
                # Anthropic's SDK expects a prompt wrapped with HUMAN_PROMPT/AI_PROMPT.
                from anthropic import HUMAN_PROMPT, AI_PROMPT  # type: ignore
                full_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
                resp = self.claude.create(prompt=full_prompt, max_tokens_to_sample=self.max_tokens, temperature=self.temperature)
                # extract response text (SDK return shapes differ by versions)
                text = resp.get("completion") or resp.get("completion", "")
                if not text:
                    text = str(resp)
                return text.strip()
            except Exception as e:
                return f"[Anthropic call failed: {e}]\n{self._fallback_answer(prompt)}"

        # If none configured or provider not ready, return grounded fallback
        return self._fallback_answer(prompt)

    def _fallback_answer(self, prompt: str) -> str:
        # Deterministic grounded fallback: return the CONTEXT block (so answers are traceable)
        marker = "CONTEXT:"
        try:
            if marker in prompt:
                ctx = prompt.split(marker,1)[1].split("QUESTION:",1)[0].strip()
                answer = "[LLM not configured — returning concatenated context as grounded answer]\n\n" + ctx
                return answer
        except Exception:
            pass
        return "[LLM not configured and context not found in prompt]"
