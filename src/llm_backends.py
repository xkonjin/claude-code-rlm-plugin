"""
LLM Backend Integration for RLM Plugin

Priority order for authentication:
1. ANTHROPIC_API_KEY (direct Anthropic API - fastest)
2. OPENAI_API_KEY (OpenAI API)
3. OPENROUTER_API_KEY (200+ models via single key - best flexibility)
4. Local models (Ollama, text-generation-webui)
5. Claude CLI (uses user's Claude Code auth via `claude -p` - zero config)
6. Rule-based fallback (always available, no LLM)

Inside Claude Code: if no API keys are set, Claude CLI backend activates
automatically using the user's existing Claude Code session authentication.
"""

import os
import json
import logging
import subprocess
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standard response structure for LLM queries"""
    content: str
    model_used: str
    provider: str
    processing_time_ms: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(available={self.is_available()})"


class AnthropicBackend(LLMBackend):
    """Anthropic API backend using standard API keys"""

    MODEL_MAP = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-6-20250610",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._client = None

        # Only use real API keys (not OAuth tokens, which need Bearer auth)
        key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if key and key.startswith("sk-ant-oat"):
            key = None  # OAuth tokens don't work with the API directly

        if key:
            try:
                import anthropic
                base_url = os.getenv('ANTHROPIC_BASE_URL')
                kwargs = {"api_key": key}
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = anthropic.Anthropic(**kwargs)
            except ImportError:
                logging.warning("anthropic library not installed. Run: pip install anthropic")

    @property
    def name(self) -> str:
        return "Anthropic"

    def is_available(self) -> bool:
        return self._client is not None

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="", model_used=model or "haiku",
                provider=self.name, processing_time_ms=0,
                error="Anthropic backend not available"
            )

        start_time = time.time()
        claude_model = self.MODEL_MAP.get(model, model or self.MODEL_MAP["haiku"])

        try:
            response = self._client.messages.create(
                model=claude_model,
                max_tokens=kwargs.get('max_tokens', 2000),
                messages=[{"role": "user", "content": prompt}]
            )
            processing_time = (time.time() - start_time) * 1000
            return LLMResponse(
                content=response.content[0].text,
                model_used=claude_model,
                provider=self.name,
                processing_time_ms=processing_time,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", model_used=claude_model,
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""

    MODEL_MAP = {
        "haiku": "gpt-4.1-mini",
        "sonnet": "gpt-4.1",
        "opus": "gpt-4.1",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._client = None
        key = api_key or os.getenv('OPENAI_API_KEY')

        if key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=key)
            except ImportError:
                logging.warning("openai library not installed. Run: pip install openai")

    @property
    def name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        return self._client is not None

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="", model_used=model or "gpt-4.1-mini",
                provider=self.name, processing_time_ms=0,
                error="OpenAI backend not available"
            )

        start_time = time.time()
        openai_model = self.MODEL_MAP.get(model, model or self.MODEL_MAP["haiku"])

        try:
            response = self._client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7)
            )
            processing_time = (time.time() - start_time) * 1000
            return LLMResponse(
                content=response.choices[0].message.content,
                model_used=openai_model,
                provider=self.name,
                processing_time_ms=processing_time,
                tokens_used=response.usage.total_tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", model_used=openai_model,
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend — access to 200+ models via single key.
    Uses OpenAI-compatible API at https://openrouter.ai/api/v1.
    Intelligent model selection: maps haiku/sonnet/opus tiers to best
    price/performance models available on OpenRouter."""

    # Tiered model selection: cheap/fast → balanced → premium
    # Each tier has a primary and fallback in case the primary is unavailable
    MODEL_MAP = {
        "haiku": "google/gemini-2.5-flash",           # fast, cheap
        "sonnet": "anthropic/claude-sonnet-4-5",       # balanced
        "opus": "anthropic/claude-opus-4-6",           # premium
    }

    # Fallbacks per tier if primary isn't routable
    MODEL_FALLBACKS = {
        "haiku": ["google/gemini-2.0-flash-001", "meta-llama/llama-4-maverick"],
        "sonnet": ["openai/gpt-4.1", "google/gemini-2.5-pro"],
        "opus": ["anthropic/claude-sonnet-4-5", "openai/gpt-4.1"],
    }

    def __init__(self, api_key: Optional[str] = None, site_url: Optional[str] = None):
        self._client = None
        key = api_key or os.getenv('OPENROUTER_API_KEY')
        self._site_url = site_url or os.getenv('OPENROUTER_SITE_URL', 'https://github.com/Plasma-Projects/claude-code-rlm-plugin')
        self._app_name = os.getenv('OPENROUTER_APP_NAME', 'RLM Plugin')

        if key:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except ImportError:
                logging.warning("openai library not installed. Run: pip install openai")

    @property
    def name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        return self._client is not None

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="", model_used=model or "openrouter",
                provider=self.name, processing_time_ms=0,
                error="OpenRouter backend not available — set OPENROUTER_API_KEY"
            )

        start_time = time.time()

        # Resolve model: abstract tier → OpenRouter model ID
        # If the user passes a full model ID (contains '/'), use it directly
        if model and '/' in model:
            or_model = model
        else:
            or_model = self.MODEL_MAP.get(model, self.MODEL_MAP["haiku"])

        extra_headers = {
            "HTTP-Referer": self._site_url,
            "X-Title": self._app_name,
        }

        try:
            response = self._client.chat.completions.create(
                model=or_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                extra_headers=extra_headers,
            )
            processing_time = (time.time() - start_time) * 1000
            return LLMResponse(
                content=response.choices[0].message.content,
                model_used=or_model,
                provider=self.name,
                processing_time_ms=processing_time,
                tokens_used=getattr(response.usage, 'total_tokens', None) if response.usage else None
            )
        except Exception as e:
            err_str = str(e)
            # If model not available, try fallbacks
            tier = model if model in self.MODEL_FALLBACKS else None
            if tier and ("not available" in err_str.lower() or "does not exist" in err_str.lower()):
                for fallback_model in self.MODEL_FALLBACKS[tier]:
                    try:
                        response = self._client.chat.completions.create(
                            model=fallback_model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=kwargs.get('max_tokens', 2000),
                            temperature=kwargs.get('temperature', 0.7),
                            extra_headers=extra_headers,
                        )
                        processing_time = (time.time() - start_time) * 1000
                        return LLMResponse(
                            content=response.choices[0].message.content,
                            model_used=fallback_model,
                            provider=self.name,
                            processing_time_ms=processing_time,
                            tokens_used=getattr(response.usage, 'total_tokens', None) if response.usage else None
                        )
                    except Exception:
                        continue

            return LLMResponse(
                content="", model_used=or_model,
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=err_str
            )


class LocalLLMBackend(LLMBackend):
    """Local LLM backend (Ollama, text-generation-webui, etc.)"""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        self.base_url = base_url
        self.default_model = model_name
        self._available = None

    @property
    def name(self) -> str:
        return "Local"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            self._available = response.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="", model_used=model or self.default_model,
                provider=self.name, processing_time_ms=0,
                error="Local LLM server not available"
            )

        start_time = time.time()
        model_to_use = model or self.default_model

        try:
            import requests

            # Ollama format
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', 0.7),
                        "num_predict": kwargs.get('max_tokens', 2000)
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result.get('response', ''),
                    model_used=model_to_use,
                    provider=self.name,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # OpenAI-compatible fallback
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model_to_use,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get('max_tokens', 2000),
                    "temperature": kwargs.get('temperature', 0.7)
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result['choices'][0]['message']['content'],
                    model_used=model_to_use,
                    provider=self.name,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=result.get('usage', {}).get('total_tokens')
                )

            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

        except Exception as e:
            return LLMResponse(
                content="", model_used=model_to_use,
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class ClaudeCLIBackend(LLMBackend):
    """Uses the claude CLI as a subprocess for LLM queries.
    Works when claude is installed and authenticated but no API key is available."""

    def __init__(self):
        self._available = None

    @property
    def name(self) -> str:
        return "Claude CLI"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=5
            )
            self._available = result.returncode == 0
            return self._available
        except Exception:
            self._available = False
            return False

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                content="", model_used="claude-cli",
                provider=self.name, processing_time_ms=0,
                error="Claude CLI not available"
            )

        start_time = time.time()
        cmd = ["claude", "-p", "--output-format", "text"]

        # Map model names to claude CLI model flags
        model_flag_map = {
            "haiku": "haiku",
            "sonnet": "sonnet",
            "opus": "opus",
        }
        if model and model in model_flag_map:
            cmd.extend(["--model", model_flag_map[model]])

        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=kwargs.get('timeout', 120)
            )
            processing_time = (time.time() - start_time) * 1000

            if result.returncode == 0:
                return LLMResponse(
                    content=result.stdout.strip(),
                    model_used=model or "claude-cli",
                    provider=self.name,
                    processing_time_ms=processing_time
                )
            else:
                # Mark unavailable to avoid repeated failures in this session
                self._available = False
                return LLMResponse(
                    content="", model_used=model or "claude-cli",
                    provider=self.name,
                    processing_time_ms=processing_time,
                    error=f"CLI error: {result.stderr.strip()[:200]}"
                )
        except subprocess.TimeoutExpired:
            self._available = False
            return LLMResponse(
                content="", model_used=model or "claude-cli",
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error="Claude CLI timed out"
            )
        except Exception as e:
            self._available = False
            return LLMResponse(
                content="", model_used=model or "claude-cli",
                provider=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class SimpleFallbackBackend(LLMBackend):
    """Rule-based fallback when no LLM is available"""

    @property
    def name(self) -> str:
        return "Fallback"

    def is_available(self) -> bool:
        return True

    def query(self, prompt: str, model: str = None, **kwargs) -> LLMResponse:
        start_time = time.time()
        prompt_lower = prompt.lower()

        if 'summarize' in prompt_lower or 'summary' in prompt_lower:
            content = self._summarize(prompt)
        elif 'extract' in prompt_lower or 'find' in prompt_lower:
            content = self._extract(prompt)
        elif 'analyze' in prompt_lower or 'analysis' in prompt_lower:
            content = self._analyze(prompt)
        else:
            content = self._generic(prompt)

        return LLMResponse(
            content=content,
            model_used=f"fallback-{model or 'default'}",
            provider=self.name,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    def _summarize(self, prompt: str) -> str:
        # Extract the data portion and provide basic stats
        lines = prompt.split('\n')
        data_lines = [l for l in lines if l.strip() and not l.startswith(('QUERY', 'INSTRUCTIONS', 'DATA', 'RESPONSE'))]
        return (
            f"[Fallback Summary] Content: {len(data_lines)} lines, "
            f"~{sum(len(l) for l in data_lines)} chars. "
            f"Configure an LLM backend for real analysis. "
            f"Set ANTHROPIC_API_KEY or run inside Claude Code for automatic auth."
        )

    def _extract(self, prompt: str) -> str:
        return (
            "[Fallback Extraction] Content chunked and ready for processing. "
            "No LLM backend available for semantic extraction. "
            "Set ANTHROPIC_API_KEY or run inside Claude Code for automatic auth."
        )

    def _analyze(self, prompt: str) -> str:
        return (
            "[Fallback Analysis] Content received and chunked. "
            "Rule-based processing only — no semantic analysis available. "
            "Set ANTHROPIC_API_KEY or run inside Claude Code for automatic auth."
        )

    def _generic(self, prompt: str) -> str:
        return (
            f"[Fallback] Processed query ({len(prompt)} chars). "
            f"For real LLM analysis, configure one of: "
            f"ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, "
            f"local Ollama, or run inside Claude Code for auto-auth."
        )


class LLMManager:
    """Manages multiple LLM backends with automatic fallback"""

    def __init__(self, preferred_backends: List[str] = None):
        self.backends: Dict[str, LLMBackend] = {}
        self.preferred_order = preferred_backends or [
            "anthropic", "openai", "openrouter", "local", "claude_cli", "fallback"
        ]
        self._initialize_backends()

    def _initialize_backends(self):
        # Anthropic API key (separate from Claude Code's OAuth)
        self.backends["anthropic"] = AnthropicBackend()

        # OpenAI
        self.backends["openai"] = OpenAIBackend()

        # OpenRouter (200+ models via single key)
        self.backends["openrouter"] = OpenRouterBackend()

        # Local models
        local_configs = [
            ("http://localhost:11434", "llama2"),   # Ollama
            ("http://localhost:8080", "default"),    # text-generation-webui
        ]
        for base_url, model in local_configs:
            backend = LocalLLMBackend(base_url, model)
            if backend.is_available():
                self.backends["local"] = backend
                break
        if "local" not in self.backends:
            self.backends["local"] = LocalLLMBackend()

        # Claude CLI subprocess
        self.backends["claude_cli"] = ClaudeCLIBackend()

        # Always-available fallback
        self.backends["fallback"] = SimpleFallbackBackend()

    def get_available_backend(self) -> LLMBackend:
        for name in self.preferred_order:
            if name in self.backends and self.backends[name].is_available():
                return self.backends[name]
        return self.backends["fallback"]

    def query(self, prompt: str, model: str = "haiku", **kwargs) -> LLMResponse:
        backend = self.get_available_backend()
        response = backend.query(prompt, model, **kwargs)

        # Auto-fallback on error
        if response.error and backend.name != "Fallback":
            logging.warning(f"LLM query failed with {backend.name}: {response.error}")
            return self.backends["fallback"].query(prompt, model, **kwargs)

        return response

    def create_query_function(self) -> Callable[[str, str], str]:
        def query_fn(prompt: str, model: str = "haiku") -> str:
            response = self.query(prompt, model)
            return response.content or f"[Error: {response.error}]"
        return query_fn

    def get_status(self) -> Dict[str, Any]:
        status = {}
        for name, backend in self.backends.items():
            status[name] = {
                "available": backend.is_available(),
                "name": backend.name
            }
        current = self.get_available_backend()
        status["current"] = current.name
        return status


# Thread-safe singleton
_llm_manager: Optional[LLMManager] = None
_manager_lock = threading.Lock()


def get_llm_manager() -> LLMManager:
    global _llm_manager
    if _llm_manager is None:
        with _manager_lock:
            if _llm_manager is None:
                _llm_manager = LLMManager()
    return _llm_manager


def create_llm_query_function() -> Callable[[str, str], str]:
    return get_llm_manager().create_query_function()
