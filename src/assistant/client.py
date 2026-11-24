import os

from ..config import (
    DEFAULT_LLM_MODELS,
    LLM_API_KEY_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM API is not configured correctly."""


class LLMError(RuntimeError):
    """Raised when the LLM API call fails or returns no text."""


def detect_provider(environ=None):
    """Read the provider from config, then fall back to whichever key is set."""
    environ = environ if environ is not None else os.environ
    configured = environ.get(LLM_PROVIDER_ENV, "").strip().lower()
    if configured:
        if configured not in LLM_API_KEY_ENV:
            raise LLMConfigurationError(
                f"Unsupported {LLM_PROVIDER_ENV}={configured!r}; "
                f"choose from {sorted(LLM_API_KEY_ENV)}."
            )
        return configured

    for provider in LLM_API_KEY_ENV:
        if environ.get(LLM_API_KEY_ENV[provider]):
            return provider

    raise LLMConfigurationError(
        "No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
        f"or set {LLM_PROVIDER_ENV} explicitly."
    )


class LLMClient:
    """Thin LLM API wrapper; API keys are read from the environment only."""

    def __init__(
        self, provider=None, model=None, api_key=None, transport=None, environ=None
    ):
        self.environ = environ if environ is not None else os.environ
        self.provider = (provider or detect_provider(self.environ)).lower()
        if self.provider not in LLM_API_KEY_ENV:
            raise LLMConfigurationError(
                f"Unsupported provider {self.provider!r}; "
                f"choose from {sorted(LLM_API_KEY_ENV)}."
            )

        self.model = model or self.environ.get(LLM_MODEL_ENV) or DEFAULT_LLM_MODELS[self.provider]
        self.api_key = api_key or self.environ.get(LLM_API_KEY_ENV[self.provider])
        if not self.api_key and transport is None:
            raise LLMConfigurationError(
                f"Set {LLM_API_KEY_ENV[self.provider]} in your environment "
                "before using the LLM API."
            )

        self._transport = transport
        self._sdk_client = None

    def __repr__(self):
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model!r}, "
            "api_key='***')"
        )

    def chat(self, messages, max_tokens=300, temperature=0.2):
        """Send messages to the provider and return the assistant text."""
        if not messages:
            raise ValueError("messages must contain at least one message")

        text = self._call(messages, max_tokens=max_tokens, temperature=temperature)
        if not isinstance(text, str) or not text.strip():
            raise LLMError("The LLM returned an empty response.")
        return text.strip()

    def _call(self, messages, max_tokens, temperature):
        if self._transport is not None:
            return self._transport(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if self.provider == "openai":
            return self._call_openai(messages, max_tokens, temperature)
        return self._call_anthropic(messages, max_tokens, temperature)

    def _call_openai(self, messages, max_tokens, temperature):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "Install the OpenAI SDK with: pip install 'openai>=1.40,<3'"
            ) from exc

        if self._sdk_client is None:
            self._sdk_client = OpenAI(api_key=self.api_key)
        response = self._sdk_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, messages, max_tokens, temperature):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "Install the Anthropic SDK with: pip install 'anthropic>=0.40,<3'"
            ) from exc

        if self._sdk_client is None:
            self._sdk_client = Anthropic(api_key=self.api_key)
        system = "\n\n".join(
            message["content"]
            for message in messages
            if message["role"] == "system"
        )
        conversation = [
            message for message in messages if message["role"] != "system"
        ]
        response = self._sdk_client.messages.create(
            model=self.model,
            system=system,
            messages=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
