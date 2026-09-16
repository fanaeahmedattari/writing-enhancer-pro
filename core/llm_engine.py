"""
core/llm_engine.py
Production-ready Multi-Provider LLM Rewriting Engine.

Features:
- Multi-provider support: Google Gemini (google-genai SDK), OpenAI (direct), OpenRouter (via openai base_url)
- Exponential backoff retry with jitter for rate-limit (429) and transient failures
- Pre-processing: Unicode/steganography stripping via unicode_cleaner
- Post-processing: AI preamble stripping, leak token removal, markdown wrapper cleanup
- Hook-and-Eye cross-chunk context stitching via prompt_rules
- Batch chunk processing with Streamlit progress callback
- Configurable temperature, max tokens, and user-selectable rewrite options
"""

from __future__ import annotations

import os
import re
import random
import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.prompt_rules import (
    SYSTEM_HUMANIZER_PROMPT,
    USER_REWRITE_TEMPLATE,
    LLM_LEAK_PATTERNS,
    get_user_rewrite_prompt,
    sanitize_ai_markers,
)
from core.unicode_cleaner import sanitize_unicode

logger = logging.getLogger(__name__)

# =========================================================================== #
# Provider & Model Definitions
# =========================================================================== #

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


# Default model per provider
DEFAULT_MODELS: Dict[str, str] = {
    LLMProvider.GEMINI: "gemini-flash-latest",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.OPENROUTER: "anthropic/claude-3.5-sonnet",
}

# Automatic migration map for deprecated Gemini model identifiers
DEPRECATED_GEMINI_MIGRATIONS: Dict[str, str] = {
    "gemini-2.5-flash": "gemini-flash-latest",
    "gemini-2.5-flash-lite": "gemini-flash-lite-latest",
    "gemini-2.0-flash": "gemini-flash-latest",
    "gemini-2.0-flash-exp": "gemini-flash-latest",
    "gemini-1.5-flash": "gemini-flash-latest",
    "gemini-1.5-pro": "gemini-flash-latest",
}

# Available model choices per provider (shown in UI)
AVAILABLE_MODELS: Dict[str, List[str]] = {
    LLMProvider.GEMINI: [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ],
    LLMProvider.OPENAI: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    LLMProvider.OPENROUTER: [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-2.0-flash-001",
        "google/gemini-1.5-pro",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
    ],
}

# =========================================================================== #
# Post-Processing: AI Output Sanitizer
# =========================================================================== #

# Patterns to strip from LLM output (preambles, wrappers, meta-talk)
_PREAMBLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:Here(?:'s| is)(?: the)? (?:the )?(?:re-?written|humanized|revised|cleaned|edited|updated)(?: version)?(?: of)?(?: the)?(?: text)?[:\s]*)", re.I),
    re.compile(r"^(?:Sure(?:,| -)? (?:here(?:'s| is))?.*?[:\s]*)", re.I),
    re.compile(r"^(?:Certainly[!,.]?\s*(?:Here(?:'s| is))?.*?[:\s]*)", re.I),
    re.compile(r"^(?:Of course[!,.]?\s*(?:Here(?:'s| is))?.*?[:\s]*)", re.I),
    re.compile(r"^(?:Below is (?:the )?(?:re-?written|humanized|revised).*?[:\s]*)", re.I),
]

_POSTAMBLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\n+---+\s*\n+(?:Note|Explanation|Changes made|Key changes|Summary of changes|I (?:have )?made).*$", re.I | re.DOTALL),
    re.compile(r"\n+\*\*(?:Note|Changes|Summary)\*\*.*$", re.I | re.DOTALL),
    re.compile(r"\n+Let me know if .*$", re.I | re.DOTALL),
    re.compile(r"\n+(?:I hope this|Feel free to|Please let me know).*$", re.I | re.DOTALL),
]


def sanitize_llm_output(text: str) -> str:
    """
    Strips AI preambles, markdown code wrappers, leak tokens, and post-explanation
    commentary from LLM output to return only the rewritten text.
    """
    if not text:
        return text

    cleaned = text.strip()

    # 1. Strip markdown code block wrappers (```text ... ``` or ```markdown ... ```)
    code_block_match = re.match(r"^```(?:text|markdown|plain)?\s*\n(.*?)```\s*$", cleaned, re.DOTALL)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    # 2. Strip wrapping quotation marks
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        inner = cleaned[1:-1].strip()
        if len(inner) > 50:  # Only unwrap if it's substantial text, not a quoted phrase
            cleaned = inner

    # 3. Strip AI preambles
    for pattern in _PREAMBLE_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()

    # 4. Strip postamble meta-commentary
    for pattern in _POSTAMBLE_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()

    # 5. Strip LLM leak tokens (oaicite, turn0search0, contentReference, etc.)
    for leak_re in LLM_LEAK_PATTERNS:
        cleaned = leak_re.sub("", cleaned)

    # 6. Collapse excessive whitespace but preserve paragraph breaks
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # 7. Post-processing Anti-AI Sanitizer: eradicate em-dashes and hidden watermarks
    cleaned = sanitize_ai_markers(cleaned)

    return cleaned.strip()


# =========================================================================== #
# LLM Humanizer Engine
# =========================================================================== #

class LLMHumanizerEngine:
    """
    Multi-provider LLM engine that processes document chunks through the
    anti-AI humanization pipeline with retry, rate-limiting, and output sanitization.

    Supported providers:
        - Google Gemini (via google-genai SDK)
        - OpenAI (direct API)
        - OpenRouter (via openai SDK with custom base_url)
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "openrouter",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.provider = LLMProvider(provider.lower())
        raw_model = model_name or DEFAULT_MODELS.get(self.provider, "gemini-3.6-flash")
        if self.provider == LLMProvider.GEMINI:
            self.model_name = DEPRECATED_GEMINI_MIGRATIONS.get(raw_model, raw_model)
        else:
            self.model_name = raw_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.last_error: Optional[str] = None

        # Initialize provider-specific client
        self._gemini_client = None
        self._openai_client = None

        if self.provider == LLMProvider.GEMINI:
            self._init_gemini()
        elif self.provider == LLMProvider.OPENAI:
            self._init_openai(base_url=None)
        elif self.provider == LLMProvider.OPENROUTER:
            self._init_openai(base_url="https://openrouter.ai/api/v1")

    # ------------------------------------------------------------------ #
    # Provider Initialization
    # ------------------------------------------------------------------ #

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client via the new google-genai SDK."""
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
            logger.info("Gemini client initialized (google-genai SDK)")
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini provider. "
                "Install it with: pip install google-genai"
            )

    def _init_openai(self, base_url: Optional[str] = None) -> None:
        """Initialize OpenAI-compatible client (works for OpenAI direct and OpenRouter)."""
        try:
            from openai import OpenAI
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_client = OpenAI(**kwargs)
            provider_label = "OpenRouter" if base_url else "OpenAI"
            logger.info(f"{provider_label} client initialized")
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

    # ------------------------------------------------------------------ #
    # Single Chunk Processing (Core API Call)
    # ------------------------------------------------------------------ #

    def process_single_chunk(
        self,
        chunk_text: str,
        section_type: str = "general",
        aggressive: bool = False,
        options: Optional[Dict[str, bool]] = None,
        prev_context_tail: str = "",
        length_mode: str = "preserve",
        target_min: Optional[int] = None,
        target_max: Optional[int] = None,
        work_mode: str = "journal",
        english_tone: str = "academic",
        strict_mode: bool = False,
        global_doc_context: str = "",
        retries: int = 3,
    ) -> str:
        """
        Processes a single text chunk through the full humanization pipeline:
        1. Pre-clean: Strip invisible Unicode & steganography markers
        2. Prompt Construction: Build section-aware, length-constrained prompt with stitching context
        3. API Call: Send to LLM with exponential backoff retry
        4. Post-clean: Strip preambles, leak tokens, and markdown wrappers
        5. Final Unicode pass: Sanitize output for any injected invisible characters
        """
        # --- Step 1: Pre-clean input ---
        pre_cleaned, pre_stats = sanitize_unicode(chunk_text, normalize_quotes=False)

        if pre_stats.get("removed_invisibles", 0) > 0:
            logger.info(
                f"Pre-clean removed {pre_stats['removed_invisibles']} invisible chars, "
                f"normalized {pre_stats['normalized_spaces']} spaces"
            )

        # --- Step 2: Build prompt ---
        user_prompt = get_user_rewrite_prompt(
            text_chunk=pre_cleaned,
            section_type=section_type,
            aggressive=aggressive,
            options=options,
            prev_context_tail=prev_context_tail,
            length_mode=length_mode,
            target_min=target_min,
            target_max=target_max,
            work_mode=work_mode,
            english_tone=english_tone,
            strict_mode=strict_mode,
            global_doc_context=global_doc_context,
        )

        # --- Step 3: API call with retry ---
        raw_output = self._call_with_retry(user_prompt, retries=retries)

        if raw_output is None:
            err_msg = self.last_error or "All retry attempts exhausted"
            logger.error(f"LLM generation failed: {err_msg}")
            raise RuntimeError(
                f"LLM processing failed ({self.provider.value} / {self.model_name}): {err_msg}. "
                "Please verify your API key, check model selection in the sidebar, or retry."
            )

        # --- Step 4: Post-clean LLM output ---
        cleaned_output = sanitize_llm_output(raw_output)

        # --- Step 5: Final Unicode sanitization pass on output ---
        final_output, post_stats = sanitize_unicode(cleaned_output, normalize_quotes=False)

        if post_stats.get("removed_invisibles", 0) > 0:
            logger.info(
                f"Post-clean removed {post_stats['removed_invisibles']} invisible chars from LLM output"
            )

        return final_output

    # ------------------------------------------------------------------ #
    # Batch Processing (All Chunks)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tail_sentence(text: str) -> str:
        clean = text.strip()
        if not clean:
            return ""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if s.strip()]
        for s in reversed(sentences):
            if len(s.split()) >= 4:
                return s
        return sentences[-1] if sentences else ""

    def process_all_chunks(
        self,
        chunks: List[Dict[str, Any]],
        aggressive: bool = False,
        options: Optional[Dict[str, bool]] = None,
        length_mode: str = "preserve",
        target_min: Optional[int] = None,
        target_max: Optional[int] = None,
        work_mode: str = "journal",
        english_tone: str = "academic",
        strict_mode: bool = False,
        global_doc_context: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        inter_chunk_delay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Iterates over all document chunks, processes them sequentially with
        dynamic context stitching (linking to the actual newly generated sentences),
        global manuscript context, and reports progress to the Streamlit frontend.
        """
        results: List[Dict[str, Any]] = []
        total_chunks = len(chunks)
        last_generated_tail = ""

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", idx)
            section_type = chunk.get("section_type", "general")
            original_text = chunk.get("text", "")

            # Dynamic Stitching: Use real newly generated tail from previous chunk if available
            prev_tail = last_generated_tail if (idx > 0 and last_generated_tail) else chunk.get("prev_context_tail", "")

            # Report progress
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_chunks,
                    f"Processing chunk {idx + 1}/{total_chunks} ({section_type})..."
                )

            # If section is an abstract and no custom range provided, default to standard IMRaD 200-300 word range
            chunk_length_mode = length_mode
            chunk_target_min = target_min
            chunk_target_max = target_max
            if section_type.lower() == "abstract" and length_mode == "preserve" and target_min is None:
                orig_w = len(original_text.split())
                if orig_w >= 100:
                    chunk_length_mode = "target_range"
                    chunk_target_min = max(200, min(orig_w - 20, 250))
                    chunk_target_max = max(250, min(orig_w + 30, 300))

            try:
                humanized_text = self.process_single_chunk(
                    chunk_text=original_text,
                    section_type=section_type,
                    aggressive=aggressive,
                    options=options,
                    prev_context_tail=prev_tail,
                    length_mode=chunk_length_mode,
                    target_min=chunk_target_min,
                    target_max=chunk_target_max,
                    work_mode=work_mode,
                    english_tone=english_tone,
                    strict_mode=strict_mode,
                    global_doc_context=global_doc_context or "",
                )
                status = "success"
                # Update stitching tail for next chunk with the REAL newly generated sentence
                last_generated_tail = self._extract_tail_sentence(humanized_text)
            except Exception as e:
                logger.error(f"Chunk {chunk_id} failed: {e}")
                humanized_text = original_text  # Fallback to original
                status = f"error: {str(e)}"
                last_generated_tail = self._extract_tail_sentence(original_text)

            results.append({
                "chunk_id": chunk_id,
                "original_text": original_text,
                "humanized_text": humanized_text,
                "section_type": section_type,
                "word_count": len(humanized_text.split()),
                "status": status,
                "para_indices": chunk.get("para_indices", []),
                "elements": chunk.get("elements", []),
            })

            # Rate-limit delay between chunks (skip after last chunk)
            if idx < total_chunks - 1 and inter_chunk_delay > 0:
                # If using Gemini, enforce minimum safe delay (4.0s) to never exceed 15 RPM free tier limit
                actual_delay = max(inter_chunk_delay, 4.0) if self.provider == LLMProvider.GEMINI else inter_chunk_delay
                time.sleep(actual_delay)


        # Final progress update
        if progress_callback:
            progress_callback(total_chunks, total_chunks, "All chunks processed.")

        return results

    # ------------------------------------------------------------------ #
    # Internal: API Call with Exponential Backoff
    # ------------------------------------------------------------------ #

    def _call_with_retry(self, user_prompt: str, retries: int = 3) -> Optional[str]:
        """
        Dispatches the API call to the configured provider with exponential
        backoff + jitter and fast per-attempt model failover for Gemini.
        """
        last_error: Optional[Exception] = None
        gemini_fallbacks = AVAILABLE_MODELS.get(LLMProvider.GEMINI, [self.model_name])

        for attempt in range(retries):
            # Select model: use self.model_name on first attempt, rotate on subsequent attempts
            current_model = self.model_name
            if self.provider == LLMProvider.GEMINI:
                current_model = DEPRECATED_GEMINI_MIGRATIONS.get(current_model, current_model)
                if attempt > 0:
                    idx = attempt % len(gemini_fallbacks)
                    current_model = gemini_fallbacks[idx]

            try:
                if self.provider == LLMProvider.GEMINI:
                    return self._call_gemini(user_prompt, model_name=current_model)
                else:
                    return self._call_openai_compatible(user_prompt)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str or "quota" in error_str
                is_server_error = any(code in error_str for code in ["500", "502", "503", "overloaded", "unavailable"])
                is_not_found = any(k in error_str for k in ["404", "not found", "no longer available"])

                if is_rate_limit or is_server_error or is_not_found:
                    base_delay = (1.5 ** attempt) * (1 + random.random() * 0.5)
                    wait_time = min(base_delay, 5.0) if not is_not_found else 0.5

                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} on {current_model} failed "
                        f"({'deprecated/not-found' if is_not_found else ('rate-limit' if is_rate_limit else 'server-error')}). "
                        f"Retrying with alternative model in {wait_time:.1f}s... Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error on attempt {attempt + 1} with {current_model}: {e}")
                    if attempt == retries - 1:
                        break
                    time.sleep(1)

        logger.error(f"All {retries} attempts failed. Last error: {last_error}")
        self.last_error = str(last_error) if last_error else "Unknown error"
        return None

    # ------------------------------------------------------------------ #
    # Provider-Specific API Calls
    # ------------------------------------------------------------------ #

    def _call_gemini(self, user_prompt: str, model_name: Optional[str] = None) -> str:
        """Call Google Gemini API via google-genai SDK."""
        from google.genai import types

        target_model = model_name or self.model_name
        response = self._gemini_client.models.generate_content(
            model=target_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_HUMANIZER_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        if response.text:
            return response.text
        raise ValueError(f"Gemini model {target_model} returned empty response")

    def _call_openai_compatible(self, user_prompt: str) -> str:
        """Call OpenAI or OpenRouter API via openai SDK."""
        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_HUMANIZER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        content = response.choices[0].message.content
        if content:
            return content.strip()
        raise ValueError("OpenAI/OpenRouter returned empty response")

    # ------------------------------------------------------------------ #
    # Utility: Quick Text Rewrite (No Chunking)
    # ------------------------------------------------------------------ #

    def rewrite_text(
        self,
        text: str,
        section_type: str = "general",
        aggressive: bool = False,
        options: Optional[Dict[str, bool]] = None,
        length_mode: str = "preserve",
        target_min: Optional[int] = None,
        target_max: Optional[int] = None,
        work_mode: str = "journal",
        english_tone: str = "academic",
        strict_mode: bool = False,
        retries: int = 3,
    ) -> str:
        """Convenience method to humanize a plain text string."""
        return self.process_single_chunk(
            chunk_text=text,
            section_type=section_type,
            aggressive=aggressive,
            options=options,
            length_mode=length_mode,
            target_min=target_min,
            target_max=target_max,
            work_mode=work_mode,
            english_tone=english_tone,
            strict_mode=strict_mode,
            retries=retries,
        )
