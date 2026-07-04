"""
LLM text generation with caching via litellm + hashstash.

Requires optional dependencies: pip install abstraction[llm]
"""

import os

from .config import PATH_STASH
from .utils import parse_json_str

DEFAULT_MODEL = "gemini/gemini-2.5-pro"


def _get_stash():
    try:
        from hashstash import HashStash
    except ImportError:
        raise ImportError("Install llm extras: pip install abstraction[llm]")
    return HashStash(os.path.join(PATH_STASH, "llms"), engine="pairtree")


def get_llm_stash():
    """Get the LLM response cache (pairtree HashStash)."""
    return _get_stash()


def generate_text(user_prompt, model=DEFAULT_MODEL, system_prompt="",
                  verbose=True, temperature=0.0, max_tokens=100_000,
                  use_cache=True, force=False, **options):
    """Generate text using an LLM, with optional caching."""
    try:
        from litellm import completion
    except ImportError:
        raise ImportError("Install llm extras: pip install abstraction[llm]")

    cache_key = {
        "user_prompt": user_prompt, "model": model,
        "system_prompt": system_prompt, "temperature": temperature,
        "max_tokens": max_tokens, **options,
    }

    if use_cache and not force:
        stash = _get_stash()
        if cache_key in stash:
            return stash[cache_key]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = completion(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
        stream=True, **options,
    )

    tokens = []
    for token_obj in response:
        token = token_obj.choices[0].delta.content or ""
        if verbose:
            print(token, end="", flush=True)
        tokens.append(token)

    result = "".join(tokens)
    if use_cache:
        _get_stash()[cache_key] = result
    return result


def generate_json(user_prompt, model=DEFAULT_MODEL, system_prompt="", **options):
    """Generate text and parse the response as JSON."""
    response = generate_text(
        user_prompt=user_prompt, model=model,
        system_prompt=system_prompt, **options,
    )
    return parse_json_str(response)
