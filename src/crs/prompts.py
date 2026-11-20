"""Prompt construction and tokenizer-derived constants."""

from __future__ import annotations

from functools import lru_cache

from .config import ModelConfig, generation_defaults
from .record import TokenInfo


@lru_cache(maxsize=8)
def load_tokenizer(path: str):  # noqa: ANN201
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def prompt_ids(problem: dict, model_cfg: ModelConfig, tokenizer, suffix: str | None = None) -> list[int]:  # noqa: ANN001
    suffix = generation_defaults()["prompt_suffix"] if suffix is None else suffix
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem["problem"] + suffix}],
        add_generation_prompt=True,
        tokenize=False,
        **model_cfg.chat_template_kwargs,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def stop_ids(tokenizer, model_path: str | None = None) -> set[int]:  # noqa: ANN001
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for t in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"):
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    if model_path:
        try:
            from transformers import GenerationConfig

            eos = GenerationConfig.from_pretrained(model_path).eos_token_id
            ids.update([eos] if isinstance(eos, int) else (eos or []))
        except Exception:  # noqa: BLE001 - no generation_config.json
            pass
    ids.discard(None)
    return ids  # type: ignore[return-value]


def token_info(model_cfg: ModelConfig, tokenizer, hesitation_ids: set[int] | None = None) -> TokenInfo:  # noqa: ANN001
    return TokenInfo(
        think_end_ids=tokenizer.encode(model_cfg.think_end, add_special_tokens=False),
        stop_ids=stop_ids(tokenizer, model_cfg.hf_id),
        hesitation_ids=hesitation_ids or set(),
    )
