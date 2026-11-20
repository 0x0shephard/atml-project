import json

import pytest
import torch

TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable (offline?): {e}")


def make_tiny(tok, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    m = Qwen2ForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = tok.eos_token_id
    m.generation_config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    return m


@pytest.fixture()
def tiny(tok):
    return make_tiny(tok)


@pytest.fixture()
def problems_file(tmp_path):
    probs = [
        {"id": "math500-000", "source": "math500", "problem": "What is 6*7?", "answer": "42", "subject": None, "level": 1},
        {"id": "gsm8k-0000", "source": "gsm8k", "problem": "Tom has 3 apples and buys 2. How many?", "answer": "5",
         "subject": None, "level": None},
        {"id": "aime24-00", "source": "aime24", "problem": "Find 2+2.", "answer": "4", "subject": None, "level": None},
    ]
    p = tmp_path / "problems.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in probs) + "\n")
    return p
