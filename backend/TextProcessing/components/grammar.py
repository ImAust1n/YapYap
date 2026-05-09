from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
# pyrefly: ignore [missing-import]
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Monkey-patch: transformers 5.x always spawns a background thread to check
# for safetensors conversion PRs on HuggingFace. Repos that have discussions
# disabled (e.g. prithivida/grammar_error_correcter_v1) respond with 403,
# crashing the thread and polluting the log. This patch silently suppresses it.
# pyrefly: ignore [missing-import]
import transformers.safetensors_conversion as _sc
_orig_auto_conversion = _sc.auto_conversion
def _patched_auto_conversion(*args, **kwargs):
    try:
        _orig_auto_conversion(*args, **kwargs)
    except Exception:
        pass  # Silently ignore 403 / discussions-disabled errors
_sc.auto_conversion = _patched_auto_conversion


MODEL_NAME = "prithivida/grammar_error_correcter_v1"


@dataclass
class GrammarCorrector:
    tokenizer: T5Tokenizer
    model: T5ForConditionalGeneration

    @classmethod
    def load(cls, model_name: str = MODEL_NAME) -> "GrammarCorrector":
        # Try loading from local cache first to avoid HF hub 403 errors
        # (prithivida/grammar_error_correcter_v1 has discussions disabled which
        #  causes transformers' auto_conversion thread to crash with 403).
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, use_safetensors=False, local_files_only=True
            )
        except Exception:
            # Cache miss — download on first run
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, use_safetensors=False
            )
        model.eval()
        return cls(tokenizer=tokenizer, model=model)

    def correct(self, text: str) -> str:
        input_text = "gec: " + text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_correct(self, sentences: Iterable[str]) -> List[str]:
        return [self.correct(sentence) for sentence in sentences]


GRAMMAR = GrammarCorrector.load()


def correct_grammar(text: str) -> str:
    return GRAMMAR.correct(text)


def batch_correct(sentences: Iterable[str]) -> List[str]:
    return GRAMMAR.batch_correct(sentences)