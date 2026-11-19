"""Explicit adapter over caller-owned Thoughtformula model/tokenizer instances."""
from dataclasses import dataclass

from .contracts import ModelDefinition


@dataclass(frozen=True)
class Token:
    token_id: int
    text: str


class ThoughtformulaModel:
    """Synchronous incremental inference; does not load or train a model.

    Model/cache access must be serialized by the caller. Iteration performs
    model computation; use an execution adapter before consuming on a UI loop.
    """

    def __init__(self, definition, model, tokenizer, *, use_int8=False):
        if type(definition) is not ModelDefinition or definition.provider != 'thoughtformula':
            raise ValueError('Select a Thoughtformula model definition explicitly')
        if type(use_int8) is not bool:
            raise TypeError('use_int8 must be a Boolean')
        if model.vocab_size != len(tokenizer.vocab):
            raise ValueError('Model and tokenizer vocabulary sizes differ')
        if type(model.max_seq_len) is not int or model.max_seq_len <= 0:
            raise ValueError('Model must declare a positive context capacity')
        self.definition = definition
        self.model = model
        self.tokenizer = tokenizer
        self.use_int8 = use_int8

    def stream(self, prompt):
        if type(prompt) is not str or len(prompt) > 65536:
            raise ValueError('Prompt requires bounded text')
        config = self.definition.generation
        if config.max_new_tokens == 0 or not prompt:
            return
        ids = self.tokenizer.encode(prompt)
        if self.tokenizer.decode(ids) != prompt:
            raise ValueError('Prompt contains text absent from the selected tokenizer')
        if not ids or any(type(value) is not int or not 0 <= value < self.model.vocab_size for value in ids):
            raise ValueError('Prompt encoding is outside the selected vocabulary')
        if len(ids) + config.max_new_tokens > self.model.max_seq_len:
            raise ValueError('Prompt plus generation budget exceeds context capacity')
        iterator = iter(self.model.generate_stream(ids, max_new_tokens=config.max_new_tokens,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p,
            stop_token_id=self.tokenizer.eos_id, use_int8=self.use_int8))
        try:
            for _ in range(config.max_new_tokens):
                try:
                    token_id = next(iterator)
                except StopIteration:
                    return
                if type(token_id) is not int or not 0 <= token_id < self.model.vocab_size:
                    raise ValueError('Generated token is outside the selected vocabulary')
                if token_id == self.tokenizer.eos_id:
                    return
                yield Token(token_id, self.tokenizer.decode([token_id], skip_special=True))
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
