# Thoughtform

Individual model definitions and model-neutral agency. Python 3.14+.

A `ModelDefinition` names an individual model and its generation policy. A
`Being` identifies a character, environmental mind or other participant that
may share that model. `Observation`, `Action` and `Policy` work with deterministic
rules and do not require a model or orchestration framework.

```python
from thoughtform import Being, GenerationConfig, ModelDefinition

model = ModelDefinition('alice-model', 'thoughtformula', 'caller-owned-checkpoint',
                        GenerationConfig(max_new_tokens=24))
alice = Being('alice', 'Alice', model=model, goals=('Explore the woods',))
```

For real token inference, install `uv sync --extra thoughtformula`, construct a
compatible Thoughtformula `ModernMicroGPT` and `BPETokenizer` explicitly, then
use `thoughtform.thoughtformula.ThoughtformulaModel(model, transformer, tokenizer)`.
Its synchronous `stream(prompt)` yields `Token(token_id, text)` and closes the
underlying generator on early exit. The caller owns artifacts, weights, model
access serialization and execution resources. This adapter never loads, downloads
or trains a model. Use Hypnosis's execution adapter before consuming on a UI loop.

The initial profile rejects unknown tokenizer text and prompt-plus-output beyond
the model's declared context capacity. Zero output budget performs no inference;
EOS ends the stream. Speculative generation, persisted lineage, memories and
specialization are later owning tasks.

Verify with `PYTHONPATH=src python -m unittest discover -s tests`; the optional
profile adds actual tiny untrained Transformer tests. These prove integration,
not language quality or agent capability.

[Full implementation plan](https://github.com/wispsy/wispsy/blob/main/docs/cohesion-block-b/IMPLEMENTATION_PLAN.md)
and [Project](https://github.com/orgs/wispsy/projects/28) track the complete family.
