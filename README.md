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

## Hero recipes and local artifacts

The optional Thoughtformula profile provides `thoughtform.recipes.HERO` and
`HERO_CODING`: small experimental general-language and coding model identities.
Recipes describe architecture; Thoughtformula owns initialization, training,
optimizer state and checkpoints. A model identity remains separate from a Being.

`artifacts.publish_artifact` writes a new passive JSON sidecar beside a selected
native `.tft` checkpoint. `load_artifact` checks complete checkpoint and tokenizer
hashes, bounded bytes, architecture, corpus identity and parent lineage before
returning a model and synchronous inference adapter. References are local single
filenames; importing Thoughtform performs no reads, training or downloads.
`specialize` loads fresh parent weights, preserves its tokenizer/architecture and
resets Adam for a separately identified child. Exact numerical resume uses the
Thoughtformula checkpoint directly and preserves Adam instead. No weights are
distributed by this package. Hypnosis owns job policy and evaluation.

`recipes.recipe_for(model_id, profile)` selects explicit `pilot-v1` or
`balanced-v2` recipes. Balanced Hero/Hero Coding use width 64, two layers, two
experts and hidden width 128; actual parameter count depends on the frozen
vocabulary. Existing pilot constants and artifact identities are unchanged.
