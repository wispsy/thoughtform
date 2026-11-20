"""Passive local model identities and explicit specialization lineage."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from .contracts import GenerationConfig, ModelDefinition
from .recipes import ModelRecipe, identity
from .thoughtformula import ThoughtformulaModel

SCHEMA = 'thoughtform-artifact/1'
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected a SHA-256 identity')
    return value


def checkpoint_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError('Select a bounded ordinary local checkpoint')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        consumed = 0
        while chunk := stream.read(65536):
            consumed += len(chunk)
            if consumed > MAX_CHECKPOINT_BYTES:
                raise ValueError('Checkpoint changed beyond the byte ceiling')
            result.update(chunk)
    return result.hexdigest(), consumed


def artifact_metadata(recipe, corpus_sha256, parent_sha256=None):
    if type(recipe) is not ModelRecipe:
        raise TypeError('Select a ModelRecipe')
    if parent_sha256 is not None:
        digest(parent_sha256)
    if (recipe.parent_model_id is None) != (parent_sha256 is None):
        raise ValueError('Specialization requires explicit parent lineage')
    return {'model_id': recipe.model_id, 'recipe_sha256': recipe.sha256,
            'corpus_sha256': digest(corpus_sha256), 'parent_sha256': parent_sha256}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate artifact field')
        result[key] = value
    return result


@dataclass
class LoadedArtifact:
    recipe: ModelRecipe
    model: object
    tokenizer: object
    training_config: object
    state: object
    receipt: dict

    def adapter(self, *, generation=None):
        definition = ModelDefinition(self.recipe.model_id, 'thoughtformula',
            'sha256:' + self.receipt['checkpoint']['sha256'],
            generation or GenerationConfig(max_new_tokens=32, temperature=0))
        return ThoughtformulaModel(definition, self.model, self.tokenizer)


def publish_artifact(destination, checkpoint, recipe, *, corpus_sha256, parent_sha256=None):
    """Write a new sidecar beside a complete native training checkpoint."""
    from thoughtformula import load_training_checkpoint
    destination, checkpoint = Path(destination), Path(checkpoint)
    if destination.parent.resolve() != checkpoint.parent.resolve() or destination == checkpoint:
        raise ValueError('Publish beside the selected checkpoint under a distinct filename')
    sha256, size = checkpoint_identity(checkpoint)
    model, tokenizer, _config, _state, metadata = load_training_checkpoint(checkpoint)
    expected = artifact_metadata(recipe, corpus_sha256, parent_sha256)
    if metadata != expected:
        raise ValueError('Checkpoint metadata does not match the selected model lineage')
    for key, value in recipe.configuration(len(tokenizer.vocab)).items():
        if getattr(model, key) != value:
            raise ValueError('Checkpoint architecture differs from the selected recipe')
    receipt = {'schema': SCHEMA, 'status': 'experimental', 'recipe': asdict(recipe),
               **expected, 'tokenizer_sha256': identity(tokenizer.to_dict()),
               'checkpoint': {'file': checkpoint.name, 'sha256': sha256, 'bytes': size}}
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(raw)
    return receipt


def load_artifact(path):
    """Resolve a caller-selected local sidecar; never import code or download."""
    from thoughtformula import load_training_checkpoint
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('Artifact sidecar exceeds its byte ceiling')
    receipt = json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda _v: (_ for _ in ()).throw(ValueError('Nonfinite artifact value')))
    fields = {'schema', 'status', 'recipe', 'model_id', 'recipe_sha256', 'corpus_sha256',
              'parent_sha256', 'tokenizer_sha256', 'checkpoint'}
    if type(receipt) is not dict or set(receipt) != fields or receipt['schema'] != SCHEMA or receipt['status'] != 'experimental':
        raise ValueError('Unsupported model artifact')
    recipe = ModelRecipe(**receipt['recipe'])
    expected = artifact_metadata(recipe, receipt['corpus_sha256'], receipt['parent_sha256'])
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError('Model identity or lineage mismatch')
    entry = receipt['checkpoint']
    if type(entry) is not dict or set(entry) != {'file', 'sha256', 'bytes'}:
        raise ValueError('Invalid checkpoint reference')
    filename = entry['file']
    if type(filename) is not str or not filename or Path(filename).name != filename or '/' in filename or '\\' in filename or filename in ('.', '..'):
        raise ValueError('Checkpoint reference must be one local filename')
    checkpoint = path.parent / filename
    sha256, size = checkpoint_identity(checkpoint)
    if (sha256, size) != (entry['sha256'], entry['bytes']):
        raise ValueError('Checkpoint bytes differ from the selected artifact')
    model, tokenizer, config, state, metadata = load_training_checkpoint(checkpoint)
    if metadata != expected or identity(tokenizer.to_dict()) != receipt['tokenizer_sha256']:
        raise ValueError('Checkpoint tokenizer or lineage mismatch')
    if any(getattr(model, key) != value for key, value in recipe.configuration(len(tokenizer.vocab)).items()):
        raise ValueError('Model architecture differs from its recipe')
    return LoadedArtifact(recipe, model, tokenizer, config, state, receipt)


def specialize(parent_artifact, recipe):
    """Load fresh parent weights and reset Adam for a new model trajectory."""
    from thoughtformula import reset_optimizer
    parent = load_artifact(parent_artifact)
    if recipe.parent_model_id != parent.recipe.model_id:
        raise ValueError('Recipe selects a different parent model')
    if recipe.configuration(len(parent.tokenizer.vocab)) != parent.recipe.configuration(len(parent.tokenizer.vocab)):
        raise ValueError('Specialization must preserve parent architecture and tokenizer')
    reset_optimizer(parent.model)
    return parent.model, parent.tokenizer, parent.receipt['checkpoint']['sha256']
