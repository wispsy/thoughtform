"""Named experimental recipes; computation remains in Thoughtformula."""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from .contracts import identifier


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModelRecipe:
    model_id: str
    role: str
    parent_model_id: str | None = None
    version: str = 'pilot-v1'
    width: int = 48
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    experts: int = 2
    hidden: int = 96
    lora_rank: int = 4

    def __post_init__(self):
        for name in ('model_id', 'role', 'version'):
            identifier(getattr(self, name), name)
        if self.parent_model_id is not None:
            identifier(self.parent_model_id, 'Parent model ID')
        self.configuration(128)

    def configuration(self, vocabulary_size):
        from thoughtformula import validate_model_config
        return validate_model_config({
            'vocab_size': vocabulary_size, 'n_embd': self.width,
            'n_heads': self.heads, 'n_kv_heads': self.kv_heads,
            'n_layers': self.layers, 'n_experts': self.experts,
            'moe_top_k': min(2, self.experts), 'expert_hidden': self.hidden,
            'lora_rank': self.lora_rank, 'lora_alpha': 8.0,
            'moe_aux_loss_coeff': 0.01, 'use_flash_attention': False,
            'flash_block_size': 32,
        })

    @property
    def sha256(self):
        return identity(asdict(self))


HERO = ModelRecipe('hero', 'general-knowledge')
HERO_CODING = ModelRecipe('hero-coding', 'coding', parent_model_id='hero')
HERO_BALANCED = ModelRecipe('hero', 'general-knowledge', version='balanced-v2', width=64, hidden=128)
HERO_CODING_BALANCED = ModelRecipe('hero-coding', 'coding', parent_model_id='hero',
                                  version='balanced-v2', width=64, hidden=128)


def recipe_for(model_id, profile='pilot-v1'):
    """Select one explicit built-in recipe; older identities remain unchanged."""
    recipes={('hero','pilot-v1'):HERO, ('hero-coding','pilot-v1'):HERO_CODING,
             ('hero','balanced-v2'):HERO_BALANCED, ('hero-coding','balanced-v2'):HERO_CODING_BALANCED}
    try: return recipes[(model_id,profile)]
    except (KeyError,TypeError): raise ValueError('Unknown named model recipe/profile') from None



def initialize(recipe, tokenizer, *, seed=1):
    """Create an explicitly seeded model without changing caller random state."""
    from thoughtformula import ModernMicroGPT
    if type(recipe) is not ModelRecipe or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Select a typed recipe and a finite initialization seed')
    prior = random.getstate()
    try:
        random.seed(seed)
        return ModernMicroGPT(**recipe.configuration(len(tokenizer.vocab)))
    finally:
        random.setstate(prior)
