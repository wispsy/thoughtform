"""Model definitions and agency values, independent of execution and world state."""
from dataclasses import dataclass, field
import math
import re
from typing import Protocol


def identifier(value, label):
    if type(value) is not str or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', value):
        raise ValueError(f'{label} must be a bounded stable identifier')


def text(value, label, limit=4096):
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError(f'{label} must be nonempty bounded text')


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.05
    top_k: int = 6
    top_p: float = 0.9

    def __post_init__(self):
        for name, minimum, maximum in (('max_new_tokens', 0, 4096), ('top_k', 1, 65536)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{name} is outside its integer budget')
        for name, minimum, maximum in (('temperature', 0, 2), ('top_p', 0, 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum or (name == 'top_p' and value == 0):
                raise ValueError(f'{name} must be a finite sampling value')


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    provider: str
    reference: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self):
        identifier(self.model_id, 'Model ID')
        identifier(self.provider, 'Provider')
        text(self.reference, 'Model reference', 1024)
        if type(self.generation) is not GenerationConfig:
            raise TypeError('generation requires GenerationConfig')


@dataclass(frozen=True)
class Being:
    being_id: str
    name: str
    role: str = 'inhabitant'
    model: ModelDefinition | None = None
    goals: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.being_id, 'Being ID')
        text(self.name, 'Name', 256)
        identifier(self.role, 'Role')
        if self.model is not None and type(self.model) is not ModelDefinition:
            raise TypeError('model requires a ModelDefinition')
        if type(self.goals) is not tuple or len(self.goals) > 32:
            raise ValueError('Goals require a tuple of at most 32 text values')
        for goal in self.goals:
            text(goal, 'Goal')


@dataclass(frozen=True)
class Observation:
    actor_id: str
    tick: int
    habitat: str
    visible: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.actor_id, 'Actor ID')
        identifier(self.habitat, 'Habitat')
        if type(self.tick) is not int or not 0 <= self.tick <= 2**53 - 1:
            raise ValueError('Tick requires a nonnegative portable integer')
        if type(self.visible) is not tuple or len(self.visible) > 256:
            raise ValueError('Visible information requires at most 256 text values')
        for value in self.visible:
            text(value, 'Visible value')


@dataclass(frozen=True)
class Action:
    actor_id: str
    kind: str
    target: str | None = None
    value: str | None = None

    def __post_init__(self):
        identifier(self.actor_id, 'Actor ID')
        identifier(self.kind, 'Action kind')
        if self.target is not None:
            identifier(self.target, 'Target')
        if self.value is not None:
            text(self.value, 'Action value')


class Policy(Protocol):
    def decide(self, observation: Observation) -> Action:
        """Propose one action; the consuming environment grants authority."""
        ...
