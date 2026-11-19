"""Individual model definitions and neutral agency contracts."""
from .contracts import Action, Being, GenerationConfig, ModelDefinition, Observation, Policy

__all__ = ['Action', 'Being', 'GenerationConfig', 'ModelDefinition', 'Observation', 'Policy']


def main():
    """Print the location of the source-oriented public API."""
    print('Thoughtform: Being, ModelDefinition, Observation, Action and Policy. See README.md.')
