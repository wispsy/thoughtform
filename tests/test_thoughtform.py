from dataclasses import FrozenInstanceError
import importlib.util
import sys
import unittest

from thoughtform import Action, Being, GenerationConfig, ModelDefinition, Observation
from thoughtform.thoughtformula import ThoughtformulaModel


class ContractTests(unittest.TestCase):
    def test_individual_model_and_being_are_distinct_immutable_values(self):
        model = ModelDefinition('shared', 'thoughtformula', 'caller-owned')
        alice, bob = Being('alice', 'Alice', model=model), Being('bob', 'Bob', model=model)
        self.assertIs(alice.model, bob.model)
        with self.assertRaises(FrozenInstanceError):
            alice.name = 'changed'
        observation = Observation('alice', 0, 'woods', ('bob',))
        self.assertEqual(Action(observation.actor_id, 'wait').actor_id, 'alice')
        self.assertNotIn('hypnosis', sys.modules)
        self.assertNotIn('paracosm', sys.modules)

    def test_refuses_invalid_bounds_and_mutable_collections(self):
        for kwargs in ({'max_new_tokens': True}, {'max_new_tokens': -1}, {'temperature': float('nan')}, {'top_p': 0}, {'top_k': 0}):
            with self.assertRaises(ValueError):
                GenerationConfig(**kwargs)
        with self.assertRaises(ValueError):
            Being('a', 'A', goals=[])
        with self.assertRaises(ValueError):
            Observation('a', True, 'woods')
        with self.assertRaises(ValueError):
            Action('../a', 'wait')


@unittest.skipUnless(importlib.util.find_spec('thoughtformula'), 'optional Thoughtformula profile is not installed')
class ModelTests(unittest.TestCase):
    def setUp(self):
        from thoughtformula import BPETokenizer, ModernMicroGPT
        self.tokenizer = BPETokenizer()
        self.tokenizer.train('hello world', max_merges=0)
        self.model = ModernMicroGPT(len(self.tokenizer.vocab), n_embd=8, n_heads=2, n_kv_heads=1,
            n_layers=1, n_experts=1, moe_top_k=1, expert_hidden=8, lora_rank=2)

    def adapter(self, count=3):
        return ThoughtformulaModel(ModelDefinition('tiny', 'thoughtformula', 'synthetic-untrained',
            GenerationConfig(max_new_tokens=count, temperature=0)), self.model, self.tokenizer)

    def test_real_transformer_stream_and_incremental_decode(self):
        tokens = list(self.adapter().stream('hello'))
        self.assertLessEqual(len(tokens), 3)
        for token in tokens:
            self.assertIs(type(token.token_id), int)
            self.assertEqual(token.text, self.tokenizer.decode([token.token_id], skip_special=True))

    def test_zero_empty_unknown_and_context_boundaries(self):
        self.model.generate_stream = lambda *args, **kwargs: self.fail('zero/empty must not invoke inference')
        self.assertEqual(list(self.adapter(0).stream('hello')), [])
        self.assertEqual(list(self.adapter().stream('')), [])
        with self.assertRaises(ValueError):
            list(self.adapter().stream('🙂'))
        with self.assertRaises(ValueError):
            list(self.adapter(256).stream('hello'))

    def test_early_close_eos_and_bad_token_close_the_upstream_generator(self):
        for value in (self.tokenizer.stoi['h'], self.tokenizer.eos_id, -1):
            closed = []
            def source(*args, **kwargs):
                try:
                    yield value
                    yield value
                finally:
                    closed.append(True)
            self.model.generate_stream = source
            stream = self.adapter().stream('hello')
            if value == -1:
                with self.assertRaises(ValueError):
                    next(stream)
            elif value == self.tokenizer.eos_id:
                self.assertEqual(list(stream), [])
            else:
                next(stream)
                stream.close()
            self.assertEqual(closed, [True])
