from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HAS_MODEL = importlib.util.find_spec('thoughtformula') is not None


@unittest.skipUnless(HAS_MODEL, 'optional Thoughtformula profile is not installed')
class ArtifactTests(unittest.TestCase):
    def fixture(self, root):
        from thoughtformula import BPETokenizer, TrainingConfig, TrainingState, save_training_checkpoint
        from thoughtform.recipes import HERO, initialize
        from thoughtform.artifacts import artifact_metadata, publish_artifact
        recipe = replace(HERO, width=8, layers=1, heads=2, kv_heads=1, hidden=12, lora_rank=2)
        tokenizer = BPETokenizer();tokenizer.train('A small model.\n', max_merges=0)
        model = initialize(recipe, tokenizer, seed=3)
        checkpoint = root/'model.tft';corpus = '1'*64
        save_training_checkpoint(checkpoint, model, tokenizer, TrainingConfig(max_steps=0), TrainingState(),
                                 metadata=artifact_metadata(recipe, corpus))
        receipt = publish_artifact(root/'artifact.json', checkpoint, recipe, corpus_sha256=corpus)
        return recipe, receipt

    def test_roundtrip_adapter_and_fresh_specialization_preserve_parent(self):
        from thoughtform.artifacts import load_artifact, specialize
        with TemporaryDirectory() as directory:
            root=Path(directory);recipe, receipt=self.fixture(root)
            before=(root/'model.tft').read_bytes()
            loaded=load_artifact(root/'artifact.json')
            self.assertEqual(loaded.adapter().definition.model_id, 'hero')
            child=replace(recipe, model_id='hero-coding', role='coding', parent_model_id='hero')
            model, tokenizer, parent_sha=specialize(root/'artifact.json', child)
            self.assertEqual(parent_sha, receipt['checkpoint']['sha256'])
            self.assertEqual(model.total_steps, 0)
            self.assertEqual(model.weights['tok_emb'].data, loaded.model.weights['tok_emb'].data)
            self.assertIsNot(model, loaded.model)
            self.assertEqual(before, (root/'model.tft').read_bytes())

    def test_passive_sidecar_rejects_lineage_traversal_tamper_and_republication(self):
        from thoughtform.artifacts import load_artifact, publish_artifact
        with TemporaryDirectory() as directory:
            root=Path(directory);recipe, receipt=self.fixture(root);path=root/'artifact.json'
            with self.assertRaises(FileExistsError):
                publish_artifact(path, root/'model.tft', recipe, corpus_sha256='1'*64)
            for change in ({'parent_sha256':'2'*64}, {'tokenizer_sha256':'2'*64},
                           {'checkpoint':{**receipt['checkpoint'],'file':'../model.tft'}}):
                path.write_text(json.dumps({**receipt, **change}))
                with self.assertRaises(ValueError): load_artifact(path)
            path.write_text(json.dumps(receipt));(root/'model.tft').write_bytes(b'tampered')
            with self.assertRaises(ValueError): load_artifact(path)

    def test_balanced_recipes_are_distinct_and_preserve_parent_shape(self):
        from thoughtform.recipes import HERO,HERO_CODING,recipe_for
        self.assertIs(recipe_for('hero'),HERO)
        self.assertIs(recipe_for('hero-coding'),HERO_CODING)
        hero=recipe_for('hero','balanced-v2');coding=recipe_for('hero-coding','balanced-v2')
        self.assertEqual(hero.width,64)
        self.assertNotEqual(hero.sha256,HERO.sha256)
        self.assertEqual(hero.configuration(256),coding.configuration(256))
        self.assertEqual(coding.parent_model_id,hero.model_id)
        with self.assertRaises(ValueError): recipe_for('hero','download-a-model')
