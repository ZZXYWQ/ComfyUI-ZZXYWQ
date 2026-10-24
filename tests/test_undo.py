import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent))
pipeline = importlib.import_module(root.name + '.Paints-UNDO.undo_pipeline')
nodes = importlib.import_module(root.name + '.Paints-UNDO.ZZX_PaintsUndo')


class Model:
    config = SimpleNamespace(scaling_factor=1.0)
    def to(self, *args, **kwargs):
        return self
    def cpu(self):
        return self
    def encode(self, image):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image))
    def decode(self, latent):
        return SimpleNamespace(sample=latent)


class ContractTests(unittest.TestCase):
    def test_invalid_operations(self):
        for value in ['', 'abc', '1,', '-1', '1000', '600,400', '400,400']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.parse_operations(value)
        self.assertEqual(pipeline.parse_operations('0, 400,999'), [0, 400, 999])

    def test_reject_batch_and_invalid_pixels(self):
        for image in [torch.zeros(2, 64, 64, 3), torch.zeros(1, 64, 64, 4),
                      torch.full((1, 64, 64, 3), float('nan'))]:
            with self.assertRaises(ValueError):
                pipeline.prepare_image(image, 256)

    def test_no_models_loaded_during_node_creation(self):
        with patch.object(pipeline.UndoPipeline, 'load', side_effect=AssertionError):
            nodes.ZZX_PaintsUndo()
            nodes.ZZX_PaintsUndoSequence()

    def test_rgb_batch_order_and_shared_noise(self):
        runner = pipeline.UndoPipeline()
        calls = []
        def load():
            runner.unet = runner.vae = runner.text_encoder = Model()
            def sampler(**kwargs):
                calls.append(kwargs)
                return torch.full_like(kwargs['initial_latent'], 0.2)
            runner.sampler = sampler
        image = torch.zeros(1, 64, 64, 3)
        image[..., 0] = 1
        with patch.object(runner, 'load', load), patch.object(runner, 'encode_prompt', return_value=torch.zeros(1, 77, 8)):
            frames, prompt = runner.run(image, 'red', [400, 900], 42,
                                        resolution=256, include_original=True, include_blank=True)
        self.assertEqual(tuple(frames.shape), (4, 256, 256, 3))
        self.assertEqual(frames.dtype, torch.float32)
        self.assertEqual(frames.device.type, 'cpu')
        self.assertEqual(frames[0, 0, 0].tolist(), [1., 0., 0.])
        self.assertTrue(torch.all(frames[-1] == 1))
        self.assertEqual([int(c['cross_attention_kwargs']['coded_conds'][0]) for c in calls], [400, 900])
        self.assertEqual([c['generator'].initial_seed() for c in calls], [42, 42])
        self.assertEqual(prompt, 'red')

    def test_failure_offloads_models(self):
        runner = pipeline.UndoPipeline()
        from unittest.mock import MagicMock
        model = MagicMock()
        def fail():
            runner.unet = model
            raise RuntimeError('load failed')
        with patch.object(runner, 'load', fail), self.assertRaisesRegex(RuntimeError, 'load failed'):
            runner.run(torch.zeros(1, 64, 64, 3), 'test', [400], 0)
        model.cpu.assert_called_once()

    def test_loading_with_invalid_stderr_and_restored_progress(self):
        from transformers import CLIPTextModel
        from transformers.utils import logging as log
        from huggingface_hub import utils as hub
        class BrokenConsole:
            def write(self, text):
                raise OSError(22, 'Invalid argument')
            def flush(self):
                raise OSError(22, 'Invalid argument')
        original = log.is_progress_bar_enabled(), hub.are_progress_bars_disabled()
        def load(*args, **kwargs):
            self.assertEqual(list(log.tqdm(range(2), desc='Loading weights')), [0, 1])
            return 'loaded'
        try:
            for enabled, disabled in [(True, False), (True, True), (False, False), (False, True)]:
                (log.enable_progress_bar if enabled else log.disable_progress_bar)()
                (hub.disable_progress_bars if disabled else hub.enable_progress_bars)()
                with patch.object(CLIPTextModel, 'from_pretrained', side_effect=load), patch('sys.stderr', BrokenConsole()):
                    self.assertEqual(pipeline.load_clip_text_encoder('unused'), 'loaded')
                self.assertEqual((log.is_progress_bar_enabled(), hub.are_progress_bars_disabled()), (enabled, disabled))
                with patch.object(CLIPTextModel, 'from_pretrained', side_effect=ValueError('bad weights')):
                    with self.assertRaisesRegex(ValueError, 'bad weights'):
                        pipeline.load_clip_text_encoder('unused')
                self.assertEqual((log.is_progress_bar_enabled(), hub.are_progress_bars_disabled()), (enabled, disabled))
        finally:
            (log.enable_progress_bar if original[0] else log.disable_progress_bar)()
            (hub.disable_progress_bars if original[1] else hub.enable_progress_bars)()


if __name__ == '__main__':
    unittest.main()
