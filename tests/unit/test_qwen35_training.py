"""Qwen3.5 training adaptations without loading a model."""
import unittest
from types import SimpleNamespace
from unittest import mock

from training import qwen35


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        self.calls.append(kwargs)
        return 'ok'


class NonThinkingTest(unittest.TestCase):
    def test_forces_enable_thinking_false_and_is_idempotent(self):
        tok = qwen35.nonthinking(FakeTokenizer())
        qwen35.nonthinking(tok)
        tok.apply_chat_template([], tokenize=False)
        self.assertEqual(tok.calls, [{'tokenize': False, 'enable_thinking': False}])


class PatchTokenizerLoadingTest(unittest.TestCase):
    def setUp(self):
        import transformers
        self.cls = transformers.AutoTokenizer
        self.saved = self.cls.__dict__['from_pretrained']
        self.had_flag = '_lifemate_patched' in self.cls.__dict__

    def tearDown(self):
        self.cls.from_pretrained = self.saved
        if '_lifemate_patched' in self.cls.__dict__:
            delattr(self.cls, '_lifemate_patched')
        if self.had_flag:
            self.cls._lifemate_patched = True

    def load(self, model_type):
        """Install a fake loader, patch it, load, then report the render kwargs actually used."""
        if '_lifemate_patched' in self.cls.__dict__:
            delattr(self.cls, '_lifemate_patched')
        self.cls.from_pretrained = classmethod(lambda cls, path, *a, **k: FakeTokenizer())
        with mock.patch('transformers.AutoConfig.from_pretrained', return_value=SimpleNamespace(model_type=model_type)):
            qwen35.patch_tokenizer_loading()
            tok = self.cls.from_pretrained('x')
        tok.apply_chat_template([])
        return tok.calls[-1]

    def test_qwen35_tokenizers_render_without_thinking(self):
        for model_type in ('qwen3_5', 'qwen3_5_text'):
            with self.subTest(model_type=model_type):
                self.assertEqual(self.load(model_type), {'enable_thinking': False})

    def test_other_tokenizers_untouched(self):
        self.assertEqual(self.load('qwen3'), {})


class LoadBaseGuardTest(unittest.TestCase):
    def build(self, model_type, params=('model.layers.0.w',)):
        model = SimpleNamespace(config=SimpleNamespace(model_type=model_type),
                                named_parameters=lambda: [(n, None) for n in params])
        with mock.patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model), \
                mock.patch('peft.prepare_model_for_kbit_training', side_effect=lambda m, **k: m):
            return qwen35.load_base('unused', training=True)

    def test_accepts_qwen35_text_model(self):
        self.assertFalse(self.build('qwen3_5_text').config.use_cache)

    def test_rejects_non_qwen35(self):
        with self.assertRaises(ValueError):
            self.build('qwen3')

    def test_rejects_vision_tower(self):
        with self.assertRaises(ValueError):
            self.build('qwen3_5', params=('model.visual.w',))


class AttentionContextTest(unittest.TestCase):
    def test_context_manager_enters_and_exits(self):
        with qwen35.efficient_attention():
            pass


if __name__ == '__main__':
    unittest.main()
