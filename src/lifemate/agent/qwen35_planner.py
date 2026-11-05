"""Qwen3.5-4B planner: production greedy loop, thinking disabled, text tower only.

Integrated from the former Experiments/015-qwen35-candidate (tag archive/experiments-011-021 in the workspace repository). Requires transformers>=5 (Qwen3.5
architecture); the baseline environment pins an older stack, so import this module only
in an environment that supports it.
"""
import functools
import json
from pathlib import Path

from .planner import LocalPlanner


def checkpoint_parameters(model_path):
    """Count every tensor element in the published checkpoint, including the unused vision tower."""
    from safetensors import safe_open
    model_path=Path(model_path)
    index=json.loads((model_path/'model.safetensors.index.json').read_text())
    total=0; vision=0
    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(str(model_path/shard),framework='pt') as handle:
            for name in handle.keys():
                shape=handle.get_slice(name).get_shape(); count=1
                for size in shape: count*=size
                total+=count
                if '.visual.' in name or name.startswith('visual.') or 'vision' in name: vision+=count
    return {'checkpoint_total':total,'checkpoint_vision':vision,'checkpoint_non_vision':total-vision}


class Qwen35Planner(LocalPlanner):
    """Reuses LocalPlanner unchanged (chunked prefill, deadline checks, argmax decode).

    Differences: `enable_thinking=False` is forced on every chat-template render, and
    the official checkpoint parameter count is recorded beside the loaded count.
    """
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        render=self.tokenizer.apply_chat_template
        self.tokenizer.apply_chat_template=functools.partial(render,enable_thinking=False)
        # The published checkpoint has no generation_config.json; config EOS is only <|endoftext|>.
        # Mirror Qwen3-4B-2507's generation_config: stop at <|im_end|> or <|endoftext|>.
        stops=[self.tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|endoftext|>')]
        if None in stops or self.tokenizer.unk_token_id in stops: raise ValueError('missing chat stop tokens')
        self.model.generation_config.eos_token_id=stops
        self.parameter_audit={'loaded_language_model':self.parameters,**checkpoint_parameters(model_path)}
        if self.parameter_audit['checkpoint_total']>=5_000_000_000:
            raise ValueError('published checkpoint exceeds competition LLM limit')
