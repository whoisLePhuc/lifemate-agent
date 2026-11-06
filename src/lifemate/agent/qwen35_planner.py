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

    Differences: `enable_thinking` is forced on every chat-template render (off by default), and
    the official checkpoint parameter count is recorded beside the loaded count.

    Thinking mode (experimental): the template ends the prompt with `<think>\n`. Decoding stays
    greedy. After `think_budget` generated tokens without a closing `</think>` the planner forces
    `\n</think>\n\n` so the model must produce an action; the returned text is only what follows
    `</think>`. Thinking tokens/text and whether closing was forced go to `last_generation`.
    """
    def __init__(self, model_path, *, thinking=False, think_budget=1024, **kwargs):
        if thinking and think_budget < 1: raise ValueError('think_budget must be positive')
        self.thinking=thinking; self.think_budget=think_budget
        if thinking: kwargs['new_tokens']=think_budget+8+kwargs.get('new_tokens',384)
        super().__init__(model_path, **kwargs)
        render=self.tokenizer.apply_chat_template
        self.tokenizer.apply_chat_template=functools.partial(render,enable_thinking=thinking)
        # The published checkpoint has no generation_config.json; config EOS is only <|endoftext|>.
        # Mirror Qwen3-4B-2507's generation_config: stop at <|im_end|> or <|endoftext|>.
        stops=[self.tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|endoftext|>')]
        if None in stops or self.tokenizer.unk_token_id in stops: raise ValueError('missing chat stop tokens')
        self.model.generation_config.eos_token_id=stops
        self._force=[]
        if thinking:
            end=self.tokenizer.encode('</think>',add_special_tokens=False)
            if len(end)!=1: raise ValueError('</think> must be a single token')
            self.think_end_id=end[0]
            self._close_ids=self.tokenizer.encode('\n</think>\n\n',add_special_tokens=False)
            if self._close_ids.count(self.think_end_id)!=1: raise ValueError('unexpected </think> encoding')
        self.parameter_audit={'loaded_language_model':self.parameters,**checkpoint_parameters(model_path)}
        if self.parameter_audit['checkpoint_total']>=5_000_000_000:
            raise ValueError('published checkpoint exceeds competition LLM limit')

    def _split(self):
        generated=self._generated
        if self.think_end_id in generated:
            index=generated.index(self.think_end_id)
            return generated[:index], generated[index+1:]
        return generated, None

    def _choose(self, token, generated):
        if not self.thinking: return token
        if self._force: return self._force.pop(0)
        if self.think_end_id in generated or token==self.think_end_id: return token
        if len(generated)>=self.think_budget:
            self._force=list(self._close_ids); self.last_generation['thinking_forced']=True
            return self._force.pop(0)
        return token

    def _decode_generated(self):
        if not self.thinking: return super()._decode_generated()
        _, answer=self._split()
        if answer is None: return ''   # still thinking: no visible answer
        return self.tokenizer.decode(answer,skip_special_tokens=self._active_tools is None).strip()

    def generate(self, messages, seconds, *, tools=None):
        self._force=[]
        try:
            return super().generate(messages, seconds, tools=tools)
        finally:
            if self.thinking:
                thought,_=self._split()
                info=self.last_generation
                info.setdefault('thinking_forced',False)
                info['thinking_tokens']=len(thought)
                info['thinking_text']=self.tokenizer.decode(thought,skip_special_tokens=True).strip()
