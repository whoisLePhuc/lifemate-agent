"""Minimal Qwen3.5 adaptations for the lifemate-agent training/data code.

Integrated from the former Experiments/016-qwen35-qlora/qwen35_compat.py (tag
archive/experiments-011-021 in the workspace repository). Importing this module has no
side effects; call `patch_tokenizer_loading()` before loading a tokenizer.

1. Every chat-template render uses `enable_thinking=False`, matching Qwen35Planner inference
   (generation prompt ends with an empty `<think>\n\n</think>\n\n` block).
2. `train_sft.load_base` accepts the Qwen3.5 text model (`qwen3_5_text`); NF4 settings unchanged.
"""
import functools

QWEN35_TYPES={'qwen3_5','qwen3_5_text'}


def nonthinking(tokenizer):
    if getattr(tokenizer,'_lifemate_nonthinking',False): return tokenizer
    tokenizer.apply_chat_template=functools.partial(tokenizer.apply_chat_template,enable_thinking=False)
    tokenizer._lifemate_nonthinking=True
    return tokenizer


def patch_tokenizer_loading():
    """Force non-thinking rendering for tokenizers whose config is Qwen3.5; others untouched."""
    from transformers import AutoConfig, AutoTokenizer
    original=AutoTokenizer.from_pretrained.__func__ if hasattr(AutoTokenizer.from_pretrained,'__func__') else AutoTokenizer.from_pretrained
    if getattr(AutoTokenizer,'_lifemate_patched',False): return
    def from_pretrained(cls,path,*args,**kwargs):
        tokenizer=original(cls,path,*args,**kwargs)
        if AutoConfig.from_pretrained(path,local_files_only=True).model_type in QWEN35_TYPES: nonthinking(tokenizer)
        return tokenizer
    AutoTokenizer.from_pretrained=classmethod(from_pretrained); AutoTokenizer._lifemate_patched=True


def load_base(path,*,training):
    """Copy of training.train_sft.load_base with the model-type guard widened to Qwen3.5 text."""
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    if model.config.model_type not in QWEN35_TYPES: raise ValueError('this loader is for Qwen3.5 only; use training.train_sft.load_base for Qwen3')
    if any('visual' in name for name,_ in model.named_parameters()): raise ValueError('vision tower must not be loaded')
    model.config.use_cache=False
    return prepare_model_for_kbit_training(model,use_gradient_checkpointing=training,
        gradient_checkpointing_kwargs={'use_reentrant':False})


from contextlib import contextmanager


@contextmanager
def efficient_attention():
    """Qwen3.5 full-attention layers (head_dim 256, 16Q/4KV).

    Measured on sm_89, Torch 2.8: EFFICIENT fails with enable_gqa=True (used when no padding mask is
    built); FLASH handles that case but not masked attention; EFFICIENT handles masked (repeat_kv) attention.
    Allow both; math attention stays excluded so memory is not quadratic.
    """
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,SDPBackend.FLASH_ATTENTION]): yield
