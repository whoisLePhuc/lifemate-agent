"""Strict action parser and local offline model backend."""
import json
from pathlib import Path
import time


def parse_action(text):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate key')
            out[k]=v
        return out
    def bad(_): raise ValueError('nonfinite number')
    value=json.loads(text,object_pairs_hook=unique,parse_constant=bad)
    if not isinstance(value,dict): raise ValueError('action must be object')
    if value == {'type':'finish'}: return value
    if (set(value) != {'type','name','arguments'} or value.get('type') != 'call'
        or not isinstance(value.get('name'),str) or not isinstance(value.get('arguments'),dict)):
        raise ValueError('expected call or finish action')
    json.dumps(value,allow_nan=False)
    return value


class LocalPlanner:
    def __init__(self, model_path, *, seed=42, context_tokens=16384, new_tokens=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not Path(model_path).is_dir(): raise ValueError('local model directory missing')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required by this backend')
        if not 0 < new_tokens < context_tokens <= 16384: raise ValueError('invalid token budget')
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
                     trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.context_tokens=context_tokens; self.new_tokens=new_tokens
        self.seed=seed
        self.parameters=sum(p.numel() for p in self.model.parameters())
        if self.parameters >= 5_000_000_000: raise ValueError('LLM exceeds competition parameter limit')

    def generate(self, messages, seconds):
        """Return the same text contract; expose diagnostics separately."""
        started=time.perf_counter()
        self.last_generation={'finish_reason':None,'prompt_tokens':None,'new_tokens':0,
                              'prefill_seconds':0.,'decode_seconds':0.,
                              'timing_method':'CPU wall time; no added CUDA synchronize'}
        self._generated=[]
        self._generation_stage=None
        self._stage_started=started
        try:
            return self._generate(messages, seconds)
        except TimeoutError:
            self.last_generation['finish_reason']='deadline'
            raise
        except Exception:
            if self.last_generation['finish_reason'] is None:
                self.last_generation['finish_reason']='backend_error'
            raise
        finally:
            if self._generation_stage:
                self.last_generation[self._generation_stage+'_seconds']=time.perf_counter()-self._stage_started
            self.last_generation['elapsed_seconds']=time.perf_counter()-started
            self.last_generation['new_tokens']=len(self._generated)
            self.last_generation['partial_output']=self.tokenizer.decode(self._generated,skip_special_tokens=True).strip()

    def _generate(self, messages, seconds):
        import torch
        deadline=time.monotonic()+seconds
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors='pt').to('cuda')
        count=inputs['input_ids'].shape[-1]
        self.last_generation['prompt_tokens']=count
        if count+self.new_tokens > self.context_tokens:
            self.last_generation['finish_reason']='context_limit'
            raise ValueError('context budget exceeded; no silent truncation')
        # Chunk long prefills to bound attention/logit temporary memory on Windows
        # CUDA builds without fused attention. Cache is new for every decision.
        cache=None; generated=self._generated
        eos=self.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        self._generation_stage='prefill'; self._stage_started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,count,256):
                if time.monotonic() >= deadline: raise TimeoutError('prefill deadline')
                end=min(count,start+256)
                output=self.model(input_ids=inputs['input_ids'][:,start:end],
                    attention_mask=inputs['attention_mask'][:,:end],past_key_values=cache,
                    use_cache=True,logits_to_keep=1)
                cache=output.past_key_values
            self.last_generation['prefill_seconds']=time.perf_counter()-self._stage_started
            self._generation_stage='decode'; self._stage_started=time.perf_counter()
            for step in range(self.new_tokens):
                if time.monotonic() >= deadline: raise TimeoutError('decode deadline')
                token=output.logits[:,-1,:].argmax(dim=-1).item()
                if token in eos:
                    self.last_generation['finish_reason']='eos'
                    break
                generated.append(token)
                if step+1==self.new_tokens:
                    self.last_generation['finish_reason']='token_limit'
                    break
                output=self.model(input_ids=torch.tensor([[token]],device='cuda'),
                    attention_mask=torch.ones((1,count+len(generated)),dtype=torch.long,device='cuda'),
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=output.past_key_values
        if time.monotonic() >= deadline: raise TimeoutError('generation deadline')
        return self.tokenizer.decode(generated,skip_special_tokens=True).strip()
