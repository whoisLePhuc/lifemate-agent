"""Explicit completion-only labels; no dependence on a patched training template."""
from lifemate.agent.planner import parse_action


class SequenceBudgetError(ValueError):
    def __init__(self,reason,*,prompt_tokens,target_tokens):
        super().__init__(reason)
        self.reason=reason;self.prompt_tokens=prompt_tokens;self.target_tokens=target_tokens


def validate_mask(row,*,eos_token_id):
    ids,labels,attention=row['input_ids'],row['labels'],row['attention_mask']
    start=row['prompt_tokens'];target=row['target_tokens']
    if type(start) is not int or type(target) is not int or start<=0 or target<=1:
        raise ValueError('invalid token boundaries')
    if len(ids)!=len(labels) or len(ids)!=len(attention) or len(ids)!=start+target:
        raise ValueError('token/label length mismatch')
    if any(type(x) is not int or x<0 for x in ids) or attention!=[1]*len(ids):
        raise ValueError('invalid unpadded token sequence')
    if labels[:start]!=[-100]*start or labels[start:]!=ids[start:]:
        raise ValueError('loss mask leaks prompt or loses completion')
    if ids[-1]!=eos_token_id or labels[-1]!=eos_token_id:raise ValueError('EOS not supervised')


def encode_action(row,tokenizer,*,max_length=16384,context_tokens=16384,new_tokens=384):
    if not 0<max_length<=context_tokens<=16384 or not 0<new_tokens<context_tokens:
        raise ValueError('invalid training/inference token budgets')
    prompt=row['prompt'];completion=row['completion']
    if len(completion)!=1 or completion[0]['role']!='assistant':raise ValueError('one assistant target required')
    action_text=completion[0]['content'];parse_action(action_text)
    for message in prompt+completion:
        if any(t in message['content'] for t in ('<|im_start|>','<|im_end|>','<|endoftext|>')):
            raise ValueError('raw chat control delimiter in data; do not silently escape training-only')
    prefix=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(prompt+completion,tokenize=False,add_generation_prompt=False)
    if not full.startswith(prefix+action_text):raise ValueError('training/inference template prefix mismatch')
    eos=tokenizer.eos_token
    if not isinstance(eos,str) or not eos:raise ValueError('tokenizer EOS text unavailable')
    closing=full[len(prefix+action_text):]
    if not closing.startswith(eos) or closing[len(eos):].strip():
        raise ValueError('unsupported assistant template ending')
    text=prefix+action_text+eos
    prefix_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    if tokenizer(prefix)['input_ids']!=prefix_ids:
        raise ValueError('tokenizer default adds tokens unlike explicit training encoding')
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if ids[:len(prefix_ids)]!=prefix_ids:raise ValueError('token boundary crosses prompt/completion')
    target=len(ids)-len(prefix_ids)
    if tokenizer.decode(ids[len(prefix_ids):-1],skip_special_tokens=False)!=action_text:
        raise ValueError('completion does not decode to the exact action JSON')
    if len(prefix_ids)+new_tokens>context_tokens:
        raise SequenceBudgetError('inference_context_budget',prompt_tokens=len(prefix_ids),target_tokens=target)
    if target>new_tokens:
        raise SequenceBudgetError('generation_target_budget',prompt_tokens=len(prefix_ids),target_tokens=target)
    if len(ids)>max_length:
        raise SequenceBudgetError('training_sequence_budget',prompt_tokens=len(prefix_ids),target_tokens=target)
    result={'id':row['id'],'sample_id':row['sample_id'],'step_index':row['step_index'],
            'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prefix_ids)+ids[len(prefix_ids):],
            'prompt_tokens':len(prefix_ids),'target_tokens':target}
    validate_mask(result,eos_token_id=tokenizer.eos_token_id)
    return result


def collate_actions(rows,*,pad_token_id,eos_token_id,pad_to_multiple_of=8,context_tokens=16384,return_tensors=None):
    if not rows or type(pad_token_id) is not int or pad_token_id<0:raise ValueError('invalid batch/pad token')
    if type(pad_to_multiple_of) is not int or pad_to_multiple_of<1:raise ValueError('invalid padding multiple')
    size=max(len(row['input_ids']) for row in rows)
    size=((size+pad_to_multiple_of-1)//pad_to_multiple_of)*pad_to_multiple_of
    if size>context_tokens:raise ValueError('padded batch exceeds context limit')
    batch={key:[] for key in ('input_ids','attention_mask','labels')}
    for row in rows:
        ids,labels,attention=row['input_ids'],row['labels'],row['attention_mask']
        if not (len(ids)==len(labels)==len(attention)):raise ValueError('invalid record shape')
        if 'prompt_tokens' not in row or 'target_tokens' not in row:raise ValueError('missing mask boundaries')
        validate_mask(row,eos_token_id=eos_token_id)
        n=size-len(ids)
        batch['input_ids'].append(ids+[pad_token_id]*n)
        batch['attention_mask'].append(attention+[0]*n)
        batch['labels'].append(labels+[-100]*n)
    if return_tensors is None:return batch
    if return_tensors!='pt':raise ValueError('only pt tensor format is supported')
    import torch
    return {key:torch.tensor(value,dtype=torch.long) for key,value in batch.items()}


def torch_mask_probe(rows,*,pad_token_id,eos_token_id):
    """CPU-only toy loss/gradient check; no LLM, optimizer or weight update."""
    import torch
    import torch.nn.functional as F
    batch=collate_actions(rows,pad_token_id=pad_token_id,eos_token_id=eos_token_id,return_tensors='pt')
    labels=batch['labels'][:,1:].clone()  # Causal LM predicts token j at logit j-1.
    active=labels.ne(-100)
    labels[active]=labels[active].remainder(7)
    logits=torch.zeros((*labels.shape,7),dtype=torch.float32,requires_grad=True)
    losses=F.cross_entropy(logits.reshape(-1,7),labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(labels)
    if not active.any() or losses[~active].ne(0).any():raise ValueError('masked positions affect toy loss')
    loss=losses.sum()/active.sum();loss.backward()
    gradient=logits.grad.abs().sum(dim=-1)
    if gradient[~active].ne(0).any() or not gradient[active].gt(0).all():
        raise ValueError('masked gradient check failed')
    return {'passed':True,'device':'cpu','toy_vocabulary':7,'supervised_shifted_tokens':int(active.sum()),
            'ignored_shifted_tokens':int((~active).sum()),'loss':float(loss.detach()),
            'model_loaded':False,'weights_updated':False}
