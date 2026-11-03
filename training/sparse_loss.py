"""Exact causal completion loss via Qwen3 selected prediction positions."""


def sparse_action_loss(model,batch,*,return_probe=False):
    import torch
    import torch.nn.functional as F
    labels=batch['labels']
    if labels.ndim!=2 or labels.shape[1]<2:raise ValueError('invalid causal labels')
    positions=torch.nonzero(labels[:,1:].ne(-100).any(dim=0),as_tuple=False).flatten()
    if not positions.numel():raise ValueError('no supervised causal targets')
    targets=labels.index_select(1,positions+1)
    output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                 logits_to_keep=positions,use_cache=False)
    logits=output.logits
    if logits.shape[:2]!=targets.shape:raise ValueError('model does not support selected logits')
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100)
    if not return_probe:return loss
    return loss,{'argmax_ids':logits.argmax(dim=-1).detach().cpu().tolist(),
                 'supervised_tokens':int(targets.ne(-100).sum()),
                 'selected_prediction_positions':int(positions.numel())}
