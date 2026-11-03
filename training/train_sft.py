"""Single-GPU NF4 QLoRA with explicit audited labels and exact sparse logits loss."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import random
import time

from lifemate.observability.provenance import collect_provenance,file_hash
from lifemate.agent.prompt import SYSTEM
from training.prepare_trajectories import safe_path,write_json
from training.sft_data import verify_dataset,TokenRows,select_rows
from training.loss_mask import collate_actions
from training.sparse_loss import sparse_action_loss


@contextmanager
def efficient_attention():
    from torch.nn.attention import sdpa_kernel,SDPBackend
    # Avoid quadratic-memory math attention on the Windows baseline CUDA build.
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION]):yield


def load_base(path,*,training):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        quantization_config=quant,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    if model.config.model_type!='qwen3':raise ValueError('selected-logits backend currently validated for Qwen3 only')
    model.config.use_cache=False
    # Same nonquantized dtype preparation during reload for a meaningful parity check.
    return prepare_model_for_kbit_training(model,use_gradient_checkpointing=training,
        gradient_checkpointing_kwargs={'use_reentrant':False})


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(model).items()}


def state_hash(state):
    from safetensors.torch import save
    return sha256(save({k:state[k] for k in sorted(state)})).hexdigest()


def to_batch(row,tokenizer):
    return {k:v.to('cuda') for k,v in collate_actions([row],pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,return_tensors='pt').items()}


def evaluate_rows(model,dataset,indices,tokenizer):
    import torch
    model.eval();weighted=total=0;losses=[]
    with torch.no_grad(),efficient_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index in indices:
            row=dataset[index];batch=to_batch(row,tokenizer)
            loss=sparse_action_loss(model,batch)
            value=float(loss);count=row['target_tokens']
            if not math.isfinite(value):raise ValueError('nonfinite validation loss')
            weighted+=value*count;total+=count;losses.append(value)
    return {'rows':len(indices),'supervised_tokens':total,'token_mean_loss':weighted/total,
            'row_mean_loss':sum(losses)/len(losses),'teacher_forced':True,'closed_loop_score':False}


def probe(model,row,tokenizer):
    import torch
    model.eval()
    with torch.no_grad(),efficient_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        loss,result=sparse_action_loss(model,to_batch(row,tokenizer),return_probe=True)
    return {'loss':float(loss),**result}


def run(args):
    import torch
    import numpy as np
    import peft,bitsandbytes
    from peft import LoraConfig,get_peft_model,PeftModel
    from transformers import AutoTokenizer
    from importlib import metadata
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA GPU required')
    if args.max_steps<1 or args.gradient_accumulation<1 or args.train_limit<0 or args.validation_limit<0:
        raise ValueError('invalid step/row settings')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or args.lora_rank<1:raise ValueError('invalid optimizer/LoRA settings')
    directory,model_path,out=map(safe_path,(args.dataset_dir,args.model_path,args.output_dir))
    if out.exists() or out.is_relative_to(directory) or out.is_relative_to(model_path) or any(
        p.lower() in {'data','models'} for p in out.parts):raise ValueError('choose a new output outside input Data/Models/dataset')
    manifest,split,dataset_report=verify_dataset(directory,args.audit_file)
    if manifest['prompt_sha256']!=sha256(SYSTEM.encode()).hexdigest():raise ValueError('training/inference SYSTEM prompt mismatch')
    model_metadata=dataset_report['provenance']['model']['metadata_sha256']
    for name,digest in model_metadata.items():
        if file_hash(model_path/name)!=digest:raise ValueError('base/tokenizer metadata differs from dataset')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    train=TokenRows(directory/'train.tokens.jsonl',partition='train',allowed_ids=set(split['train_ids']),eos_token_id=tokenizer.eos_token_id)
    validation=TokenRows(directory/'validation.tokens.jsonl',partition='validation',allowed_ids=set(split['validation_ids']),eos_token_id=tokenizer.eos_token_id)
    if len(train)!=dataset_report['partitions']['train']['rows'] or len(validation)!=dataset_report['partitions']['validation']['rows']:
        raise ValueError('dataset/report row counts differ')
    fit_indices=select_rows(train,args.train_limit,seed=args.seed,longest_first=True)
    val_indices=select_rows(validation,args.validation_limit,seed=args.seed+1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    out.mkdir(parents=True,exist_ok=False)
    config={'seed':args.seed,'max_steps':args.max_steps,'train_limit':args.train_limit,'validation_limit':args.validation_limit,
      'learning_rate':args.learning_rate,'gradient_accumulation':args.gradient_accumulation,
      'lora_rank':args.lora_rank,'lora_alpha':2*args.lora_rank,'lora_dropout':.05,'target_modules':'all-linear',
      'quantization':'NF4 double quant','compute_dtype':'bfloat16','micro_batch_size':1,
      'gradient_checkpointing':'nonreentrant','loss':'exact selected-logits action/EOS cross entropy',
      'attention_backends':['efficient','cudnn'],'source_dataset_manifest_sha256':file_hash(directory/'dataset_manifest.json')}
    provenance=collect_provenance(Path(__file__).resolve().parents[1],config=config,model_path=model_path,prompt=SYSTEM)
    provenance['training_packages']={n:metadata.version(n) for n in ['peft','bitsandbytes','torch','transformers','accelerate']}
    write_json(out/'provenance.json',provenance)
    write_json(out/'selection.json',{'train_rows':[train.entries[i] for i in fit_indices],
        'validation_rows':[validation.entries[i] for i in val_indices],'longest_train_row_included':True})
    write_json(out/'inner_validation_ids.json',split['validation_ids'])
    def emit(event,**fields):
        with (out/'progress.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'event':event,**fields},ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    emit('start',config=config)
    started=time.perf_counter();print('Loading local NF4 base model',flush=True)
    model=load_base(model_path,training=True)
    model=get_peft_model(model,LoraConfig(r=args.lora_rank,lora_alpha=2*args.lora_rank,lora_dropout=.05,
        target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    trainable=[p for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('unexpected trainable base weights')
    initial_state=adapter_state(model);initial_hash=state_hash(initial_state)
    counts=model.get_nb_trainable_parameters()
    if counts[1]>=5_000_000_000:raise ValueError('LLM plus unmerged adapter exceeds parameter budget')
    before=evaluate_rows(model,validation,val_indices,tokenizer);emit('validation_before',metrics=before)
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=0.,foreach=False)
    cursor=0;order=list(fit_indices);rng=random.Random(args.seed);step_rows=[]
    for step in range(1,args.max_steps+1):
        tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();loss_sum=0;tokens=0;lengths=[]
        for micro in range(args.gradient_accumulation):
            if cursor==len(order):rng.shuffle(order);cursor=0
            row=train[order[cursor]];cursor+=1
            batch=to_batch(row,tokenizer)
            with efficient_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=sparse_action_loss(model,batch)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                (loss/args.gradient_accumulation).backward()
            loss_sum+=float(loss.detach());tokens+=row['target_tokens'];lengths.append(len(row['input_ids']))
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        metrics={'step':step,'loss':loss_sum/args.gradient_accumulation,'gradient_norm':float(norm),
            'target_tokens':tokens,'sequence_lengths':lengths,'elapsed_seconds':time.perf_counter()-tick,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        step_rows.append(metrics);emit('train_step',**metrics)
        print(f"step {step}/{args.max_steps}: loss={metrics['loss']:.4f}, peak={metrics['peak_reserved_bytes']/2**30:.2f} GiB",flush=True)
        if args.save_every and step%args.save_every==0:model.save_pretrained(out/f'checkpoint-{step}',safe_serialization=True)
    after=evaluate_rows(model,validation,val_indices,tokenizer);emit('validation_after',metrics=after)
    final_state=adapter_state(model);changed=sum(not torch.equal(initial_state[k],final_state[k]) for k in initial_state)
    if not changed:raise ValueError('adapter weights did not change')
    adapter=out/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    probe_row=train[min(fit_indices,key=lambda i:train.entries[i]['sequence_tokens'])]
    expected_probe=probe(model,probe_row,tokenizer)
    emit('adapter_saved',changed_tensors=changed)
    # Release all live GPU tensors before creating the verification model.
    del model,optimizer,trainable,loss,batch;gc.collect();torch.cuda.empty_cache()
    print('Reloading saved adapter on the same NF4 backend',flush=True)
    reloaded=PeftModel.from_pretrained(load_base(model_path,training=False),adapter,local_files_only=True,is_trainable=False)
    restored_state=adapter_state(reloaded)
    equal=final_state.keys()==restored_state.keys() and all(torch.equal(final_state[k],restored_state[k]) for k in final_state)
    restored_probe=probe(reloaded,probe_row,tokenizer)
    if not equal or expected_probe['argmax_ids']!=restored_probe['argmax_ids'] or not math.isclose(
        expected_probe['loss'],restored_probe['loss'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('adapter reload parity failed')
    verify_dataset(directory,args.audit_file)
    report={'schema_version':1,'status':'complete','training_performed':True,'optimizer_steps':args.max_steps,
        'train_rows_selected':len(fit_indices),'validation_rows_selected':len(val_indices),
        'trainable_parameters':counts[0],'total_parameters_including_adapter':counts[1],
        'initial_adapter_sha256':initial_hash,'final_adapter_sha256':state_hash(final_state),'changed_adapter_tensors':changed,
        'validation_before':before,'validation_after':after,'steps':step_rows,
        'reload':{'state_tensors_exact':equal,'greedy_teacher_forced_ids_equal':True,
                  'loss_before_reload':expected_probe['loss'],'loss_after_reload':restored_probe['loss'],
                  'backend':'same NF4 and kbit dtype preparation; BF16 inference parity not implied'},
        'adapter_files_sha256':{p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()},
        'dataset_manifest_sha256':config['source_dataset_manifest_sha256'],
        'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':time.perf_counter()-started,
        'limitations':['Short training smoke is not an agent-quality improvement claim.',
            'Validation may be a selected row subset; no closed-loop score is computed here.',
            'Micro-loss means are averaged across accumulation steps; rows are unweighted.',
            'Seeds fixed; CUDA reductions need not be bit-identical across hosts.',
            'Adapter checkpoints only; exact optimizer/RNG resume is not implemented.']}
    write_json(out/'report.json',report);emit('complete',report_sha256=file_hash(out/'report.json'))
    print(json.dumps({k:v for k,v in report.items() if k not in {'steps','adapter_files_sha256'}},indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--audit-file',type=Path)
    p.add_argument('--max-steps',type=int,default=4)
    p.add_argument('--train-limit',type=int,default=4,help='0 = all fit rows; longest row is always included')
    p.add_argument('--validation-limit',type=int,default=4,help='0 = all inner validation rows')
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every',type=int,default=0)
    a=p.parse_args(argv)
    if a.save_every<0:p.error('save-every must be nonnegative')
    output_existed=safe_path(a.output_dir).exists()
    try:run(a)
    except Exception as error:
        out=safe_path(a.output_dir)
        if not output_existed and out.is_dir() and (out/'provenance.json').exists() and not (out/'report.json').exists():
            write_json(out/'failure.json',{'status':'failed','error_type':type(error).__name__,'message':str(error)})
        raise


if __name__=='__main__':main()
