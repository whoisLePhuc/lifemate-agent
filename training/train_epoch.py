"""One-epoch resumable Qwen3.5 QLoRA trainer reusing the audited NF4/loss/data primitives.

Integrated from the former Experiments/020-one-epoch-training/train_epoch.py (tag
archive/experiments-011-021 in the workspace repository). Training logic is unchanged;
workspace-specific paths became arguments (--dataset-dir, --model-path).
"""
import argparse,json,sys,os,time,math,random,gc,shutil
from pathlib import Path
from hashlib import sha256
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1]
from training import qwen35 as compat
from training.epoch_schedule import epoch_order,groups,lr_factor
from training.sft_data import verify_dataset,TokenRows
from training.prepare_trajectories import safe_path
from lifemate.observability.provenance import file_hash,collect_provenance
from lifemate.agent.prompt import SYSTEM

def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def fingerprint(x):return sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sources():
    paths=sorted((REPO/'training').glob('*.py'))
    return {str(p.relative_to(REPO)):file_hash(p) for p in paths}

def prepare_data(DS,MODEL,seed,smoke_rows,validation_limit):
    compat.patch_tokenizer_loading()
    from transformers import AutoTokenizer
    m,split,report=verify_dataset(DS)
    if m['prompt_sha256']!=sha256(SYSTEM.encode()).hexdigest():raise ValueError('prompt mismatch')
    for n,h in report['provenance']['model']['metadata_sha256'].items():
        if file_hash(MODEL/n)!=h:raise ValueError('model metadata mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    fit=TokenRows(DS/'train.tokens.jsonl',partition='train',allowed_ids=set(split['train_ids']),eos_token_id=tok.eos_token_id)
    val=TokenRows(DS/'validation.tokens.jsonl',partition='validation',allowed_ids=set(split['validation_ids']),eos_token_id=tok.eos_token_id)
    if len(fit)!=3517 or len(val)!=392:raise ValueError('frozen row counts differ')
    order=epoch_order(fit.entries,seed,smoke_rows);vorder=list(range(len(val)))[:validation_limit or None]
    return tok,fit,val,order,vorder,split

def config_for(args,order,total):
    DS=args.dataset_dir
    from importlib.metadata import version
    packages={n:version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','flash-linear-attention']}
    warmup=min(total-1,math.ceil(total*args.warmup_ratio))
    return {'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'packages':packages,'seed':args.seed,'smoke_rows':args.smoke_rows,'validation_limit':args.validation_limit,'accumulation':args.accumulation,'learning_rate':args.learning_rate,'warmup_steps':warmup,'scheduler':'explicit step-indexed warmup cosine','min_lr_ratio':.1,'optimizer_steps':total,'order':order,'dataset_manifest_sha256':file_hash(DS/'dataset_manifest.json'),'model_metadata':read(DS/'report.json')['provenance']['model']['metadata_sha256'],'system_sha256':sha256(SYSTEM.encode()).hexdigest(),'source_sha256':sources(),'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,'attention':['efficient','flash'],'loss':'mean micro-row action loss, final partial accumulation normalized by its actual row count'}

def rng_state():
    import torch,numpy as np
    s=np.random.get_state()
    return {'python':random.getstate(),'numpy':(s[0],s[1].tolist(),s[2],s[3],s[4]),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s):
    import torch,numpy as np
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def save_checkpoint(out,step,model,optimizer,config,before,elapsed,history):
    import torch
    dest=out/f'checkpoint-{step:04d}';tmp=out/f'checkpoint-{step:04d}.partial'
    if dest.exists() or tmp.exists():raise ValueError('checkpoint already exists; never overwrite')
    tmp.mkdir();model.save_pretrained(tmp/'adapter',safe_serialization=True)
    # Optimizer buffers and RNG are local trusted artifacts; load with weights_only=True.
    torch.save({'optimizer':optimizer.state_dict(),'rng':rng_state(),'completed_steps':step,'consumed_rows':sum(len(c) for c in groups(config['order'],config['accumulation'])[:step]),'scheduler':{'completed_steps':step,'total_steps':config['optimizer_steps'],'warmup_steps':config['warmup_steps'],'next_lr':config['learning_rate']*lr_factor(step,config['optimizer_steps'],config['warmup_steps']) if step<config['optimizer_steps'] else None},'config_fingerprint':fingerprint(config),'validation_before':before,'elapsed_seconds':elapsed,'history':history},tmp/'training_state.pt')
    write(tmp/'config.json',config)
    hashes={str(p.relative_to(tmp)):file_hash(p) for p in tmp.rglob('*') if p.is_file()}
    write(tmp/'checkpoint.json',{'status':'complete','step':step,'files_sha256':hashes,'config_fingerprint':fingerprint(config)})
    for p in tmp.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,dest);return dest

def verify_checkpoint(path,config):
    path=safe_path(path)
    if any(p.lower() in {'data','models'} for p in path.parts):raise ValueError('resume checkpoint must not live under Data/Models')
    marker=read(path/'checkpoint.json')
    if marker['status']!='complete' or marker['config_fingerprint']!=fingerprint(config):raise ValueError('resume configuration mismatch')
    for n,h in marker['files_sha256'].items():
        p=safe_path(path/n)
        if not p.is_relative_to(path) or file_hash(p)!=h:raise ValueError('checkpoint fingerprint/path mismatch')
    return marker

def tensor_tree_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b

def run(args):
    import torch,numpy as np
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    import training.train_sft as helpers
    from peft import LoraConfig,get_peft_model,PeftModel
    if args.seed<0 or args.smoke_rows<0 or args.validation_limit<0 or args.accumulation<1 or not 0<=args.warmup_ratio<1 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:raise ValueError('invalid training settings')
    if args.save_every<1 or args.stop_after<0 or args.max_train_seconds<0 or not math.isfinite(args.max_train_seconds) or args.validation_every<0:raise ValueError('invalid operational limits')
    out,DS,MODEL=map(safe_path,(args.output_dir,args.dataset_dir,args.model_path));args.dataset_dir,args.model_path=DS,MODEL
    if out.exists() or out.is_relative_to(DS) or out.is_relative_to(MODEL) or any(p.lower() in {'data','models'} for p in out.parts):raise ValueError('choose a new output directory outside Data/Models/dataset/model')
    tok,fit,val,order,vorder,split=prepare_data(DS,MODEL,args.seed,args.smoke_rows,args.validation_limit);chunks=groups(order,args.accumulation);cfg=config_for(args,order,len(chunks))
    out.mkdir(parents=True);write(out/'config.json',cfg);write(out/'selection.json',{'train_rows':[fit.entries[i] for i in order],'validation_rows':[val.entries[i] for i in vorder]});write(out/'inner_validation_ids.json',split['validation_ids'])
    if args.preflight:
        from training.audit_trajectories import audit_dataset
        audit=audit_dataset(DS,tok,torch_probe=True);write(out/'full_mask_audit.json',audit)
        write(out/'report.json',{'status':'preflight_complete','training_performed':False,'fit_rows':len(fit),'validation_rows':len(val),'optimizer_steps':len(chunks),'final_accumulation_rows':len(chunks[-1]),'order_unique':len(set(order))==len(order),'config_fingerprint':fingerprint(cfg)});return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
    if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('need at least20GiB free for full resumable checkpoints')
    if args.resume:verify_checkpoint(args.resume,cfg)
    helpers.efficient_attention=compat.efficient_attention
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=False
    write(out/'provenance.json',collect_provenance(REPO,config={**cfg,'order':'selection.json','resume':str(args.resume) if args.resume else None},model_path=MODEL,prompt=SYSTEM))
    model=compat.load_base(MODEL,training=True)
    if args.resume:model=PeftModel.from_pretrained(model,args.resume/'adapter',is_trainable=True,local_files_only=True)
    else:model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    params=[p for p in model.parameters() if p.requires_grad]
    if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad) or model.get_nb_trainable_parameters()[1]>=5_000_000_000:raise ValueError('parameter budget/trainable weights')
    optimizer=torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=0.,foreach=False)
    started=time.perf_counter();history=[];completed=0;previous_elapsed=0
    if args.resume:
        state=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=True)
        if state['config_fingerprint']!=fingerprint(cfg) or state['completed_steps']!=read(args.resume/'checkpoint.json')['step']:raise ValueError('checkpoint cursor mismatch')
        completed=state['completed_steps'];optimizer.load_state_dict(state['optimizer']);before=state['validation_before'];history=state['history'];previous_elapsed=state['elapsed_seconds']
        expected_rows=sum(len(c) for c in chunks[:completed])
        if state['consumed_rows']!=expected_rows or state['scheduler']['completed_steps']!=completed:raise ValueError('resume row/scheduler mismatch')
        if completed>=len(chunks):raise ValueError('checkpoint already completed epoch')
        expected_next=args.learning_rate*lr_factor(completed,len(chunks),cfg['warmup_steps'])
        if state['scheduler']['next_lr']!=expected_next:raise ValueError('saved scheduler next LR mismatch')
        from safetensors.torch import load_file
        saved_adapter=load_file(args.resume/'adapter/adapter_model.safetensors')
        restored_adapter=helpers.adapter_state(model)
        adapter_exact=tensor_tree_equal(saved_adapter,restored_adapter)
        optimizer_exact=tensor_tree_equal(state['optimizer'],optimizer.state_dict())
        restore_rng(state['rng']);rng_exact=tensor_tree_equal(state['rng'],rng_state())
        if not (adapter_exact and optimizer_exact and rng_exact):raise ValueError('initial resume state does not match checkpoint')
        write(out/'resume_restore.json',{'adapter_exact':adapter_exact,'optimizer_exact':optimizer_exact,'rng_exact':rng_exact,'cursor_verified':True,'scheduler_next_lr_verified':True,'source_checkpoint':str(args.resume)})
        del state,saved_adapter,restored_adapter
    else:before=helpers.evaluate_rows(model,val,vorder,tok)
    def emit(event,**values):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':event,**values},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    emit('start',completed_steps=completed,total_steps=len(chunks),total_rows=len(order),restored_optimizer_rng=bool(args.resume))
    last_saved=None;training_start=time.perf_counter()
    for step_index in range(completed,len(chunks)):
        if args.stop_after and completed>=args.stop_after:break
        if args.max_train_seconds and time.perf_counter()-training_start>=args.max_train_seconds:break
        tick=time.perf_counter();batch_ids=chunks[step_index];model.train();optimizer.zero_grad(set_to_none=True);lr=args.learning_rate*lr_factor(step_index,len(chunks),cfg['warmup_steps'])
        for group in optimizer.param_groups:group['lr']=lr
        torch.cuda.reset_peak_memory_stats();loss_sum=0.;lengths=[]
        for i in batch_ids:
            row=fit[i];batch=helpers.to_batch(row,tok)
            with compat.efficient_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=helpers.sparse_action_loss(model,batch)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                (loss/len(batch_ids)).backward()
            loss_sum+=float(loss.detach());lengths.append(len(row['input_ids']))
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();completed=step_index+1
        metric={'step':completed,'lr':lr,'row_indices':batch_ids,'sequence_lengths':lengths,'micro_rows':len(batch_ids),'loss':loss_sum/len(batch_ids),'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-tick,'peak_reserved_bytes':torch.cuda.max_memory_reserved()};history.append(metric);emit('train_step',**metric);print(f"step{completed}/{len(chunks)} rows={len(batch_ids)} lr={lr:.8g} loss={metric['loss']:.4f}",flush=True)
        if args.validation_every and completed%args.validation_every==0 and completed<len(chunks):
            saved_rng=rng_state();validation=helpers.evaluate_rows(model,val,vorder,tok);restore_rng(saved_rng);emit('validation',step=completed,metrics=validation)
        if completed%args.save_every==0:
            last_saved=save_checkpoint(out,completed,model,optimizer,cfg,before,previous_elapsed+time.perf_counter()-started,history);emit('checkpoint',step=completed,path=str(last_saved))
    if completed==0:raise ValueError('no optimizer steps completed')
    if last_saved is None or read(last_saved/'checkpoint.json')['step']!=completed:last_saved=save_checkpoint(out,completed,model,optimizer,cfg,before,previous_elapsed+time.perf_counter()-started,history)
    after=helpers.evaluate_rows(model,val,vorder,tok);final=helpers.adapter_state(model);probe_row=fit[min(order,key=lambda i:fit.entries[i]['sequence_tokens'])];expected=helpers.probe(model,probe_row,tok)
    state_hash=helpers.state_hash(final);adapter=last_saved/'adapter'
    del model,optimizer,params,loss,batch;gc.collect();torch.cuda.empty_cache()
    reload=PeftModel.from_pretrained(compat.load_base(MODEL,training=False),adapter,is_trainable=False,local_files_only=True);restored=helpers.adapter_state(reload);actual=helpers.probe(reload,probe_row,tok)
    exact=final.keys()==restored.keys() and all(torch.equal(final[k],restored[k]) for k in final)
    if not exact or expected['argmax_ids']!=actual['argmax_ids'] or not math.isclose(expected['loss'],actual['loss'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('adapter reload parity failed')
    verify_dataset(DS)
    if sources()!=cfg['source_sha256']:raise ValueError('experiment sources changed during run')
    consumed=[i for h in history for i in h['row_indices']]
    if consumed!=order[:len(consumed)] or len(set(consumed))!=len(consumed):raise ValueError('row order/repetition failure')
    result={'status':'complete' if completed==len(chunks) else 'stopped_checkpointed','optimizer_steps':completed,'total_optimizer_steps':len(chunks),'consumed_rows':len(consumed),'planned_rows':len(order),'final_accumulation_rows':len(chunks[-1]),'validation_before':before,'validation_after':after,'checkpoint':str(last_saved),'final_adapter_sha256':state_hash,'reload_state_exact':exact,'elapsed_seconds':previous_elapsed+time.perf_counter()-started,'dataset_manifest_sha256':cfg['dataset_manifest_sha256'],'training_performed':True,'resume_supported':True,'closed_loop_score_computed':False};write(out/'report.json',result);emit('complete',status=result['status']);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke-rows',type=int,default=0);p.add_argument('--validation-limit',type=int,default=0);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--warmup-ratio',type=float,default=.05);p.add_argument('--save-every',type=int,default=100);p.add_argument('--validation-every',type=int,default=400);p.add_argument('--stop-after',type=int,default=0);p.add_argument('--max-train-seconds',type=float,default=0);a=p.parse_args()
    if a.resume:a.resume=safe_path(a.resume)
    existed=safe_path(a.output_dir).exists()
    try:run(a)
    except Exception as e:
        if not existed and a.output_dir.exists() and not (a.output_dir/'failure.json').exists():write(a.output_dir/'failure.json',{'status':'failed','error':repr(e)})
        raise
if __name__=='__main__':main()
