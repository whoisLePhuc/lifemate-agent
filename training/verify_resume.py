"""Three fresh processes: uninterrupted, stop at checkpoint 1, resume; compare all states.

Integrated from the former Experiments/020-one-epoch-training/run_smoke.py (tag
archive/experiments-011-021 in the workspace repository). Needs a GPU and the Qwen3.5
training environment; it is a smoke diagnostic, not a task-quality evaluation.
"""
import json,subprocess,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
from lifemate.observability.provenance import file_hash
from training.prepare_trajectories import safe_path

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def verify(out,dataset,model):
 if out.exists() or out.is_relative_to(dataset) or out.is_relative_to(model) or any(p.lower() in {'data','models'} for p in out.parts):raise ValueError('choose a new output directory outside Data/Models/dataset/model')
 out.mkdir(parents=True);env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0',PYTHONPATH=os.pathsep.join([str(REPO/'src'),str(REPO)]))
 for name,extra in [('continuous',[]),('interrupted',['--stop-after','1']),('resumed',['--resume',str(out/'continuous/checkpoint-0001')])]:
  write(out/'progress.json',{'phase':name})
  with (out/(name+'.log')).open('x') as f:subprocess.run([sys.executable,'-u','-m','training.train_epoch','--dataset-dir',str(dataset),'--model-path',str(model),'--output-dir',str(out/name),'--smoke-rows','9','--validation-limit','4','--save-every','1','--validation-every','0',*extra],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(out/'progress.json',{'phase':'verifying'})
 import torch
 from safetensors.torch import load_file
 a=out/'continuous/checkpoint-0003';b=out/'resumed/checkpoint-0003';x=load_file(a/'adapter/adapter_model.safetensors');y=load_file(b/'adapter/adapter_model.safetensors')
 exact=x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x);maxdiff=max(float((x[k]-y[k]).abs().max()) for k in x)
 sa=torch.load(a/'training_state.pt',map_location='cpu',weights_only=True);sb=torch.load(b/'training_state.pt',map_location='cpu',weights_only=True)
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 optim=equal(sa['optimizer'],sb['optimizer']);rng=equal(sa['rng'],sb['rng']);schedule=sa['scheduler']==sb['scheduler']
 consumed=lambda s:[i for h in s['history'] for i in h['row_indices']]
 if consumed(sa)!=consumed(sb) or len(set(consumed(sa)))!=9:raise ValueError('resume row mismatch')
 restored=read(out/'resumed/resume_restore.json')
 if not all(restored[k] for k in ['adapter_exact','optimizer_exact','rng_exact','cursor_verified','scheduler_next_lr_verified']):raise ValueError('resume restoration failed')
 if not (rng and schedule):raise ValueError('RNG/scheduler progression changed')
 va=read(out/'continuous/report.json')['validation_after']['token_mean_loss'];vb=read(out/'resumed/report.json')['validation_after']['token_mean_loss']
 relative_loss=abs(va-vb)/max(abs(va),1e-12)
 if relative_loss>.05 or maxdiff>.001:raise ValueError(f'continuation diverged beyond smoke tolerances: loss={relative_loss},maxdiff={maxdiff}')
 report={'status':'RESTORE_AND_CONTINUATION_VERIFIED','initial_restore':restored,'bit_exact_continuation':exact and optim,'validation_relative_difference':relative_loss,'tolerances':{'validation_relative_difference':.05,'max_adapter_abs_difference':.001},'limits':['GPU backward can be numerically nondeterministic; final bit equality is reported, not assumed.','Smoke tolerances are diagnostics, not proof of task-score parity.'],'adapter_tensors_exact':exact,'optimizer_state_exact':optim,'rng_state_exact':rng,'scheduler_state_exact':schedule,'max_adapter_abs_difference':maxdiff,'consumed_rows':9,'micro_rows_per_step':[h['micro_rows'] for h in sa['history']],'learning_rates':[h['lr'] for h in sa['history']],'step_seconds':[h['elapsed_seconds'] for h in sa['history']],'peak_reserved_bytes':max(h['peak_reserved_bytes'] for h in sa['history']),'longest_row_tokens':max(n for h in sa['history'] for n in h['sequence_lengths']),'report_hashes':{n:file_hash(out/n/'report.json') for n in ['continuous','interrupted','resumed']},'training_performed':'smoke only; not full epoch','quality_improvement_claim':False};write(out/'report.json',report);write(out/'progress.json',{'phase':'complete'});print(json.dumps(report,indent=2))
def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 try:verify(*map(safe_path,(a.output_dir,a.dataset_dir,a.model_path)))
 except Exception as e:
  if a.output_dir.exists():write(a.output_dir/'failure.json',{'error':repr(e)})
  raise

if __name__=='__main__':main()
