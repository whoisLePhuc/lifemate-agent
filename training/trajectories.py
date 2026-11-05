"""Gold train prefixes only; exact current custom-JSON inference representation."""
from copy import deepcopy
import json

from lifemate.agent.planner import parse_action
from lifemate.agent.prompt import messages
from lifemate.context.manager import text_input
from training.preparation import split_groups


def unique_ids(values,label):
    if not isinstance(values,list) or not values or any(not isinstance(x,str) or not x for x in values):
        raise ValueError(label+': expected nonempty string ID list')
    if len(values)!=len(set(values)):raise ValueError(label+': duplicate ID')
    return set(values)


def inner_split(manifest,*,validation_fraction=.1,seed=43,seen_ids=()):
    if not 0<validation_fraction<1:raise ValueError('invalid validation fraction')
    train=unique_ids(manifest['train_ids'],'outer train');dev=unique_ids(manifest['dev_ids'],'outer dev')
    if train&dev:raise ValueError('outer partitions overlap')
    groups=manifest['groups']
    if not isinstance(groups,list):raise ValueError('groups must be an array')
    covered=set();training_groups=[]
    for group in groups:
        ids=unique_ids(group,'group')
        if ids&covered or not ids<=train|dev:raise ValueError('invalid group coverage')
        if ids&train and ids&dev:raise ValueError('outer group crosses train/dev')
        covered|=ids
        if ids<=train:training_groups.append(sorted(ids))
    if covered!=train|dev:raise ValueError('groups do not cover outer partitions')
    seen=set(seen_ids)
    if not seen<=train:raise ValueError('seen IDs must belong to outer train')
    forced=[g for g in training_groups if set(g)&seen]
    eligible=[g for g in training_groups if not set(g)&seen]
    available=sum(map(len,eligible));target=max(1,round(len(train)*validation_fraction))
    if target>=available or len(eligible)<2:raise ValueError('not enough eligible groups for validation')
    fit,validation=split_groups(eligible,dev_fraction=target/available,seed=seed)
    fit=sorted(fit+[i for group in forced for i in group])
    if set(fit)&set(validation) or set(fit)|set(validation)!=train:raise AssertionError('inner split integrity failed')
    return {'schema_version':1,'seed':seed,'validation_fraction_target':validation_fraction,
            'train_ids':fit,'validation_ids':validation,'outer_dev_ids':sorted(dev),
            'seen_ids':sorted(seen),'forced_train_ids':sorted(i for g in forced for i in g),
            'groups':sorted(training_groups,key=lambda g:g[0]),
            'limitations':'Retrospective validation from previously inspected train; grouping is heuristic, not semantic proof.'}


def explicit_inner_split(manifest,supplied):
    """Validate a previously fixed inner split (for example one recovered from a completed run).

    The supplied split must partition the outer train IDs exactly, reproduce the outer dev IDs
    and keep every manifest group on one side. Nothing is re-derived from a seed.
    """
    train=unique_ids(manifest['train_ids'],'outer train');dev=unique_ids(manifest['dev_ids'],'outer dev')
    if train&dev:raise ValueError('outer partitions overlap')
    fit=unique_ids(supplied['train_ids'],'inner train');validation=unique_ids(supplied['validation_ids'],'inner validation')
    if unique_ids(supplied['outer_dev_ids'],'supplied outer dev')!=dev:raise ValueError('supplied outer dev differs from manifest')
    if fit&validation or fit|validation!=train:raise ValueError('inner split must partition outer train exactly')
    groups=[sorted(g) for g in manifest['groups'] if set(g)<=train]
    for group in groups:
        if set(group)&fit and set(group)&validation:raise ValueError('group crosses the supplied inner split')
    return {'schema_version':1,'seed':None,'validation_fraction_target':None,
            'train_ids':sorted(fit),'validation_ids':sorted(validation),'outer_dev_ids':sorted(dev),
            'seen_ids':[],'forced_train_ids':[],'groups':sorted(groups,key=lambda g:g[0]),
            'limitations':'Explicit inner split supplied by the caller; grouping is heuristic, not semantic proof.'}


def trajectory_rows(example):
    """Yield independent next-action examples. Never consult final_state."""
    item,ref=example.agent_input,example.reference
    if len(ref.tool_calls)!=len(ref.tool_results):raise ValueError('call/result mismatch')
    allowed={tool['name'] for tool in item.tools}
    for i,call in enumerate(ref.tool_calls):
        if call['name'] not in allowed:raise ValueError('gold tool outside sample allowlist')
        parse_action(json.dumps({'type':'call',**call},ensure_ascii=False,allow_nan=False))
        result=ref.tool_results[i]
        if not isinstance(result,dict) or type(result.get('ok')) is not bool:
            raise ValueError('invalid gold observation')
        if result['ok'] is False:
            raise ValueError('rejected gold call cannot form a validated controller prefix')
        if call['name']=='sys.ask_missing_info' and (i!=len(ref.tool_calls)-1 or not result['ok']):
            raise ValueError('ask termination differs from current controller; no silent truncation')
    payload=text_input(item,ref.audio_text,ref.history_audio_text)
    actions=[{'type':'call',**deepcopy(c)} for c in ref.tool_calls]
    asked=bool(actions and actions[-1]['name']=='sys.ask_missing_info')
    if not asked:actions.append({'type':'finish'})
    events=[]
    for i,action in enumerate(actions):
        completion=json.dumps(action,ensure_ascii=False,allow_nan=False)
        row={'schema_version':1,'id':f'{item.id}:{i}','sample_id':item.id,'step_index':i,
             'trajectory_rows':len(actions),'target_type':action['type'],
             'target_tool':action.get('name'),'prompt':messages(payload,events),
             'completion':[{'role':'assistant','content':completion}]}
        # Detach prefixes: later appends must not mutate already-yielded rows.
        yield deepcopy(row)
        if action['type']=='call':
            events.append({'action':deepcopy(action),'observation':deepcopy(ref.tool_results[i])})
