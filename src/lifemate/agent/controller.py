import time

from lifemate.contracts import Outcome
from .protocol import build_messages, parse_output, system_prompt, tool_schemas


def run_controller(payload, env, planner, *, seconds=30, max_decisions=12, max_calls=8, max_repairs=2,
                   trace=None, protocol='custom_json'):
    system_prompt(protocol)
    planner_kwargs={'tools':tool_schemas(payload)} if protocol=='qwen_native' else {}
    start=time.monotonic(); deadline=start+seconds
    events=[]; errors=calls=decisions=0; stop='max_decisions'
    allowed={t['name'] for t in payload['tools']}
    def emit(event, **fields):
        if trace is not None: trace(event, decision=decisions, **fields)
    for _ in range(max_decisions):
        remaining=deadline-time.monotonic()
        if remaining <= .5: stop='deadline'; break
        decisions += 1
        emit('decision_start', remaining_seconds=remaining, calls=calls, errors=errors)
        generation_start=time.perf_counter()
        try:
            raw=planner.generate(build_messages(payload,events,protocol), remaining-.3, **planner_kwargs)
        except TimeoutError as error:
            emit('generation', raw_output=getattr(planner,'last_generation',{}).get('partial_output'),
                 error_type=type(error).__name__, error_category='deadline',
                 generation=getattr(planner,'last_generation',{}),
                 elapsed_seconds=time.perf_counter()-generation_start)
            stop='deadline'; break
        except (ValueError,RuntimeError) as error:
            emit('generation', error_type=type(error).__name__, error_category='backend_error',
                 generation=getattr(planner,'last_generation',{}),
                 elapsed_seconds=time.perf_counter()-generation_start)
            error.controller_outcome=Outcome('error',time.monotonic()-start,decisions,errors,calls)
            raise
        emit('generation', raw_output=raw, generation=getattr(planner,'last_generation',{}),
             elapsed_seconds=time.perf_counter()-generation_start)
        # Do not dispatch a late-generated action.
        if time.monotonic() >= deadline-.2: stop='deadline'; break
        try:
            # A cut-off native text response is not evidence of a deliberate finish.
            if protocol=='qwen_native' and getattr(planner,'last_generation',{}).get('finish_reason')=='token_limit':
                raise ValueError('native response reached token limit; do not dispatch partial output')
            action=parse_output(raw,protocol)
        except (ValueError,TypeError,RecursionError) as error:
            emit('parse_error', raw_output=raw, error_type=type(error).__name__,
                 error_message=str(error), error_category='parse_error')
            errors+=1
            feedback={'validation_error':'Output must be one valid action JSON object.'}
            if protocol=='qwen_native':
                feedback={'validation_error':str(error),'raw_output':raw}
            events.append(feedback)
            if errors>max_repairs: stop='invalid_action'; break
            continue
        emit('parsed_action', action=action)
        if action['type']=='finish': stop='finished'; break
        if calls >= max_calls: stop='max_calls'; break
        validation_start=time.perf_counter()
        if action['name'] not in allowed:
            valid={'ok':False,'error':'Tool not in current allowlist'}
        else: valid=env.validate(action['name'],action['arguments'])
        emit('validation', action=action, result=valid,
             elapsed_seconds=time.perf_counter()-validation_start)
        if not valid['ok']:
            errors+=1
            feedback={'action':action,'validation_error':valid.get('error')}
            if protocol=='qwen_native': feedback['raw_output']=raw
            events.append(feedback)
            if errors>max_repairs: stop='validation_limit'; break
            continue
        emit('call_start', action=action, call_index=calls)
        call_start=time.perf_counter()
        try:
            observation=env.call(action['name'],action['arguments'])
        except Exception as error:
            emit('call_error', action=action, error_type=type(error).__name__,
                 indeterminate=True, elapsed_seconds=time.perf_counter()-call_start)
            raise
        calls+=1
        emit('call_result', action=action, call_index=calls-1, observation=observation,
             schema_rejected=observation.get('ok') is False,
             business_failure=observation.get('ok') is True and observation.get('done') is False,
             elapsed_seconds=time.perf_counter()-call_start)
        events.append({'action':action,'observation':observation})
        if action['name']=='sys.ask_missing_info' and observation.get('ok'):
            stop='asked_missing_info'; break
    outcome=Outcome(stop,time.monotonic()-start,decisions,errors,calls)
    emit('controller_end', stop=stop, elapsed_seconds=outcome.elapsed_seconds,
         calls=calls, errors=errors)
    return outcome
