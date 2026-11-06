import json
import time

from lifemate.contracts import Outcome
from .protocol import NATIVE_PROTOCOLS, build_messages, parse_output, system_prompt, tool_schemas
from .policy import GroundedPolicy, READ_POLICIES


def reference_sources(tool, observation, tools):
    """Link schema references to ID paths present in a successful read."""
    if tool.get('kind') != 'read' or observation.get('ok') is not True or observation.get('done') is False:
        return []
    app = tool.get('app') or tool['name'].split('.')[0]
    sources = []
    for target in tools.values():
        for field, spec in target.get('parameters', {}).get('properties', {}).items():
            domain = spec.get('x-ref')
            if not isinstance(domain, str):
                continue
            namespace, _, collection = domain.partition('.')
            records = observation.get(collection)
            if namespace != app or not isinstance(records, list):
                continue
            count = sum(isinstance(record, dict) and isinstance(record.get('id'), str)
                        and bool(record['id']) for record in records)
            sources.append({'tool': target['name'], 'field': field,
                            'candidate_id_path': f'{collection}[].id', 'candidate_count': count})
    return sources


def recovery_options(tool, observation, tools_by_name):
    options = ['Dùng kết quả nếu nó xác định được đối tượng cần thao tác.',
               'Tra theo dữ kiện khác chỉ khi dữ kiện đó có trong yêu cầu hoặc lịch sử.',
               'Nếu yêu cầu đã hoàn tất hoặc không có tool phù hợp, dùng finish.']
    if 'sys.ask_missing_info' not in tools_by_name:
        return options
    domains = {f"{tool.get('app', tool['name'].split('.')[0])}.{key}"
               for key, value in observation.items() if isinstance(value, list)}
    for target in tools_by_name.values():
        if target.get('kind') != 'write':
            continue
        schema = target.get('parameters', {})
        required = set(schema.get('required', ()))
        required.update(field for group in schema.get('x-one-of', ()) for field in group)
        for field, spec in schema.get('properties', {}).items():
            if field in required and spec.get('x-ref') in domains:
                options.append({'condition': 'Chỉ hỏi nếu người dùng yêu cầu thao tác này và trường vẫn thiếu.',
                                'name': 'sys.ask_missing_info',
                                'arguments': {'tool_name': target['name'], 'params': [field]}})
    return options


def run_controller(payload, env, planner, *, seconds=30, max_decisions=12, max_calls=8, max_repairs=2,
                   trace=None, protocol='custom_json',policy='off',lang='vi'):
    system_prompt(protocol,policy,lang)
    native=protocol in NATIVE_PROTOCOLS
    guard=GroundedPolicy(payload) if policy=='grounded_v1' else None
    shadow = GroundedPolicy(payload) if policy == 'clarification_shadow_v1' else None
    repeat_reads = policy in READ_POLICIES
    recovery = repeat_reads and policy != 'repeat_reads_v1'
    tools_by_name = {tool['name']: tool for tool in payload['tools']}
    successful_reads = {}
    recovery_attempted = False
    planner_kwargs={'tools':tool_schemas(payload)} if native else {}
    start=time.monotonic(); deadline=start+seconds
    events=[]; errors=calls=decisions=0; stop='max_decisions'
    allowed={t['name'] for t in payload['tools']}
    def emit(event, **fields):
        if trace is not None: trace(event, decision=decisions, **fields)
    def policy_check(action):
        nonlocal errors
        if guard is None:return True
        verdict=guard.inspect(action)
        emit('policy_check',action=action,verdict=verdict)
        if verdict['ok']:return True
        guard.blocked(action,verdict)
        errors+=1
        events.append({'policy_error':verdict['issues'],'executed':False})
        return False
    for _ in range(max_decisions):
        remaining=deadline-time.monotonic()
        if remaining <= .5: stop='deadline'; break
        decisions += 1
        emit('decision_start', remaining_seconds=remaining, calls=calls, errors=errors)
        generation_start=time.perf_counter()
        try:
            model_payload=payload if guard is None else {
                **{k:payload[k] for k in ('request','context','history','tools')},'controller_context':guard.summary()}
            raw=planner.generate(build_messages(model_payload,events,protocol,policy,lang), remaining-.3, **planner_kwargs)
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
            if native and getattr(planner,'last_generation',{}).get('finish_reason')=='token_limit':
                raise ValueError('native response reached token limit; do not dispatch partial output')
            action=parse_output(raw,protocol,tools_by_name)
        except (ValueError,TypeError,RecursionError) as error:
            emit('parse_error', raw_output=raw, error_type=type(error).__name__,
                 error_message=str(error), error_category='parse_error')
            errors+=1
            feedback={'validation_error':'Output must be one valid action JSON object.'}
            if native:
                feedback={'validation_error':str(error),'raw_output':raw}
            events.append(feedback)
            if errors>max_repairs: stop='invalid_action'; break
            continue
        emit('parsed_action', action=action)
        if action['type']=='finish':
            if not policy_check(action):
                if errors>max_repairs:stop='policy_limit';break
                continue
            stop='finished'; break
        if calls >= max_calls: stop='max_calls'; break
        validation_start=time.perf_counter()
        if action['name'] not in allowed:
            valid={'ok':False,'error':'Tool not in current allowlist'}
        else: valid=env.validate(action['name'],action['arguments'])
        emit('validation', action=action, result=valid,
             elapsed_seconds=time.perf_counter()-validation_start)
        if not valid['ok']:
            if guard is not None:guard.validation_failed(action)
            errors+=1
            feedback={'action':action,'validation_error':valid.get('error')}
            if native: feedback['raw_output']=raw
            events.append(feedback)
            if errors>max_repairs: stop='validation_limit'; break
            continue
        if not policy_check(action):
            if errors>max_repairs:stop='policy_limit';break
            continue
        if shadow is not None and (action['name'] == 'sys.ask_missing_info' or
                                   tools_by_name[action['name']].get('kind') == 'write'):
            verdict = shadow.inspect(action)
            issues = [issue for issue in verdict['issues'] if issue['code'] in {
                'invalid_ask_target', 'invalid_ask_fields', 'lookup_before_asking',
                'reference_already_resolved', 'ungrounded_id', 'ambiguous_reference',
                'different_lookup_candidate', 'ungrounded_or_ambiguous_phone'}]
            emit('clarification_shadow', action=action, issues=issues, sources=verdict['sources'])
        kind = tools_by_name[action['name']].get('kind')
        signature = None
        if repeat_reads and kind == 'read':
            signature = json.dumps([action['name'], action['arguments']], sort_keys=True,
                                   ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            if signature in successful_reads:
                errors += 1
                feedback = {'action': action, 'validation_error':
                               'Truy vấn này đã thành công và dữ liệu chưa thay đổi. '
                               'Hãy dùng observation trước hoặc đổi điều kiện tìm kiếm.',
                               'executed': False}
                if recovery:
                    previous = successful_reads[signature]
                    feedback['previous_observation'] = previous
                    if policy == 'repeat_reads_options':
                        feedback['next_options'] = recovery_options(
                            tools_by_name[action['name']], previous, tools_by_name)
                events.append(feedback)
                emit('repeated_read_blocked', action=action)
                if errors > max_repairs or (recovery and recovery_attempted):
                    stop = 'repeat_read_limit'
                    break
                recovery_attempted = recovery
                continue
        elif repeat_reads:
            # Unknown tool kinds may change state as well. Invalidate conservatively.
            successful_reads.clear()
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
        event = {'action': action, 'observation': observation}
        if policy == 'xref_observation_v1':
            sources = reference_sources(tools_by_name[action['name']], observation, tools_by_name)
            if sources:
                event['reference_sources'] = sources
        events.append(event)
        if signature is not None and observation.get('ok') is True and observation.get('done') is not False:
            successful_reads[signature] = observation
        if guard is not None:
            guard.observe(action,observation)
            emit('progress_update',summary=guard.summary())
        if shadow is not None:
            shadow.observe(action, observation)
        if action['name']=='sys.ask_missing_info' and observation.get('ok'):
            stop='asked_missing_info'; break
    outcome=Outcome(stop,time.monotonic()-start,decisions,errors,calls)
    emit('controller_end', stop=stop, elapsed_seconds=outcome.elapsed_seconds,
         calls=calls, errors=errors)
    return outcome
