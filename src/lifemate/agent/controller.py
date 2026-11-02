import time

from lifemate.contracts import Outcome
from .prompt import messages
from .planner import parse_action


def run_controller(payload, env, planner, *, seconds=30, max_decisions=12, max_calls=8, max_repairs=2):
    start=time.monotonic(); deadline=start+seconds
    events=[]; errors=calls=decisions=0; stop='max_decisions'
    allowed={t['name'] for t in payload['tools']}
    for _ in range(max_decisions):
        remaining=deadline-time.monotonic()
        if remaining <= .5: stop='deadline'; break
        decisions += 1
        try:
            raw=planner.generate(messages(payload,events), remaining-.3)
        except TimeoutError: stop='deadline'; break
        # Do not dispatch a late-generated action.
        if time.monotonic() >= deadline-.2: stop='deadline'; break
        try:
            action=parse_action(raw)
        except (ValueError,TypeError,RecursionError):
            errors+=1; events.append({'validation_error':'Output must be one valid action JSON object.'})
            if errors>max_repairs: stop='invalid_action'; break
            continue
        if action['type']=='finish': stop='finished'; break
        if calls >= max_calls: stop='max_calls'; break
        if action['name'] not in allowed:
            valid={'ok':False,'error':'Tool not in current allowlist'}
        else: valid=env.validate(action['name'],action['arguments'])
        if not valid['ok']:
            errors+=1; events.append({'action':action,'validation_error':valid.get('error')})
            if errors>max_repairs: stop='validation_limit'; break
            continue
        observation=env.call(action['name'],action['arguments']); calls+=1
        events.append({'action':action,'observation':observation})
        if action['name']=='sys.ask_missing_info' and observation.get('ok'):
            stop='asked_missing_info'; break
    return Outcome(stop,time.monotonic()-start,decisions,errors,calls)
