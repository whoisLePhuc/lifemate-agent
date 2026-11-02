"""Evidence checks using public input/tool observations only; never app state.

This is a bounded heuristic policy, not a semantic proof of task correctness.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import re
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

POLICIES=('off','grounded_v1')
POLICY_PROMPT='''Trước mỗi action, xác định bước còn thiếu từ yêu cầu và observation; chỉ xuất action JSON như quy định, không xuất kế hoạch hoặc giải thích.
controller_context là dữ kiện kiểm tra từ controller, không phải toàn bộ trạng thái app.
ID phải lấy từ user/context hợp lệ hoặc kết quả tool; biết một ID không có nghĩa ID đó đúng người người dùng yêu cầu.
Ưu tiên tra cứu phù hợp nếu có thể giải quyết dữ kiện thiếu. Kết quả nhiều người cùng phù hợp hoặc không tìm được người cần gọi/nhắn: gọi sys.ask_missing_info cho tool đích, params đúng tên schema. Không chọn người đầu tiên hoặc dùng số điện thoại của người trùng tên để né hỏi.
Số tiền phải dựa yêu cầu user; số dư không phải số tiền muốn chi. Ngày tương đối dựa context.current_time/timezone. Không tự điền dữ kiện chưa có căn cứ.
Đủ dữ kiện mới thực thi; làm từng mục của yêu cầu, đọc kết quả trước bước phụ thuộc. sys.ask_missing_info cần tool_name là tool đích và params là trường required/x-one-of chưa xác định.
ok=true, done=false là chưa hoàn tất; không lặp lại cùng write thất bại nếu không có dữ kiện mới. Không lặp hành động đã thực hiện trừ yêu cầu lặp rõ ràng.
policy_error có executed=false: action đó chưa chạy, sửa theo issues/lookup_tools hoặc hỏi thiếu. Không coi finish là bằng chứng đã hoàn tất mọi mục.'''


def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD',text.lower().replace('đ','d'))
                   if unicodedata.category(c)!='Mn')


def _literal(value,text):
    return bool(re.search(r'(?<![\w])'+re.escape(str(value))+r'(?![\w])',text,re.I))


def _signature(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def money_values(text):
    """Supported numeric/spoken VND forms; excludes balances/other observations."""
    text=fold(text);values=set()
    scales={'k':1000,'nghin':1000,'ngan':1000,'tr':1000000,'trieu':1000000,'ty':1000000000,'dong':1,'vnd':1}
    for match in re.finditer(r'(?<![\w])([0-9]+(?:[.,][0-9]+)*)\s*(ty|trieu|tr|nghin|ngan|k|dong|vnd)?(?![\w])',text):
        raw,unit=match.groups()
        if unit is None:
            before,after=text[:match.start()],text[match.end():]
            if (raw.startswith('0') and len(raw)>=7) or before.endswith(('-','/')) or after.startswith(('-','/')):
                continue
            if re.search(r'\bnam\s*$',before):continue
        parts=re.split('[.,]',raw)
        if len(parts)>1 and all(len(p)==3 for p in parts[1:]):raw=''.join(parts)
        else:raw=raw.replace(',','.')
        try:
            value=Decimal(raw)*scales.get(unit,1)
            if value==value.to_integral_value():values.add(int(value))
        except InvalidOperation:pass
    digits={'khong':0,'mot':1,'hai':2,'ba':3,'bon':4,'tu':4,'nam':5,'lam':5,'sau':6,'bay':7,'tam':8,'chin':9}
    words=set(digits)|{'muoi','tram','nghin','ngan','trieu','ty','le','linh','ruoi','dong','nua'}
    runs=[];run=[]
    for token in re.findall(r'\w+',text)+['STOP']:
        if token in words:run.append(token)
        elif run:runs.append(run);run=[]
    for run in runs:
        if not any(t in {'nghin','ngan','trieu','ty','dong'} for t in run):continue
        if not any(t in digits or t in {'muoi','tram','nua'} for t in run):continue
        total=group=last=0;last_scale=1
        for token in run:
            if token in digits:last=digits[token]
            elif token=='nua':last=Decimal('0.5')
            elif token=='muoi':group+=(last or 1)*10;last=0
            elif token=='tram':group+=(last or 1)*100;last=0
            elif token in {'nghin','ngan','trieu','ty'}:
                last_scale=scales[token];total+=(group+last or 1)*last_scale;group=last=0
            elif token=='ruoi':total+=last_scale//2
        value=Decimal(total+group+last)
        if value==value.to_integral_value():values.add(int(value))
    return values


def time_anchors(context):
    try:
        current=datetime.fromisoformat(context['current_time'].replace('Z','+00:00'))
        if current.tzinfo is None:return {}
        try:current=current.astimezone(ZoneInfo(context['timezone']))
        except ZoneInfoNotFoundError:
            # Modern contest timestamps; fixed-offset fallback, not historical tz rules.
            fixed={'Asia/Ho_Chi_Minh':timezone(timedelta(hours=7)),
                   'Asia/Bangkok':timezone(timedelta(hours=7)),'UTC':timezone.utc}
            if context['timezone'] not in fixed:return {}
            current=current.astimezone(fixed[context['timezone']])
        return {name:(current+timedelta(days=offset)).date().isoformat()
                for name,offset in [('today',0),('tomorrow',1),('day_after_tomorrow',2)]}
    except (ValueError,KeyError,TypeError):return {}


class GroundedPolicy:
    def __init__(self,payload):
        self.payload=payload;self.tools={t['name']:t for t in payload['tools']}
        self.domains={p['x-ref'] for t in self.tools.values()
                      for p in t.get('parameters',{}).get('properties',{}).values() if 'x-ref' in p}
        self.evidence={domain:{} for domain in self.domains}
        self.latest={};self.steps=[];self.pending=None;self.seen={};self.versions={}
        self.read_versions={};self.read_results={}
        self.anchors=time_anchors(payload.get('context',{}))
        self.request=payload['request']
        self.amounts=money_values(self.request)
        if not self.amounts:
            for turn in reversed(payload.get('history',[])):
                if turn['role']=='user' and money_values(turn['text']):
                    self.amounts=money_values(turn['text']);break
        # Context is supplied input, not hidden app state. Only collection-shaped records count.
        for domain in self.domains:
            app=domain.split('.')[0]
            context=payload.get('context',{})
            self._collect(domain,context.get(app,{}),f'context:{app}')
            self._collect(domain,context.get('user_profile',{}).get(app,{}),f'context:user_profile:{app}')
        for turn in payload.get('history',[]):
            if turn['role']!='assistant':continue
            for call,result in zip(turn.get('tool_calls',[]),turn.get('tool_results',[])):
                self._ingest(call,result,'history',current=False)

    def _collect(self,domain,value,source):
        collection=domain.split('.')[-1]
        def visit(value):
            if isinstance(value,dict):
                records=value.get(collection)
                if isinstance(records,list):
                    for record in records:
                        if isinstance(record,dict) and isinstance(record.get('id'),str):
                            self.evidence[domain][record['id']]={'record':deepcopy(record),'source':source}
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(value)

    def _ingest(self,call,result,source,*,current):
        if result.get('ok') is not True or result.get('done') is False:return
        app=call['name'].split('.')[0]
        domains=[d for d in self.domains if d.split('.')[0]==app]
        for domain in domains:
            self._collect(domain,result,source+':'+call['name'])
            collection=domain.split('.')[-1]
            records=result.get(collection)
            if current and isinstance(records,list):
                self.latest[domain]={'records':deepcopy(records),'query':call['arguments'].get('query'),
                                     'tool':call['name']}
            # Creation endpoints often return a generic item; infer only a single domain.
            item=result.get('item')
            if len(domains)==1 and isinstance(item,dict) and isinstance(item.get('id'),str):
                self.evidence[domain][item['id']]={'record':deepcopy(item),'source':source+':'+call['name']}

    def lookup_tools(self,domain):
        app,collection=domain.split('.',1)
        result=[]
        for name,tool in self.tools.items():
            if tool.get('kind')!='read' or tool.get('app',name.split('.')[0])!=app:continue
            params=tool['parameters'];props=params.get('properties',{})
            if any(props.get(k,{}).get('x-ref')==domain for k in params.get('required',[])):continue
            if collection in tool.get('returns',''):result.append(name)
        return sorted(result)

    def _explicit_id(self,value):
        texts=[self.request]+[t['text'] for t in self.payload.get('history',[]) if t['role']=='user']
        return any(_literal(value,text) for text in texts)

    def _resolved_candidate(self,domain):
        lookup=self.latest.get(domain)
        if not lookup or not isinstance(lookup.get('query'),str) or not lookup['query'].strip():return None
        records=lookup['records'];query=fold(lookup['query'].strip())
        named=[r for r in records if isinstance(r,dict) and any(isinstance(r.get(k),str) and len(r[k])>=3
               and _literal(fold(r[k]),fold(self.request)) for k in ('name','phone'))]
        if len(named)==1:return named[0].get('id')
        exact=[r for r in records if isinstance(r,dict) and any(
            isinstance(r.get(k),str) and fold(r[k].strip())==query for k in ('name','relation','phone'))]
        if len(exact)==1:return exact[0].get('id')
        return records[0].get('id') if len(records)==1 and isinstance(records[0],dict) else None

    def _ambiguous(self,domain):
        lookup=self.latest.get(domain)
        return bool(lookup and isinstance(lookup.get('query'),str) and lookup['query'].strip()
                    and len(lookup['records'])>1 and self._resolved_candidate(domain) is None)

    def _refs(self,tool):
        return {k:p['x-ref'] for k,p in tool['parameters'].get('properties',{}).items() if 'x-ref' in p}

    def _read_stamp(self,tool):
        apps={tool['name'].split('.')[0]}|{d.split('.')[0] for d in self._refs(tool).values()}
        return tuple((app,self.read_versions.get(app,0)) for app in sorted(apps))

    def inspect(self,action):
        issues=[];sources={}
        if action['type']=='finish':
            if self.pending:issues.append({'code':'unresolved_action','target_tool':self.pending['tool'],
                                           'fields':self.pending['fields']})
            return {'ok':not issues,'issues':issues,'sources':sources}
        name,args=action['name'],action['arguments'];tool=self.tools[name]
        for field in sorted(set(args)-set(tool['parameters'].get('properties',{}))):
            issues.append({'code':'unknown_argument','field':field})
        if name=='sys.ask_missing_info':
            target=self.tools.get(args.get('tool_name'))
            if target is None or args.get('tool_name')==name:
                issues.append({'code':'invalid_ask_target'})
            else:
                spec=target['parameters'];required=set(spec.get('required',[]))
                required.update(k for group in spec.get('x-one-of',[]) for k in group)
                params=args.get('params',[])
                if len(set(params))!=len(params) or not params or not set(params)<=required:
                    issues.append({'code':'invalid_ask_fields','allowed_fields':sorted(required)})
                for field in params:
                    domain=self._refs(target).get(field)
                    if domain and domain not in self.latest and self.lookup_tools(domain):
                        issues.append({'code':'lookup_before_asking','field':field,
                                       'lookup_tools':self.lookup_tools(domain)})
                    elif domain and self._resolved_candidate(domain) is not None:
                        issues.append({'code':'reference_already_resolved','field':field,'domain':domain})
            return {'ok':not issues,'issues':issues,'sources':sources}
        for field,domain in self._refs(tool).items():
            value=args.get(field)
            if value in (None,'',[],{}):continue
            explicit=self._explicit_id(value)
            evidence=self.evidence.get(domain,{}).get(value)
            if not explicit and evidence is None:
                issues.append({'code':'ungrounded_id','field':field,'domain':domain,
                               'lookup_tools':self.lookup_tools(domain)})
            elif not explicit and tool.get('kind')=='write' and self._ambiguous(domain):
                issues.append({'code':'ambiguous_reference','field':field,'domain':domain,
                               'ask':{'tool_name':name,'params':[field]}})
            elif not explicit and tool.get('kind')=='write' and self._resolved_candidate(domain) not in (None,value):
                issues.append({'code':'different_lookup_candidate','field':field,'domain':domain})
            else:sources[field]='user_literal' if explicit else evidence['source']
        for field in ('phone','phone_number'):
            value=args.get(field)
            if not isinstance(value,str) or not value:continue
            digits=re.sub(r'\D','',value)
            texts=[self.request]+[t['text'] for t in self.payload.get('history',[]) if t['role']=='user']
            explicit=any(re.sub(r'\D','',phone)==digits for text in texts
                         for phone in re.findall(r'(?<!\d)\+?\d[\d ().-]{5,}\d(?!\d)',text)) if len(digits)>=7 else False
            matches=[(domain,e) for domain,items in self.evidence.items() for e in items.values()
                     if str(e['record'].get('phone',''))==value]
            if not explicit and (not matches or any(self._ambiguous(d) or self._resolved_candidate(d) not in (None,e['record']['id']) for d,e in matches)):
                ask_field='contact_id' if self._refs(tool).get('contact_id')=='contacts.contacts' else field
                issues.append({'code':'ungrounded_or_ambiguous_phone','field':field,
                               'ask':{'tool_name':name,'params':[ask_field]}})
            else:sources[field]='user_literal' if explicit else matches[0][1]['source']
        if 'amount' in args and args['amount'] not in (None,''):
            if args['amount'] not in self.amounts:issues.append({'code':'ungrounded_amount','field':'amount'})
            else:sources['amount']='user_money_expression'
        request=fold(self.request)
        relative=[key for key,pattern in [('tomorrow',r'\b(ngay mai|sang mai|chieu mai|toi mai|trua mai)\b'),
                    ('today',r'\bhom nay\b'),('day_after_tomorrow',r'\bngay kia\b')] if re.search(pattern,request)]
        if len(relative)==1 and not re.search(r'\b(thu|tuan)\b|\d{4}-\d\d-\d\d',request) and self.anchors:
            for field in ('start','time','date'):
                value=args.get(field)
                if tool.get('kind')=='write' and isinstance(value,str) and re.match(r'^\d{4}-\d\d-\d\d',value):
                    expected=self.anchors[relative[0]]
                    if value[:10]!=expected:issues.append({'code':'relative_date_mismatch','field':field,'expected_date':expected})
                    else:sources[field]='context_time_anchor'
        signature=_signature({'name':name,'arguments':args})
        seen=self.seen.get(signature)
        if seen and seen['version']==self.versions.get(name.split('.')[0],0):
            count=seen['count'];repeat=re.search(r'\b(\d+|hai|ba|bon) lan\b',request)
            allowance=1
            if repeat and not re.search(r'\b(khong|dung|tranh)\b',request[:repeat.start()]):
                allowance=int(repeat[1]) if repeat[1].isdigit() else {'hai':2,'ba':3,'bon':4}[repeat[1]]
            failed_without_new_evidence=seen['status']=='failed' and seen.get('read_version')==self._read_stamp(tool)
            if (tool.get('kind')=='write' and (failed_without_new_evidence or (seen['status']!='failed' and count>=allowance))) or (
                    tool.get('kind')=='read' and count>=2):
                issues.append({'code':'no_progress_repeat','field':None,'previous_status':seen['status']})
        return {'ok':not issues,'issues':issues,'sources':sources}

    def blocked(self,action,verdict):
        needs_resolution=any(i['code']!='no_progress_repeat' for i in verdict['issues'])
        if action['type']=='call' and action['name']!='sys.ask_missing_info' and needs_resolution:
            self.pending={'tool':action['name'],'fields':sorted({i['field'] for i in verdict['issues'] if i.get('field')})}

    def validation_failed(self,action):
        """Remember a failed proposed step, without pretending it executed."""
        name=action['name']
        if name not in self.tools or name=='sys.ask_missing_info':return
        spec=self.tools[name]['parameters'];args=action['arguments'];empty=(None,'',[],{})
        missing=[k for k in spec.get('required',[]) if args.get(k) in empty]
        groups=spec.get('x-one-of',[])
        if groups and not any(all(args.get(k) not in empty for k in g) for g in groups):
            missing+=sorted({k for g in groups for k in g if args.get(k) in empty})
        self.pending={'tool':name,'fields':sorted(set(missing))}

    def observe(self,action,observation):
        name=action['name'];tool=self.tools[name];kind=tool.get('kind');app=name.split('.')[0]
        status='failed' if observation.get('ok') is not True or observation.get('done') is False else (
            'observed' if kind=='read' else 'executed')
        if kind=='write' and status!='failed':self.versions[app]=self.versions.get(app,0)+1
        sig=_signature({'name':name,'arguments':action['arguments']});previous=self.seen.get(sig,{})
        if kind=='read' and status!='failed':
            result_signature=_signature(observation)
            if self.read_results.get(sig)!=result_signature:self.read_versions[app]=self.read_versions.get(app,0)+1
            self.read_results[sig]=result_signature
        self.seen[sig]={'count':previous.get('count',0)+1,'status':status,'version':self.versions.get(app,0),
                        'read_version':self._read_stamp(tool)}
        self.steps.append({'tool':name,'kind':kind,'status':status})
        self._ingest(action,observation,'current',current=True)
        if kind=='write' and status=='failed':self.pending={'tool':name,'fields':[]}
        elif self.pending and (name==self.pending['tool'] or name=='sys.ask_missing_info') and status!='failed':self.pending=None

    def summary(self):
        return {'time_anchors':self.anchors,'pending':deepcopy(self.pending),'execution_progress':deepcopy(self.steps),
                'known_references':{domain:{'count':len(items),'example_ids':list(items)[:8],
                    'lookup_tools':self.lookup_tools(domain),'ambiguous_latest_lookup':self._ambiguous(domain)}
                    for domain,items in sorted(self.evidence.items()) if items or domain in self.latest},
                'limits':'Evidence proves origin, not intent correctness. No hidden app state is available.'}
