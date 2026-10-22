"""Frozen ARC likelihood semantics from core.quality; no model dependencies."""
import hashlib
import math
from ..core.io import require,digest
from ..core.quality import score_document,likelihood_from_response
from .base import TaskRequest

TASK='arc_challenge';VERSION='1.0';SCORER='arc_choice_sum_character_norm_v1'
SELECTION='BENCH05-B3-v1|'

def validate_sample(row):
    n=len(row['choices'])
    require(n>1 and all(isinstance(c,str) and len(c)>0 for c in row['choices']),'Empty ARC choice')
    require(len(row['choice_labels'])==n and len(set(row['choice_labels']))==n,'Duplicate/missing choice labels')
    require(type(row['gold']) is int and 0<=row['gold']<n,'Invalid ARC gold index')
    require(row['sample_id']==row['document']['id'] and row['choice_labels'][row['gold']]==row['document']['answerKey'],'External answerKey/label/index mismatch')
    require(row['document'].get('question')==row['question'] and row['document'].get('choices',{}).get('text')==row['choices'] and row['document']['choices'].get('label')==row['choice_labels'],'Original document/choice order mismatch')
    reqs=row['requests'];require(len(reqs)==n and [r['choice_index'] for r in reqs]==list(range(n)),'Missing/duplicate/unknown choice')
    for req in reqs:
        TaskRequest('validation',TASK,VERSION,'choice_loglikelihood',row['sample_id'],req,{'task_id':TASK,'task_version':VERSION}).validate()
        require(req['context']==row['prompt'] and req['continuation']==' '+row['choices'][req['choice_index']],'Original ARC delimiter/prompt changed')
    return row

def select_pool(pool):
    require(isinstance(pool,list) and len(pool)>=68,'Insufficient prepared ARC pool')
    require(len({r['sample_id'] for r in pool})==len(pool),'Duplicate ARC document')
    for row in pool:validate_sample(row)
    ordered=sorted(pool,key=lambda r:hashlib.sha256((SELECTION+r['sample_id']).encode('utf-8')).hexdigest())
    rows=[dict(r,usage_partition='SMOKE' if i<4 else 'REGRESSION',usage_scope='REGRESSION_REUSE') for i,r in enumerate(ordered[:68])]
    return {'task_id':TASK,'task_version':VERSION,'scorer_version':SCORER,'pool_count':len(pool),'selection_rule':'sha256('+SELECTION+'original_sample_id) ascending; first4 SMOKE, next64 REGRESSION',
            'rows':rows,'rows_identity':digest(rows),'usage_scope':'REGRESSION_REUSE','original_splits':sorted({r['split'] for r in pool})}

def body(request):
    from ..adapters.likelihood import request_body
    return request_body(request)

def decode_record(request,record):
    require(request.request_kind=='choice_loglikelihood' and record.get('request_kind')=='choice_loglikelihood','Wrong collector/request kind')
    require(record['request_id']==request.request_id and record['request_body']==body(request),'Raw request identity/body mismatch')
    require(record['identity']==request.identity and record['request_body_identity']==digest(record['request_body']),'Raw task/tokenizer/prompt identity mismatch')
    if not record['transport_complete']:return {'status':'INCOMPLETE','sample_id':request.sample_id,'identity':request.identity,'choice_index':request.payload['choice_index'],'tokens':[],'request_id':request.request_id,'error':record.get('error')}
    from ..core.io import loads
    require(record.get('http_status')==200 and loads(record['raw_response_text'])==record['response'],'Raw HTTP text/response/status mismatch')
    require(record['queued']<=record['semaphore_acquired']<=record['http_submit']<=record['response_complete'],'Timing boundaries out of order')
    try:
        total,values=likelihood_from_response(record['response'],request.payload)
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'INCOMPLETE','choice_index':request.payload['choice_index'],'sample_id':request.sample_id,'identity':request.identity,'tokens':[],'request_id':request.request_id,'error':{'type':type(exc).__name__,'message':str(exc)}}
    start=len(request.payload['context_ids']);ids=request.payload['input_ids']
    tokens=[{'token_id':ids[t],'token_position':t,'logit_position':t-1,'prefix_hash':digest({'token_ids':ids[:t],'token_id_space':request.identity['token_id_space']}),'logprob':float(lp),'nll_nat':-float(lp),
             'token_id_space':request.identity['token_id_space'],'request_id':request.request_id,'sample_id':request.sample_id,'choice_index':request.payload['choice_index']} for t,lp in zip(range(start,len(ids)),values)]
    return {'status':'COMPLETE','sample_id':request.sample_id,'identity':request.identity,'request_id':request.request_id,'choice_index':request.payload['choice_index'],'choice_loglikelihood':total,
      'continuation_token_count':len(tokens),'tokens':tokens,'audit_generated_tokens':1,'request_latency_http':record['response_complete']-record['http_submit'],
      'request_latency_task':record['response_complete']-record['queued'],'queued':record['queued'],'response_complete':record['response_complete']}

class ArcChallengeAdapter:
    task_id=TASK;task_version=VERSION;request_kind='choice_loglikelihood';scorer_version=SCORER
    def requests(self,sample,execution,tokenizer_identity,token_id_space='unspecified',runtime_identity='unspecified'):
        validate_sample(sample)
        identity={'task_id':TASK,'task_version':VERSION,'scorer_version':SCORER,'sample_id':sample['sample_id'],'split':sample['split'],
          'source_identity':sample['source_text_hash'],'prompt_identity':digest(sample['prompt']),'tokenizer_identity':tokenizer_identity,
          'token_id_space':token_id_space,'runtime_identity':runtime_identity}
        return [TaskRequest(execution+'|'+TASK+'|'+sample['sample_id']+'|choice'+str(r['choice_index']),TASK,VERSION,self.request_kind,sample['sample_id'],r,identity).validate() for r in sample['requests']]
    def reference(self,sample):
        validate_sample(sample)
        return {'reference_source':'dataset_answer_key_choice','choice_index':sample['gold'],'label':sample['choice_labels'][sample['gold']],
                'text':sample['requests'][sample['gold']]['continuation'],'answerKey':sample['document']['answerKey']}
    def score(self,sample,records):
        validate_sample(sample); n=len(sample['choices'])
        require(len({r['choice_index'] for r in records})==len(records) and all(type(r['choice_index']) is int and 0<=r['choice_index']<n for r in records),'Duplicate/unknown ARC score choice')
        require(all(r.get('sample_id')==sample['sample_id'] for r in records),'Score/sample identity mismatch')
        complete=len(records)==n and all(r['status']=='COMPLETE' for r in records)
        if not complete:return {'sample_id':sample['sample_id'],'status':'INCOMPLETE' if records else 'NOT_RUN','planned_choices':n,'complete_choices':sum(r['status']=='COMPLETE' for r in records),'acc':None,'acc_norm':None}
        by={r['choice_index']:r for r in records}; result=score_document(sample,[by[i]['choice_loglikelihood'] for i in range(n)])
        g=sample['gold'];raw=result['choice_loglikelihood'];norm=result['choice_normalized_scores']
        result.update(status='COMPLETE',choice_labels=sample['choice_labels'],usage_scope='REGRESSION_REUSE',
          gold_margin_raw=raw[g]-max(v for i,v in enumerate(raw) if i!=g),gold_margin_norm=norm[g]-max(v for i,v in enumerate(norm) if i!=g),
          choice_token_counts=[by[i]['continuation_token_count'] for i in range(n)],gold_mean_nll_nat=-raw[g]/by[g]['continuation_token_count'],
          gold_token_count=by[g]['continuation_token_count'],
          document_choice_http_sum_seconds=sum(r['request_latency_http'] for r in by.values()) if all('request_latency_http' in r for r in by.values()) else None,
          document_serial_elapsed_seconds=max(r['response_complete'] for r in by.values())-min(r['queued'] for r in by.values()) if all('response_complete' in r and 'queued' in r for r in by.values()) else None)
        return result
    def metric_specs(self):return [{'metric_id':'acc_norm','version':SCORER,'formula':'argmax(sum continuation logprob / original choice character length) == external gold; first-index tie','unit':'document fraction'},
      {'metric_id':'acc','version':SCORER,'formula':'argmax(sum continuation logprob) == external gold; audit generation excluded','unit':'document fraction'},
      {'metric_id':'native_target_nll','version':'b3-v1','formula':'-logprob at logit t-1 predicting continuation token t; ln','unit':'nat','aggregation':'document macro and token weighted separately'},
      {'metric_id':'http_latency','version':'b3-v1','formula':'response_complete-http_submit','unit':'seconds','TTFT':'NOT_COLLECTED_NON_STREAM'}]
