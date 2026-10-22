import copy
import json
from pathlib import Path
from inference_kit.core.io import read,write,digest,sha
from inference_kit.tasks.arc_challenge import select_pool,ArcChallengeAdapter,body,decode_record
from inference_kit.probes.arc_tokens import POLICY
from inference_kit.suites.bench05b1_run import seal
from bench05b1.helpers import local_fixture as b1_local

def sample(n=4,sid='문항-1'):
    choices=['x','long answer','third','fourth','fifth'][:n];prompt='Question: 기록?\nAnswer:\n'
    requests=[{'choice_index':i,'context':prompt,'continuation':' '+c,'context_ids':[1,2],'continuation_ids':[3,4+i],'input_ids':[1,2,3,4+i]} for i,c in enumerate(choices)]
    return {'sample_id':sid,'original_index':1,'split':'test','document':{'id':sid,'answerKey':str(n),'question':'기록?','choices':{'text':choices,'label':[str(i+1) for i in range(n)]}},'question':'기록?',
      'choices':choices,'choice_labels':[str(i+1) for i in range(n)],'gold':n-1,'prompt':prompt,'source_text_hash':digest(sid),'request_identity':digest(requests),'requests':requests}

def raw(request,values=None):
    ids=request.payload['input_ids'];values=[None,-.2,-1.,-2.,-999.] if values is None else values
    response={'choices':[{'logprobs':{'token_logprobs':values,'tokens':['token_id:'+str(i) for i in ids]+['token_id:7']}}],
              'usage':{'prompt_tokens':len(ids),'completion_tokens':1}}
    payload=body(request)
    return {'request_kind':'choice_loglikelihood','request_id':request.request_id,'identity':request.identity,'request_body':payload,'request_body_identity':digest(payload),'response':response,
            'raw_response_text':json.dumps(response,ensure_ascii=False),'http_status':200,'transport_complete':True,'queued':1.,'semaphore_acquired':1.1,'http_submit':1.2,'response_complete':1.5,'evidence_kind':'synthetic_contract'}

def local_fixture(root):
    suite,pair,local=b1_local(root);cfg=read(root/'prepared.json')['model_config'];data=select_pool([sample(sid='文-'+str(i)) for i in range(68)])
    prepared={'kind':'prepared_arc_regression_v1','preparation_id':POLICY,'data_identity':data['rows_identity'],'model_config':cfg,'rows':[], 'rows_identity':digest([]),
      'native_boundary_checks':[{'sample_id':s['sample_id'],'choice_index':r['choice_index']} for s in data['rows'] for r in s['requests']],
      'tokenizer_files':{},'tokenizer_identity':'synthetic-tokenizer','token_id_space':'synthetic-space','vocabulary_size':256,'probe_selection':{'sample_ids':[]}}
    s=read(Path(__file__).parents[2]/'configs/bench05b3/suite.json');s['budget'].update(minimum_cpu_available_bytes=1,minimum_disk_free_bytes=1)
    for p,value in ((suite,s),(root/'data.json',data),(root/'prepared.json',prepared)):p.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    loc=read(local);loc['arc_data']=loc.pop('authored_data');loc['serving_runtime_identity']='synthetic-runtime';local.write_text(json.dumps(loc),encoding='utf-8');return suite,pair,local

def synthetic_run(root,partial=False):
    from inference_kit.suites.bench05b3_plan import plan
    root=Path(root);root.mkdir();inputs=root.parent/(root.name+'-inputs');inputs.mkdir();p=plan(*local_fixture(inputs),root/'plan');p.pop('plan_identity');p['evidence_kind']='synthetic_contract';p['plan_identity']=digest(p);write(root/'PLAN_FROZEN.json',p)
    jobs={};adapter=ArcChallengeAdapter()
    for arm in p['arms']:
        name=arm['arm_id'];d=root/'attempts'/('serve-'+name+'-attempt1');d.mkdir(parents=True);records=[]
        for sample_ in p['data']['rows'][:1 if partial else 68]:
            for req in adapter.requests(sample_,p['plan_identity'][:16]+'|attempt1|'+name,p['samples']['tokenizer_identity'],p['samples']['token_id_space'],p['serving_runtime_identity']):
                r=raw(req);r.update(sample_id=sample_['sample_id'],choice_index=req.payload['choice_index'],arm_id=name,artifact_identity=arm['artifact']['artifact_identity']);records.append(r)
        (d/'http-responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8');jobs['serve-'+name]={'status':'SYNTHETIC_CONTRACT_ONLY','directory':str(d.relative_to(root))}
    write(root/'progress.json',{'status':'SYNTHETIC_CONTRACT_ONLY','jobs':jobs});seal(root,scope='bench05b3_arc_raw');return root
