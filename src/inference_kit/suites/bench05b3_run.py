"""B3 adds a sequential typed likelihood driver to the existing owned server."""
import asyncio
import json
from pathlib import Path
import sys
import time
from ..core.io import read,write,require,digest
from ..core.state import new_output,atomic
from ..tasks.registry import get_task
from ..tasks.arc_challenge import decode_record,body
from ..adapters.task_executor import execute
from ..adapters.run import guarded
from .bench05b1_run import serving_job,seal
from .bench05a_run import launch_job
from .bench05b3_plan import check_frozen

async def likelihood_driver(plan,arm,directory,url,process,abort,phase):
    import aiohttp
    adapter=get_task(plan['task_id'],plan['task_version']);all_records=[]
    execution=plan['plan_identity'][:16]+'|attempt1|'+arm['arm_id']
    write(directory/'collector-contract.json',{'request_kind':'choice_loglikelihood','stream':False,'audit_generation_cap':1,'audit_excluded_from_score':True,'concurrency':1,'TTFT':'NOT_COLLECTED_NON_STREAM'})
    with (directory/'http-responses.jsonl').open('x',encoding='utf-8') as raw,(directory/'choice-scores.jsonl').open('x',encoding='utf-8') as choices,(directory/'document-scores.jsonl').open('x',encoding='utf-8') as documents,(directory/'request-journal.jsonl').open('x',encoding='utf-8') as journal:
        def event_sink(event):
            journal.write(json.dumps(event|{'arm_id':arm['arm_id'],'artifact_identity':arm['artifact']['artifact_identity']},ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
        async with aiohttp.ClientSession() as session:
            for sample in plan['data']['rows']:
                phase(sample['usage_partition']);scored=[]
                for req in adapter.requests(sample,execution,plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity']):
                    require(process.poll() is None and not abort.is_set(),'Server/ownership unsafe')
                    require(time.monotonic()<plan['clock']['deadline_monotonic']-plan['suite']['budget']['cleanup_reserve_seconds'],'Session cleanup reserve')
                    record=await guarded(execute(req,session=session,url=url,timeout=30,event_sink=event_sink),abort)
                    record.update(arm_id=arm['arm_id'],sample_id=sample['sample_id'],usage_partition=sample['usage_partition'],choice_index=req.payload['choice_index'],
                      artifact_identity=arm['artifact']['artifact_identity'],execution_id=execution,execution_mode='DEPLOYED_PATH_OBSERVATION')
                    raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();all_records.append(record)
                    value=decode_record(req,record);choices.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');choices.flush();scored.append(value)
                    require(value['status']=='COMPLETE','Likelihood transport/echo failed; preserve partial, no retry')
                value=adapter.score(sample,scored);documents.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');documents.flush()
    return all_records

async def managed(plan_file,output):
    plan=read(Path(plan_file));check_frozen(plan)
    root=new_output(output);write(root/'PLAN_FROZEN.json',plan)
    import inference_kit
    write(root/'controller-environment.json',{'python':sys.executable,'package_import_origin':inference_kit.__file__,'package_version':inference_kit.__version__,'source_identity':plan['source_identity']})
    ledger=root/'budget-continuation.json';previous=plan['campaign']['used_seconds']
    atomic(ledger,{'schema_version':1,'task_id':'BENCH05-B3','campaign_cap_seconds':7200,'cap_seconds':previous+plan['gpu_allowance_seconds'],'used_seconds':previous,
      'task_origin_used_seconds':plan['task_origin_used_seconds'],'running':None,'jobs':[],
      'inherited_from':{'path':plan['campaign']['path'],'sha256':plan['campaign']['sha256'],'used_seconds':previous}})
    progress={'status':'RUNNING','jobs':{},'automatic_retries':0};atomic(root/'progress.json',progress)
    try:
        for phase in ('serve','main'):
            if phase=='main' and len(plan['samples']['rows'])!=4:
                progress['attention_status']='NOT_RUN_INSUFFICIENT_FROZEN_PROBE_DATA';break
            for name in plan['arm_order']:
                if time.monotonic()>=plan['clock']['no_new_gpu_after']:
                    progress['status']='BUDGET_SKIPPED';return progress
                arm=next(a for a in plan['arms'] if a['arm_id']==name)
                okay=await serving_job(plan,root,ledger,progress,arm,request_driver=likelihood_driver) if phase=='serve' else await launch_job(plan,root,ledger,progress,'main',arm)
                atomic(root/'progress.json',progress)
                if not okay:
                    progress['status']='PARTIAL_OR_BLOCKED';return progress
        progress['status']='COMPLETE';return progress
    except BaseException as exc:
        progress['status']='FAILED_OR_PARTIAL';write(root/'controller-failure.json',{'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        atomic(root/'progress.json',progress);seal(root,scope='bench05b3_arc_raw')

def run(plan,output):return asyncio.run(managed(plan,output))
