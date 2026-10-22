"""B4 typed natural collector, reusing managed serving and diagnostic lifecycle."""
import asyncio
import json
from pathlib import Path
import sys
import time
from ..core.io import read, write, require, digest
from ..core.state import new_output, atomic
from ..tasks.registry import get_task
from ..adapters import http
from ..adapters.protocol import Adapter
from ..adapters.task_executor import execute
from ..adapters.run import guarded
from .bench05a_run import launch_job
from .bench05b1_run import serving_job, seal
from .bench05b4_plan import check_frozen


def execution_id(plan, arm):
    return plan['plan_identity'][:16]+'|attempt1|'+arm['arm_id']


async def generation_driver(plan,arm,directory,url,process,abort,phase):
    adapter=get_task(plan['task_id'],plan['task_version']);by={s['sample_id']:s for s in plan['data']['rows']}
    prepared={s['sample_id']:s for s in plan['samples']['generation_rows']};records=[]
    with (directory/'request-journal.jsonl').open('x',encoding='utf-8') as journal, (directory/'task-generations.jsonl').open('x',encoding='utf-8') as raw, (directory/'task-scores.jsonl').open('x',encoding='utf-8') as scored:
        for sid in plan['request_order']:
            require(process.poll() is None and not abort.is_set(),'B4 owned server/GPU ownership unsafe')
            require(time.monotonic()<plan['clock']['deadline_monotonic']-plan['suite']['budget']['cleanup_reserve_seconds'],'B4 session cleanup reserve')
            sample=by[sid];row=prepared[sid];phase(sample['split'])
            request=adapter.requests(sample|{'generation_input_ids':row['generation_input_ids'],'tokenizer_identity':plan['samples']['tokenizer_identity']},
                execution_id(plan,arm),plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])[0]
            def sink(event):
                event=event|{'identity':request.identity,'arm_id':arm['arm_id'],'sample_id':sid,'artifact_identity':arm['artifact']['artifact_identity'],
                             'request_kind':request.request_kind,'task_id':request.task_id,'task_version':request.task_version,'prompt_hash':row['prompt_hash']}
                journal.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
            async def natural(req):
                payload={'id':req.request_id,'prompt_hash':row['prompt_hash'],'text_hash':row['document_hash'],'token_ids':req.payload['token_ids'],'input_length':len(req.payload['token_ids'])}
                block=await http.block(url,Adapter('vllm'),[payload],1,output_tokens=64,profile='natural_generation_v1',timeout=30,abort_event=abort,event_sink=sink)
                return block['records'][0]
            wire=await guarded(execute(request,natural_collector=natural),abort)
            record=wire|{'request_identity':http.sha(wire['request_body']),'identity':request.identity,'request_kind':request.request_kind,
                'task_id':request.task_id,'task_version':request.task_version,'sample_id':sid,'sample_identity':sample['sample_identity'],'split':sample['split'],
                'arm_id':arm['arm_id'],'artifact_identity':arm['artifact']['artifact_identity'],'tokenizer_identity':plan['samples']['tokenizer_identity'],
                'task_prompt_hash':row['task_prompt_hash'],'prompt_identity':request.identity['prompt_identity'],'scorer_version':adapter.scorer_version,
                'reference_used_for_generation':False,'reference_source':'task_author_gold_answer_evidence_v1','evidence_kind':'live_task_generation',
                'execution_mode':'DEPLOYED_PATH_OBSERVATION','execution_id':execution_id(plan,arm)}
            # Existing serving lifecycle reads generations.jsonl only in the default driver.
            raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();records.append(record)
            value=adapter.score(sample,record);scored.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');scored.flush()
            require(record['transport_complete'],'B4 transport/CPU contract failure; no further submissions, no retry')
    return records


async def managed(plan_file,output):
    plan=read(Path(plan_file));check_frozen(plan)
    root=new_output(output);write(root/'PLAN_FROZEN.json',plan)
    import inference_kit
    write(root/'controller-environment.json',{'python':sys.executable,'package_version':inference_kit.__version__,'package_import_origin':inference_kit.__file__,'source_identity':plan['source_identity']})
    ledger=root/'budget-continuation.json';previous=plan['campaign']['used_seconds']
    atomic(ledger,{'schema_version':1,'task_id':'BENCH05-B4','campaign_cap_seconds':7200,'cap_seconds':previous+plan['gpu_allowance_seconds'],'used_seconds':previous,
      'task_origin_used_seconds':plan['task_origin_used_seconds'],'running':None,'jobs':[],
      'inherited_from':{'path':plan['campaign']['path'],'sha256':plan['campaign']['sha256'],'used_seconds':previous}})
    progress={'status':'RUNNING','jobs':{},'automatic_retries':0};atomic(root/'progress.json',progress)
    try:
        for phase in ('serve','main'):
            for name in plan['arm_order']:
                if time.monotonic()>=plan['clock']['no_new_gpu_after']:
                    progress['status']='BUDGET_SKIPPED';return progress
                arm=next(a for a in plan['arms'] if a['arm_id']==name)
                okay=await serving_job(plan,root,ledger,progress,arm,request_driver=generation_driver,collector_contract=plan['collector_contract']) if phase=='serve' else await launch_job(plan,root,ledger,progress,'main',arm)
                atomic(root/'progress.json',progress)
                if not okay:progress['status']='PARTIAL_OR_BLOCKED';return progress
        progress['status']='COMPLETE';return progress
    except BaseException as exc:
        progress['status']='FAILED_OR_PARTIAL';write(root/'controller-failure.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:
        atomic(root/'progress.json',progress);seal(root,scope='bench05b4_context_raw')


def run(plan,output):return asyncio.run(managed(plan,output))
