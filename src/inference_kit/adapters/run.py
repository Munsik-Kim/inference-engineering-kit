"""Explicit opt-in single-server execution using the extracted client/lifecycle.

This maintained path is CPU contract verified; GPU was NOT rerun for release01.
"""
import asyncio
import json
from pathlib import Path
import time
from ..core.config import make_plan
from ..core.io import read,write,require
from ..core.state import new_output,atomic,Budget
from . import owned,http
from .protocol import Adapter
from .nvml import gpu_query,gpu_monitor,proc_identity
from .lifecycle_evidence import observed_shutdown


async def guarded(operation,abort):
    task=asyncio.create_task(operation);signal=asyncio.create_task(abort.wait())
    done,_=await asyncio.wait((task,signal),return_when=asyncio.FIRST_COMPLETED)
    if signal in done and abort.is_set():
        task.cancel();await asyncio.gather(task,return_exceptions=True);raise ValueError('Foreign/unknown GPU process observed; submissions stopped')
    signal.cancel();await asyncio.gather(signal,return_exceptions=True);return await task


async def managed(spec_file,local_file,output):
    try:import aiohttp
    except ImportError as exc:raise RuntimeError('Install the optional HTTP extra in the client environment; no GPU packages are installed by this toolkit') from exc
    plan=make_plan(spec_file,local_file);require(plan['spec']['backend']=='vllm','run requires explicit vLLM backend')
    from .vllm import runtime_snapshot,runtime_gate
    snapshot=runtime_snapshot(plan['paths']['server_python'])
    require(runtime_gate(snapshot) is None,'Installed serving runtime differs from checked version tuple')
    local=read(Path(local_file));s=plan['spec'];profile=s['profile'];launch=plan['launch']
    observed=gpu_query(local['gpu']);apps=gpu_query(local['gpu'],True)
    require(observed['returncode']==apps['returncode']==0 and not apps.get('values'),'BLOCKED_SHARED_GPU_OR_ACCESS')
    require(observed['values'].split(',')[1].strip()==local['gpu_uuid'],'GPU UUID differs from explicit request')
    out=new_output(output);launch['environment']['VLLM_CACHE_ROOT']=str(out/'runtime-cache')
    await owned.ensure_port_free(launch['url'])
    write(out/'plan.json',plan);write(out/'runtime.json',snapshot);write(out/'gpu-preflight.json',{'device':observed,'applications':apps})
    rows=read(Path(plan['paths']['workload']))
    if isinstance(rows,dict):rows=rows['rows']
    require(rows and len(rows)>=profile['concurrency'],'Insufficient real warmup overlap inputs')
    require(all(len(r['token_ids'])==r['input_length'] and r['input_length']+profile['output_tokens']<=profile['context'] for r in rows),'Workload token/context mismatch')
    cap=s['budget']['wall_seconds'];reserve=s['budget']['cleanup_reserve_seconds']
    atomic(out/'budget.json',{'schema_version':1,'cap_seconds':cap,'used_seconds':0,'running':None,'jobs':[]})
    budget=Budget(out/'budget.json');budget.begin('server-attempt1',cap)
    stop,abort=asyncio.Event(),asyncio.Event();owner={};process=log=None;started=time.monotonic();result={};readiness=[]
    monitor_spec={'runtime':{'gpu':local['gpu']},'instrumentation':{'ownership_poll_seconds':2,'memory_poll_seconds':1}}
    monitor=asyncio.create_task(gpu_monitor(stop,abort,out,monitor_spec,owner))
    class TokenTextAdapter(Adapter):
        def body(self,row,model,output_tokens,wire_profile):return super().body(row,model,output_tokens,'token_ids')
    adapter=TokenTextAdapter('vllm')
    async def exercise():
        await guarded(owned.ready(process,launch['url'],timeout=min(240,cap-reserve),observations=readiness),abort)
        for phase,chosen in [('warmup',rows[:profile['concurrency']]),('timed',rows)]:
            block=await guarded(http.block(launch['url'],adapter,chosen,profile['concurrency'],output_tokens=profile['output_tokens'],
                               profile='text',timeout=min(120,cap-reserve),abort_event=abort),abort)
            write(out/(phase+'.json'),block);require(block['summary']['status']=='COMPLETE','Request failure; raw preserved')
            result[phase]=block['summary']
    try:
        process,log=owned.start(launch,out);owner.update(parent_pid=process.pid,parent_proc_identity=proc_identity(process.pid))
        write(out/'owned-server.json',owner)
        await asyncio.wait_for(exercise(),max(.1,cap-reserve))
        write(out/'result.json',{'kind':'live_measurement','request_status':'COMPLETE','profile':s['profile'],'metrics':result,
                                 'quality_status':'NOT_EVALUATED','kernel_level':'NOT_VERIFIED','customer_validation':'NOT_RUN'})
    except BaseException as exc:
        write(out/'failure.json',{'status':'FAILED_OR_BLOCKED','error':type(exc).__name__+': '+str(exc)});raise
    finally:
        try:
            if process is not None:
                exit_record=await asyncio.to_thread(owned.stop,process,log,grace=profile['cleanup_grace'])
                lifecycle=observed_shutdown(exit_record,(out/'server.log').read_text(),profile['shutdown_timeout'])
                await owned.ensure_port_free(launch['url']);lifecycle['owned_port_cleanup']='FREE'
                write(out/'lifecycle.json',lifecycle)
            write(out/'readiness-http.json',readiness)
        finally:
            stop.set();await monitor
            after=gpu_query(local['gpu'],True);write(out/'gpu-after.json',after)
            budget.end(time.monotonic()-started,returncode=process.returncode if process else None)
        require(after['returncode']==0 and not after.get('values'),'Unresolved owned/shared GPU resource; no retry')
    return {'status':'REQUESTS_COMPLETE_QUALITY_NOT_EVALUATED','output':str(out),'new_gpu_launches':1}


def run(spec,local,output):return asyncio.run(managed(spec,local,output))
