"""Non-stream likelihood collector. No task scoring, SDK or model imports."""
import asyncio
import time
from ..core.io import digest,require


def request_body(request):
    require(request.request_kind=='choice_loglikelihood','Likelihood collector rejects generation request')
    request.validate()
    return {'model':'serving-lab','prompt':request.payload['input_ids'],'max_tokens':1,'temperature':0,'top_p':1,'n':1,
      'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,'ignore_eos':True,'echo':True,'stream':False,
      'logprobs':1,'prompt_logprobs':1,'add_special_tokens':False,'return_tokens_as_token_ids':True,'return_token_ids':True}

async def collect(session,url,request,timeout=30,event_sink=None):
    payload=request_body(request);queued=time.monotonic();acquired=time.monotonic();submit=time.monotonic()
    response=None;raw_text=None;error=None;http_status=None
    journal={'request_id':request.request_id,'sample_id':request.sample_id,'choice_index':request.payload['choice_index'],'identity':request.identity,'request_body':payload,'request_body_identity':digest(payload),'queued':queued,'semaphore_acquired':acquired,'http_submit':submit}
    if event_sink:event_sink(journal|{'event':'http_submit'})
    try:
        async with session.post(url+'/v1/completions',json=payload,timeout=timeout) as result:
            http_status=result.status;raw_text=await result.text(encoding='utf-8')
            if result.status!=200:raise ValueError('HTTP status '+str(result.status))
            from ..core.io import loads
            response=loads(raw_text)
    except asyncio.CancelledError:
        error={'type':'CancelledError','message':'Owned request cancelled; submission journal retained'}
        raise
    except (Exception,asyncio.TimeoutError) as exc:error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        completed=time.monotonic()
        if event_sink:event_sink(journal|{'event':'response_end','response_complete':completed,'transport_complete':error is None,'error':error})
    return {'request_id':request.request_id,'request_kind':request.request_kind,'identity':request.identity,'request_body':payload,'request_body_identity':digest(payload),
      'queued':queued,'semaphore_acquired':acquired,'http_submit':submit,'response_complete':completed,'http_status':http_status,
      'response':response,'raw_response_text':raw_text,'transport_complete':error is None,'error':error,'evidence_kind':'live_native_likelihood'}
