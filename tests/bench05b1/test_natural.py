import asyncio
import json
import unittest
from inference_kit.adapters.protocol import Adapter
from inference_kit.adapters.http import request


class Content:
    def __init__(self,frames):self.frames=frames
    async def iter_any(self):
        for frame in self.frames:yield ('data: '+frame+'\n\n').encode('utf-8')


class Response:
    content_type='text/event-stream'
    def __init__(self,frames,status=200,timeout=False):self.content=Content(frames);self.status=status;self.timeout=timeout
    async def __aenter__(self):
        if self.timeout:raise asyncio.TimeoutError()
        return self
    async def __aexit__(self,*args):pass
    async def text(self):return 'fixture failure'


class Session:
    def __init__(self,frames,status=200,timeout=False):self.frames=frames;self.status=status;self.timeout=timeout;self.bodies=[]
    def post(self,url,json):self.bodies.append(json);return Response(self.frames,self.status,self.timeout)


def frames(finish='stop',empty=False,done=True):
    n=0 if empty else 2
    parts=[json.dumps({'choices':[{'index':0,'text':'' if empty else '{"x":1}', 'token_ids':[] if empty else [9,10],'prompt_token_ids':[1,2,3],'finish_reason':None}]}),
           json.dumps({'choices':[{'index':0,'text':'','finish_reason':finish}]}),
           json.dumps({'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':n,'total_tokens':3+n}})]
    return parts+(['[DONE]'] if done else [])


class NaturalTests(unittest.IsolatedAsyncioTestCase):
    async def call(self,session,profile='natural_generation_v1'):
        return await request(session,'http://127.0.0.1',Adapter('vllm'),{'id':'request-1','prompt_hash':'hash','token_ids':[1,2,3],'input_length':3},asyncio.Semaphore(1),'serving-lab',192,profile)
    async def test_natural_stop_accepts_short_actual_count(self):
        session=Session(frames());r=await self.call(session);self.assertTrue(r['transport_complete']);self.assertEqual(r['termination'],'NATURAL_STOP');self.assertEqual(r['usage']['completion_tokens'],2)
        self.assertFalse(session.bodies[0]['ignore_eos']);self.assertNotIn('stop',session.bodies[0]);self.assertNotIn('min_tokens',session.bodies[0]);self.assertNotIn('guided_json',session.bodies[0])
    async def test_length_cap_is_not_transport_failure(self):
        r=await self.call(Session(frames('length')));self.assertTrue(r['transport_complete']);self.assertEqual(r['termination'],'LENGTH_CAP')
    async def test_empty_complete_output_is_distinct(self):
        r=await self.call(Session(frames(empty=True)));self.assertTrue(r['transport_complete']);self.assertEqual(r['termination'],'EMPTY_OUTPUT');self.assertIsNone(r['tpot_stream_estimate_seconds']);self.assertIsNone(r['ttft_http'])
    async def test_truncated_stream_retains_raw_but_not_complete(self):
        r=await self.call(Session(frames(done=False)));self.assertFalse(r['transport_complete']);self.assertEqual(r['termination'],'TRUNCATED_OR_EOF');self.assertEqual(r['output_text'],'{"x":1}')
    async def test_http_error_and_timeout_distinct(self):
        self.assertEqual((await self.call(Session([],status=503)))['termination'],'HTTP_ERROR')
        self.assertEqual((await self.call(Session([],timeout=True)))['termination'],'TIMEOUT')
    async def test_fixed_length_contract_still_rejects_natural_stop(self):
        r=await self.call(Session(frames()),'token_ids');self.assertFalse(r['transport_complete']);self.assertEqual(r['status'],'ERROR')
    async def test_http_and_slot_timestamps_not_reinterpreted(self):
        r=await self.call(Session(frames()));self.assertEqual(r['request_latency_http'],r['completed_monotonic']-r['http_submit_monotonic']);self.assertEqual(r['request_latency_task'],r['completed_monotonic']-r['queued_monotonic']);self.assertEqual(r['ttft_seconds'],r['first_content_monotonic']-r['semaphore_acquired_monotonic'])
