"""B4 synthetic HTTP and numeric blobs exercise data-only replay contracts."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import inference_kit
from inference_kit import bench_b4_replay
from inference_kit.adapters.http import sha as wire_hash
from inference_kit.adapters.protocol import Adapter
from inference_kit.core.io import read,write,digest
from inference_kit.probes.blobs import save
from inference_kit.probes.contracts import Execution,Observation
from inference_kit.suites.bench05b1_run import seal
from inference_kit.suites.bench05b4_plan import plan as make_plan
from inference_kit.suites.bench05b4_replay import requests_for
from inference_kit.tasks import context_evidence as task
from .test_plan import local_fixture


def write_lines(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8')


def reseal(root):
    (root/'RAW_MANIFEST.json').unlink()
    seal(root,scope='bench05b4_context_raw')


def synthetic_run(root,*,probes=False,partial=False):
    root.mkdir();inputs=root.parent/(root.name+'-inputs');inputs.mkdir()
    frozen=make_plan(*local_fixture(inputs),root/'plan')
    frozen['evidence_kind']='synthetic_contract'
    for arm in frozen['arms']:arm['execution_mode']='REIMPLEMENTED_OR_SYNTHETIC'
    frozen.pop('plan_identity');frozen['plan_identity']=digest(frozen)
    write(root/'PLAN_FROZEN.json',frozen);jobs={}
    for arm in frozen['arms']:
        name=arm['arm_id'];directory=root/'attempts'/('serve-'+name+'-attempt1');directory.mkdir(parents=True)
        reqs=requests_for(frozen,arm);prepared={r['sample_id']:r for r in frozen['samples']['generation_rows']}
        generations=[];journal=[]
        chosen=frozen['data']['rows'][:1] if partial else frozen['data']['rows']
        for sample in chosen:
            request=reqs[sample['sample_id']];row=prepared[sample['sample_id']]
            body=Adapter('vllm').body({'token_ids':request.payload['token_ids']},'serving-lab',64,'natural_generation_v1')
            begin={'event':'http_submit','request_id':request.request_id,'identity':request.identity,'request_kind':'generate_until',
                   'arm_id':name,'sample_id':sample['sample_id'],'artifact_identity':arm['artifact']['artifact_identity'],
                   'prompt_hash':row['prompt_hash'],'request_body':body,'queued_monotonic':1.,'semaphore_acquired_monotonic':1.1,'http_submit_monotonic':1.2}
            journal.append(begin)
            if partial:continue
            usage={'prompt_tokens':len(request.payload['token_ids']),'completion_tokens':2,'total_tokens':len(request.payload['token_ids'])+2}
            text=task.canonical_reference(sample)['text']
            frames=[json.dumps({'choices':[{'index':0,'text':text,'token_ids':[11,12],
                                           'prompt_token_ids':request.payload['token_ids'],'finish_reason':'stop'}]},ensure_ascii=False),
                    json.dumps({'choices':[],'usage':usage}),'[DONE]']
            record={k:v for k,v in begin.items() if k!='event'}
            record.update(task_id=task.TASK,task_version=task.VERSION,sample_identity=sample['sample_identity'],split=sample['split'],
                          evidence_kind='synthetic_contract',tokenizer_identity=frozen['samples']['tokenizer_identity'],reference_used_for_generation=False,
                          request_identity=wire_hash(body),raw_sse_frames=frames,output_text=text,usage=usage,transport_complete=True,termination='NATURAL_STOP',
                          first_content_monotonic=1.3,last_content_monotonic=1.4,completed_monotonic=1.5,
                          request_latency_http=1.5-1.2,request_latency_task=.5,ttft_http=1.3-1.2)
            generations.append(record);journal.append(begin|{'event':'response_end','completed_monotonic':1.5,'error':None})
        write_lines(directory/'task-generations.jsonl',generations);write_lines(directory/'request-journal.jsonl',journal)
        jobs['serve-'+name]={'status':'FAILED_OR_PARTIAL' if partial else 'COMPLETE','directory':str(directory.relative_to(root))}
        if probes:
            directory=root/'attempts'/('main-'+name+'-attempt1');directory.mkdir(parents=True)
            execution=Execution('synthetic-run',directory.name,name,frozen['pair']['model_id'],frozen['pair']['revision'],
                arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],frozen['samples']['tokenizer_identity'],frozen['runtime_identity'],
                'REIMPLEMENTED_OR_SYNTHETIC','float32','toy','synthetic_contract','MEASURED','observations.jsonl').validate()
            execution['phase']='main';execution['execution_id']=digest(execution)
            write(directory/'execution.json',execution);observations=[]
            def emit(row,kind,coordinates,values,details):
                coordinate={'sample_id':row['sample_id'],'pass_id':'ON','token_id_space':frozen['samples']['token_id_space'],
                            'tokenizer_identity':frozen['samples']['tokenizer_identity'],'sequence_identity':row['sequence_identity'],**coordinates}
                prefix=coordinate['token_position'] if 'token_position' in coordinate else coordinate['query_position']+1
                coordinate['prefix_hash']=digest({'token_ids':row['input_ids'][:prefix],'token_id_space':frozen['samples']['token_id_space']})
                identifier=digest({'arm':name,'kind':kind,'coordinates':coordinate})[:28]
                observations.append(Observation(identifier,kind,execution['execution_id'],row['sample_id'],'synthetic_contract','REIMPLEMENTED_OR_SYNTHETIC',
                    'MEASURED',coordinate,save(directory,identifier,values,[len(values)]),details).validate())
            for row in frozen['samples']['rows']:
                for target in row['targets']:
                    emit(row,'target_logprob',target,[-math.log(256)],{'raw_argmax_token_id':0,'normalization_dtype':'float32'})
                    if target['token_position'] in row['full_vocab_target_positions']:
                        emit(row,'full_logits',target,[0.]*256,{'full_vocabulary':True})
                for head in row['attention']['heads']:
                    for query in row['attention']['queries']:
                        c={'layer':row['attention']['layer'],'head':head,'kv_head':head,'query_position':query,
                           'key_positions':list(range(len(row['input_ids']))),'allowed_keys':[k<=query for k in range(len(row['input_ids']))]}
                        emit(row,'attention_row',c,[1/(query+1) if k<=query else 0. for k in range(len(row['input_ids']))],{'probability_source':'synthetic_full_key'})
                        emit(row,'attention_output',c,[1.]+[0.]*63,{})
            write_lines(directory/'observations.jsonl',observations);write(directory/'result.json',{'status':'MEASURED'})
            jobs['main-'+name]={'status':'COMPLETE','directory':str(directory.relative_to(root))}
    write(root/'progress.json',{'status':'SYNTHETIC_CONTRACT_ONLY','jobs':jobs});seal(root,scope='bench05b4_context_raw')
    return root


class RetrievalReplayTests(unittest.TestCase):
    def test_full28_pair_cpu_scoring_does_not_promote_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw');summary=bench_b4_replay(raw,root/'replay')
            for arm in ('bf16','rtn'):
                result=summary['arms'][arm]['splits']['EVAL_LOCKED']
                self.assertEqual(result['observed'],24);self.assertEqual(result['metrics']['task_success']['correct'],24)
            self.assertEqual(summary['paired']['complete_documents'],24)
            self.assertEqual(summary['live_status'],'NOT_RUN_OR_SYNTHETIC')
            self.assertNotEqual(summary['stage_status']['TASK_LIVE'],'COMPLETE_PAIR24')
            self.assertEqual(len((root/'replay/sample-linked.jsonl').read_text(encoding='utf-8').splitlines()),4)

    def test_partial_submission_journal_recovers_incomplete_not_not_run(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw',partial=True);summary=bench_b4_replay(raw,root/'replay')
            result=summary['arms']['bf16']['splits']['SMOKE']
            self.assertEqual((result['observed'],result['not_run']),(1,3));self.assertFalse(result['transport_complete'])
            rows=[json.loads(x) for x in (root/'replay/bf16-generations-replayed.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['termination'],'INCOMPLETE_SUBMISSION');self.assertIsNone(rows[0]['request_latency_http'])
            self.assertEqual(summary['paired']['incomplete_documents'],24)

    def test_numeric_replay_identity_and_absent_evidence_mass_is_null(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw',probes=True);summary=bench_b4_replay(raw,root/'replay')
            self.assertEqual(summary['probe']['status'],'SYNTHETIC_CONTRACT_ONLY')
            probe=read(root/'replay/probe-comparisons.json')
            for row in probe['records']:
                if row['kind']=='target_logprob':self.assertEqual(row['delta_nll_nat'],0)
                if row['kind']=='attention_row':self.assertEqual(row['js_nat'],0);self.assertEqual(row['l1'],0)
                if row['kind']=='attention_output':self.assertEqual(row['nmse'],0);self.assertEqual(row['cosine'],1)
            masses=[json.loads(x) for x in (root/'replay/evidence-content-mass.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r['status']=='NOT_APPLICABLE' for r in masses))
            self.assertTrue(all(r['mass'] is None for r in masses if r['status']=='NOT_APPLICABLE'))

    def test_raw_byte_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw');path=next(raw.glob('attempts/serve-*/task-generations.jsonl'))
            path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):bench_b4_replay(raw,root/'replay')

    def test_semantic_raw_token_and_identity_mismatches_rejected(self):
        for change in ('tokenizer','task','body','echo'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                root=Path(t);raw=synthetic_run(root/'raw');path=next(raw.glob('attempts/serve-*/task-generations.jsonl'))
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                if change=='tokenizer':rows[0]['tokenizer_identity']='different'
                if change=='task':rows[0]['task_id']='tech_record_extract'
                if change=='body':rows[0]['request_body']['ignore_eos']=True;rows[0]['request_identity']=wire_hash(rows[0]['request_body'])
                if change=='echo':
                    frame=json.loads(rows[0]['raw_sse_frames'][0]);frame['choices'][0]['prompt_token_ids']=[999];rows[0]['raw_sse_frames'][0]=json.dumps(frame)
                write_lines(path,rows);reseal(raw)
                with self.assertRaises(ValueError):bench_b4_replay(raw,root/'replay')

    def test_blob_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw',probes=True);path=next(raw.glob('attempts/main-*/blobs/*'))
            value=bytearray(path.read_bytes());value[0]^=1;path.write_bytes(value)
            with self.assertRaises(ValueError):bench_b4_replay(raw,root/'replay')

    def test_immutable_output_and_data_only_safe_zip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw');archive=root/'recorded.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for path in raw.rglob('*'):
                    if path.is_file():z.write(path,path.relative_to(raw).as_posix())
            summary=bench_b4_replay(archive,root/'replay');self.assertEqual(summary['paired']['complete_documents'],24)
            with self.assertRaises(ValueError):bench_b4_replay(archive,root/'replay')
            for name in ('../escape','/absolute'):
                unsafe=root/('unsafe-'+str(len(name))+'.zip')
                with zipfile.ZipFile(unsafe,'w') as z:z.writestr(name,'unsafe')
                with self.assertRaises(ValueError):bench_b4_replay(unsafe,root/('bad-'+str(len(name))))
            duplicate=root/'duplicate.zip'
            with zipfile.ZipFile(duplicate,'w') as z:z.writestr('A','1');z.writestr('a','2')
            with self.assertRaises(ValueError):bench_b4_replay(duplicate,root/'bad-duplicate')

    def test_ascii_locale_cli_replay_no_sdk_utf8_files(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);synthetic_run(root/'raw')
            package_root=str(Path(inference_kit.__file__).absolute().parent.parent)
            code='import sys,locale;sys.path.insert(0,'+repr(package_root)+');assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");'
            code+='from inference_kit.cli import main;result=main(["bench","b4","replay","--run","raw","--output","replay"]);assert not any(n.split(".")[0] in ("torch","vllm","transformers","numpy","aiohttp") for n in sys.modules);raise SystemExit(result)'
            child=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',code],cwd=root,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True,encoding='utf-8')
            self.assertEqual(child.returncode,0,child.stderr);self.assertEqual(json.loads(child.stdout)['paired']['complete_documents'],24)
            self.assertIn('문맥',(root/'replay/REPORT_KO.md').read_text(encoding='utf-8'))
            self.assertIn('context_evidence_retrieval',(root/'replay/sample-linked.csv').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
