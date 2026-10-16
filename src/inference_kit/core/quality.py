# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Official ARC acc/acc_norm semantics with deployed-artifact likelihood scores."""
import asyncio
import json
import math
from pathlib import Path
import random
import statistics
import time

from .io import digest, require, write, read, loads


def score_document(row, scores):
    require(len(scores) == len(row['choices']) and all(math.isfinite(x) for x in scores), 'Missing/nonfinite choice score')
    lengths = [len(c) for c in row['choices']]
    require(all(n > 0 for n in lengths), 'Zero choice character length')
    normalized = [score / length for score,length in zip(scores,lengths)]
    pred = max(range(len(scores)), key=scores.__getitem__)
    pred_norm = max(range(len(scores)), key=normalized.__getitem__)
    return {'sample_id': row['sample_id'], 'original_index': row['original_index'], 'split': row['split'],
            'gold': row['gold'], 'choice_loglikelihood': scores, 'normalization_char_lengths': lengths,
            'choice_normalized_scores': normalized, 'prediction_acc': pred, 'prediction_acc_norm': pred_norm,
            'acc': float(pred == row['gold']), 'acc_norm': float(pred_norm == row['gold']),
            'question': row['question'], 'choices': row['choices'], 'prompt': row['prompt'],
            'request_identity': row['request_identity'], 'source_text_hash': row['source_text_hash'], 'status': 'OK'}


def likelihood_from_response(response, request):
    require(response.get('error') is None and len(response.get('choices', [])) == 1, 'Likelihood HTTP response invalid')
    lp = response['choices'][0]['logprobs']
    ids = request['input_ids']; count = len(ids)
    require(response['usage']['prompt_tokens'] == count, 'Likelihood prompt token count differs')
    require(response['usage']['completion_tokens'] == 1, 'Likelihood transport expected one audit generation token')
    require(len(lp['token_logprobs']) == len(lp['tokens']) == count + 1, 'Prompt echo logprobs missing or truncated')
    require(lp['tokens'][:count] == ['token_id:' + str(i) for i in ids], 'Echo token identity differs')
    start = len(request['context_ids'])
    scores = lp['token_logprobs'][start:count]
    require(len(scores) == len(request['continuation_ids']) and all(type(x) in (int,float) and math.isfinite(x) for x in scores),
            'Continuation logprobs incomplete/nonfinite')
    return math.fsum(scores), scores


async def live_quality(url, rows, directory, concurrency=4):
    import aiohttp
    directory=Path(directory); require(not directory.exists(), 'Refuse quality output overwrite'); directory.mkdir()
    semaphore=asyncio.Semaphore(concurrency)
    raw=(directory/'http-responses.jsonl').open('x',encoding='utf-8'); records=[]; requests=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        async def one(row, request):
            body={'model':'serving-lab','prompt':request['input_ids'],'max_tokens':1,'temperature':0,
                  'ignore_eos':True,'n':1,'echo':True,'stream':False,'logprobs':1,'prompt_logprobs':1,
                  'add_special_tokens':False,'return_tokens_as_token_ids':True,'return_token_ids':True}
            result={'sample_id':row['sample_id'],'choice_index':request['choice_index'], 'request_identity':digest(body),
                    'request':body, 'context':request['context'],'continuation':request['continuation'],
                    'context_ids':request['context_ids'],'continuation_ids':request['continuation_ids']}
            started=time.monotonic()
            async with semaphore:
                try:
                    async with session.post(url+'/v1/completions',json=body) as response:
                        payload=await response.json(); result['http_status']=response.status;result['response']=payload
                        require(response.status == 200, 'Likelihood endpoint returned '+str(response.status))
                        score,tokens=likelihood_from_response(payload,request)
                        result.update(status='OK',loglikelihood=score,continuation_token_logprobs=tokens,usage=payload['usage'])
                except Exception as exc:
                    result.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
                result['wall_seconds']=time.monotonic()-started
                raw.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');raw.flush()
                requests.append({k:v for k,v in result.items() if k not in ('response','request')})
                return result
        try:
            for row in rows:
                responses=await asyncio.gather(*(one(row,request) for request in row['requests']))
                if any(r['status']!='OK' for r in responses):
                    records.append({'sample_id':row['sample_id'],'status':'ERROR','errors':[r.get('error') for r in responses if r['status']!='OK']})
                    break  # Preserve error and incomplete coverage; never substitute another benchmark.
                records.append(score_document(row,[r['loglikelihood'] for r in responses]))
        finally:raw.close()
    complete=len(records)==len(rows) and all(r['status']=='OK' for r in records)
    out={'schema_version':2,'kind':'live_quality','task':'arc_challenge','split':rows[0]['split'],
         'status':'VALID_LIVE_SUBSET' if complete else 'BLOCKED_LIKELIHOOD_OR_INCOMPLETE',
         'primary_metric':'acc_norm','auxiliary_metric':'acc','n_expected_documents':len(rows),
         'n_documents':len(records),'n_requests':len(requests),'request_errors':sum(r['status']!='OK' for r in requests),
         'records':records,'requests':requests,'normalization':'sum choice continuation logprob / choice text character length',
         'full_benchmark':False,'backend':'vLLM0.29 HTTP prompt-logprobs echo; deployed artifact loader and kernel path',
         'generation_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in requests),
         'processed_input_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in requests)}
    if complete:
        out['metrics']={metric:statistics.fmean(r[metric] for r in records) for metric in ('acc_norm','acc')}
    write(directory/'scores.json',out)
    return out


def paired_quality(baseline, candidate, replicates=5000, seed=909223, threshold=-.02):
    require(baseline.get('kind')==candidate.get('kind')=='live_quality','Only explicitly typed live quality may use this legacy live-subset view')
    require(baseline['status']==candidate['status']=='VALID_LIVE_SUBSET','Incomplete/historical quality is not eligible')
    left,right=baseline['records'],candidate['records']
    for records in (left,right):
        require(len({r['sample_id'] for r in records})==len(records),'Duplicate quality sample ID')
    identities=lambda rows:[(r['sample_id'],r['source_text_hash'],r['request_identity'],r['gold']) for r in rows]
    require(identities(left)==identities(right) and left,'Different quality documents/prompts/tokenization')
    result={'kind':'paired_live_subset','n':len(left),'unit':'input/document','bootstrap_replicates':replicates,'seed':seed,
            'scope':'Sampling uncertainty within selected subset only; no full benchmark or production generalization','metrics':{}}
    for metric in ('acc_norm','acc'):
        delta=[r[metric]-l[metric] for l,r in zip(left,right)]
        rng=random.Random(seed); draws=sorted(statistics.fmean(delta[rng.randrange(len(delta))] for _ in delta) for _ in range(replicates))
        lo,hi=draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]
        gate='PASS' if lo>=threshold else 'FAIL' if hi<threshold else 'INCONCLUSIVE'
        result['metrics'][metric]={'baseline':statistics.fmean(l[metric] for l in left),'candidate':statistics.fmean(r[metric] for r in right),
                'delta':statistics.fmean(delta),'ci95':[lo,hi],'demo_loss_floor':threshold,'screen':gate,
                'correct_to_wrong':sum(l[metric]==1 and r[metric]==0 for l,r in zip(left,right)),
                'wrong_to_correct':sum(l[metric]==0 and r[metric]==1 for l,r in zip(left,right)),
                'correct_retained':sum(l[metric]==r[metric]==1 for l,r in zip(left,right))}
    result['per_input']=[{'sample_id':l['sample_id'],'gold':l['gold'],'baseline_acc_norm':l['acc_norm'],'candidate_acc_norm':r['acc_norm'],
                         'choice_score_changes':[b-a for a,b in zip(l['choice_loglikelihood'],r['choice_loglikelihood'])]} for l,r in zip(left,right)]
    # Explicit official metric views over the library's generic binary-correctness contract.
    # acc_norm uses the normalized prediction; it is never relabeled as ARC acc.
    from .._vendor.diova_compare import core as compare_core_module
    views={}
    for metric in ('acc_norm','acc'):
        def exported(out,tag):
            return [{'schema_version':1,'sample_id':r['sample_id'],'task':'arc_challenge/'+metric,'task_version':'1.0/STEP02-subset',
                     'prompt_hash':digest({'prompt':r['prompt'],'request_identity':r['request_identity']}),
                     'gold':[str(r['gold'])],'gold_definition':'Official choice index; '+metric+' decision rule',
                     'model_id':out.get('artifact',{}).get('model_id','same-model-paired-subset'),'artifact_id':tag,
                     'output_type':'choice','answer':str(r['prediction_'+metric]),'correct':bool(r[metric]),'evidence_kind':'measurement',
                     'official_metric':metric,'normalization':'choice character length' if metric=='acc_norm' else 'raw summed continuation likelihood'} for r in out['records']]
        views[metric]=compare_core_module.compare(exported(baseline,'baseline'),exported(candidate,'candidate'),replicates=20,seed=seed)
    result['diova_compare_views']={'scope':'Generic paired correctness/transition connection check; official metric names kept as distinct task views; primary intervals above use5000 subset draws','views':views}
    return result


def jsonl(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def replay_scores(directory,rows,strict_order=False):
    saved=read(directory/'scores.json');raw=jsonl(directory/'http-responses.jsonl')
    require(saved['kind']=='live_quality' and saved['status']=='VALID_LIVE_SUBSET' and saved['n_expected_documents']==len(rows), 'Incomplete/historical quality is not new valid DEV')
    require(len(raw)==sum(len(r['requests']) for r in rows),'Raw choice coverage incomplete')
    mapping={(r["sample_id"],r["choice_index"]):r for r in raw}
    require(len(mapping)==len(raw),"Duplicate raw sample/choice")
    cursor=0;records=[]
    for row in rows:
        scores=[]
        for request in row['requests']:
            key=(row['sample_id'],request['choice_index'])
            require(key in mapping,'Missing raw sample/choice')
            r=mapping[key]
            if strict_order:require(raw[cursor] is r,'C1 raw request order differs')
            cursor+=1
            require(r['status']=='OK' and r['http_status']==200 and r['sample_id']==row['sample_id'] and r['choice_index']==request['choice_index'],'Raw request pairing differs')
            require(r['request']['prompt']==request['input_ids'] and r['context_ids']==request['context_ids'] and r['continuation_ids']==request['continuation_ids'],'Raw token boundary differs')
            require(r['request_identity']==digest(r['request']) and r['request']['max_tokens']==1 and r['request']['echo'] is True,'Likelihood transport changed')
            score,_=likelihood_from_response(r['response'],request)
            require(score==r['loglikelihood'],'Raw stored score differs')
            scores.append(score)
        records.append(score_document(row,scores))
    require(records==saved['records'],'Official scoring replay differs')
    require(saved['metrics']=={m:statistics.fmean(r[m] for r in records) for m in ('acc_norm','acc')},'Aggregate differs')
    return records

