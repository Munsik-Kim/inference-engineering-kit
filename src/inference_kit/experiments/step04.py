# Derived from inference-lab; import/path boundaries changed for internal release01.
"""CPU replay of raw likelihoods and joint paired document contrasts."""
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics as st

from ..core.io import read, write, require, digest, sha
from ..core.quality import score_document, likelihood_from_response, replay_scores
from .step04_recipe import H, ARMS, COEFFICIENTS


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def retain(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Replay numeric/semantic mismatch: '+path.name)
    else:write(path,value)


def margin(record):
    scores=record['choice_normalized_scores'];gold=record['gold']
    return scores[gold]-max(x for i,x in enumerate(scores) if i!=gold)


def paired_contrast(records,coefficients,metric,replicates=5000,seed=909233):
    arms=list(coefficients);n=len(records[arms[0]])
    identity=lambda rows:[(r['sample_id'],r['request_identity'],r['source_text_hash'],r['gold']) for r in rows]
    require(n>0 and all(identity(records[a])==identity(records[arms[0]]) for a in arms),'Unpaired identity')
    get=lambda r:margin(r) if metric=='gold_margin' else r[metric]
    delta=[math.fsum(coefficients[a]*get(records[a][i]) for a in arms) for i in range(n)]
    # Each contrast resets to the same seed, giving identical document resamples across all arms/contrasts.
    rng=random.Random(seed);draws=sorted(math.fsum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(replicates))
    return {'delta':st.fmean(delta),'ci95':[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]],
            'n_documents':n,'unit':'document ID','replicates':replicates,'seed':seed,
            'interval_scope':'Pointwise exploratory selected-DEV sampling uncertainty; joint paired resamples, not multiplicity-adjusted or full-benchmark certification'}


def screen(delta_norm,delta_acc,delta_margin,panels,ci):
    signal=delta_norm>=.02 and delta_acc>=-.02 and delta_margin>0 and all(x>=0 for x in panels)
    return {'status':'FOLLOWUP_SIGNAL' if signal else 'NO_FOLLOWUP_SIGNAL',
            'uncertainty':'EXPLORATORY_SIGNAL_ONLY' if signal and ci[0]<=0<=ci[1] else 'SINGLE_MODEL_DEV_EXPLORATION',
            'criteria':{'acc_norm_delta_at_least_0.02':delta_norm>=.02,'acc_delta_at_least_minus0.02':delta_acc>=-.02,
                        'mean_gold_margin_positive':delta_margin>0,'both_panels_nonnegative':all(x>=0 for x in panels)}}




def local_summary(root,directory,output):
    if not (directory/'local-result.json').exists():
        return {'status':'BLOCKED_OR_INCOMPLETE','evidence':str(directory.relative_to(root))}
    receipt=read(directory/'local-result.json');rows=jsonl(directory/'module-source.jsonl')
    expected=read(root/'data/holdout.json');byid={r['id']:r for r in expected}
    modules=set(read(root/'PROTOCOL_FROZEN.json')['selected_ratios'])
    require(receipt['status']=='COMPLETE_PACKED_LOCAL_HOLDOUT' and len(rows)==len(expected)*len(modules),'Local incomplete')
    require(len({(r['source_id'],r['module']) for r in rows})==len(rows),'Duplicate local units')
    require(all(r['source_id'] in byid and r['module'] in modules and r['rows']==8 and r['panel']==byid[r['source_id']]['panel'] for r in rows),'Local identity differs')
    def totals(selected):
        rtn=math.fsum(r['sse_rtn'] for r in selected);clip=math.fsum(r['sse_clip'] for r in selected);energy=math.fsum(r['reference_energy'] for r in selected)
        return {'sse_rtn':rtn,'sse_clip':clip,'gain':rtn-clip,'relative_reduction':(rtn-clip)/rtn if rtn else None,
                'reference_energy':energy,'normalized_rtn':rtn/energy if energy else None,'normalized_clip':clip/energy if energy else None,
                'actual_sampled_rows':sum(r['rows'] for r in selected)}
    def partition(selected):
        h=totals([r for r in selected if r['module']==H]);rest=totals([r for r in selected if r['module']!=H]);all_=totals(selected)
        permod={m:totals([r for r in selected if r['module']==m]) for m in sorted(modules)}
        normr=[v['normalized_rtn'] for m,v in permod.items() if m!=H and v['normalized_rtn'] is not None]
        normc=[v['normalized_clip'] for m,v in permod.items() if m!=H and v['normalized_clip'] is not None]
        h_sources=[r for r in selected if r['module']==H]
        return {'H':h,'R':rest,'total':all_,'H_share_of_total_gain':h['gain']/all_['gain'] if all_['gain'] else None,
                'H_sources_improved':sum(r['sse_clip']<r['sse_rtn'] for r in h_sources),
                'H_sources_worsened':sum(r['sse_clip']>r['sse_rtn'] for r in h_sources),
                'H_sources_equal':sum(r['sse_clip']==r['sse_rtn'] for r in h_sources),
                'R_normalized_mean_rtn':st.fmean(normr),'R_normalized_mean_clip':st.fmean(normc),
                'R_normalized_median_rtn':st.median(normr),'R_normalized_median_clip':st.median(normc),
                'R_modules_improved':sum(v['gain']>0 for m,v in permod.items() if m!=H),
                'R_modules_worsened':sum(v['gain']<0 for m,v in permod.items() if m!=H),
                'R_modules_equal':sum(v['gain']==0 for m,v in permod.items() if m!=H),'per_module':permod}
    result={'status':'COMPLETE_PACKED_LOCAL_HOLDOUT','all':partition(rows),
            'panels':{p:partition([r for r in rows if r['panel']==p]) for p in ('H1','H2')},
            'per_source':{r['id']:partition([x for x in rows if x['source_id']==r['id']]) for r in expected},
            'scope':'Common BF16 inputs and actual packed decode; not propagated quantized activations or end-task causal proof',
            'bootstrap':'Not used for local descriptive totals; source/article is the unit if added',
            'local_HxR_interaction':'Zero by additive definition; no claim about whole-model interaction'}
    retain(output/'local-summary.json',result)
    return result


def memory(directory):
    if not (directory/'memory.jsonl').exists():return {'scope':'whole_device_mib','status':'NOT_AVAILABLE'}
    samples=jsonl(directory/'memory.jsonl');valid=[r for r in samples if r['returncode']==0 and len(r.get('values','').split(','))==10]
    markers=read(directory/'memory-phases.json') if (directory/'memory-phases.json').exists() else []
    phases={}
    for left,right in zip(markers,markers[1:]):
        values=[float(r['values'].split(',')[3]) for r in valid if left['monotonic']<=r['monotonic']<right['monotonic']]
        phases[left['phase']]={'n':len(values),'max_mib':max(values) if values else None,'min_mib':min(values) if values else None}
    return {'scope':'whole_device_mib','purpose':'Resource/diagnostic only; no serving performance claim',
            'status':'OBSERVED' if valid else 'NOT_AVAILABLE','max_mib':max(float(r['values'].split(',')[3]) for r in valid) if valid else None,
            'samples':len(valid),'failed_samples':len(samples)-len(valid),'phase':phases,'nominal_interval_seconds':1}


def analyze(root,output):
    root=Path(root);output=Path(output);p=read(root/'PROTOCOL_FROZEN.json');rows=read(root/'data/dev-new.json');sent=read(root/'data/sentinels.json')
    for name,h in p['data_files'].items():require(sha(root/'data'/name)==h,'Input changed after freeze')
    progress=read(root/'progress.json');records={};arm_summary={};sentinel_summary={}
    for arm in ARMS:
        selected=progress.get('jobs',{}).get(arm)
        if not selected:
            arm_summary[arm]={'status':'NOT_RUN_OR_BLOCKED'};continue
        d=root/selected['directory']
        if selected['status']!='COMPLETE' or not (d/'evaluation.json').exists():
            arm_summary[arm]={'status':'FAILED_OR_BLOCKED','evidence':str(d.relative_to(root))};continue
        ev=read(d/'evaluation.json');fr=read(d/'freeze.json');life=read(d/'lifecycle.json');after=read(d/'gpu-after.json')
        require(fr['plan_sha256']==sha(root/'PROTOCOL_FROZEN.json') and fr['artifact']['artifact_identity']==p['artifacts'][arm]['artifact_identity'],'Live result source identity differs')
        require(ev['kind']=='new_live_dev_quality' and ev['status']=='VALID_LIVE_DEV_NEW' and ev['input_data_sha256']==sha(root/'data/dev-new.json'),'Not new frozen DEV')
        require(life['owned_process_cleanup']=='CLEANED' and life['owned_port_cleanup']=='FREE' and life['parent_returncode']==0,'Unsafe/incomplete lifecycle')
        require(after['returncode']==0 and not after.get('values'),'Post-job GPU interference/unresolved')
        for name,key in [('quality/scores.json','main_score_sha256'),('quality/http-responses.jsonl','raw_http_sha256'),
                         ('sentinel/scores.json','sentinel_score_sha256'),('sentinel/http-responses.jsonl','sentinel_raw_sha256')]:
            require(sha(d/name)==ev[key],'Sealed evaluation evidence changed')
        records[arm]=replay_scores(d/'quality',rows,strict_order=True);sr=replay_scores(d/'sentinel',sent,strict_order=True)
        comparisons=[]
        for a,b in zip(records[arm][:len(sr)],sr):
            changes=[y-x for x,y in zip(a['choice_normalized_scores'],b['choice_normalized_scores'])]
            comparisons.append({'sample_id':a['sample_id'],'normalized_choice_deltas':changes,
                     'raw_choice_deltas':[y-x for x,y in zip(a['choice_loglikelihood'],b['choice_loglikelihood'])],
                     'prediction_acc_match':a['prediction_acc']==b['prediction_acc'],
                     'prediction_acc_norm_match':a['prediction_acc_norm']==b['prediction_acc_norm']})
        sentinel_summary[arm]={'documents':len(sr),'http_requests':sum(len(x['requests']) for x in sent),
                     'changed_normalized_predictions':sum(not r['prediction_acc_norm_match'] for r in comparisons),
                     'max_abs_normalized_score_change':max(abs(v) for r in comparisons for v in r['normalized_choice_deltas']),
                     'comparisons':comparisons,'unit':'Repeatability probe only; excluded from main accuracy/CI'}
        arm_summary[arm]={'status':'VALID_LIVE_DEV_NEW','n_documents':len(records[arm]),
             'acc_norm':st.fmean(r['acc_norm'] for r in records[arm]),'acc':st.fmean(r['acc'] for r in records[arm]),
             'correct_acc_norm':sum(r['acc_norm'] for r in records[arm]),'correct_acc':sum(r['acc'] for r in records[arm]),
             'mean_gold_margin':st.fmean(margin(r) for r in records[arm]),'main_http':ev['n_main_http'],
             'sentinel_http':ev['n_sentinel_http'],'warmup_http':1,'startup_seconds':ev['startup_seconds'],
             'artifact_identity':ev['artifact_identity'],'memory':memory(d),'evidence':str(d.relative_to(root)),
             'lifecycle':life,'panel_acc_norm':{panel:st.fmean(r['acc_norm'] for r,row in zip(records[arm],rows) if row['panel']==panel) for panel in ('A','B')}}
    complete=set(records)==set(ARMS);contrasts={};pairings={};decisions={}
    if complete:
        for name,co in COEFFICIENTS.items():
            contrasts[name]={m:paired_contrast(records,co,m) for m in ('acc_norm','acc','gold_margin')}
        for arm in ('R10','R01','R11','BF16_REF','GPTQ_REF'):
            base,cur=records['R00'],records[arm]
            comparisons={m:paired_contrast(records,{arm:1,'R00':-1},m) for m in ('acc_norm','acc','gold_margin')}
            panel_delta=[arm_summary[arm]['panel_acc_norm'][pan]-arm_summary['R00']['panel_acc_norm'][pan] for pan in ('A','B')]
            transitions={'correct_to_wrong':sum(a['acc_norm']==1 and b['acc_norm']==0 for a,b in zip(base,cur)),
                         'wrong_to_correct':sum(a['acc_norm']==0 and b['acc_norm']==1 for a,b in zip(base,cur)),
                         'correct_retained':sum(a['acc_norm']==b['acc_norm']==1 for a,b in zip(base,cur)),
                         'prediction_changed':sum(a['prediction_acc_norm']!=b['prediction_acc_norm'] for a,b in zip(base,cur))}
            pairings[arm]={'vs_R00':comparisons,'panels_A_B_acc_norm_delta':panel_delta,'transitions_acc_norm':transitions,
                          'gold_margin_delta_quantiles':quantiles([margin(b)-margin(a) for a,b in zip(base,cur)]),
                          'per_input':[{'sample_id':a['sample_id'],'acc_norm_delta':b['acc_norm']-a['acc_norm'],'acc_delta':b['acc']-a['acc'],
                             'gold_margin_delta':margin(b)-margin(a),'choice_raw_score_delta':[y-x for x,y in zip(a['choice_loglikelihood'],b['choice_loglikelihood'])],
                             'choice_norm_score_delta':[y-x for x,y in zip(a['choice_normalized_scores'],b['choice_normalized_scores'])]} for a,b in zip(base,cur)]}
            if arm in ('R10','R01','R11'):
                decisions[arm]=screen(comparisons['acc_norm']['delta'],comparisons['acc']['delta'],comparisons['gold_margin']['delta'],panel_delta,comparisons['acc_norm']['ci95'])
        for arm in ('R00','R10','R01','R11'):
            pairings.setdefault(arm,{})['vs_references']={ref:{m:paired_contrast(records,{arm:1,ref:-1},m) for m in ('acc_norm','acc','gold_margin')} for ref in ('BF16_REF','GPTQ_REF')}
    candidates=[a for a in ('R10','R01','R11') if decisions.get(a,{}).get('status')=='FOLLOWUP_SIGNAL']
    selected=max(candidates,key=lambda a:arm_summary[a]['acc_norm']) if candidates else None
    decision={'status':('FOLLOWUP_SIGNAL' if candidates else 'HOLD_CLIPPING_VARIANT') if complete else 'BLOCKED_OR_INCONCLUSIVE',
              'candidates':decisions,'selected_one_for_future_independent_check':selected,
              'selection_rule':'Highest acc_norm among screen pass; ties R10,R01,R11; DEV resource screen only',
              'vs_GPTQ':('BETTER_POINT_ESTIMATE' if arm_summary[selected]['acc_norm']>arm_summary['GPTQ_REF']['acc_norm'] else 'NOT_BETTER_THAN_THIS_GPTQ_RECIPE') if selected else 'NO_SELECTED_VARIANT',
              'production_recommendation':'NOT_ELIGIBLE_RESEARCH_DEV_ONLY','customer_validation':'NOT_RUN',
              'threshold_role':'Prospective STEP04 resource-allocation DEMO policy, not universal scientific or customer acceptance threshold'}
    local_dir=progress.get('jobs',{}).get('local',{}).get('directory','jobs/local-attempt1')
    local=local_summary(root,root/local_dir,output)
    comparisons={'schema_version':4,'status':'COMPLETE_FROZEN_SIX_ARM' if complete else 'PARTIAL_OR_BLOCKED','contrasts':contrasts,
                 'pairings':pairings,'sentinel':sentinel_summary,'intervals':'Pointwise exploratory paired document bootstrap5000; same resamples; no TEST used'}
    retain(output/'comparisons.json',comparisons);retain(output/'decision.json',decision)
    budget=read(root/'gpu_budget.json')
    summary={'schema_version':4,'task':'step04','engineering_status':'SIX_LIVE_ARTIFACTS_VALIDATED' if complete else 'PARTIAL_OR_BLOCKED',
             'quality_status':'NEW_DEV_ONLY_NO_TEST_NO_CERTIFICATION' if complete else 'INCOMPLETE',
             'local_status':local['status'],'research_decision':decision['status'],'deployment_readiness':'NOT_ELIGIBLE_RESEARCH_DEV_ONLY',
             'customer_validation':'NOT_RUN','novel_algorithm':'NOT_ESTABLISHED','arms':arm_summary,
             'gpu_task_used_seconds':budget['used_seconds']-p['prior_ledger']['used_seconds'],
             'campaign_used_seconds':budget['used_seconds'],'campaign_remaining_seconds':7200-budget['used_seconds'],
             'task_remaining_seconds':budget['cap_seconds']-budget['used_seconds'],
             'local_memory':memory(root/local_dir),'protocol_sha256':sha(root/'PROTOCOL_FROZEN.json'),
             'new_main_http':sum(r.get('main_http',0) for r in arm_summary.values()),
             'new_sentinel_http':sum(r.get('sentinel_http',0) for r in arm_summary.values()),
             'new_warmup_http':sum(r.get('warmup_http',0) for r in arm_summary.values())}
    retain(output/'summary.json',summary)
    buf=io.StringIO(newline='');w=csv.writer(buf);w.writerow(['arm','status','n','acc_norm','acc','margin','whole_device_max_mib'])
    for arm,a in arm_summary.items():w.writerow([arm,a['status'],a.get('n_documents'),a.get('acc_norm'),a.get('acc'),a.get('mean_gold_margin'),a.get('memory',{}).get('max_mib')])
    csv_path=output/'comparisons.csv';blob=buf.getvalue().encode()
    if csv_path.exists():require(csv_path.read_bytes()==blob,'CSV replay differs (bytes, intentional CRLF)')
    else:csv_path.write_bytes(blob)
    return summary


def quantiles(values):
    ordered=sorted(values)
    return {str(q):ordered[round((len(ordered)-1)*q)] for q in (0,.25,.5,.75,1)}
