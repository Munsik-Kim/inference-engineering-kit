"""Saved multi-model quality and performance replay; no acquisition or model import."""
from pathlib import Path
from ..core.io import read,write,require
from ..core.quality import replay_scores,paired_quality
from ..core.assess import assess
from ..core.parity import equal
from ..core.index import check_index
from ..adapters.http import summary as request_summary,quantile


def replay(root,selected,output):
    root,selected,output=Path(root),Path(selected),Path(output)
    bases=[p.parent for p in root.rglob('CONFIG_FREEZE.json') if (p.parent/'data/qwen3_4b-test.json').exists()]
    require(len(bases)==1,'Missing/ambiguous STEP02 frozen data');base=bases[0]
    source_spec=read(base/'CONFIG_FREEZE.json');qualities={};quality_docs=0;paired={}
    for directory in sorted((base/'test').glob('*')):
        file=directory/'quality/scores.json'
        if not file.exists():continue
        saved=read(file);key=directory.name.split('--')[0];rows=read(base/'data'/(key+'-test.json'))
        records=replay_scores(directory/'quality',rows);quality_docs+=len(records);qualities[directory.name]=saved
    for key,saved in qualities.items():
        model=key.split('--')[0];candidate=paired_quality(qualities[model+'--bf16'],saved)
        old=read(base/'comparisons'/(key+'-test.json'))
        equal(candidate,old);paired[key]=candidate
    runs=[];requests=0;all_retained_runs=[]
    # Latest repeat is selected by its explicit index/source, while older runs remain historical evidence.
    for directory in sorted((selected/'performance').glob('*')):
        f=directory/'result.json'
        if not f.exists():continue
        result=read(f);life=read(directory/'lifecycle.json')
        require(result['status']=='VALID_LIVE_PROFILE' and life['owned_process_cleanup']=='CLEANED' and life['owned_port_cleanup']=='FREE','Incomplete selected run')
        for cell in result['cells']:
            for phase in ('warmup','timed'):
                block=cell[phase];calculated=request_summary(block['records'],block['summary']['elapsed_seconds'])
                for key,value in calculated.items():equal(value,block['summary'][key],phase+'.'+key)
                records=block['records'];requests+=len(records)
                for stat,q in [('p50',.5),('p95',.95)]:equal(quantile([r['http_complete_seconds'] for r in records],q),block['summary']['http_complete_seconds'][stat])
        runs.append(result)
    index=check_index(read(selected/'results-index.json'),root,selected)
    assessments={}
    for file in sorted((selected/'requirements').glob('*.json')):
        value=assess(index,read(file));equal(value,read(selected/'assessments'/file.name));assessments[file.stem]=value
    if len(runs)==3:
        row=next(r for r in index['results'] if r['configuration_id']=='qwen3_4b--gptq_w4')
        for cell in row['performance']['cells']:
            selected_cells=[next(c for c in r['cells'] if c['input_length']==cell['input_length'] and c['concurrency']==cell['concurrency']) for r in runs]
            for metric in ('ttft_seconds','tpot_stream_estimate_seconds','http_complete_seconds'):
                for stat in ('p50','p95'):equal(max(c['timed']['summary'][metric][stat] for c in selected_cells),cell[metric][stat])
    original_review=base/'review/results-index.json'
    previous_status=None
    if original_review.exists():
        old=next(r for r in read(original_review)['results'] if r['configuration_id']=='qwen3_4b--gptq_w4')
        previous_status={'measurement_status':old['measurement_status'],'memory_comparison_status':old['memory'].get('comparison_status'),'memory_max_mib':old['memory'].get('max_mib')}
        require(old['measurement_status']=='PARTIAL_UNCERTIFIED' and old['memory'].get('comparison_status')=='SHARED_GPU_INTERFERENCE','Old interference/partial was promoted')
    result={'status':'CPU_REPLAY_PARITY_PASS','case':'step02','historical_quality_documents':quality_docs,
            'selected_performance_requests':requests,'selected_server_rounds':len(runs),'quality_pairings':len(paired),
            'assessment_statuses':{k:v['status'] for k,v in assessments.items()},'preserved_previous_status':previous_status,
            'new_gpu_runs':0,'quality_metrics':{k:v['metrics'] for k,v in qualities.items()}}
    write(output/'quality-comparisons.json',paired);write(output/'assessment-results.json',assessments)
    return result,index
