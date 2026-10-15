"""Explicit recorded slice replay; never claims completion of a full frozen recipe."""
from ..core.io import require
from ..core.quality import likelihood_from_response,score_document
from ..core.parity import equal


def replay_fixture(fixture,case):
    require(fixture['case']==case and fixture['evidence_kind']=='historical_recorded_fixture','Wrong recorded fixture identity')
    rows=fixture['rows'];require(len({r['sample_id'] for r in rows})==len(rows),'Duplicate recorded sample')
    results={};requests=0
    for arm,value in fixture['arms'].items():
        raw=value['http'];lookup={(r['sample_id'],r['choice_index']):r for r in raw}
        require(len(raw)==len(lookup)==sum(len(r['requests']) for r in rows),'Missing/duplicate recorded response')
        records=[]
        for row in rows:
            scores=[]
            for q in row['requests']:
                require((row['sample_id'],q['choice_index']) in lookup,'Missing recorded choice')
                r=lookup[(row['sample_id'],q['choice_index'])]
                require(r['status']=='OK' and r['request']['prompt']==q['input_ids'],'Recorded HTTP identity differs')
                score,_=likelihood_from_response(r['response'],q);scores.append(score);requests+=1
            records.append(score_document(row,scores))
        equal(records,value['expected_records']);results[arm]={'documents':len(records),'correct_acc_norm':sum(r['acc_norm'] for r in records)}
    return {'status':'RECORDED_FIXTURE_REPLAY_PASS','case':case,'evidence_scope':'historical_recorded_slice_only',
            'full_recipe_replayed':False,'documents_per_arm':len(rows),'http_responses_recomputed':requests,
            'arms':results,'new_gpu_runs':0,'new_model_downloads':0,'production_recommendation':'NOT_ELIGIBLE'}
