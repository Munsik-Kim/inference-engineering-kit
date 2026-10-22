"""Explicit negative in-memory projections of stored data; raw bytes stay untouched."""
import argparse
import copy
import json
from pathlib import Path
from inference_kit.analysis.tokens import jsonl
from inference_kit.metrics.token_diagnostics import compare_token
from inference_kit.core.io import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--kit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('New output required')
 view=a.kit/'fixtures/smol-token';paths=[view/'token_records.jsonl',view/'sequences.jsonl'];before={str(x):sha(x) for x in paths}
 row=next(jsonl(paths[0]));sequences={x['sequence_id']:x for x in jsonl(paths[1])};seq=sequences[row['sequence_id']]
 normal=compare_token(row,row,seq,seq);assert normal['status']=='MEASURED'
 wrong=copy.deepcopy(row);wrong['prefix_hash']='0'*64
 denied=compare_token(row,wrong,seq,seq);assert denied['status']=='INCOMPARABLE' and any('PREFIX' in x for x in denied['reasons'])
 assert not denied['coverage']['target_logprob'];assert before=={str(x):sha(x) for x in paths}
 a.output.mkdir(parents=True);value={'status':'PASS','kind':'SYNTHETIC_NEGATIVE_PROJECTION_OF_REAL_ROW','normal_control':normal['status'],'different_prefix':denied['status'],'reasons':denied['reasons'],'raw_unchanged':True,'metric_api':'compare_token','new_observations':0}
 (a.output/'NEGATIVE_CONTRACTS.json').write_text(json.dumps(value,ensure_ascii=True,indent=2)+'\n');print(json.dumps(value,ensure_ascii=True))
if __name__=='__main__':main()
