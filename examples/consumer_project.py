"""Self-owned synthetic business task adapter; no model/network is run."""
import argparse
import copy
from pathlib import Path
from inference_kit import compare, assess
from inference_kit.core.io import digest, write
from inference_kit.core.state import new_output

def main(output):
    out=new_output(output);baseline=[]
    for i in range(3):
        baseline.append({'schema_version':1,'sample_id':str(i),'task':'synthetic-invoice-category','task_version':'consumer-v1',
          'prompt_hash':digest({'own_synthetic_input':i,'profile':'own-workload-v1'}),'gold':['invoice'],'gold_definition':'Synthetic expected label only',
          'model_id':'synthetic-consumer','artifact_id':'baseline','output_type':'choice','answer':'invoice' if i==0 else 'other',
          'correct':i==0,'evidence_kind':'synthetic_test'})
    candidate=copy.deepcopy(baseline)
    for r in candidate:r.update(artifact_id='candidate',answer='invoice',correct=True)
    result=compare(baseline,candidate,out/'comparison',replicates=100)
    assessment=assess({'evidence_scope':'research_dev_only','task':'synthetic-invoice-category'}, {},out/'assessment')
    write(out/'baseline.json',baseline);write(out/'candidate.json',candidate)
    write(out/'consumer.json',{'scope':'SYNTHETIC_CONSUMER_EXAMPLE','core_modified':False,'assessment':assessment['status'],'new_gpu_runs':0})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)
