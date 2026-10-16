"""unittest runner recording unique IDs, invocations and explicit skips."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import unittest
import inference_kit

class Result(unittest.TextTestResult):
    def startTest(self,test):
        self.ids.append(test.id());super().startTest(test)
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.ids=[]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--pattern',default='test_*.py');a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise SystemExit('Refuse test receipt overwrite')
    suite=unittest.defaultTestLoader.discover('tests',pattern=a.pattern)
    r=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interpreter':sys.executable,'python_version':sys.version,
             'test_invocations':r.testsRun,'unique_tests':len(set(r.ids)),'ids':r.ids,
             'failures':[(t.id(),s) for t,s in r.failures],'errors':[(t.id(),s) for t,s in r.errors],
             'skips':[(t.id(),s) for t,s in r.skipped], 'successful':r.wasSuccessful() and r.testsRun>0,
             'collection_status':'COLLECTED' if r.testsRun else 'NO_TESTS_COLLECTED',
             'package_version':inference_kit.__version__, 'package_import_origin':inference_kit.__file__,
             'gpu_inference':0,'model_downloads':0,
             'remote_ci':'GITHUB_ACTIONS_CPU_RUN' if os.environ.get('GITHUB_ACTIONS')=='true' else 'CI_NOT_RUN',
             'github_run_id':os.environ.get('GITHUB_RUN_ID') if os.environ.get('GITHUB_ACTIONS')=='true' else None}
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return not receipt['successful']
if __name__=='__main__':raise SystemExit(main())
