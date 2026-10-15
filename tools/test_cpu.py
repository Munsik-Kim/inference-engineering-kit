"""unittest runner recording unique IDs, invocations and explicit skips."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import unittest

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
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interpreter':sys.executable,
             'test_invocations':r.testsRun,'unique_tests':len(set(r.ids)),'ids':r.ids,
             'failures':[(t.id(),s) for t,s in r.failures],'errors':[(t.id(),s) for t,s in r.errors],
             'skips':[(t.id(),s) for t,s in r.skipped], 'successful':r.wasSuccessful(),
             'gpu_inference':0,'model_downloads':0,'remote_ci':'CI_NOT_RUN'}
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(receipt,indent=2)+'\n')
    return not r.wasSuccessful()
if __name__=='__main__':raise SystemExit(main())
