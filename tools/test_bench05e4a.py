"""Explicit closed E4A unit subset; optional legacy impact tests are local-only."""
import argparse
import json
from pathlib import Path
import sys
import unittest
import inference_kit
from inference_kit.core.io import digest, write
from inference_kit.suites.bench05a_plan import source_identity

class Result(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.ids=[]
    def startTest(self, test):
        self.ids.append(test.id()); super().startTest(test)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--legacy',action='store_true');args=parser.parse_args()
    sys.path.insert(0,str(Path('tests').absolute()))
    modules=['bench05e4a.test_raw_tasks','bench05e4a.test_token_preparation','bench05e4a.test_suite_prepared',
             'bench05e4a.test_preparation_api','bench05e4a.test_golden_parity']
    if args.legacy:
        modules += ['bench05e3.test_execution','bench05e3.test_terminal_checkpoint','bench05e2.test_inputs',
                    'bench05e2.test_plan','bench05c2.test_inputs','bench05c1.test_metrics','bench05c1.test_generation']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    receipt={'interpreter':sys.executable,'python_version':sys.version,'package_version':inference_kit.__version__,
      'package_origin':inference_kit.__file__,'source_identity':digest(source_identity()),'legacy':args.legacy,
      'invocations':result.testsRun,'unique_ids':len(set(result.ids)),'ids':result.ids,
      'failures':[(t.id(),msg) for t,msg in result.failures], 'errors':[(t.id(),msg) for t,msg in result.errors],
      'skips':[(t.id(),why) for t,why in result.skipped],
      'status':'PASS' if result.wasSuccessful() and result.testsRun else 'FAIL',
      'scope':'SYNTHETIC_CONTRACT_TEST; no actual tokenizer/model calls'}
    write(args.output,receipt)
    return int(receipt['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
