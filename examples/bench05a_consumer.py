"""Recalculate explicit saved observations with an ordinarily installed wheel."""
import argparse
import json
from inference_kit import bench_replay

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
result = bench_replay(args.run, args.output)
print(json.dumps({'counts': result['counts'], 'metrics': result['metrics'],
                  'scope': result['scope'], 'replay_gpu_actions': result['replay_gpu_actions']}, indent=2))
