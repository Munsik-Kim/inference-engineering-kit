"""Export selected *recorded* observations as data, never execute bundle code.

Full vocabulary logits are deliberately omitted. The derived receipt retains
the original manifest/source identities; missing distributions stay missing.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from inference_kit.core.evidence import open_bundle
from inference_kit.core.io import read, write, sha, safe_name
from inference_kit.core.state import derived_output
from inference_kit.probes.blobs import load
from inference_kit.suites.bench05a_replay import verify_raw
from inference_kit.suites.bench05a_run import raw_manifest


def export(bundle, output):
    out = derived_output(bundle, output)
    with open_bundle(bundle) as source:
        verify_raw(source)
        for name in ('PLAN_FROZEN.json', 'data.json', 'budget-continuation.json'):
            shutil.copyfile(source / name, out / name)
        progress = copy.deepcopy(read(source / 'progress.json'))
        progress['jobs'] = {k: v for k, v in progress['jobs'].items() if k.startswith('main-')}
        progress['bridge_status'] = 'NOT_INCLUDED_IN_DERIVED_FIXTURE'
        write(out / 'progress.json', progress)
        omitted, changed = [], []
        for job in progress['jobs'].values():
            rel = safe_name(job['directory'])
            src, dst = source / rel, out / rel
            dst.mkdir(parents=True)
            for name in ('execution.json', 'result.json', 'loader.json', 'lifecycle.json'):
                shutil.copyfile(src / name, dst / name)
            retained = []
            for line in (src / 'observations.jsonl').read_text().splitlines():
                record = json.loads(line)
                if record['kind'] == 'full_logits':
                    omitted.append(record['observation_id'])
                    continue
                if record.get('numeric'):
                    load(src, record['numeric'])  # verify bounds/hash before copying
                    name = safe_name(record['numeric']['path'])
                    target = dst / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src / name, target)
                retained.append(record)
            target = dst / 'observations.jsonl'
            with target.open('x') as stream:
                for record in retained:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            changed.append({'path': rel + '/observations.jsonl', 'original_sha256': sha(src / 'observations.jsonl'),
                            'derived_sha256': sha(target), 'retained_records': len(retained)})
        write(out / 'DERIVATION.json', {
            'schema_version': 1, 'kind': 'selected_recorded_probe_fixture',
            'evidence_kind': 'live_model_diagnostic', 'source_raw_manifest_sha256': sha(source / 'RAW_MANIFEST.json'),
            'operation': 'Data-only selection; retained numeric bytes unchanged; no model or GPU execution',
            'changed_files': changed, 'omitted_full_logits_observation_ids': omitted,
            'omitted': ['full-vocabulary logits', 'native bridge HTTP', 'memory traces', 'smoke jobs'],
            'limits': 'Full KL/JS and complete observer-logit checks cannot be recalculated from this slice. '
                      'Original result.json counts describe the full historical execution, not this export.',
            'private_local_only': True,
        })
        raw_manifest(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(export(args.run, args.output))
