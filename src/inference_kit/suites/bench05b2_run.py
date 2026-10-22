"""B2 policy scheduling over existing owned serving/diagnostic primitives."""
import asyncio
from pathlib import Path
import sys
import time
from ..core.io import read, write, digest
from ..core.state import new_output, atomic
from ..tasks.contract_policy import diagnostics
from .bench05b2_plan import check_frozen
from .bench05b1_run import serving_job, seal
from .bench05a_run import launch_job


def diagnostic_plan(plan, policy):
    result = dict(plan)
    result.pop('plan_identity')
    samples = dict(plan['samples'])
    samples['rows'] = plan['samples']['by_policy'][policy]
    samples['rows_identity'] = digest(samples['rows'])
    result.update(samples=samples, diagnostic_policy=policy)
    result['counts'] = plan['counts'] | {
      'probe_target_positions_per_arm': sum(len(r['targets']) for r in samples['rows']),
      'probe_attention_rows_per_arm': sum(len(r['attention']['heads']) * len(r['attention']['queries']) for r in samples['rows'])}
    result['plan_identity'] = digest(result)
    return result


async def managed(plan_file, output):
    plan = read(Path(plan_file))
    check_frozen(plan)  # Before GPU queries, output creation, or continuation mutation.
    root = new_output(output); write(root / 'PLAN_FROZEN.json', plan)
    import inference_kit
    write(root / 'controller-environment.json', {'python': sys.executable, 'package_version': inference_kit.__version__,
      'package_import_origin': inference_kit.__file__, 'package_site': plan['paths']['package_site'], 'source_identity': plan['source_identity']})
    ledger = root / 'budget-continuation.json'; previous = plan['campaign']['used_seconds']
    atomic(ledger, {'schema_version': 1, 'task_id': 'BENCH05-B2-CONTRACT', 'campaign_cap_seconds': 7200,
      'cap_seconds': previous + plan['gpu_allowance_seconds'], 'used_seconds': previous,
      'task_origin_used_seconds': plan['task_origin_used_seconds'], 'running': None, 'jobs': [],
      'inherited_from': {'path': plan['campaign']['path'], 'sha256': plan['campaign']['sha256'], 'used_seconds': previous}})
    progress = {'status': 'RUNNING', 'jobs': {}, 'diagnostics': {}, 'automatic_retries': 0}
    atomic(root / 'progress.json', progress)
    try:
        for arm_id in plan['arm_order']:
            if time.monotonic() >= plan['clock']['no_new_gpu_after']:
                progress['status'] = 'BUDGET_SKIPPED'; return progress
            arm = next(a for a in plan['arms'] if a['arm_id'] == arm_id)
            okay = await serving_job(plan, root, ledger, progress, arm, score_function=diagnostics)
            atomic(root / 'progress.json', progress)
            if not okay:
                progress['status'] = 'PARTIAL_OR_BLOCKED'; return progress
        for policy in plan['suite']['probe_policies']:
            subroot = root / 'diagnostics' / policy; subroot.mkdir(parents=True)
            subplan = diagnostic_plan(plan, policy); write(subroot / 'PLAN_FROZEN.json', subplan)
            subprogress = {'status': 'RUNNING', 'jobs': {}}
            for arm_id in plan['arm_order']:
                if time.monotonic() >= plan['clock']['no_new_gpu_after']:
                    subprogress['status'] = 'BUDGET_SKIPPED'; break
                arm = next(a for a in plan['arms'] if a['arm_id'] == arm_id)
                okay = await launch_job(subplan, subroot, ledger, subprogress, 'main', arm, budget_namespace=policy + '-')
                atomic(subroot / 'progress.json', subprogress)
                if not okay:
                    subprogress['status'] = 'PARTIAL_OR_BLOCKED'; break
            else:
                subprogress['status'] = 'COMPLETE'
            atomic(subroot / 'progress.json', subprogress)
            progress['diagnostics'][policy] = {'status': subprogress['status'], 'directory': str(subroot.relative_to(root))}
            atomic(root / 'progress.json', progress)
            if subprogress['status'] != 'COMPLETE':
                break
        progress['status'] = 'NATURAL_COMPLETE'
        return progress
    except BaseException as exc:
        progress['status']='FAILED_OR_PARTIAL'
        write(root/'controller-failure.json',{'status':progress['status'],'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        atomic(root / 'progress.json', progress)
        seal(root, scope='bench05b2_contract_raw')


def run(plan, output):
    return asyncio.run(managed(plan, output))
