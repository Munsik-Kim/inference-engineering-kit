"""Closed E4B CPU contract subset, plus retained E3/R1 executor regressions.

The five module names are fixed. Test fixtures create only synthetic temporary
acquisitions. No actual model, tokenizer, inference or historical run is used.
The explicit tests root may be a dependency-closed copy outside the checkout;
it is never used as a package-source fallback.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

MODULES = ('bench05e4b.test_binding', 'bench05e4b.test_execution_binding',
           'bench05e4b.test_results', 'bench05e3.test_execution',
           'bench05e3.test_terminal_checkpoint')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_closure(root):
    """Enumerate only static local test imports and package initializers."""
    pending = list(MODULES); seen = set(); files = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        relative = Path(*name.split('.')).with_suffix('.py'); path = root / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                'Missing or unsafe closed test dependency: ' + relative.as_posix())
        files.add(relative.as_posix())
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=relative.as_posix())
        for node in ast.walk(tree):
            candidates = [node.module] if isinstance(node, ast.ImportFrom) and node.module else (
                [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for child in candidates:
                if child.startswith('bench') and (root / Path(*child.split('.')).with_suffix('.py')).is_file():
                    pending.append(child)
        for directory in relative.parents:
            if directory == Path('.'):
                continue
            init = directory / '__init__.py'; init_path = root / init
            require(init_path.is_file() and not init_path.is_symlink(), 'Missing closed test initializer: ' + init.as_posix())
            files.add(init.as_posix())
    return {name: {'bytes': (root/name).stat().st_size, 'sha256': sha(root/name)} for name in sorted(files)}


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = []

    def startTest(self, test):
        self.ids.append(test.id())
        super().startTest(test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-source-digest')
    parser.add_argument('--expected-origin', type=Path)
    parser.add_argument('--installed', action='store_true')
    args = parser.parse_args(argv)
    root = args.tests_root.absolute().resolve(); require(root.is_dir(), 'Explicit closed tests root required')
    inventory = dependency_closure(root)
    require(not args.output.exists(), 'Refusing existing test receipt')
    sys.path.insert(0, str(root))
    import inference_kit
    from inference_kit.analysis.tokens import source_identity
    origin = Path(inference_kit.__file__).absolute(); source = source_identity()
    if args.expected_origin is not None:
        require(origin.is_relative_to(args.expected_origin.absolute()), 'Package import origin differs from requested environment')
    if args.installed:
        require('site-packages' in origin.parts, 'Installed wheel must import from site-packages')
    require(args.expected_source_digest is None or source == args.expected_source_digest, 'Package source differs from frozen test snapshot')
    loader = unittest.TestLoader(); suites = [loader.loadTestsFromName(module) for module in MODULES]
    counts = {name: suite.countTestCases() for name, suite in zip(MODULES, suites)}
    require(all(count > 0 for count in counts.values()), 'Zero-test module cannot pass')
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(unittest.TestSuite(suites))
    failures = [{'test_id': test.id(), 'traceback': message} for test, message in result.failures]
    errors = [{'test_id': test.id(), 'traceback': message} for test, message in result.errors]
    error_tests = sorted({row['test_id'].split(' (', 1)[0] for row in errors})
    failure_tests = sorted({row['test_id'].split(' (', 1)[0] for row in failures})
    success = result.wasSuccessful() and result.testsRun > 0 and result.testsRun == len(result.ids) == len(set(result.ids))
    receipt = {'schema_version': 1, 'status': 'PASS' if success else 'FAIL',
        'scope': 'SYNTHETIC_CONTRACT_TEST; prepared metadata and temporary fake execution only; no new live raw replay',
        'modules': list(MODULES), 'module_collected_counts': counts,
        'collected': sum(counts.values()), 'invocations': result.testsRun, 'unique_ids': len(set(result.ids)),
        'test_ids': result.ids, 'pass_count': result.testsRun - len(error_tests) - len(failure_tests) - len(result.skipped),
        'failure_test_count': len(failure_tests), 'failure_events': len(failures), 'failures': failures,
        'error_test_count': len(error_tests), 'error_events': len(errors), 'errors': errors,
        'skip_count': len(result.skipped), 'skips': [{'test_id': test.id(), 'reason': reason} for test, reason in result.skipped],
        'interpreter': sys.executable, 'python_version': sys.version, 'package_version': inference_kit.__version__,
        'package_origin': str(origin), 'source_digest': source, 'installed_wheel': args.installed,
        'isolated_interpreter': bool(sys.flags.isolated), 'test_root_is_package_source': False,
        'test_dependency_inventory': inventory, 'new_live_acquisitions': 0, 'new_model_measurements': 0,
        'guard_scope': 'Boundary guards within individual tests, not an OS sandbox or inherited child guard'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'collected', 'invocations', 'unique_ids', 'pass_count',
        'failure_test_count', 'error_test_count', 'error_events', 'skip_count', 'source_digest')}, ensure_ascii=True))
    return int(not success)


if __name__ == '__main__':
    raise SystemExit(main())
