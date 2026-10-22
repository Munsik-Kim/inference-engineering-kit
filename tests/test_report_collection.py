"""Small finite-cap regressions for the product's offline collection path."""
from copy import deepcopy
import errno
from html.parser import HTMLParser
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from inference_kit.analysis.index_report import render_report
from inference_kit.analysis.report_packaging import (
    LIMITS, validate_report_collection, validate_report_packaging,
)
from inference_kit.core.io import read, sha


def example(root, sizes=(3, 3, 3)):
    source = root / 'source'
    source.mkdir()
    result = {k: [] for k in ('sources', 'acquisitions', 'analyses', 'samples',
                              'executions', 'measurements', 'evidence', 'relations')}
    result['sources'] = [{'id': 'source-a', 'validation': {'status': 'RAW_HASH_VERIFIED'}}]
    for i, size in enumerate(sizes):
        name = 'raw/' + str(i) + '.json'
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes([65 + i]) * size)
        result['evidence'].append({'id': 'evidence-' + str(i), 'source_id': 'source-a',
                                  'relative_path': name, 'bytes': size, 'sha256': sha(path),
                                  'approved_copy': {'source_root': str(source), 'relative_path': name,
                                                    'portable_path': 'evidence/source-a/' + name,
                                                    'sha256': sha(path)}})
    return result


def policy(volume=6, **extra):
    return {'schema_version': 1, 'mode': 'bounded_collection_v1',
            'volume_payload_bytes': volume, **extra}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a' and 'href' in dict(attrs): self.links.append(dict(attrs)['href'])


class ReportCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_policy_is_explicit_and_only_downward(self):
        self.assertEqual(validate_report_packaging(), {'schema_version': 1, 'mode': 'single_report_v1'})
        normalized = validate_report_packaging(policy())
        self.assertEqual(normalized['volume_payload_bytes'], 6)
        for key, maximum in LIMITS.items():
            for bad in (0, True, maximum + 1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_report_packaging(policy(**{key: bad}) if key != 'volume_payload_bytes'
                                              else policy(volume=bad))
        for invalid in ({'schema_version': True, 'mode': 'bounded_collection_v1'},
                        {'schema_version': 1, 'mode': 'unknown'},
                        {'schema_version': 1, 'mode': 'single_report_v1', 'max_volumes': 1},
                        {'schema_version': 1, 'mode': 'bounded_collection_v1', 'ignore_hash': True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_report_packaging(invalid)

    def test_two_parts_exact_cap_and_legacy_numeric_parity(self):
        fixture = example(self.root)
        fixture['measurements'] = [{'id': 'metric', 'metric': 'target_nll', 'value': 1.25,
                                    'status': 'MEASURED', 'scope': 'token_pair', 'evidence_ids': ['evidence-2']}]
        legacy = render_report(fixture, self.root / 'legacy')
        result = render_report(fixture, self.root / 'collection', policy())
        verified = validate_report_collection(self.root / 'collection')
        self.assertEqual(verified['volume_payload_bytes'], [6, 3])
        self.assertEqual(verified['logical_member_count'], 3)
        self.assertEqual(verified['logical_payload_bytes'], 9)
        self.assertEqual(result['table_counts'], legacy['table_counts'])
        self.assertEqual((self.root / 'legacy/display.csv').read_bytes(), (self.root / 'collection/display.csv').read_bytes())
        self.assertEqual(len(result['portable_evidence']), 3)

    def test_under_exact_and_one_byte_over_volume_boundary(self):
        for sizes, expected in (((5,), [5]), ((6,), [6]), ((6, 1), [6, 1])):
            location = self.root / str(len(list(self.root.iterdir())))
            location.mkdir()
            fixture = example(location, sizes)
            result = render_report(fixture, location / 'out', policy())
            self.assertEqual(result['collection']['volume_payload_bytes'], expected)

    def test_total_payload_member_and_volume_caps_fail_before_output(self):
        fixture = example(self.root)
        for bound in (policy(volume=2), policy(volume=3, max_volumes=1),
                      policy(evidence_payload_bytes=8), policy(member_bytes=2),
                      policy(max_members=2), policy(collection_bytes=8)):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                render_report(fixture, self.root / 'out', bound)
            self.assertFalse((self.root / 'out').exists())

    def test_presentation_and_whole_output_caps_are_separate(self):
        fixture = example(self.root)
        for bound in (policy(presentation_bytes=1), policy(collection_bytes=100)):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                render_report(fixture, self.root / 'out', bound)
            self.assertFalse((self.root / 'out').exists())
            self.assertEqual(list(self.root.glob('.out.*.tmp')), [])

    def test_order_and_mapping_deterministic_and_excluded_status_preserved(self):
        fixture = example(self.root)
        fixture['evidence'].append({'id': 'stale', 'source_id': 'historical', 'relative_path': 'old.json',
                                    'copy_exclusion_status': 'STALE', 'copy_exclusion_reason': 'preserved failure'})
        before = deepcopy(fixture)
        render_report(fixture, self.root / 'one', policy())
        fixture['evidence'].reverse()
        render_report(fixture, self.root / 'two', policy())
        left = read(self.root / 'one/REPORT_COLLECTION_MANIFEST.json')
        right = read(self.root / 'two/REPORT_COLLECTION_MANIFEST.json')
        self.assertEqual(left['logical_evidence'], right['logical_evidence'])
        self.assertEqual(left['parts'], right['parts'])
        self.assertEqual(left['logical_inventory_identity'], right['logical_inventory_identity'])
        self.assertEqual(left['excluded_evidence'][0]['status'], 'STALE')
        self.assertEqual(before['evidence'], list(reversed(fixture['evidence'])))

    def test_relocation_preserves_all_relative_raw_links_and_identity(self):
        fixture = example(self.root)
        render_report(fixture, self.root / 'one', policy())
        before = validate_report_collection(self.root / 'one')
        relocated = self.root / '한글 공백' / '보고서'
        relocated.parent.mkdir()
        shutil.move(self.root / 'one', relocated)
        self.assertEqual(before, validate_report_collection(relocated))
        parser = Links(); parser.feed((relocated / 'index.html').read_text(encoding='utf-8'))
        for link in parser.links:
            if not link.startswith('#'):
                self.assertTrue((relocated / unquote(link)).is_file())
                self.assertFalse('://' in link)

    def test_missing_part_member_changed_byte_and_extra_file_fail(self):
        fixture = example(self.root)
        render_report(fixture, self.root / 'original', policy())
        manifest = read(self.root / 'original/REPORT_COLLECTION_MANIFEST.json')
        for i, (name, action) in enumerate(((manifest['parts'][1]['path'], 'delete'),
                                           (manifest['logical_evidence'][2]['path'], 'delete'),
                                           (manifest['logical_evidence'][2]['path'], 'change'),
                                           ('derived.sqlite', 'extra'))):
            target = self.root / ('copy-' + str(i))
            shutil.copytree(self.root / 'original', target)
            if action == 'delete': (target / name).unlink()
            else: (target / name).write_bytes(b'wrong')
            with self.subTest(name=name, action=action), self.assertRaises(ValueError):
                validate_report_collection(target)

    def test_traversal_symlink_conflicting_key_and_output_inside_source_rejected(self):
        fixture = example(self.root)
        for change in ('traversal', 'symlink', 'duplicate', 'inside'):
            altered = deepcopy(fixture)
            output = self.root / 'out'
            if change == 'traversal': altered['evidence'][0]['approved_copy']['relative_path'] = '../wrong.json'
            elif change == 'symlink':
                link = self.root / 'source/link.json'
                link.symlink_to(self.root / 'source/raw/0.json')
                altered['evidence'][0]['relative_path'] = 'link.json'
                altered['evidence'][0]['approved_copy']['relative_path'] = 'link.json'
            elif change == 'duplicate':
                duplicate = deepcopy(altered['evidence'][0]); duplicate['id'] = 'another'
                altered['evidence'].append(duplicate)
            else: output = self.root / 'source/report'
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_report(altered, output, policy())
            self.assertFalse(output.exists())

    def test_atomic_copy_and_rename_failure_never_publish_partial_output(self):
        fixture = example(self.root)
        original = (self.root / 'source/raw/0.json').read_bytes()
        for patched in ('inference_kit.analysis.index_report.copy_evidence',
                        'inference_kit.analysis.report_packaging.os.rename'):
            with self.subTest(patched=patched), patch(patched, side_effect=OSError(errno.ENOSPC, 'injected')):
                with self.assertRaises(OSError) as caught:
                    render_report(fixture, self.root / 'out', policy())
                self.assertEqual(caught.exception.errno, errno.ENOSPC)
            self.assertFalse((self.root / 'out').exists())
            self.assertEqual(list(self.root.glob('.out.*.tmp')), [])
            self.assertEqual((self.root / 'source/raw/0.json').read_bytes(), original)

    def test_destination_collision_preserves_existing_bytes(self):
        fixture = example(self.root)
        output = self.root / 'out'; output.mkdir(); (output / 'sentinel').write_bytes(b'keep')
        with self.assertRaises(ValueError): render_report(fixture, output, policy())
        self.assertEqual((output / 'sentinel').read_bytes(), b'keep')


if __name__ == '__main__': unittest.main()
