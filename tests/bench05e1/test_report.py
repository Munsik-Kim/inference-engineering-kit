import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest

import inference_kit

from inference_kit.analysis.index_report import (
    CAUTIONS, copy_evidence, display_csv_cell, render_report,
)


class ParsedReport(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.ids = set()
        self.links = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "href" in attrs:
            self.links.append(attrs["href"])
        if tag not in {"meta", "style"}:
            assert not any(key.startswith("on") for key in attrs), attrs

    def handle_data(self, data):
        self.text.append(data)


def report_fixture():
    return {
        "sources": [{"id": "source-a", "validation": {"status": "VERIFIED"},
                     "payload": {"physical_locator": "/home/private/secret", "message": "한글"}}],
        "acquisitions": [{"id": "acq-old", "payload": {"status": "HISTORICAL"}},
                         {"id": "acq-new", "payload": {"status": "DIAGNOSTIC"}}],
        "analyses": [{"id": "analysis-1", "source_id": "source-a"}],
        "samples": [{"id": "sample-a", "task_id": "context_evidence_retrieval", "source_id": "원문"}],
        "executions": [{"id": "execution-a", "payload": {"method": "DECODED_WEIGHT_DIAGNOSTIC"}}],
        "measurements": [
            {"id": "quality", "sample_id": "sample-a", "execution_id": "execution-a",
             "metric": "task_success", "value": 0, "status": "MEASURED", "scope": {"planned": 24}},
            {"id": "token", "sample_id": "sample-a", "metric": "target_nll", "value": -0.25,
             "status": "MEASURED", "coordinates": {"t": 9, "logit_position": 8}, "evidence_ids": ["raw-a"]},
            {"id": "row", "sample_id": "sample-a", "metric": "attention_JS", "value": None,
             "status": "MISSING_SPAN_METADATA", "coordinates": {"layer": 12, "head": 16, "query_position": 8}},
            {"id": "projection", "sample_id": "sample-a", "metric": "output_projection_nmse", "value": 2.0,
             "status": "MEASURED", "coordinates": {"layer": 12, "query_position": 8}},
            {"id": "control", "sample_id": "sample-a", "metric": "observer_difference", "value": 0.0,
             "status": "MEASURED", "coordinates": {"pass_id": "OFF2"}},
        ],
        "evidence": [{"id": "raw-a", "payload": {"sha256": "a" * 64, "path": "raw/한글.jsonl"}}],
        "relations": [{"id": "relation-a", "kind": "HISTORICAL_OUTCOME_LINK", "status": "RELATED_ONLY",
                       "payload": {"reason": "different execution"}}],
        "issues": [{"classification": "EVIDENCE_GAP", "message": "QKV not stored"}],
        "compatibility": [{"status": "INCOMPARABLE", "reason": "prefix differs"}],
    }


class OfflineReportTests(unittest.TestCase):
    def test_static_escape_and_required_boundaries(self):
        fixture = report_fixture()
        attack = '<script>alert(1)</script><img src=x onerror="alert(2)"></script>'
        fixture["samples"][0]["payload"] = {"model_output": attack,
                                               "message": "at /home/private/credentials/token"}
        before = copy.deepcopy(fixture)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "보고서"
            receipt = render_report(fixture, output)
            content = (output / "index.html").read_text(encoding="utf-8")
            parser = ParsedReport()
            parser.feed(content)
            self.assertNotIn("script", parser.tags)
            self.assertNotIn("img", parser.tags)
            # Metadata is JSON text; its quote escaping is preserved rather than
            # interpreted as HTML attributes or executable model output.
            self.assertIn(json.dumps(attack, ensure_ascii=False)[1:-1], "".join(parser.text))
            self.assertNotIn("/home/private", content)
            self.assertNotIn("https://", content)
            self.assertNotIn("fetch(", content)
            self.assertNotIn("innerHTML", content)
            self.assertTrue(all(caution in content for caution in CAUTIONS))
            for link in parser.links:
                self.assertTrue(link.startswith("#"))
                self.assertIn(link[1:], parser.ids)
            self.assertEqual(receipt["scripts"], 0)
            self.assertEqual(receipt["network_requests"], 0)
            self.assertEqual(receipt["browser_visual_validation"], "NOT_RUN")
            self.assertEqual(fixture, before)

    def test_family_separation_null_and_headless_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            receipt = render_report(report_fixture(), output)
            self.assertEqual(receipt["measurement_families"],
                             {"quality": 1, "token": 1, "structure": 1, "projection": 1, "control": 1, "other": 0})
            with (output / "display.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            projection = next(row for row in rows if row["family"] == "projection")
            self.assertNotIn("head", json.loads(projection["coordinates"]))
            missing = next(row for row in rows if row["family"] == "structure")
            self.assertEqual(missing["status"], "MISSING_SPAN_METADATA")
            self.assertEqual(missing["value"], "")
            self.assertEqual(next(row for row in rows if row["family"] == "quality")["value"], "0")

    def test_csv_formula_escaping_does_not_change_numeric_negatives(self):
        for value in ("=1+2", "+cmd", "-text", "@A1", "\tformula", "\rformula", "  =1", " -1"):
            self.assertEqual(display_csv_cell(value), "'" + value)
        self.assertEqual(display_csv_cell(-1.25), -1.25)
        self.assertEqual(display_csv_cell(0), 0)
        self.assertEqual(display_csv_cell("NOT_FOUND"), "NOT_FOUND")

    def test_new_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            render_report(report_fixture(), output)
            before = (output / "index.html").read_bytes()
            with self.assertRaises(ValueError):
                render_report(report_fixture(), output)
            self.assertEqual((output / "index.html").read_bytes(), before)

    def test_approved_evidence_copy_and_link_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "source"
            evidence.mkdir()
            raw = evidence / "원문.jsonl"
            raw.write_text('{"output":"<script>한글</script>"}\n', encoding="utf-8")
            identity = hashlib.sha256(raw.read_bytes()).hexdigest()
            fixture = report_fixture()
            fixture["evidence"][0]["approved_copy"] = {
                "source_root": str(evidence), "relative_path": raw.name,
                "portable_path": "evidence/원문.jsonl", "sha256": identity,
            }
            output = root / "report"
            receipt = render_report(fixture, output)
            self.assertEqual((output / "evidence/원문.jsonl").read_bytes(), raw.read_bytes())
            self.assertEqual(receipt["portable_evidence"]["raw-a"]["sha256"], identity)
            content = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn('href="evidence/%EC%9B%90%EB%AC%B8.jsonl"', content)
            self.assertNotIn(str(evidence), content)

    def test_missing_evidence_reference_is_visible_without_broken_link(self):
        fixture = report_fixture()
        fixture["measurements"][1]["evidence_ids"] = ["unknown"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            render_report(fixture, output)
            content = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("MISSING_EVIDENCE_REFERENCE", content)
            parser = ParsedReport()
            parser.feed(content)
            self.assertTrue(all(link[1:] in parser.ids for link in parser.links if link.startswith("#")))

    def test_copy_rejects_traversal_symlink_wrong_hash_and_executable_type(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "data.json").write_text("{}", encoding="utf-8")
            identity = hashlib.sha256(b"{}").hexdigest()
            (source / "link.json").symlink_to(source / "data.json")
            (source / "code.py").write_text("raise RuntimeError()", encoding="utf-8")
            for relative, portable, expected in (("../source/data.json", "evidence/data.json", identity),
                                                 ("link.json", "evidence/link.json", identity),
                                                 ("data.json", "../escape.json", identity),
                                                 ("data.json", "outside.json", identity),
                                                 ("data.json", "evidence/wrong.json", "0" * 64),
                                                 ("data.json", "evidence/missing.json", None),
                                                 ("data.json", "evidence/unsafe.html", identity),
                                                 ("code.py", "evidence/code.py", identity)):
                with self.subTest(relative=relative, portable=portable, expected=expected):
                    with self.assertRaises(ValueError):
                        copy_evidence(source, relative, destination, portable, expected)

    def test_copy_rejects_oversize_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "data.json").write_bytes(b"{}")
            identity = hashlib.sha256(b"{}").hexdigest()
            with self.assertRaises(ValueError):
                copy_evidence(source, "data.json", destination, "evidence/data.json", identity, 1)
            copy_evidence(source, "data.json", destination, "evidence/data.json", identity)
            with self.assertRaises(ValueError):
                copy_evidence(source, "data.json", destination, "evidence/data.json", identity)

    def test_export_requires_lists_and_objects(self):
        for value in (None, {"measurements": {}}, {"samples": [1]}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    render_report(value, Path(temporary) / "report")

    def test_report_utf8_under_actual_non_utf8_locale_without_ml_or_network(self):
        package_root = str(Path(inference_kit.__file__).absolute().parent.parent)
        code = (
            "import sys,locale,json;sys.path.insert(0," + repr(package_root) + ");\n"
            "assert locale.getpreferredencoding(False).lower() not in ('utf-8','utf8')\n"
            "def guard(event, args):\n"
            " if event == 'import' and args and str(args[0]).split('.')[0] in "
            "('torch','vllm','transformers','numpy','aiohttp','compressed_tensors'): "
            "raise AssertionError('ML SDK forbidden')\n"
            " if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'): "
            "raise AssertionError('network/subprocess forbidden')\n"
            "sys.addaudithook(guard)\n"
            "from pathlib import Path\n"
            "from inference_kit.core.io import read\n"
            "from inference_kit.analysis.index_report import render_report\n"
            "result=render_report(read(Path('input.json')),Path('report'))\n"
            "print(json.dumps(result,ensure_ascii=True))\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.json").write_text(json.dumps(report_fixture(), ensure_ascii=False), encoding="utf-8")
            env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONDONTWRITEBYTECODE="1")
            for key in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(key, None)
            completed = subprocess.run([sys.executable, "-I", "-X", "utf8=0", "-B", "-c", code],
                                       cwd=root, env=env, capture_output=True, encoding="utf-8")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["network_requests"], 0)
            self.assertIn("한글", (root / "report/index.html").read_text(encoding="utf-8"))
            with (root / "report/display.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)


if __name__ == "__main__":
    unittest.main()
