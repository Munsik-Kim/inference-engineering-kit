"""Static, escaped E1 reports. No server, network, SDK or archive code loader.

Only explicitly approved, bounded evidence copies become links. The canonical
index export is never normalized for display; path redaction and CSV formula
escaping apply solely to these derived presentation files.
"""
import csv
import hashlib
import html
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from ..core.io import require, safe_name, sha, write
from ..core.state import new_output


TABLES = ("sources", "acquisitions", "analyses", "samples", "executions",
          "measurements", "evidence", "relations")
CAUTIONS = (
    "과거 업무 결과와 별도 진단을 문항 단위로 연결함. 동일 실행의 품질·내부상태 인증이 아님.",
    "RTN 내부 관측: DECODED_WEIGHT_DIAGNOSTIC/effective BF16.",
    "OFF/OFF/ON control은 본 task 표본에 합산하지 않음.",
    "단일 모델 pair·재사용 입력의 분석이며 신규 TEST/배포/인과 검증이 아님.",
)
_PRIVATE_PATH = re.compile(
    r"(?<![\w])/(?:home|mnt|tmp|root|Users|private|var|opt)(?:/[^\s\"'<>\[\]{}()]+)+"
    r"|[A-Za-z]:[\\/][^\s\"'<>\[\]{}()]+")
_COPY_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".log", ".f32", ".bin", ".md"}


def _get(row, key, default=None):
    if key in row:
        return row[key]
    payload = row.get("payload")
    return payload.get(key, default) if isinstance(payload, dict) else default


def _identifier(row):
    return str(row.get("id", row.get("source_id", "UNKNOWN")))


def _redact(value):
    if isinstance(value, dict):
        return {_redact(str(k)): _redact(v) for k, v in value.items()
                if str(k) not in {"approved_copy", "source_root", "physical_locator", "locator", "root_map"}}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        if value.startswith("/") and not value.startswith("</"):
            return "[local path withheld]"
        return _PRIVATE_PATH.sub("[local path withheld]", value)
    return value


def _text(value):
    value = _redact(value)
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return str(value)


def _escape(value):
    return html.escape(_text(value), quote=True)


def _anchor(kind, identifier):
    return kind + "-" + hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()[:20]


def display_csv_cell(value):
    """Escape formula-capable display strings, retaining numeric negatives.

    Canonical JSON/raw and metric numbers are unchanged. Formula prefixes are
    tested after leading ASCII spaces, since spreadsheet readers can trim them.
    """
    if isinstance(value, str):
        if value.startswith(("\t", "\r")) or value.lstrip(" ").startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
    return value


def _bounded_file(root, relative):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), "Evidence root must be an explicit directory")
    relative = safe_name(str(relative))
    current = root
    for component in Path(relative).parts:
        current = current / component
        require(not current.is_symlink(), "Evidence symlink refused")
    require(current.resolve().is_relative_to(root.resolve()) and current.is_file(),
            "Missing or escaping evidence file")
    return current


def copy_evidence(source_root, relative_path, destination_root, portable_path,
                  expected_sha256=None, max_bytes=8 * 1024 * 1024):
    """Copy one parent-approved evidence file, never arbitrary inferred paths."""
    source = _bounded_file(source_root, relative_path)
    portable_path = safe_name(str(portable_path))
    require(Path(portable_path).parts[0] == "evidence", "Portable evidence must be under evidence/")
    require(source.suffix.lower() in _COPY_SUFFIXES, "Unsupported evidence file type")
    require(Path(portable_path).suffix.lower() == source.suffix.lower(),
            "Portable evidence extension must retain the approved source type")
    require(isinstance(max_bytes, int) and 0 < max_bytes <= 64 * 1024 * 1024,
            "Invalid evidence copy bound")
    require(source.stat().st_size <= max_bytes, "Evidence copy exceeds byte bound")
    identity = sha(source)
    require(expected_sha256 is not None and identity == expected_sha256,
            "Approved evidence SHA differs or is absent")
    target_root = Path(destination_root).absolute()
    require(target_root.is_dir() and not target_root.is_symlink(), "Missing report root")
    target = target_root
    for component in Path(portable_path).parts:
        target = target / component
        require(not target.is_symlink(), "Destination symlink refused")
    require(target.resolve().is_relative_to(target_root.resolve()) and not target.exists(),
            "Refuse overwrite or escaping evidence destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        incoming = os.fdopen(descriptor, 'rb')
        descriptor = None
        with incoming, target.open("xb") as outgoing:
            opened = os.fstat(incoming.fileno())
            require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) ==
                    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                    "Evidence changed before copy")
            copied_hash = hashlib.sha256()
            copied_bytes = 0
            while True:
                block = incoming.read(1024 * 1024)
                if not block: break
                copied_bytes += len(block)
                require(copied_bytes <= max_bytes, "Evidence grew beyond byte bound")
                copied_hash.update(block)
                outgoing.write(block)
    finally:
        if descriptor is not None: os.close(descriptor)
    after = source.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and
            copied_bytes == before.st_size and copied_hash.hexdigest() == identity and
            sha(target) == identity and sha(source) == identity,
            "Evidence changed during copy")
    return {"path": portable_path, "sha256": identity, "bytes": target.stat().st_size}


def _details(value):
    return '<details><summary>metadata</summary><pre>' + _escape(value) + "</pre></details>"


def _coordinates(row):
    coordinate = _get(row, "coordinates", {})
    if not isinstance(coordinate, dict):
        return coordinate
    keys = ("layer", "layer_index", "head", "query_head", "kv_head", "query_position", "query_roles",
            "pass_id", "reference_pass", "candidate_pass", "token_position", "target_t", "t",
            "logit_position", "target_token_id", "choice_index", "gold_choice", "reference_scope",
            "arm_id", "policy_id", "kind")
    return {key: coordinate[key] for key in keys if key in coordinate and coordinate[key] is not None}


def _measurement_details(row):
    """Keep repeated table rows bounded; complete payload remains in JSON export."""
    payload = row.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    keys = ("left_record_id", "right_record_id", "reference_observation_id", "candidate_observation_id",
            "observation_id", "raw_reference", "request_reference", "request_id", "reference_source",
            "reasons", "reason", "status", "evidence_ids", "evidence_refs", "termination", "finish_reason")
    result = {key: row[key] if key in row else payload[key] for key in keys if key in row or key in payload}
    return _details(result) if result else ""


def _quality_overview(rows, task):
    counts = defaultdict(list)
    responses = {}
    count_metrics = {"acc", "acc_norm", "task_success", "strict_task_success", "task_success_within_10s",
                     "strict_task_success_within_10s", "format_valid", "schema_valid", "strict_schema_valid",
                     "json_valid", "answer_exact", "evidence_exact", "transport_complete", "core_values_correct",
                     "evidence_correct_temperature", "evidence_correct_status"}
    for row in rows:
        coordinate = _get(row, "coordinates", {})
        coordinate = coordinate if isinstance(coordinate, dict) else {}
        arm = coordinate.get("arm_id", _get(row, "arm_id", "UNKNOWN"))
        metric = str(row.get("metric", _get(row, "metric_id", "UNKNOWN")))
        # ARC choice HTTP requests are not independent task documents. Their
        # individual timing stays in the detail table, outside this document
        # summary and outside the generation timing denominator.
        if metric.startswith(("native_choice.", "native_token.")):
            continue
        name = metric.split(".", 1)[-1]
        value = row.get("value")
        if name in count_metrics or name.startswith(("field_correct.", "evidence_correct.")):
            counts[(task(row), str(_get(row,"scope","UNKNOWN")), str(row.get("analysis_id","UNKNOWN")), str(arm), metric)].append(value)
        elif name in ("request_latency_http", "request_latency_task", "document_choice_http_sum_seconds", "document_serial_elapsed_seconds"):
            counts[(task(row), str(_get(row,"scope","UNKNOWN")), str(row.get("analysis_id","UNKNOWN")), str(arm), metric)].append(value)
        payload = row.get("payload", {})
        if isinstance(payload, dict) and "output_text" in payload:
            responses[(str(row.get("sample_id")), str(arm), coordinate.get("policy_id"))] = payload
    summary = []
    for (task_id, scope, analysis, arm, metric), values in sorted(counts.items()):
        present = [v for v in values if v is not None]
        latency = metric.rsplit(".", 1)[-1] in ("request_latency_http", "request_latency_task", "document_choice_http_sum_seconds", "document_serial_elapsed_seconds")
        if latency:
            result = {"observed_p50_seconds": statistics.median(present) if present else None,
                      "observed_max_seconds": max(present) if present else None}
        else:
            result = {"correct_or_true_count": sum(1 for value in present if value is True or value == 1),
                      "observed_row_denominator": len(values)}
        summary.append([_escape(task_id), _escape(scope), _escape(analysis), _escape(arm), _escape(metric), _escape(result),
                        str(len(present)), str(len(values) - len(present))])
    text = "<h3>기존 결과의 문항별 count·시간 요약</h3><p>분모는 이 export에 포함한 문항 수입니다. Slice의 수치를 전체24/64문항 결과로 확대하지 않습니다. Timing metric별 표본은 따로 유지합니다.</p>"
    text += _table(["Task", "Scope", "Analysis revision", "Arm", "Metric", "Count 또는 p50/max", "유효 n", "null n"], summary)
    if responses:
        text += "<h3>과거 자유 생성 응답</h3>" + _table(["문항", "Arm", "Policy", "원시 응답 텍스트", "종료"],
            [[_escape(key[0]), _escape(key[1]), _escape(key[2]), "<pre>" + _escape(value.get("output_text")) + "</pre>",
              _escape({k: value.get(k) for k in ("termination", "finish_reason")})] for key, value in sorted(responses.items(), key=lambda item: str(item[0]))])
    return text


def _table(headers, rows):
    return ("<div class=scroll><table><thead><tr>" +
            "".join("<th>" + html.escape(h) + "</th>" for h in headers) +
            "</tr></thead><tbody>" +
            "".join("<tr>" + "".join("<td>" + c + "</td>" for c in row) + "</tr>" for row in rows) +
            "</tbody></table></div>")


def _family(row):
    coordinate = _get(row, "coordinates", {})
    coordinate = coordinate if isinstance(coordinate, dict) else {}
    description = " ".join(str(_get(row, k, "")) for k in ("metric", "scope", "kind", "family", "metric_family")).lower()
    pass_id = str(coordinate.get("pass_id", _get(row, "pass_id", ""))).upper()
    if pass_id in {"OFF1", "OFF2"} or "observer" in description or "control" in description:
        return "control"
    if "projection" in description or "projection" in str(coordinate.get("kind", "")):
        return "projection"
    if str(_get(row, "scope", "")) == "historical_task":
        return "quality"
    if any(k in coordinate for k in ("layer", "layer_index", "query_head", "head", "query_position")) or any(k in description for k in ("attention", "reconstructed", "native_av", "sdpa")):
        return "structure"
    if any(k in coordinate for k in ("target_position", "target_t", "logit_position", "t")) or any(k in description for k in ("nll", "logprob", "token", "entropy", "top_k", "top1", "rank", "kl", "js")):
        return "token"
    if any(k in description for k in ("acc", "task_success", "strict", "answer", "evidence_exact", "format", "quality", "latency", "transport", "field_correct", "correct", "temperature", "status_enum")):
        return "quality"
    return "other"


def _bounded_write(stream, text, budget=None):
    if budget is not None:
        count = len(text.encode('utf-8'))
        require(budget[1] + count <= budget[0], 'Report presentation byte cap exceeded')
        budget[1] += count
    return stream.write(text)


def _bounded_json(path, value, budget=None):
    if budget is None:
        write(path, value)
    else:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            _bounded_write(stream, text, budget)


class _CSVWriterBudget:
    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def write(self, value):
        return _bounded_write(self.stream, value, self.budget)


def render_report(export, output, packaging=None, *, _presentation_budget=None):
    """Render an immutable offline report from a validated canonical export.

    This presentation layer does not certify numerical comparability or raw
    validity. It displays the statuses/relations supplied by the index validator.
    """
    require(isinstance(export, dict), "Expected index export object")
    for table in TABLES:
        require(isinstance(export.get(table, []), list), "Expected export table list: " + table)
        require(all(isinstance(row, dict) for row in export.get(table, [])), "Expected export row objects")
    from .report_packaging import validate_report_packaging, render_collection
    policy = validate_report_packaging(packaging)
    if policy['mode'] == 'bounded_collection_v1':
        return render_collection(export, output, policy)
    destination = new_output(output)
    copies = {}
    for row in export.get("evidence", []):
        declaration = row.get("approved_copy")
        if declaration is None:
            continue
        require(isinstance(declaration, dict), "Invalid approved evidence copy")
        copied = copy_evidence(declaration["source_root"], declaration["relative_path"], destination,
                               declaration["portable_path"], declaration.get("sha256"),
                               declaration.get("max_bytes", 8 * 1024 * 1024))
        copies[_identifier(row)] = copied

    sample_rows = {_identifier(row): row for row in export.get("samples", [])}
    execution_rows = {_identifier(row): row for row in export.get("executions", [])}
    evidence_rows = {_identifier(row): row for row in export.get("evidence", [])}
    task_counts = Counter()
    groups = defaultdict(list)

    def task(row):
        sample = sample_rows.get(str(row.get("sample_id")), {})
        return str(_get(row, "task_id", _get(sample, "task_id", "UNKNOWN")))

    def evidence_link(identifier):
        identifier = str(identifier)
        if identifier in copies:
            return '<a href="' + quote(copies[identifier]["path"], safe="/") + '">' + _escape(identifier) + "</a>"
        if identifier not in evidence_rows:
            return _escape(identifier) + " / MISSING_EVIDENCE_REFERENCE"
        return '<a href="#' + _anchor("evidence", identifier) + '">' + _escape(identifier) + "</a>"

    for row in export.get("measurements", []):
        groups[_family(row)].append(row)
        task_counts[task(row)] += 1

    current_suite=any(str(_get(r,"scope",""))=="current_task" for r in export.get("measurements",[]))
    fragments = ["<!doctype html><html lang=ko><head><meta charset=utf-8>",
                 '<meta name="viewport" content="width=device-width,initial-scale=1">',
                 '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'none\'; base-uri \'none\'; form-action \'none\'">',
                 "<title>BENCH05-E1 로컬 근거 보고서</title>",
                 "<style>body{font:15px system-ui,sans-serif;margin:2rem;line-height:1.5;color:#18202a;background:#fff}h1,h2,h3{line-height:1.25}table{border-collapse:collapse;font-size:13px;min-width:100%}td,th{border:1px solid #ccd3da;padding:.45rem;text-align:left;vertical-align:top}th{background:#eef2f6}a{color:#1655a1}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:75rem}.scroll{overflow:auto;margin:1rem 0}section{margin:2rem 0}.caution{padding:1rem;background:#fff6da;border:1px solid #d5b65c}code{overflow-wrap:anywhere}.status{font-weight:600}small{color:#485768}</style></head><body>",
                 "<h1>BENCH05-E1 로컬 근거 보고서</h1><p>정적 HTML · 외부 요청 없음 · 신규 GPU/업무 품질 측정 없음</p>",
                 '<div class=caution>' + "".join("<p>" + html.escape(c) + "</p>" for c in CAUTIONS) + "</div>",
                 '<nav><a href="#runs">Run·검증</a> | <a href="#tasks">Task·문항</a> | <a href="#quality">과거 품질·시간</a> | <a href="#token">Token</a> | <a href="#structure">Structure·AV</a> | <a href="#projection">Output projection</a> | <a href="#control">Control</a> | <a href="#issues">오류·비교 조건</a> | <a href="#evidence">근거</a></nav>',
                 '<section id=runs><h2>Run·검증 개요</h2><p>Acquisition과 analysis revision은 별도이며, DB는 원시 근거를 대체하지 않는 재생성 가능한 index입니다.</p>']
    for key, title in (("sources", "Source"), ("acquisitions", "Acquisition"), ("analyses", "Analysis revision"), ("executions", "Execution")):
        fragments.append("<h3>" + title + "</h3>" + _table(["ID", "상태", "metadata"],
                         [['<span id="' + _anchor("execution" if key == "executions" else key, _identifier(r)) + '">' + _escape(_identifier(r)) + "</span>",
                           _escape(_get(r, "status", r.get("validation", {}).get("status", "UNKNOWN"))), _details(r)] for r in export.get(key, [])]))
    fragments.append("</section><section id=tasks><h2>Task → 문항</h2>" + _table(["Task", "측정 row 수"],
                     [['<a href="#' + _anchor("task", t) + '">' + _escape(t) + "</a>", str(n)] for t, n in sorted(task_counts.items())]))
    by_task = defaultdict(list)
    for identifier, row in sample_rows.items():
        by_task[str(_get(row, "task_id", "UNKNOWN"))].append((identifier, row))
    for task_id, samples in sorted(by_task.items()):
        fragments.append('<h3 id="' + _anchor("task", task_id) + '">' + _escape(task_id) + "</h3>")
        fragments.append(_table(["문항", "Source ID", "dataset / split / policy", "reference / metadata"],
                          [['<span id="' + _anchor("sample", identifier) + '">' + _escape(identifier) + "</span>",
                            _escape(_get(row, "source_id")), _escape({k: _get(row, k) for k in ("dataset_id", "split", "policy_id")}),
                            _details(row)] for identifier, row in samples]))
    fragments.append("</section>")
    if current_suite:
        fragments=[part.replace("BENCH05-E1 로컬 근거 보고서","BENCH05 Suite 로컬 근거 보고서").replace("신규 GPU/업무 품질 측정 없음","보고서 생성은 CPU 재계산; 각 source의 acquisition 시점·scope를 따름").replace("과거 품질·시간","Task 품질·시간") for part in fragments]
    csv_rows = []
    sections = (("quality", "Task 품질·응답·시간" if current_suite else "과거 task 품질·응답·시간", "Scope와 analysis revision별로 분모를 분리합니다. 과거 결과와 E3 재사용2문항 관측을 합산하지 않습니다." if current_suite else "원래 task·분모·시간 경계를 유지합니다. D1의 신규 task quality는 NOT_EVALUATED입니다."),
                ("token", "Token", "동일 prefix에서 검증된 pair만 해석합니다. 선택 target/full coverage와 gold/all-choice scope는 별도입니다."),
                ("structure", "Structure: attention·AV", "Task → sample → layer/head/query 좌표를 유지합니다. 같은-arm reconstructed/native AV와 모델 pair 차이는 별개입니다."),
                ("projection", "Output projection: head 축 없음", "Layer/query 벡터이며 두 head에 복제한 독립 관측으로 세지 않습니다."),
                ("control", "OFF1/OFF2/ON control", "Control은 본 task 표본이나 ON token 분모에 합산하지 않습니다. 검사 좌표에서의 차이는 전체 모델 무간섭 인증이 아닙니다."),
                ("other", "기타 관측·지원 범위", "누락·미지원·비교 불가 상태는 null 및 이유로 유지합니다."))
    for family, title, note in sections:
        fragments.append('<section id="' + family + '"><h2>' + title + "</h2><p>" + html.escape(note) + "</p>")
        if family == "quality":
            fragments.append(_quality_overview(groups[family], task))
        rows = []
        for row in groups[family]:
            sample_id = str(row.get("sample_id", "UNKNOWN"))
            task_id = task(row)
            execution_id = str(row.get("execution_id"))
            execution = execution_rows.get(execution_id, {})
            refs = row.get("evidence_ids", _get(row, "evidence_refs", []))
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list):
                refs = []
            ev = "<br>".join(evidence_link(r) for r in refs)
            task_link = '<a href="#' + _anchor("task", task_id) + '">' + _escape(task_id) + "</a>" if task_id in by_task else _escape(task_id)
            sample_link = '<a href="#' + _anchor("sample", sample_id) + '">' + _escape(sample_id) + "</a>" if sample_id in sample_rows else _escape(sample_id)
            execution_link = '<a href="#' + _anchor("execution", execution_id) + '">' + _escape(_get(execution, "arm_id", execution_id)) + "</a>" if execution else _escape(_get(row, "arm_id", "pair / UNKNOWN"))
            rows.append([task_link, sample_link,
                         _escape(row.get("metric", _get(row, "metric_id", "UNKNOWN"))),
                         '<span class=status>' + _escape(row.get("status", "UNKNOWN")) + "</span>",
                         _escape(row.get("value")), _escape(row.get("scope")),
                         _escape(_coordinates(row)), execution_link,
                         ev + _measurement_details(row)])
            csv_rows.append([family, task_id, sample_id, _text(row.get("metric", _get(row, "metric_id", "UNKNOWN"))),
                             row.get("status", "UNKNOWN"), row.get("value"), _text(row.get("scope")),
                             _text(_get(row, "coordinates", {})), _identifier(row)])
        fragments.append(_table(["Task", "문항", "Metric", "상태", "값", "Scope·분모", "Layer/head/query·token 좌표", "Execution", "원시 근거·metadata"], rows))
        if not rows:
            fragments.append("<p>이 export에서 해당 측정 없음. 정상값 0을 의미하지 않습니다.</p>")
        fragments.append("</section>")
    fragments.append('<section id=issues><h2>근거 상태·오류·비교 가능성</h2><p>관계로 연결하는 것과 수치 비교를 허용하는 것은 다릅니다. INCOMPARABLE/MISSING/INVALID 상태는 정상 paired 계산으로 승격하지 않습니다.</p>')
    fragments.append(_table(["관계 ID", "종류·상태", "근거·이유"],
                     [[_escape(_identifier(r)), _escape(_get(r, "kind", _get(r, "relation_type", "UNKNOWN"))) + " / " + _escape(_get(r, "status", "UNKNOWN")), _details(r)] for r in export.get("relations", [])]))
    for key in ("issues", "compatibility", "coverage", "review"):
        if key in export:
            fragments.append("<h3>" + html.escape(key) + "</h3>" + _details(export[key]))
    fragments.append('</section><section id=evidence><h2>원시 근거</h2><p>승인된 portable copy만 상대 링크로 제공합니다. 나머지는 로컬 외부 근거이며, 경로를 자동 열거나 복사하지 않습니다.</p>')
    fragments.append(_table(["ID", "상태·파일", "SHA / observation / locator metadata"],
                     [['<span id="' + _anchor("evidence", identifier) + '">' + _escape(identifier) + "</span>",
                       evidence_link(identifier) if identifier in copies else "로컬 외부 근거 / " + _escape(_get(row, "status", "UNKNOWN")),
                       _details(row)] for identifier, row in evidence_rows.items()]))
    fragments.append("</section><footer><p>브라우저 시각 확인: NOT_RUN. HTML parse·escape·로컬 링크 검사는 별도 receipt 범위입니다. CSV는 표시용 formula escaping을 적용하며 canonical JSON/raw는 변경하지 않습니다.</p></footer></body></html>")
    with (destination / "index.html").open("x", encoding="utf-8", newline="\n") as stream:
        _bounded_write(stream, "\n".join(fragments) + "\n", _presentation_budget)
    with (destination / "display.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(_CSVWriterBudget(stream, _presentation_budget))
        writer.writerow(["family", "task_id", "sample_id", "metric", "status", "value", "scope", "coordinates", "measurement_id"])
        for row in csv_rows:
            writer.writerow([display_csv_cell(_text(v) if isinstance(v, (dict, list)) else _redact(v)) for v in row])
    result = {"schema_version": 1, "kind": "offline_evidence_report_v1", "entrypoint": "index.html",
              "table_counts": {key: len(export.get(key, [])) for key in TABLES},
              "measurement_families": {key: len(value) for key, value in groups.items()},
              "portable_evidence": copies, "network_requests": 0, "scripts": 0,
              "browser_visual_validation": "NOT_RUN",
              "display_transformations": ["HTML text/attribute escaping", "local absolute path withholding", "display-only CSV formula prefix escaping"],
              "files": {name: {"sha256": sha(destination / name), "bytes": (destination / name).stat().st_size}
                        for name in ("index.html", "display.csv")}}
    _bounded_json(destination / "REPORT_MANIFEST.json", result, _presentation_budget)
    return result
