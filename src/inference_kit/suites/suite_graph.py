"""Strict, data-only operation graphs for the E2 suite compiler.

This module validates dependencies and logical result contracts.  It neither
imports execution adapters nor runs operations; a validated GPU operation still
needs a new authorization and runtime preflight in a future executor.
"""
import copy
import math
import re

from inference_kit.core.io import digest, fields, require


ADAPTERS = {
    "validate_prepared_inputs": "prepared_manifest",
    "serve_generate_until": "task_executor",
    "serve_choice_loglikelihood": "task_executor",
    "collect_token_diagnostic": "token_worker",
    "collect_structure_diagnostic": "structure_worker",
    "score_task": "task_adapter",
    "analyze_tokens": "token_diagnostics",
    "analyze_structure": "structure",
    "index_add": "result_index",
    "report": "index_report",
}
GPU_OPERATIONS = frozenset({
    "serve_generate_until", "serve_choice_loglikelihood",
    "collect_token_diagnostic", "collect_structure_diagnostic",
})
NODE_FIELDS = frozenset({
    "node_id", "operation_kind", "adapter", "adapter_version", "dependencies",
    "inputs", "outputs", "arm_id", "task_id", "sample_ids", "required",
    "authorization", "state", "counts", "failure_policy", "output_contract",
})
REF_FIELDS = frozenset({"ref_id", "schema"})
REF_OPTIONAL_FIELDS = frozenset({
    "arm_id", "task_id", "sample_ids", "identity", "external",
})
STATES = frozenset({
    "NOT_EXECUTED", "BLOCKED_INPUT", "BLOCKED_DEPENDENCY",
    "UNSUPPORTED_CAPABILITY", "NEEDS_PREPARATION",
})
AUTHORIZATIONS = frozenset({"CPU", "NEEDS_GPU_AUTHORIZATION", "NEEDS_PREPARATION"})


def _text(value, label, nullable=False):
    if nullable and value is None:
        return
    require(isinstance(value, str) and bool(value) and value == value.strip()
            and not any(ord(c) < 32 for c in value), "Invalid " + label)


def _ids(value, label, allow_empty=True):
    require(isinstance(value, list), label + " must be a list")
    require(allow_empty or bool(value), "INVALID_EMPTY_PLAN: " + label)
    for item in value:
        _text(item, label)
    require(len(value) == len(set(value)), "Duplicate " + label)


def _json(value, label):
    """Reject non-JSON and non-finite values without coercing identity data."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        require(math.isfinite(value), "Non-finite " + label)
        return
    if isinstance(value, list):
        for item in value:
            _json(item, label)
        return
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "Invalid JSON key in " + label)
        for item in value.values():
            _json(item, label)
        return
    raise ValueError("Non-JSON value in " + label)


def _ref(value, node, output=False):
    fields(value, REF_FIELDS, REF_OPTIONAL_FIELDS)
    _text(value["ref_id"], "logical reference ID")
    _text(value["schema"], "logical reference schema")
    for name in ("arm_id", "task_id"):
        if name in value:
            _text(value[name], "reference " + name, nullable=True)
            if value[name] is not None and node[name] is not None:
                require(value[name] == node[name], "Reference " + name + " outside node scope")
    if "sample_ids" in value:
        _ids(value["sample_ids"], "reference sample IDs")
        require(set(value["sample_ids"]) <= set(node["sample_ids"]),
                "Reference samples outside node scope")
    if "identity" in value:
        require(isinstance(value["identity"], (str, dict)) and bool(value["identity"]),
                "Empty logical reference identity")
        _json(value["identity"], "logical reference identity")
    if "external" in value:
        require(type(value["external"]) is bool, "external must be a boolean")
    require(not output or not value.get("external", False), "Output cannot be external")
    if value.get("external", False):
        require(bool(value.get("identity")), "External reference requires identity")


def _node(value):
    fields(value, NODE_FIELDS)
    _text(value["node_id"], "node ID")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value["node_id"])),
            "Unsafe node ID")
    require(isinstance(value["operation_kind"], str) and value["operation_kind"] in ADAPTERS,
            "Unknown operation kind")
    require(value["adapter"] == ADAPTERS[value["operation_kind"]],
            "Unsupported internal adapter for operation")
    _text(value["adapter_version"], "adapter version")
    _ids(value["dependencies"], "dependencies")
    _ids(value["sample_ids"], "sample IDs",
         allow_empty=value["operation_kind"] not in GPU_OPERATIONS | {"score_task"})
    for name in ("arm_id", "task_id"):
        _text(value[name], name, nullable=True)
    require(type(value["required"]) is bool, "required must be a boolean")
    require(isinstance(value["authorization"], str) and value["authorization"] in AUTHORIZATIONS,
            "Unknown authorization")
    state = value["state"]
    require(isinstance(state, str) and (state in STATES or
            bool(re.fullmatch(r"BLOCKED_[A-Z0-9_]+", state))), "Unknown operation state")
    require(value["failure_policy"] == "BLOCKED_DEPENDENCY", "Unsupported failure policy")
    require(isinstance(value["counts"], dict), "counts must be an object")
    for key, count in value["counts"].items():
        _text(key, "count key")
        require(type(count) is int and count >= 0, "Counts must be nonnegative integers")
    require(isinstance(value["output_contract"], dict) and bool(value["output_contract"]),
            "Missing output contract")
    _json(value["output_contract"], "output contract")
    if value["operation_kind"] in GPU_OPERATIONS:
        require(value["authorization"] in {"NEEDS_GPU_AUTHORIZATION", "NEEDS_PREPARATION"},
                "GPU operation requires new GPU authorization")
        require(value["arm_id"] is not None, "GPU operation requires model arm")
        coverage = value["output_contract"].get("minimum_coverage")
        require(type(coverage) is int and coverage > 0,
                "INVALID_EMPTY_PLAN: GPU output requires positive minimum coverage")
        require(any(value["counts"].values()), "INVALID_EMPTY_PLAN: empty GPU work")
    else:
        require(value["authorization"] in {"CPU", "NEEDS_PREPARATION"},
                "CPU operation has invalid authorization")
    require(isinstance(value["inputs"], list) and isinstance(value["outputs"], list),
            "inputs and outputs must be lists")
    require(bool(value["inputs"]), "Node requires input contract reference")
    require(bool(value["outputs"]), "Node has no output contract reference")
    if value["authorization"] == "NEEDS_PREPARATION":
        require(state != "NOT_EXECUTED", "Preparation requirement must block operation")
    for output, refs in ((False, value["inputs"]), (True, value["outputs"])):
        for ref in refs:
            _ref(ref, value, output)
        require(len(refs) == len({r["ref_id"] for r in refs}), "Duplicate logical references in node")


def validate_graph(nodes):
    """Validate and topologically describe operations, without executing them.

    ``inputs`` refer either to a verified external identity or to a producer in
    this graph.  Internal references must retain producer schema and declared
    scope/identity, and that producer must be an ancestor dependency.  A blocked
    prerequisite blocks its dependents.  No status returned here means a model
    has executed or that a GPU is presently available.
    """
    require(isinstance(nodes, list) and bool(nodes), "INVALID_EMPTY_PLAN: empty operations")
    result = copy.deepcopy(nodes)
    for node in result:
        _node(node)
    by_id = {node["node_id"]: node for node in result}
    require(len(by_id) == len(result), "Duplicate node ID")
    require(any(node["required"] for node in result), "INVALID_EMPTY_PLAN: no required jobs")
    producers, external_refs = {}, {}
    for node in result:
        require(set(node["dependencies"]) <= by_id.keys(), "Missing dependency")
        require(node["node_id"] not in node["dependencies"], "Dependency cycle")
        for ref in node["outputs"]:
            require(ref["ref_id"] not in producers, "Duplicate output reference")
            producers[ref["ref_id"]] = (node["node_id"], ref)
        for ref in node["inputs"]:
            if ref.get("external", False):
                old = external_refs.setdefault(ref["ref_id"], ref)
                require(digest(old) == digest(ref), "Conflicting external reference identity")
    order, visiting, done, ancestors = [], set(), set(), {}

    def visit(node_id):
        require(node_id not in visiting, "Dependency cycle")
        if node_id in done:
            return
        visiting.add(node_id)
        inherited = set()
        for parent in sorted(by_id[node_id]["dependencies"]):
            visit(parent)
            inherited.add(parent)
            inherited.update(ancestors[parent])
        visiting.remove(node_id)
        done.add(node_id)
        ancestors[node_id] = inherited
        order.append(node_id)

    for node_id in sorted(by_id):
        visit(node_id)
    for node_id in order:
        node = by_id[node_id]
        for ref in node["inputs"]:
            if ref.get("external", False):
                require(ref["ref_id"] not in producers, "External reference shadows graph output")
                continue
            require(ref["ref_id"] in producers, "Input reference has no producer")
            producer_id, output = producers[ref["ref_id"]]
            require(producer_id in ancestors[node_id], "Input producer is not a dependency")
            for name in ("schema", "arm_id", "task_id", "sample_ids", "identity"):
                require(digest(ref.get(name)) == digest(output.get(name)),
                        "Input/output " + name + " mismatch")
        if any(by_id[parent]["state"] != "NOT_EXECUTED" for parent in node["dependencies"]):
            node["state"] = "BLOCKED_DEPENDENCY"
    blocked = [node_id for node_id in order
               if by_id[node_id]["required"] and by_id[node_id]["state"] != "NOT_EXECUTED"]
    acquisition_ids = [node_id for node_id in order
                       if by_id[node_id]["operation_kind"] in GPU_OPERATIONS]
    blocked_acquisitions = [node_id for node_id in acquisition_ids
                            if by_id[node_id]["state"] != "NOT_EXECUTED"]
    # Required CPU validation alone cannot make an entirely blocked acquisition
    # proposal executable.  Optional flags preserve their declared meanings;
    # they do not convert unsupported model work into a ready suite.
    all_acquisitions_blocked = bool(acquisition_ids) and len(blocked_acquisitions) == len(acquisition_ids)
    return {
        "schema_version": 1,
        "status": "BLOCKED_REQUIRED_OPERATIONS" if blocked or all_acquisitions_blocked else "VALIDATED_NOT_EXECUTED",
        "topological_order": order,
        "nodes": [by_id[node_id] for node_id in order],
        "required_node_count": sum(node["required"] for node in result),
        "blocked_required_node_ids": blocked,
        "blocked_acquisition_node_ids": blocked_acquisitions,
        "all_acquisition_operations_blocked": all_acquisitions_blocked,
        "execution_count": 0,
        "acquisition_count": 0,
    }
