"""Closed, model-free tests for the E2 operation graph contract."""
import copy
import unittest

from inference_kit.suites.suite_graph import ADAPTERS, validate_graph


def reference(name, *, external=False, arm=None, task="example_task", samples=None, identity=None):
    value = {"ref_id": name, "schema": "example_v1", "arm_id": arm,
             "task_id": task, "sample_ids": ["sample-1"] if samples is None else samples,
             "identity": {"prefix": "fixed-prefix"} if identity is None else identity}
    if external:
        value["external"] = True
    return value


def operation(name, kind="validate_prepared_inputs", *, dependencies=None, inputs=None,
              outputs=None, arm=None, task="example_task", samples=None, required=True,
              state="NOT_EXECUTED"):
    gpu = kind.startswith("serve_") or kind.startswith("collect_")
    return {"node_id": name, "operation_kind": kind, "adapter": ADAPTERS[kind],
            "adapter_version": "1.0", "dependencies": dependencies or [],
            "inputs": [reference(name + ":input", external=True, arm=arm, task=task, samples=samples)] if inputs is None else inputs,
            "outputs": outputs or [reference(name + ":output", arm=arm, task=task, samples=samples)],
            "arm_id": arm, "task_id": task,
            "sample_ids": ["sample-1"] if samples is None else samples,
            "required": required, "authorization": "NEEDS_GPU_AUTHORIZATION" if gpu else "CPU",
            "state": state, "counts": {"documents": 1},
            "failure_policy": "BLOCKED_DEPENDENCY", "output_contract": {"minimum_coverage": 1}}


def graph():
    source = operation("validate", inputs=[reference("prepared", external=True)])
    served = operation("serve", "serve_generate_until", dependencies=["validate"],
                       inputs=copy.deepcopy(source["outputs"]), arm="candidate")
    scored = operation("score", "score_task", dependencies=["serve"],
                       inputs=copy.deepcopy(served["outputs"]), arm="candidate")
    return [source, served, scored]


class SuiteGraphTests(unittest.TestCase):
    def test_valid_graph_is_data_only_and_topological(self):
        value = graph(); before = copy.deepcopy(value)
        result = validate_graph(value)
        self.assertEqual(value, before)
        self.assertEqual(result["status"], "VALIDATED_NOT_EXECUTED")
        self.assertEqual(result["topological_order"], ["validate", "serve", "score"])
        self.assertEqual(result["execution_count"], 0)
        self.assertEqual(result["acquisition_count"], 0)
        self.assertTrue(all(n["state"] == "NOT_EXECUTED" for n in result["nodes"]))

    def test_empty_graph_or_required_jobs_never_verifies(self):
        for value in ([], [operation("optional", required=False)]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "INVALID_EMPTY_PLAN"):
                validate_graph(value)

    def test_empty_required_gpu_samples_or_coverage_rejected(self):
        for field, bad in (("sample_ids", []), ("counts", {"documents": 0}),
                           ("output_contract", {"minimum_coverage": 0})):
            value = operation("serve", "serve_generate_until", arm="candidate")
            value[field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "INVALID_EMPTY_PLAN"):
                validate_graph([value])

    def test_duplicate_node_output_or_reference_is_rejected(self):
        variants = []
        value = graph(); value.append(copy.deepcopy(value[0])); variants.append(value)
        value = graph(); value[1]["outputs"] = copy.deepcopy(value[0]["outputs"]); variants.append(value)
        value = graph(); value[0]["inputs"] *= 2; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_graph(value)

    def test_missing_dependency_and_cycles_rejected(self):
        value = graph(); value[0]["dependencies"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Missing dependency"):
            validate_graph(value)
        value = graph(); value[0]["dependencies"] = ["score"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_graph(value)
        value = [operation("self", dependencies=["self"])]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_graph(value)

    def test_input_requires_producer_and_dependency(self):
        value = graph(); value[2]["inputs"][0]["ref_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "no producer"):
            validate_graph(value)
        value = graph(); value[2]["dependencies"] = ["validate"]
        with self.assertRaisesRegex(ValueError, "not a dependency"):
            validate_graph(value)

    def test_scorer_or_collector_cannot_be_planned_without_inputs(self):
        for kind in ("score_task", "collect_token_diagnostic", "analyze_tokens"):
            value = operation("empty", kind, arm="candidate", inputs=[])
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "requires input"):
                validate_graph([value])

    def test_preparation_authorization_without_blocked_state_rejected(self):
        value = operation("prepare"); value["authorization"] = "NEEDS_PREPARATION"
        with self.assertRaisesRegex(ValueError, "Preparation requirement"):
            validate_graph([value])

    def test_bool_number_identity_not_equal_and_external_identity_conflict_rejected(self):
        value = graph()
        value[1]["outputs"][0]["identity"] = {"token": 1}
        value[2]["inputs"][0]["identity"] = {"token": True}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_graph(value)
        left, right = operation("left"), operation("right")
        right["inputs"][0]["ref_id"] = left["inputs"][0]["ref_id"]
        right["inputs"][0]["identity"] = {"prefix": "changed"}
        with self.assertRaisesRegex(ValueError, "Conflicting external"):
            validate_graph([left, right])

    def test_transitive_dependency_is_valid(self):
        value = graph()
        value[2]["inputs"].append(copy.deepcopy(value[0]["outputs"][0]))
        self.assertEqual(validate_graph(value)["status"], "VALIDATED_NOT_EXECUTED")

    def test_scope_and_prefix_mismatch_rejected(self):
        for field, bad in (("arm_id", "baseline"), ("task_id", "other_task"),
                           ("sample_ids", ["other-sample"]), ("schema", "other_v1"),
                           ("identity", {"prefix": "different-prefix"})):
            value = graph(); value[2]["inputs"][0][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "scope|mismatch"):
                validate_graph(value)

    def test_external_reference_requires_identity_and_cannot_shadow_output(self):
        value = graph(); value[0]["inputs"][0].pop("identity")
        with self.assertRaisesRegex(ValueError, "requires identity"):
            validate_graph(value)
        value = graph(); value[0]["inputs"][0]["ref_id"] = "serve:output"
        with self.assertRaisesRegex(ValueError, "shadows"):
            validate_graph(value)

    def test_graph_failure_blocks_dependents_without_marking_execution(self):
        value = graph(); value[0]["state"] = "BLOCKED_INPUT"
        result = validate_graph(value)
        self.assertEqual(result["status"], "BLOCKED_REQUIRED_OPERATIONS")
        self.assertEqual([n["state"] for n in result["nodes"]],
                         ["BLOCKED_INPUT", "BLOCKED_DEPENDENCY", "BLOCKED_DEPENDENCY"])
        self.assertEqual(result["execution_count"], 0)

    def test_optional_blocked_node_does_not_fail_independent_required_work(self):
        value = [operation("required"), operation("optional", required=False,
                                                  state="UNSUPPORTED_CAPABILITY")]
        result = validate_graph(value)
        self.assertEqual(result["status"], "VALIDATED_NOT_EXECUTED")
        self.assertEqual(result["blocked_required_node_ids"], [])
        value[0]["dependencies"] = ["optional"]
        self.assertEqual(validate_graph(value)["status"], "BLOCKED_REQUIRED_OPERATIONS")

    def test_unknown_fields_dynamic_adapter_and_execution_status_rejected(self):
        for field, bad in (("shell", "echo unsafe"), ("adapter", "untrusted.module"),
                           ("state", "COMPLETED"), ("operation_kind", "arbitrary_python")):
            value = operation("node"); value[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_graph([value])
        value = operation("node"); value["outputs"][0]["python"] = "untrusted.module"
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            validate_graph([value])

    def test_invalid_types_nonfinite_counts_and_identity_rejected(self):
        for field, bad in (("required", 1), ("counts", {"documents": True}),
                           ("counts", {"documents": -1}), ("counts", {"documents": float("nan")}),
                           ("dependencies", "validate"), ("operation_kind", []),
                           ("authorization", []), ("sample_ids", [True])):
            value = operation("node"); value[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                validate_graph([value])
        value = operation("node"); value["outputs"][0]["identity"] = {"threshold": float("inf")}
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_graph([value])

    def test_gpu_operations_cannot_claim_cpu_authorization(self):
        for kind in ("serve_generate_until", "serve_choice_loglikelihood",
                     "collect_token_diagnostic", "collect_structure_diagnostic"):
            value = operation("gpu", kind, arm="candidate"); value["authorization"] = "CPU"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "GPU authorization"):
                validate_graph([value])

    def test_union_index_preserves_distinct_task_and_arm_references(self):
        left = operation("a", "serve_generate_until", arm="baseline", task="task_one")
        right = operation("b", "serve_choice_loglikelihood", arm="candidate", task="task_two",
                          samples=["sample-2"])
        index = operation("index", "index_add", dependencies=["a", "b"],
                          inputs=copy.deepcopy(left["outputs"] + right["outputs"]),
                          task=None, samples=["sample-1", "sample-2"])
        result = validate_graph([index, right, left])
        self.assertEqual(result["status"], "VALIDATED_NOT_EXECUTED")
        self.assertEqual(result["topological_order"], ["a", "b", "index"])

    def test_all_preparation_blocked_is_not_a_verified_empty_plan(self):
        value = operation("input", state="NEEDS_PREPARATION")
        value["authorization"] = "NEEDS_PREPARATION"
        result = validate_graph([value])
        self.assertEqual(result["status"], "BLOCKED_REQUIRED_OPERATIONS")
        self.assertEqual(result["blocked_required_node_ids"], ["input"])

    def test_required_validation_does_not_hide_all_optional_acquisitions_blocked(self):
        validation = operation("validate")
        acquire = operation("optional", "collect_token_diagnostic", arm="candidate", required=False,
                            dependencies=["validate"], inputs=copy.deepcopy(validation["outputs"]),
                            state="UNSUPPORTED_CAPABILITY")
        result = validate_graph([validation, acquire])
        self.assertEqual(result["status"], "BLOCKED_REQUIRED_OPERATIONS")
        self.assertEqual(result["required_node_count"], 1)
        self.assertEqual(result["blocked_required_node_ids"], [])
        self.assertEqual(result["blocked_acquisition_node_ids"], ["optional"])
        self.assertTrue(result["all_acquisition_operations_blocked"])

    def test_optional_acquisition_failure_does_not_block_valid_independent_work(self):
        blocked = operation("blocked", "collect_token_diagnostic", arm="candidate", required=False,
                            state="UNSUPPORTED_CAPABILITY")
        active = operation("supported", "collect_token_diagnostic", arm="baseline")
        result = validate_graph([blocked, active])
        self.assertEqual(result["status"], "VALIDATED_NOT_EXECUTED")
        self.assertFalse(result["all_acquisition_operations_blocked"])
        self.assertEqual(result["blocked_acquisition_node_ids"], ["blocked"])
        cpu = validate_graph([operation("cpu-only")])
        self.assertEqual(cpu["status"], "VALIDATED_NOT_EXECUTED")
        self.assertFalse(cpu["all_acquisition_operations_blocked"])
        self.assertEqual(cpu["blocked_acquisition_node_ids"], [])

    def test_independent_order_is_canonical_without_mutating_declared_inputs(self):
        a, b = operation("z-last"), operation("a-first")
        self.assertEqual(validate_graph([a, b]), validate_graph([b, a]))

    def test_unknown_reference_field_or_external_output_rejected(self):
        value = operation("node"); value["outputs"][0]["external"] = True
        with self.assertRaisesRegex(ValueError, "Output cannot be external"):
            validate_graph([value])
        value = operation("node"); value["outputs"][0]["source_path"] = "/private/model"
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            validate_graph([value])


if __name__ == "__main__":
    unittest.main()
