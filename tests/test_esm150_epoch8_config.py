import ast
import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "configs/iteration17_esm35_last4_full_epoch10.json"
CONFIG_PATH = ROOT / "configs/iteration18_esm150_last4_full_epoch8.json"
PLAN_PATH = ROOT / "reports/experiments/EXP-20261001-129-esm150-epoch8-plan.md"
EXPERIMENT_ID = "EXP-20261001-129-esm150-last4-full-epoch8"
MODEL_REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
EXPECTED_CHANGES = {
    "model_name": "facebook/esm2_t30_150M_UR50D",
    "model_revision": MODEL_REVISION,
    "experiment_id": EXPERIMENT_ID,
    "output_dir": f"artifacts/runs/{EXPERIMENT_ID}",
    "training.epochs": 8,
}


def flatten_parameters(values, prefix=""):
    flattened = {}
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(flatten_parameters(value, path))
        else:
            flattened[path] = value
    return flattened


class ESM150Epoch8ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(BASE_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_exact_allowlist_diff_without_added_or_removed_parameters(self):
        expected = json.loads(json.dumps(self.baseline))
        for path, value in EXPECTED_CHANGES.items():
            target = expected
            components = path.split(".")
            for component in components[:-1]:
                target = target[component]
            target[components[-1]] = value
        self.assertEqual(self.config, expected)
        baseline = flatten_parameters(self.baseline)
        candidate = flatten_parameters(self.config)
        self.assertEqual(set(candidate), set(baseline))
        changed = {
            path: value
            for path, value in candidate.items()
            if value != baseline[path]
        }
        self.assertEqual(changed, EXPECTED_CHANGES)

    def test_every_unmodified_parameter_is_identical_in_value_and_type(self):
        candidate = flatten_parameters(self.config)
        for path, value in flatten_parameters(self.baseline).items():
            if path in EXPECTED_CHANGES:
                continue
            with self.subTest(parameter=path):
                self.assertIs(type(candidate[path]), type(value))
                self.assertEqual(candidate[path], value)

    def test_training_protocol_is_preserved(self):
        training = self.config["training"]
        self.assertEqual(training["epochs"], 8)
        self.assertIs(training["save_epoch_scores"], True)
        self.assertIs(training["require_cuda"], True)
        self.assertEqual(training["batch_size"], 8)
        self.assertEqual(training["gradient_accumulation"], 2)
        self.assertEqual(training["learning_rate"], 0.0005)
        self.assertEqual(training["encoder_learning_rate"], 0.00001)
        self.assertEqual(training["loss"], {
            "name": "asymmetric_bce",
            "gamma_positive": 0.0,
            "gamma_negative": 1.0,
            "probability_clip": 0.05,
        })
        for name, value in {
            "seed": 42,
            "unfreeze_last_n_layers": 4,
            "pooling": "mean_max",
            "head_hidden_size": 512,
            "window_size": 1022,
            "overlap": 128,
        }.items():
            with self.subTest(parameter=name):
                self.assertEqual(self.config[name], value)
        self.assertNotIn("max_train_rows", self.config)
        self.assertNotIn("max_validation_rows", self.config)

    def test_model_identity_uses_verified_150m_commit_not_35m_commit(self):
        self.assertEqual(self.config["model_name"], EXPECTED_CHANGES["model_name"])
        self.assertEqual(self.config["model_revision"], MODEL_REVISION)
        self.assertRegex(self.config["model_revision"], r"^[0-9a-f]{40}$")
        self.assertNotEqual(self.config["model_revision"], self.baseline["model_revision"])

    def test_fixed_full_split_counts_hashes_and_disjoint_ids(self):
        self.assertEqual(self.config["split"], self.baseline["split"])
        self.assertEqual(self.config["split"]["kind"], "iteration4_tail")
        groups = {}
        for name, filename, count, checksum in (
            ("train_ids", "train_ids.csv", 112734,
             "bc0eadf7517dabd62361801ae971245eba08ae4c8c9974cd4a243c928b27bcce"),
            ("validation_ids", "validation_ids.csv", 1062,
             "275a340b0b242c5d394f6f129d391806695a50c0c7f23edd58a1a2dc66f3b9db"),
        ):
            with self.subTest(split=name):
                relative_path = f"artifacts/metrics/splits/iteration4_tail/{filename}"
                self.assertEqual(self.config["split"][name], relative_path)
                path = ROOT / relative_path
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, ["protein_id"])
                    ids = [row["protein_id"] for row in reader]
                self.assertEqual(len(ids), count)
                self.assertEqual(len(set(ids)), count)
                self.assertTrue(all(ids))
                groups[name] = set(ids)
        self.assertFalse(groups["train_ids"] & groups["validation_ids"])

    def test_run_directory_has_unique_expected_identity(self):
        self.assertEqual(self.config["output_dir"], EXPECTED_CHANGES["output_dir"])
        self.assertNotEqual(self.config["output_dir"], self.baseline["output_dir"])

    def test_plan_preserves_preparation_evidence_and_records_authorized_queue(self):
        plan = PLAN_PATH.read_text(encoding="utf-8")
        for evidence in (
            "preparation_status: `prepared`",
            "execution_status: `not_started`",
            "等待用户明确授权启动",
            "execution_status: `queued`",
            "用户已明确授权启动",
            "2026-10-01T03:19:54.5467855Z",
            "https://huggingface.co/api/models/facebook/esm2_t30_150M_UR50D",
            MODEL_REVISION,
            "GPU 显存",
            "不登记假成绩",
            "尚无150M实验成绩可报告",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, plan)

    def test_trainer_adapts_architecture_and_revision_by_static_inspection(self):
        source = (ROOT / "src/train_esm_finetune.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn(
            'getattr(getattr(encoder, "config", None), "hidden_size", 0)',
            functions["build_esm_classifier"],
        )
        self.assertIn('hidden_size * 2 if pooling == "mean_max"', functions["build_esm_classifier"])
        self.assertIn('getattr(getattr(encoder, "encoder", None), "layer", None)', functions["_transformer_layers"])
        self.assertIn('unfreeze_last_n_layers > len(layers)', functions["configure_encoder_trainability"])
        self.assertIn('layers[-unfreeze_last_n_layers:]', functions["configure_encoder_trainability"])
        self.assertIn('config.get("model_revision")', functions["_load_model_and_tokenizer"])
        self.assertIn('AutoModel.from_pretrained(model_name, **load_kwargs)', functions["_load_model_and_tokenizer"])
        self.assertIn('AutoTokenizer.from_pretrained(model_name, **load_kwargs)', functions["_load_model_and_tokenizer"])


if __name__ == "__main__":
    unittest.main()
