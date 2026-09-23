import json
import tempfile
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_load_config_requires_seed(self):
        from src.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"model": {"type": "random_forest"}}))

            with self.assertRaisesRegex(ValueError, "seed"):
                load_config(path)

    def test_load_config_returns_nested_values(self):
        from src.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "split": {"validation_size": 0.2},
                        "model": {"type": "random_forest"},
                    }
                )
            )

            config = load_config(path)

        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["split"]["validation_size"], 0.2)
        self.assertEqual(config["model"]["type"], "random_forest")


if __name__ == "__main__":
    unittest.main()
