import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from common import atomic_json, exclusive_lock, sha256, verified_completion
from manifest import expand, make_manifest, options
from full_manifest import paper_manifest
from streaming import StreamingMetrics


class HarnessTests(unittest.TestCase):
    def test_nested_loops_and_comments(self):
        text = '''model_name=EMAformer
# python -u run.py --bad
for length in 48 96
do
for pred in 12 24
do
python -u run.py \\
 --seq_len $length --pred_len ${pred} --model $model_name # inactive
done
done'''
        rows = expand(text)
        self.assertEqual([(options(r["argv"])["seq_len"], options(r["argv"])["pred_len"]) for r in rows],
                         [("48", "12"), ("48", "24"), ("96", "12"), ("96", "24")])

    def test_no_shell_execution(self):
        with self.assertRaises(ValueError):
            expand("rm -rf /tmp/test")
        with self.assertRaises(ValueError):
            expand("python -u run.py --model $undefined")

    def test_released_manifest(self):
        repo = Path(__file__).resolve().parent.parent / "EMAformer"
        data = make_manifest(repo)
        self.assertEqual(data["source_script_command_counts"],
                         {"boost_performance": 48, "increasing_lookback": 38, "multivariate_forecasting": 52})
        self.assertEqual(len(data["runs"]), 378)
        self.assertEqual(len({r["id"] for r in data["runs"]}), 378)
        main = [r for r in data["runs"] if r["group"] == "multivariate_forecasting"]
        self.assertEqual(len(main), 288)
        self.assertEqual(len({options(r["argv"])["model_id"] for r in main}), 48)

    def test_paper_matrix(self):
        repo = Path(__file__).resolve().parent.parent / "EMAformer"
        data = paper_manifest(repo)
        self.assertEqual(len(data["runs"]), 822)
        self.assertEqual(data["expected_counts"], {"alternative_period": 32, "armor_ablation": 336,
            "lookback_emaformer": 10, "mean_input": 16, "mlp_backbone": 32,
            "multivariate_forecasting": 288, "transfer_armor": 36,
            "transfer_inverted": 36, "transfer_original": 36})
        main = [r for r in data["runs"] if r["group"] == "multivariate_forecasting"]
        self.assertTrue(all(r["seed"] == 2023 for r in main[:48]))
        self.assertEqual({r["seed"] for r in main[48:96]}, {2024})

    def test_streaming_unequal_batches(self):
        import numpy as np
        rng = np.random.RandomState(7)
        true = rng.normal(size=(23, 11, 7)).astype(np.float32)
        pred = (true + rng.normal(scale=.4, size=true.shape)).astype(np.float32)
        accumulator = StreamingMetrics()
        for start, end in ((0, 5), (5, 10), (10, 23)):
            accumulator.update(pred[start:end], true[start:end])
        record = accumulator.result()
        error = pred - true
        self.assertEqual(record["metric_elements"], true.size)
        self.assertAlmostEqual(record["mse"], float(np.square(error).mean(dtype=np.float64)), places=14)
        self.assertAlmostEqual(record["mae"], float(np.abs(error).mean(dtype=np.float64)), places=14)
        # Difference from original float32 reduction is tiny and disclosed.
        self.assertLess(abs(record["mse"] - float(np.square(error).mean())), 1e-7)

    def test_streaming_zero_target_percentage(self):
        import numpy as np
        accumulator = StreamingMetrics()
        accumulator.update(np.array([1., 2.], dtype=np.float32), np.array([0., 2.], dtype=np.float32))
        self.assertEqual(accumulator.result()["mse"], .5)
        self.assertIsNone(accumulator.result()["mape"])

    def test_lock_released_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            code = "from pathlib import Path; from common import exclusive_lock; import sys\nwith exclusive_lock(Path(sys.argv[1])): print('acquired')"
            with exclusive_lock(path):
                other = subprocess.run([sys.executable, "-c", code, str(path)], cwd=Path(__file__).parent,
                                       capture_output=True, text=True)
                self.assertNotEqual(other.returncode, 0)
                self.assertIn("Another queue", other.stderr)
            with exclusive_lock(path):
                pass

    def test_child_lock_survives_supervisor_exit(self):
        # A real child owns the OS lock after its supervisor exits. This is the
        # SIGKILL/orphan failure mode that a parent-only queue lock misses.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_script = root / "child_lock.py"
            child_script.write_text("from pathlib import Path\nfrom common import exclusive_lock\nimport sys,time\n"
                                    "with exclusive_lock(Path(sys.argv[1])):\n Path(sys.argv[2]).write_text('ready')\n time.sleep(30)\n", encoding="utf-8")
            lock = root / "lock"
            ready = root / "ready"
            # Spawn the lock owner separately; no inherited parent lock state.
            env = dict(__import__('os').environ)
            env['PYTHONPATH'] = str(Path(__file__).parent)
            owner = subprocess.Popen([sys.executable, str(child_script), str(lock), str(ready)], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(lock):
                        pass
            finally:
                owner.terminate()
                owner.communicate(timeout=5)
            with exclusive_lock(lock):
                pass

    def test_resume_checks_integrity_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "metrics.json", {"mae": .2, "mse": .1, "rmse": .316})
            (root / "checkpoint.pth").write_bytes(b"real-test-placeholder")
            state = {"state": "complete", "signature": "s1", "metrics": {"path": "metrics.json", "sha256": sha256(root / "metrics.json")},
                     "checkpoints": [{"path": "checkpoint.pth", "sha256": sha256(root / "checkpoint.pth")}]}
            atomic_json(root / "status.json", state)
            self.assertTrue(verified_completion(root, "s1"))
            self.assertFalse(verified_completion(root, "s2"))
            (root / "checkpoint.pth").write_bytes(b"corrupted")
            self.assertFalse(verified_completion(root, "s1"))


if __name__ == "__main__":
    unittest.main()
