"""Checks that run from a new clone without downloads: the model on the example night, and every table from results/."""
import subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from dipdepth import model


def test_example_night():
    z = np.load(ROOT / "examples/tr03-0092.npz")
    row, a = model.model("c2018:tr03-0092", z["spo2"])
    assert row["converged"] and not row["fine"]
    assert len(a["events"]) == 433 and len(a["count_events"]) == 156 and len(a["rule3_nadirs"]) == 172
    np.testing.assert_array_equal(a["events"][:, 0], z["runs"][:, 0])
    # the example's arrays come from the paper's run; floating-point order moves depths by under 1e-6
    np.testing.assert_allclose(a["events"][:, 1], z["runs"][:, 1], atol=1e-5)
    np.testing.assert_allclose(a["depth_severity"], z["expected_drop"], atol=1e-5)
    # the comparator and the count as this release computes them: the rule's 172 nadirs and the 156 counted events
    np.testing.assert_array_equal(a["rule3_nadirs"], z["rule_nadirs"])
    np.testing.assert_array_equal(a["count_events"][:, 0], z["count_events"][:, 0])
    np.testing.assert_allclose(a["count_events"][:, 1], z["count_events"][:, 1], atol=1e-5)


def test_tables():
    r = subprocess.run([sys.executable, str(ROOT / "tables.py")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
