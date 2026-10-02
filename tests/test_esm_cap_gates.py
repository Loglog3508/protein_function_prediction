from copy import deepcopy

from src import compare_esm_controls as compare


def metrics(rate, over, ap=.2, auc=.75):
    return {"macro_ap": ap, "macro_auc": auc,
            "crossfit": {"predicted_positive_rate": rate,
                         "true_positive_rate": .0592, "overpredicted_labels": over}}


def test_cap_gate_requires_ap_density_and_label_improvements_with_auc_constraint():
    baseline, candidate = metrics(.12, 472), metrics(.08, 440, .21, .75)
    assert compare.cap_decision_gates(baseline, candidate)["cap_joint_criterion"]
    for field, value in [("macro_ap", .2), ("macro_auc", .749)]:
        changed = deepcopy(candidate)
        changed[field] = value
        assert not compare.cap_decision_gates(baseline, changed)["cap_joint_criterion"]
    for field, value in [("predicted_positive_rate", .13), ("overpredicted_labels", 472)]:
        changed = deepcopy(candidate)
        changed["crossfit"][field] = value
        assert not compare.cap_decision_gates(baseline, changed)["cap_joint_criterion"]


def test_lower_rate_is_not_convergence_when_it_overshoots_farther_below_truth():
    gates = compare.cap_decision_gates(metrics(.065, 200), metrics(.01, 10, .21, .76))
    assert not gates["calibrated_rate_closer_to_true"]
    assert not gates["cap_joint_criterion"]
