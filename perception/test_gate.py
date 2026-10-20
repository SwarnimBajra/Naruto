from perception.gate import StabilityGate, frame_decision, update_armed


def test_stability_gate_confirms_after_need_hits():
    gate = StabilityGate(window=3, need=2)
    assert gate.update("Rat")[2] is False
    assert gate.update("Rat") == ("Rat", 1.0, True)


def test_target_candidate_requires_arm_and_threshold():
    assert frame_decision({"Rat": .8}, "Rat", False)[0] is None
    assert frame_decision({"Rat": .8}, "Rat", True)[0] == "Rat"
    assert frame_decision({"Rat": .69}, "Rat", True)[0] is None


def test_free_mode_uses_free_threshold():
    assert frame_decision({"Snake": .6}, None, True)[0] == "Snake"


def test_relax_rule_requires_four_low_frames():
    armed, low = False, 0
    for _ in range(3):
        armed, low = update_armed("Rat", .2, armed, low)
        assert not armed
    assert update_armed("Rat", .2, armed, low) == (True, 4)

