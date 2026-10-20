import json

from perception.replay import analyze, main, replay_records


def synthetic_log():
    records = [{"type": "header", "gate": {"WINDOW": 2, "NEED": 2, "HARD_NEED": 2,
                  "thresholds": {"Rat": .5}, "free_thresholds": {},
                  "ARM_BELOW": .35, "ARM_FRAMES": 4}},
               {"type": "expect", "t": 0, "seal": "Rat"}]
    for i, score in enumerate([.2, .2, .2, .2, .8, .8, 0, 0]):
        records.append({"type": "frame", "i": i, "t": i * .5, "expect": "Rat",
                        "scores": {"Rat": score}, "top": ["Rat", score],
                        "confirm": i == 5})
    records.append({"type": "expect", "t": 4, "seal": None})
    for i in (8, 9):
        records.append({"type": "frame", "i": i, "t": i * .5, "expect": None,
                        "scores": {"Rat": .8}, "top": ["Rat", .8], "confirm": i == 9})
    return records


def test_replay_attempt_success_and_outside_false_confirm():
    report = analyze(replay_records(synthetic_log()))
    assert len(report["attempts"]) == 1
    assert report["per_seal"]["Rat"]["successes"] == 1
    assert report["false_outside"] == 1


def test_check_cli_reports_zero_mismatches(tmp_path, capsys):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in synthetic_log()), encoding="utf-8")
    assert main(["--log", str(path), "--check"]) == 0
    assert "--check mismatches: 0" in capsys.readouterr().out

