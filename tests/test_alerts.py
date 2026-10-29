"""Unit tests for the Kasoku alert rule renderer and ``kasoku alerts`` CLI."""

from __future__ import annotations

import yaml

from kasoku.alerts import render_alerts_yaml
from kasoku.config import Config, parse_config

COMMITTED_PATH = "monitoring/prometheus/rules/kasoku_alerts.yml"


def _load_yaml(text: str) -> dict:
    return yaml.safe_load(text)


def test_render_defaults_match_committed_rules():
    rendered = render_alerts_yaml(Config().alerts)
    committed = _load_yaml(open(COMMITTED_PATH, encoding="utf-8").read())
    assert _load_yaml(rendered) == committed


def test_render_all_rules_present_by_default():
    doc = _load_yaml(render_alerts_yaml(Config().alerts))
    names = [rule["alert"] for rule in doc["groups"][0]["rules"]]
    assert names == [
        "KasokuXlaNoSpeedupWarning",
        "KasokuCompileOverheadWarning",
        "KasokuXlaRegressionCritical",
        "KasokuThroughputRegressionCritical",
    ]


def test_render_ignoring_mode_preserved():
    doc = _load_yaml(render_alerts_yaml(Config().alerts))
    xla = next(r for r in doc["groups"][0]["rules"] if r["alert"] == "KasokuXlaRegressionCritical")
    assert "ignoring(mode)" in xla["expr"]


def test_render_disabled_rule_omitted():
    config = parse_config(
        {"alerts": {"xla_no_speedup": {"enabled": False}, "compile_overhead": {"enabled": False}}}
    )
    doc = _load_yaml(render_alerts_yaml(config.alerts))
    names = [rule["alert"] for rule in doc["groups"][0]["rules"]]
    assert "KasokuXlaNoSpeedupWarning" not in names
    assert "KasokuCompileOverheadWarning" not in names
    assert "KasokuXlaRegressionCritical" in names
    assert "KasokuThroughputRegressionCritical" in names


def test_render_custom_threshold_substituted():
    config = parse_config({"alerts": {"xla_regression": {"enabled": True, "multiplier": 2.0}}})
    doc = _load_yaml(render_alerts_yaml(config.alerts))
    xla = next(r for r in doc["groups"][0]["rules"] if r["alert"] == "KasokuXlaRegressionCritical")
    assert "2.0 * kasoku_wall_clock_seconds" in xla["expr"]
    assert "> 2.0x eager wall-clock" in xla["annotations"]["summary"]


def test_render_compile_overhead_ratio_substituted():
    config = parse_config({"alerts": {"compile_overhead": {"enabled": True, "ratio": 0.75}}})
    doc = _load_yaml(render_alerts_yaml(config.alerts))
    overhead = next(r for r in doc["groups"][0]["rules"] if r["alert"] == "KasokuCompileOverheadWarning")
    assert "> 0.75 * kasoku_wall_clock_seconds" in overhead["expr"]


def test_render_output_is_valid_prometheus_yaml():
    doc = _load_yaml(render_alerts_yaml(Config().alerts))
    assert doc["groups"][0]["name"] == "kasoku_alerts"
    for rule in doc["groups"][0]["rules"]:
        assert "alert" in rule
        assert "expr" in rule
        assert rule["for"] == "0s"
        assert rule["labels"]["severity"] in ("warning", "critical")


def test_render_prometheus_labels_template_preserved():
    rendered = render_alerts_yaml(Config().alerts)
    assert "{{ $labels.model }}/{{ $labels.model_size }}" in rendered


def test_show_uses_defaults_when_no_config(monkeypatch, tmp_path, capsys):
    from kasoku.cli.main import main

    assert main(["alerts", "show"]) == 0
    out = capsys.readouterr().out
    assert "xla_no_speedup: enabled=True" in out
    assert "compile_overhead: enabled=True, ratio=0.5" in out
    assert "xla_regression: enabled=True, multiplier=1.25" in out


def test_render_writes_to_file(tmp_path):
    from kasoku.cli.main import main

    output = tmp_path / "kasoku_alerts.yml"
    assert main(["alerts", "render", "--output", str(output)]) == 0
    doc = _load_yaml(output.read_text(encoding="utf-8"))
    assert doc["groups"][0]["name"] == "kasoku_alerts"


def test_validate_valid_config_returns_zero(tmp_path):
    from kasoku.cli.main import main

    cfg = tmp_path / "config.toml"
    cfg.write_text("[alerts]\nxla_regression = { multiplier = 1.5 }\n", encoding="utf-8")
    assert main(["alerts", "validate", "--config", str(cfg)]) == 0


def test_validate_invalid_config_returns_one(tmp_path):
    from kasoku.cli.main import main

    cfg = tmp_path / "config.toml"
    cfg.write_text("[alerts]\nxla_regression = { multiplier = -1.0 }\n", encoding="utf-8")
    assert main(["alerts", "validate", "--config", str(cfg)]) == 1
