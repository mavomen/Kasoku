"""Alert rule rendering from Kasoku configuration.

Renders Prometheus alerting rules from the ``[alerts]`` section of a
Kasoku TOML config using a Jinja2 template, so thresholds stay
config-driven rather than hardcoded in YAML.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateError, select_autoescape

from kasoku.config import AlertsConfig, ConfigError

_TEMPLATE_NAME = "kasoku_alerts.yml.j2"


class AlertsError(ConfigError):
    """Raised when alert configuration or rendering is invalid."""


def render_alerts_yaml(config: AlertsConfig) -> str:
    """Render Prometheus alerting rules YAML from an :class:`AlertsConfig`.

    Raises :class:`AlertsError` if the template cannot be loaded or rendered.
    """
    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    try:
        template = env.get_template(_TEMPLATE_NAME)
    except TemplateError as exc:
        raise AlertsError(f"failed to load alert template {_TEMPLATE_NAME!r}: {exc}") from exc
    try:
        return template.render(alerts=config)
    except TemplateError as exc:
        raise AlertsError(f"failed to render alert template {_TEMPLATE_NAME!r}: {exc}") from exc


__all__ = ["AlertsError", "render_alerts_yaml"]
