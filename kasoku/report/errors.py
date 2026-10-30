"""Report generation errors."""

from kasoku.config import ConfigError


class ReportError(ConfigError):
    """Raised when report generation or data loading fails."""
