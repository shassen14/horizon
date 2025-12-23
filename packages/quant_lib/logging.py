# packages/quant_lib/logging.py

import sys
from pathlib import Path
from loguru import logger as _logger  # Aliased to avoid conflict
from packages.quant_lib.config import settings


class LogManager:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._configure()

    def _configure(self):
        # Clear default handlers
        _logger.remove()

        # Define Log Path
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.service_name}.json.log"

        # 1. Console Handler
        _logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[context]}</cyan> | <level>{message}</level>",
            level="INFO",
            colorize=True,
        )

        # 2. File Handler
        _logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level="DEBUG" if settings.system.debug else "INFO",
            serialize=True,
            enqueue=True,
        )

    def get_logger(self, context_name: str):
        """Returns a logger bound to a specific class or module context."""
        return _logger.bind(app=self.service_name, context=context_name)
