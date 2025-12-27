# packages/quant_lib/logging.py

import sys
from pathlib import Path
from loguru import logger as _logger  # Aliased to avoid conflict


class LogManager:
    # Pass 'debug' flag directly to decouple from settings
    def __init__(self, service_name: str, debug: bool = False):
        self.service_name = service_name
        self.debug = debug
        self._configure()

    def _configure(self):
        _logger.remove()

        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.service_name}.json.log"

        # Console Handler
        _logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[context]}</cyan> | <level>{message}</level>",
            level="INFO",
            colorize=True,
        )

        # File Handler (uses the passed-in debug flag)
        _logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level="DEBUG" if self.debug else "INFO",
            serialize=True,
            enqueue=True,
        )

    def get_logger(self, context_name: str):
        return _logger.bind(app=self.service_name, context=context_name)
