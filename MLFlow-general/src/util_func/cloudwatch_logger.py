import configparser
import logging
import logging.config
import sys

dynamic_config = configparser.ConfigParser()
log_level = "INFO"


def get_logger_for_cloudwatch(name):
    class _ExcludeErrorsFilter(logging.Filter):
        def filter(self, record):
            """Filters out log messages with log level ERROR (numeric value: 40) or higher."""
            return record.levelno < 40

    logging_config = {
        "version": 1,
        "filters": {"exclude_errors": {"()": _ExcludeErrorsFilter}},
        "formatters": {
            # Modify log message format here
            "basic_formatter": {
                "format": "(%(process)d) %(asctime)s %(name)s (line %(lineno)s) | %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console_stderr": {
                # Sends log messages with log level ERROR or higher to stderr
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "basic_formatter",
                "stream": sys.stderr,
            },
            "console_stdout": {
                # Sends log messages with log level lower than ERROR to stdout
                "class": "logging.StreamHandler",
                "level": f"{log_level}",
                "formatter": "basic_formatter",
                "filters": ["exclude_errors"],
                "stream": sys.stdout,
            },
        },
        "root": {
            # In general, this should be kept at 'NOTSET'.
            # Otherwise it would interfere with the log levels set for each handler.
            "level": "NOTSET",
            "handlers": ["console_stderr", "console_stdout"],
        },
        "disable_existing_loggers": False,
    }
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(name)
    return logger
