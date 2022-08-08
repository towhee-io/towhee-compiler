import logging
import logging.config

logging.config.dictConfig(dict(
    version=1,
    formatters={
        "info": {
            "format": "%(asctime)s|%(levelname)s|%(module)s:%(lineno)s| %(message)s"
        },
    },
    loggers={
        "": dict(level="NOTSET", handlers=["console"]),
    },
    handlers={
        "console": {
            "level": "INFO",
            "formatter": "info",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
))

def set_log_level(level):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    for h in logging.getLogger().handlers:
        h.setLevel(level)

def get_logger(name):
    return logging.getLogger()
