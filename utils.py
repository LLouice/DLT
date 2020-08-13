def get_logger(config):
    import sys
    from loguru import logger
    fmt = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {module}:{function}:{line} \n\t {message}\n"
    logger.add(config.log_file, level="DEBUG", format=fmt)
    logger.add(sys.stderr, level="ERROR", format=fmt)
    return logger
