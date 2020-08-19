def get_logger(config):
    import sys
    from loguru import logger
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <lvl>{level}</lvl> | <c>{module}:{function}:{line}</c> \n\t {message}\n"
    logger.add(sys.stdout, level="DEBUG", colorize=True, format=fmt)
    logger.add(config.log_file + "_{time:YYYY-MM-DD_HH:mm:ss}.log",
               level="DEBUG",
               format=fmt)
    # logger.add(sys.stderr, level="ERROR", format=fmt)
    return logger


def log_dict(logger, log):
    msg = "\n"
    for i, (k, v) in enumerate(log.items()):
        msg += f"\n\t{k} : {v}"
    logger.info(msg)
