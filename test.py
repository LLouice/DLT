import time
from config import config
from utils import get_logger


def test_template():
    logger = get_logger(config)
    logger.info("this is a info")
    logger.debug(
        "from test template so many balabala balabala........................."
    )
    logger.error("this is a error!")
    # tc = time.perf_counter()
    # time.sleep(2)
    # logger.debug(f"Elapsed Time {time.perf_counter()-tc:.4f}")
    logger.info(f"{'='*42} \n\t {config} \n\t {'='*42}")
    test_accept_logger(logger)


def test_accept_logger(logger):
    logger.info("from a accept logger!")


if __name__ == "__main__":
    test_template()
