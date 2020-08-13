from config import config
from utils import get_logger
import time

def test_template():
    logger = get_logger(config)
    logger.debug("from test template so mamy balabala balabala.........................")
    # tc = time.perf_counter()
    # time.sleep(2)
    # logger.debug(f"Elapsed Time {time.perf_counter()-tc:.4f}")
    logger.info(f"{'='*42} \n\t {config} \n\t {'='*42}")
    

if __name__ == "__main__":
    test_template()
    