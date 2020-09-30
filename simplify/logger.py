"""
Logger template
"""

import logging
import atexit

# create logger
logger = logging.getLogger(__name__) # name should be given, otherwise we are configuring the root logger

# configure logging if no handlers exist
if (len(logger.handlers) < 1
    and len(logging.getLogger().handlers) < 1):

    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()

    # create formatter
    formatter = logging.Formatter('<%(levelname)s> %(name)s %(funcName)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # based on stackoverflow post - quick and dirty hack for colors ;)
    # -> Maybe switch to a solution where this is only configured for this particular logger
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = [30+_i for _i in range(8)]
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    colorMapping = {
        logging.INFO : GREEN,
        logging.DEBUG : BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
    }
    for loglevel, color in colorMapping.items():
        logging.addLevelName(loglevel, "{COLOR_SEQ}{LEVELNAME}{RESET_SEQ}".format(COLOR_SEQ=COLOR_SEQ % color,
                                                                                  LEVELNAME=logging.getLevelName(loglevel),
                                                                                  RESET_SEQ=RESET_SEQ))

def done():
    import sys
    if hasattr(sys, "last_value"):
        logger.info("Error.")
    else:
        logger.info("Done.")

atexit.register(done)
