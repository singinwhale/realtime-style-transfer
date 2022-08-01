import sys
import logging


class ColorFormatter(logging.Formatter):
    """from https://stackoverflow.com/a/56944256"""

    white = "\x1b[38;20m"
    grey = "\x1b[90;1m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(name)s - %(levelname)s - %(message)s"
    format_file = " (%(filename)s:%(lineno)d @ %(asctime)s)"
    format_file_detail = " File \"%(pathname)s\", line %(lineno)d @ %(asctime)s"

    FORMATS = {
        logging.DEBUG: grey + format + grey + format_file + reset,
        logging.INFO: white + format + grey + format_file + reset,
        logging.WARNING: yellow + format + grey + format_file + reset,
        logging.ERROR: red + format + grey + format_file_detail + reset,
        logging.CRITICAL: bold_red + format + grey + format_file_detail + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


ch = logging.StreamHandler(stream=sys.stdout)
ch.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[ch])

logging.getLogger('h5py').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('PIL').setLevel(logging.INFO)


# make numpy arrays smallerl
import numpy
numpy.set_printoptions(threshold=10, edgeitems=1)