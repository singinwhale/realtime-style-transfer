import sys
import logging
import time
from pathlib import Path
import os


class ErrorStreamWrapper:
    def __init__(self, stream):
        self.stream = stream
        self.line = ""
        self.last_print_time = 0

    def write(self, s):
        self.line += s
        if time.time_ns() - self.last_print_time > 10e6:
            self._print(self.line)
            self.line = ""

    def _print(self, s):
        self.stream.write(s)
        logging.error(s)
        self.last_print_time = time.time_ns()

    def flush(self):
        if len(self.line) > 0:
            self._print(self.line)
            self.line = ""
        self.stream.flush()

    def close(self):
        self.stream.close()


def enable_logfile(log_dir: Path):
    handler = logging.FileHandler(log_dir / "style-transfer.log", encoding='utf-8', delay=False)
    handler.setFormatter(DefaultFormatter(False))
    logging.root.addHandler(handler)
    sys.stderr = ErrorStreamWrapper(sys.stderr)


class DefaultFormatter(logging.Formatter):
    def __init__(self, with_file_info=True):
        super().__init__()
        self.format_prefix = "%(name)s - %(levelname)s - %(message)s"
        self.format_file = " (%(filename)s:%(lineno)d"
        self.format_file_detail = " File \"%(pathname)s\", line %(lineno)d"
        self.time_format = " @ %(asctime)s"

        self.formats = {
            logging.DEBUG: self.format_prefix + (self.format_file if with_file_info else "") + self.time_format,
            logging.INFO: self.format_prefix + (self.format_file if with_file_info else "") + self.time_format,
            logging.WARNING: self.format_prefix + (self.format_file if with_file_info else "") + self.time_format,
            logging.ERROR: self.format_prefix + (self.format_file_detail if with_file_info else "") + self.time_format,
            logging.CRITICAL: self.format_prefix + (self.format_file_detail if with_file_info else "") + self.time_format
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColorFormatter(DefaultFormatter):
    def __init__(self):
        super().__init__()

        white = "\x1b[38;20m"
        grey = "\x1b[90;1m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.formats = {
            logging.DEBUG: grey + self.format_prefix + grey + self.format_file + self.time_format + reset,
            logging.INFO: white + self.format_prefix + grey + self.format_file + self.time_format + reset,
            logging.WARNING: yellow + self.format_prefix + grey + self.format_file + self.time_format + reset,
            logging.ERROR: red + self.format_prefix + grey + self.format_file_detail + self.time_format + reset,
            logging.CRITICAL: bold_red + self.format_prefix + grey + self.format_file_detail + self.time_format + reset
        }


stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
stdout_stream_handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[stdout_stream_handler])

logging.getLogger('h5py').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('tf2onnx').setLevel(logging.INFO)

# make numpy arrays smallerl
import numpy

numpy.set_printoptions(threshold=10, edgeitems=1)
