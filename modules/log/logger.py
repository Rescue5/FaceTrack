import logging
import threading


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        self.logger = logging.getLogger("default")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(threadName)s,%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, extra={"threadName": threading.current_thread().name}, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, extra={"threadName": threading.current_thread().name}, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, extra={"threadName": threading.current_thread().name}, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, extra={"threadName": threading.current_thread().name}, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, extra={"threadName": threading.current_thread().name}, **kwargs)
