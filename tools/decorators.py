import time
from functools import wraps
import logging


def timing(func):
    @wraps(func)
    def wrapper(*args, **kw):
        t1 = time.time()
        result = func(*args, **kw)
        t2 = time.time()
        logging.info(f"{func.__name__}() run time: {(t2-t1):.5f}s")
        print(f"{func.__name__}() run time: {(t2-t1):.5f}s")
        return result
    return wrapper


def print_return(func):
    @wraps(func)
    def wrapper(*args, **kw):
        result = func(*args, **kw)
        logging.info(f"{func.__name__}() return: {result}")
        print(f"{func.__name__}() return: {result}")
        return result
    return wrapper


def print_input(func):
    @wraps(func)
    def wrapper(*args, **kw):
        logging.info(f"{func.__name__}() input: {args, kw}")
        print(f"{func.__name__}() input: {args, kw}")
        result = func(*args, **kw)
        return result
    return wrapper
