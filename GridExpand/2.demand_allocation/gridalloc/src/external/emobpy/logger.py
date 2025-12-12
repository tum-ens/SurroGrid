import logging
import sys

# def get_logger(logger_name='main'):
#     '''
#     Logger for emobpy.
#     Two possible location for logging:
#     - in the current working directory, else;
#     - in the user's home directory.
#     '''
#     user_dir = USER_PATH or DEFAULT_DATA_DIR
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.DEBUG)
#     if os.path.isdir(os.path.join(CWD,'log')):
#         log_filename = os.path.join(CWD,'log','emobpy.log')
#     else:
#         os.makedirs(os.path.join(user_dir,'log'),exist_ok=True)
#         log_filename = os.path.join(user_dir,'log','emobpy.log')
#     file_handler = logging.FileHandler(log_filename)
#     file_formatter = logging.Formatter("%(asctime)s:%(name)s:%(funcName)s:%(message)s")
#     file_handler.setFormatter(file_formatter)
#     file_handler.setLevel(logging.DEBUG)
#     logger.addHandler(file_handler)
#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setLevel(logging.INFO)
#     logger.addHandler(stream_handler)
#     return logger

def get_logger(logger_name='main'):
    '''
    Logger for emobpy.
    Only prints to console; does not write to a log file.
    '''
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add console output only
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(funcName)s:%(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def change_file_handler(logger, log_filename=None, log_level=None, log_format=None):
    '''
    Change file handler of a logger.
    '''
    flag = False
    for handler in logger.handlers:
        if handler.__class__.__name__ == 'FileHandler':
            flag = True
            old_handler = handler
            break
    if flag:
        old_filename = old_handler.baseFilename
        old_level = old_handler.level
        old_format = old_handler.formatter
        if log_filename is not None:
            new_handler = logging.FileHandler(log_filename)
        else:
            new_handler = logging.FileHandler(old_filename)
        if log_level is not None:
            new_handler.setLevel(log_level)
        else:
            new_handler.setLevel(old_level)
        if log_format is not None:
            new_formatter = logging.Formatter(log_format)
            new_handler.setFormatter(new_formatter)
        else:
            new_handler.setFormatter(old_format)
        old_handler.close()
        logger.removeHandler(old_handler)
        logger.addHandler(new_handler)
    else:
        logger.error('No file handler found.')
    return logger

def change_stream_handler(logger, stream=None, log_level=None, log_format=None):
    '''
    Change file handler of a logger.
    '''
    flag = False
    for handler in logger.handlers:
        if handler.__class__.__name__ == 'StreamHandler':
            flag = True
            old_handler = handler
            break
    if flag:
        old_level = old_handler.level
        old_format = old_handler.formatter
        if stream is not None:
            new_handler = logging.StreamHandler(stream)
        else:
            new_handler = logging.StreamHandler()
        if log_level is not None:
            new_handler.setLevel(log_level)
        else:
            new_handler.setLevel(old_level)
        if log_format is not None:
            new_formatter = logging.Formatter(log_format)
            new_handler.setFormatter(new_formatter)
        else:
            new_handler.setFormatter(old_format)
        old_handler.close()
        logger.removeHandler(old_handler)
        logger.addHandler(new_handler)
    else:
        logger.error('No file handler found.')
    return logger