import logging
logging.basicConfig(level=logging.INFO)

def conditional_log(message, should_log):
    if should_log:
        logging.info(f"> {message}")