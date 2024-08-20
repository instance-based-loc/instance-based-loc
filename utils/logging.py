import logging

BLUE = "\033[94m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO)

def conditional_log(message, should_log):
    if should_log:
        logging.info(f"{BLUE}> {message}{RESET}")