import os

def get_user():
    return os.environ.get('USER', os.environ.get('USERNAME'))