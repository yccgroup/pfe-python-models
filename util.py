import sys
import os
import subprocess
from datetime import datetime,timezone


class AppError(Exception):
    def __init__(self, msg):
        self.msg = msg


def log(fd, msg=''):
    fd.write(msg + "\n")


def error(msg):
    raise AppError(msg)


def get_git_revision():
    try:
        return subprocess.check_output(
            ['git', 'log', '-n1', '--pretty=tformat:%h (%ai)'],
            cwd = os.path.dirname(os.path.realpath(__file__))
            ).decode('ascii').strip()
    except:
        return 'N/A'


def timestamp():
    now = datetime.now(timezone.utc).astimezone()
    return now.strftime('%F %T %z')
