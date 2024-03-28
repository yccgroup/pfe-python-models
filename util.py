import sys
import os
import subprocess


def log(fd, msg=''):
    fd.write(msg + "\n")


def error(msg, rc=1):
    sys.stdout.write(f"ERROR: {msg}\n")
    sys.exit(rc)


def get_git_revision():
    try:
        return subprocess.check_output(
            ['git', 'log', '-n1', '--pretty=tformat:%h (%ai)'],
            cwd = os.path.dirname(os.path.realpath(__file__))
            ).decode('ascii').strip()
    except OSError:
        return 'N/A'

