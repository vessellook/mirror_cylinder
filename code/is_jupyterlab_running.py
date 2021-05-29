import re

import psutil


def is_jupyterlab_running():
    return any(re.search('jupyter-lab', x)
               for x in psutil.Process().parent().cmdline())
