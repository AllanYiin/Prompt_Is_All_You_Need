"""trident api"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
from importlib import reload
from sys import stderr

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

__version__ = '0.0.1'
stderr.write('Prompt_Is_All_You_Need {0}\n'.format(__version__))

from Prompt_Is_All_You_Need.api import *
from Prompt_Is_All_You_Need.utils import *
import threading
import random
import glob
from tqdm import tqdm
import numpy as np



