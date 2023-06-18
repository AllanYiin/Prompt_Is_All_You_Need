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

__version__ = '0.0.2'
stderr.write('prompt4all {0}\n'.format(__version__))

from prompt4all import api
from prompt4all import utils
#from . import gradio_chatbot_patch
#from . import theme
import threading
import random
import glob
from tqdm import tqdm
import numpy as np



