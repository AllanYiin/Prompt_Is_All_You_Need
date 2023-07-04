import inspect
import json
import locale
import os
import platform
import sys
import threading
from collections import OrderedDict
from functools import partial
import numpy as np

_prompt4all_context=None




def sanitize_path(path):
    """Sanitize the file or folder path, a same-format and absoluted path will return.

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    Examples:
        >>> print(sanitize_path('~/.prompt4all/datasets'))
        C:/Users/allan/.prompt4all/datasets

    """
    if path.startswith('~/'):
        path=os.path.join(os.path.expanduser("~"),path[2:])
    path=os.path.abspath(path)
    return path.strip().replace('\\', '/')
    # if isinstance(path, str):
    #     return os.path.normpath(path.strip()).replace('\\', '/')
    # else:
    #     return path

def split_path(path:str):
    """split path into folder, filename and ext 3 parts clearly.

    Args:
        path (str): a path of file or folder

    Returns:
        folder, filename and ext

    Examples:
        >>> print(split_path('C:/.prompt4all/datasets/cat.jpg'))
        ('C:/.prompt4all/datasets', 'cat', '.jpg')
        >>> print(split_path('C:/.prompt4all/models/resnet.pth.tar'))
        ('C:/.prompt4all/models', 'resnet', '.pth.tar')

    """
    if path is None or len(path) == 0:
        return '', '', ''
    path = sanitize_path(path)
    folder, filename = os.path.split(path)
    ext = ''
    if '.' in filename:
        filename, ext = os.path.splitext(filename)
        # handle double ext, like 'mode.pth.tar'
        filename, ext2 = os.path.splitext(filename)
        ext = ext2 + ext
    else:
        folder = os.path.join(folder, filename)
        filename = ''
    return folder, filename, ext


def make_dir_if_need(path):
    """Check the base folder in input path whether exist, if not , then create it.

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    """
    folder, filename, ext = split_path(path)
    if len(folder) > 0 and not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(e)
            sys.stderr.write('folder:{0} is not valid path'.format(folder))
    return sanitize_path(path)


def get_sitepackages():  # pragma: no cover
    installed_packages=None
    try:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        return installed_packages

    # virtualenv does not ship with a getsitepackages impl so we fallback
    # to using distutils if we can
    # https://github.com/pypa/virtualenv/issues/355
    except Exception as e:
        print(e)
        try:
            from distutils.sysconfig import get_python_lib

            return [get_python_lib()]

        # just incase, don't fail here, it's not worth it
        except Exception:
            return []


class _ThreadLocalInfo(threading.local):
    """
    Thread local Info used for store thread local attributes.
    """

    def __init__(self):
        super(_ThreadLocalInfo, self).__init__()
        self._reserve_class_name_in_scope = True

    @property
    def reserve_class_name_in_scope(self):
        """Gets whether to save the network class name in the scope."""
        return self._reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Sets whether to save the network class name in the scope."""
        if not isinstance(reserve_class_name_in_scope, bool):
            raise ValueError(
                "Set reserve_class_name_in_scope value must be bool!")
        self._reserve_class_name_in_scope = reserve_class_name_in_scope


class _Context:
    """
    _Context is the environment in which operations are executed
    Note:
        Create a context through instantiating Context object is not recommended.
        should use context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_handle = OrderedDict()
        self._errors_config= OrderedDict()
        self.whisper_model=None
        self.initial_context()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def get_prompt4all_dir(self):
        """Get or create prompt4all directory
        1)  read from
         enviorment variable 'PROMPT4ALL_HOME'
        2) use default directory '~/.prompt4all'
        3) if the directory not exist, create it!

        Returns:
            the  prompt4all directory path

        """
        _prompt4all_dir = ''
        if 'PROMPT4ALL_HOME' in os.environ:
            _prompt4all_dir = os.environ.get('PROMPT4ALL_HOME')
        else:
            _prompt4all_base_dir = os.path.expanduser('~')
            if not os.access(_prompt4all_base_dir, os.W_OK):
                _prompt4all_dir = '/tmp/.prompt4all'
            else:
                _prompt4all_dir = os.path.expanduser('~/.prompt4all')

        _prompt4all_dir = sanitize_path(_prompt4all_dir)
        if not os.path.exists(_prompt4all_dir):
            try:
                os.makedirs(_prompt4all_dir)
            except OSError as e:
                # Except permission denied and potential race conditions
                # in multi-threaded environments.
                print(e)

        return _prompt4all_dir

    def get_plateform(self):
        """

        Returns:
            check current system os plateform.

        """
        plateform_str = platform.system().lower()
        if 'darwin' in plateform_str:
            return 'mac'
        elif 'linux' in plateform_str:
            return 'linux'
        elif 'win' in plateform_str:
            return 'windows'
        else:
            return plateform_str

    def initial_context(self):
        self._module_dict = dict()
        self.prompt4all_dir = self.get_prompt4all_dir()
        self.backend =None
        self.print=partial(print,flush=True)
        site_packages=get_sitepackages()

        self.locale = locale.getdefaultlocale()[0].lower()

        self.working_directory = os.getcwd()
        self.plateform = self.get_plateform()
        self.numpy_print_format = '{0:.4e}'



        if 'COLAB_TPU_ADDR' in os.environ:
            self.tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            print('TPU is available.')

        _config_path = os.path.expanduser(os.path.join(self.prompt4all_dir, 'prompt4all.json'))
        _config = {}

        if os.path.exists(_config_path):
            try:
                with open(_config_path) as f:
                    _config = json.load(f)
                    for k, v in _config.items():
                        try:
                            if k == 'floatx':
                                assert v in {'float16', 'float32', 'float64'}
                            if k not in ['prompt4all_dir', 'device', 'working_directory']:
                                self.__setattr__(k, v)
                        except Exception as e:
                            print(e)
            except ValueError as ve:
                print(ve)
        if 'PROMPT4ALL_WORKING_DIR' in os.environ:
            self.working_directory = os.environ['PROMPT4ALL_WORKING_DIR']
            os.chdir(os.environ['PROMPT4ALL_WORKING_DIR'])
        np.set_printoptions(formatter={'float_kind': lambda x: self.numpy_print_format.format(x)},precision=4,suppress=True)
        self.device = None

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "_context_handle" and value is None:
            raise ValueError("Context handle is none in context!!!")
        return value

    @property
    def module_dict(self):
        return self._module_dict

    def get_module(self, cls_name, module_name='module'):
        """Get the registry record.
        Args:
            module_name ():
            cls_name ():
        Returns:
            class: The corresponding class.
        """
        if module_name not in self._module_dict:
            raise KeyError('{module_name} is not in registry')
        dd = self._module_dict[module_name]
        if cls_name not in dd:
            raise KeyError('{cls_name} is not registered in {module_name}')

        return dd[cls_name]




    def regist_resources(self,resource_name,resource ):
        if not hasattr(self._thread_local_info,'resources'):
            self._thread_local_info.resources=OrderedDict()
        self._thread_local_info.resources[resource_name]=resource
        return self._thread_local_info.resources[resource_name]

    def get_resources(self,resource_name):
        if not hasattr(self._thread_local_info, 'resources'):
            self._thread_local_info.resources = OrderedDict()
        if resource_name in self._thread_local_info.resources:
            return self._thread_local_info.resources[resource_name]
        else:
            return None








def _context():
    """
    Get the global _context, if context is not created, create a new one.
    Returns:
        _Context, the global context in PyNative mode.
    """
    global _prompt4all_context
    if _prompt4all_context is None:
        _prompt4all_context = _Context()
    return _prompt4all_context
