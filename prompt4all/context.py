import inspect
import json
import locale
import os
import platform
import sys
import threading
import traceback
import linecache
from collections import OrderedDict
from functools import partial
import numpy as np

_prompt4all_context = None

__all__ = ["sanitize_path", "split_path", "make_dir_if_need", "_context", "get_sitepackages", "PrintException"]


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
        path = os.path.join(os.path.expanduser("~"), path[2:])
    path = os.path.abspath(path)
    return path.strip().replace('\\', '/')
    # if isinstance(path, str):
    #     return os.path.normpath(path.strip()).replace('\\', '/')
    # else:
    #     return path


def split_path(path: str):
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
        if ext in ['.tar', '.pth', '.pkl', '.ckpt', '.bin', '.pt', '.zip']:
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


def is_instance(instance, check_class):
    if not inspect.isclass(instance) and inspect.isclass(check_class):
        mro_list = [b.__module__ + '.' + b.__qualname__ for b in instance.__class__.__mro__]
        return check_class.__module__ + '.' + check_class.__qualname__ in mro_list
    elif inspect.isclass(instance) and isinstance(check_class, str):
        mro_list = [b.__module__ + '.' + b.__qualname__ for b in instance.__mro__]
        mro_list2 = [b.__qualname__ for b in instance.__mro__]
        return check_class in mro_list or check_class in mro_list2
    elif not inspect.isclass(instance) and isinstance(check_class, str):
        mro_list = [b.__module__ + '.' + b.__qualname__ for b in instance.__class__.__mro__]
        mro_list2 = [b.__qualname__ for b in instance.__class__.__mro__]
        return check_class in mro_list or check_class in mro_list2
    elif isinstance(check_class, tuple):
        return any([is_instance(instance, cc) for cc in check_class])
    else:
        if not inspect.isclass(check_class):
            print(red_color('Input check_class {0} should a class, but {1}'.format(check_class, type(check_class))))
        return False


def PrintException():
    """
        Print exception with the line_no.

    """
    exc_type, exc_obj, tb = sys.exc_info()
    traceback.print_exception(*sys.exc_info())
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}\n'.format(filename, lineno, line.strip(), exc_obj))
    traceback.print_exc(limit=None, file=sys.stderr)
    # traceback.print_tb(tb, limit=1, file=sys.stdout)
    # traceback.print_exception(exc_type, exc_obj, tb, limit=2, file=sys.stdout)


def get_sitepackages():  # pragma: no cover
    installed_packages = None
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
        super().__setattr__('is_initialized', False)
        self._thread_local_info = _ThreadLocalInfo()
        self._context_handle = OrderedDict()
        self._errors_config = OrderedDict()
        self._module_dict = dict()
        self.conversation_history = None
        self.print = partial(print, flush=True)

        self.prompt4all_dir = self.get_prompt4all_dir()
        self.service_type = None
        self.deployments = []
        self.baseChatGpt = None
        self.summaryChatGpt = None
        self.imageChatGpt = None
        self.otherChatGpt = None
        self.state = None
        self.assistant_state = None
        self.status_word = ''
        self.counter = 0
        self.sql_engine = None
        self.conn_string = None
        self.databse_schema = None
        self.is_db_enable = False
        self.numpy_print_format = '{0:.4e}'
        self.plateform = None
        self.whisper_model = None
        self.current_assistant = None
        self.assistants = []
        self.placeholder_lookup = {}
        self.citations = []

        if os.path.exists(os.path.join(self.get_prompt4all_dir(), 'session.json')):
            self.load_session(os.path.join(self.get_prompt4all_dir(), 'session.json'))

        else:
            self.conversation_history = None
            self.locale = locale.getdefaultlocale()[0].lower()

            self.plateform = self.get_plateform()
            self.numpy_print_format = '{0:.4e}'
            np.set_printoptions(formatter={'float_kind': lambda x: self.numpy_print_format.format(x)}, precision=4,
                                suppress=True)
            self.is_db_enable = False
        self.conn_string = 'mssql+pyodbc://@' + 'localhost' + '/' + 'AdventureWorksDW2022' + '?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
        self.databse_schema = open("examples/query_database/schema.txt", encoding="utf-8").read()

        if 'PROMPT4ALL_WORKING_DIR' in os.environ:
            self.working_directory = os.environ['PROMPT4ALL_WORKING_DIR']
            os.chdir(os.environ['PROMPT4ALL_WORKING_DIR'])
        else:
            self.working_directory = os.getcwd()

        super().__setattr__('is_initialized', True)
        self.write_session()

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

    def write_session(self, session_path=None):
        if session_path is None:
            session_path = os.path.join(self.get_prompt4all_dir(), 'session.json')
        try:
            with open(session_path, 'w') as f:
                session = self.__dict__.copy()
                session.pop('_thread_local_info')
                session.pop('_context_handle')
                session.pop('_module_dict')
                session.pop('conversation_history')
                session.pop('is_initialized')
                session.pop('sql_engine')
                session.pop('status_word')
                session.pop('counter')
                session.pop('print')
                session.pop('current_assistant')
                session.pop('assistant_state')
                session.pop('placeholder_lookup')
                session.pop('citations')
                session['assistants'] = [a.json() if is_instance(a, "Assistant") else a for a in self.assistants]
                session['baseChatGpt'] = self.baseChatGpt if isinstance(self.baseChatGpt,
                                                                        str) else self.baseChatGpt.api_model if self.baseChatGpt else None
                session['summaryChatGpt'] = self.summaryChatGpt if isinstance(self.summaryChatGpt,
                                                                              str) else self.summaryChatGpt.api_model if self.summaryChatGpt else None
                session['imageChatGpt'] = self.imageChatGpt if isinstance(self.imageChatGpt,
                                                                          str) else self.imageChatGpt.api_model if self.imageChatGpt else None
                session['otherChatGpt'] = self.otherChatGpt if isinstance(self.otherChatGpt,
                                                                          str) else self.otherChatGpt.api_model if self.otherChatGpt else None
                session['state'] = self.state if isinstance(self.state, str) else str(
                    self.state.value) if self.state else None

                f.write(json.dumps(session, indent=4, ensure_ascii=False))
        except IOError:
            # Except permission denied.
            pass

    def load_session(self, session_path=None):
        if session_path is None:
            session_path = os.path.join(self.get_prompt4all_dir(), 'session.json')

        if os.path.exists(session_path):
            try:
                with open(session_path) as f:
                    _session = json.load(f)
                    for k, v in _session.items():
                        try:
                            self.__setattr__(k, v)

                            if k == 'service_type' and self.service_type is None:
                                self.__setattr__(k, 'openai')

                        except Exception as e:
                            print(e)
            except ValueError as ve:
                print(ve)

    def regist_resources(self, resource_name, resource):
        if not hasattr(self._thread_local_info, 'resources'):
            self._thread_local_info.resources = OrderedDict()
        self._thread_local_info.resources[resource_name] = resource
        return self._thread_local_info.resources[resource_name]

    def get_resources(self, resource_name):
        if not hasattr(self._thread_local_info, 'resources'):
            self._thread_local_info.resources = OrderedDict()
        if resource_name in self._thread_local_info.resources:
            return self._thread_local_info.resources[resource_name]
        else:
            return None

    def __setattr__(self, name: str, value) -> None:
        try:
            object.__setattr__(self, name, value)
            if self.is_initialized:
                self.write_session(os.path.join(self.get_prompt4all_dir(), 'session.json'))
        except Exception as e:
            print(name, value, e)
            PrintException()


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
