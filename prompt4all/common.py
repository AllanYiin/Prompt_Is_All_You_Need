import socket
from typing import List, Union
import numpy as np
from pydoc import locate
import importlib
import inspect

__all__ = ["find_available_port", "unpack_singleton", "red_color", "green_color", "blue_color", "cyan_color",
           "yellow_color", "orange_color", "gray_color", "violet_color", "magenta_color", "get_tool"]


def find_available_port(priority: Union[int, List[int]] = None) -> int:
    """
    Find an available port on the system.

    :param priority: Optional. A port number or a list of port numbers that will be checked first.
    :return: An available port number.
    """

    def _is_port_available(port: int) -> bool:
        """
        Check if a port is available.

        :param port: A port number.
        :return: True if the port is available; False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0

    if priority is not None:
        if isinstance(priority, int):
            if _is_port_available(priority):
                return priority
        elif isinstance(priority, list):
            for port in priority:
                if _is_port_available(port):
                    return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def unpack_singleton(x):
    if x is None:
        return None
    elif 'tensor' in x.__class__.__name__.lower() or isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (tuple, list)) and len(x) == 1:
        return x[0]
    return x


def red_color(text, bolder=False):
    if bolder:
        return '\033[1;31m{0}\033[0;0m'.format(text)
    else:
        return '\033[31m{0}\033[0;0m'.format(text)


def green_color(text, bolder=False):
    if bolder:
        return '\033[1;32m{0}\033[0;0m'.format(text)
    else:
        return '\033[32m{0}\033[0;0m'.format(text)


def blue_color(text, bolder=False):
    if bolder:
        return '\033[1;34m{0}\033[0m'.format(text)
    else:
        return '\033[34m{0}\033[0;0m'.format(text)


def cyan_color(text, bolder=False):
    if bolder:
        return '\033[1;36m{0}\033[0m'.format(text)
    else:
        return '\033[36m{0}\033[0;0m'.format(text)


def yellow_color(text, bolder=False):
    if bolder:
        return '\033[1;93m{0}\033[0m'.format(text)
    else:
        return '\033[93m{0}\033[0;0m'.format(text)


def orange_color(text, bolder=False):
    if bolder:
        return u'\033[1;33m%s\033[0m' % text
    else:
        return '\033[33m{0}\033[0;0m'.format(text)


def gray_color(text, bolder=False):
    if bolder:
        return u'\033[1;337m%s\033[0m' % text
    else:
        return '\033[37m{0}\033[0;0m'.format(text)


def violet_color(text, bolder=False):
    if bolder:
        return u'\033[1;35m%s\033[0m' % text
    else:
        return '\033[35m{0}\033[0;0m'.format(text)


def magenta_color(text, bolder=False):
    if bolder:
        return u'\033[1;35m%s\033[0m' % text
    else:
        return '\033[35m{0}\033[0;0m'.format(text)


def get_function(fn_name, module_paths=None):
    """
    Returns the function based on function name.

    Args:
        fn_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.

    Returns:
        The target function.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.

    """
    if callable(fn_name):
        return fn_name
    fn = None
    if (fn_name is not None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_name]))
            if fn is not None:
                break

    if fn is None:
        fn = locate(fn_name)
        if fn is not None:
            return fn
        else:
            return None
    else:
        return fn  # type: ignore


def get_tool(tool_name: str):
    if tool_name is None:
        return None
    fn_modules = ['prompt4all.tools.web_tools', 'prompt4all.tools.diagram_tools', 'prompt4all.tools.image_tools',
                  'prompt4all.tools.database_tools']

    try:
        if isinstance(tool_name, str):

            tool_fn = get_function(tool_name, fn_modules)
            return tool_fn
        # else:
        #     try:
        #         activation_fn = get_class(snake2camel(tool_name), fn_modules)
        #         return activation_fn()
        #     except Exception:
        #         activation_fn = get_class(tool_name, fn_modules)
        #         return activation_fn()

        else:
            raise ValueError('Unknown activation function/ class')
    except Exception as e:
        print(e)
        return None
