import _pickle as pickle
import os
import shutil
from tqdm import tqdm
from prompt4all import context
from prompt4all.utils import pdf_utils

try:
    import urllib.request
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve

__all__ = ['process_file', 'write_file', 'read_file', 'make_directory', 'replace_content', 'copy_file', 'move_file',
           'delete_file', 'append_file', 'unpickle', 'pickle_it', 'download_file']


def pickle_it(file_path, obj):
    """Pickle the obj

    Args:
        file_path (str):
        obj (obj):
    """

    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def process_file(file, state):
    if file is None:
        return '', state
    else:
        folder, filename, ext = context.split_path(file.name)
        if file.name.lower().endswith('.pdf'):
            _pdfdoc = pdf_utils.PDFDoc(file.name)
            _pdfdoc.parsing_save()
            _pdf.vectorization()
            yield return_text, state
        else:
            with open(file.name, encoding="utf-8") as f:
                content = f.read()
                print(content)
            yield content, state


def write_file(relative_path: str, content: str) -> str:
    path = absolute(relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(content)
    return f"Written content to file at {relative_path}"


def read_file(relative_path: str) -> str:
    path = absolute(relative_path)
    with open(path, "r") as file:
        content = file.read()
    return content


def make_directory(relative_path: str) -> str:
    path = absolute(relative_path)
    os.makedirs(path, exist_ok=True)
    return f"Created directory at {path}"


def replace_content(relative_path: str, pattern: str, replacement: str) -> str:
    path = absolute(relative_path)
    with open(path, "r") as file:
        content = file.read()
    content = re.sub(pattern, replacement, content)
    with open(path, "w") as file:
        file.write(content)
    return f"Replaced content in file at {relative_path}"


def copy_file(source_path: str, destination_path: str) -> str:
    src_path = absolute(source_path)
    dest_path = absolute(destination_path)
    shutil.copy(src_path, dest_path)
    return f"Copied file from {source_path} to {destination_path}"


def move_file(source_path: str, destination_path: str) -> str:
    src_path = absolute(source_path)
    dest_path = absolute(destination_path)
    shutil.move(src_path, dest_path)
    return f"Moved file from {source_path} to {destination_path}"


def delete_file(relative_path: str) -> str:
    path = absolute(relative_path)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        return "Invalid path."
    return f"Deleted {relative_path}"


def append_file(relative_path: str, content: str) -> str:
    path = absolute(relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        file.write(content)
    return f"Appended content to file at {relative_path}"


class TqdmProgress(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_file(src_url: str, destination_folder: str, filename: str = None) -> str:
    if os.path.exists(os.path.join(destination_folder, filename)):
        print('archive file is already existing, donnot need download again.')
        return True
    else:
        try:
            with TqdmProgress(unit='B', unit_scale=True, leave=True, miniters=10, desc='') as t:  # all optional kwargs
                urlretrieve(src_url, filename=os.path.join(destination_folder, filename), reporthook=t.update_to,
                            data=None)
            return True
        except Exception as e:
            print('***Cannot download data,.\n', flush=True)
            print(e)
            return False
