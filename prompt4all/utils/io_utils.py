from prompt4all.utils.pdf_utils import get_document_text
from prompt4all import context

def process_file(file, state):
    if file is None:
        return '', state
    else:
        folder, filename, ext = context.split_path(file.name)
        if file.name.lower().endswith('.pdf'):
            doc_map = get_document_text(file.name)
            return_text = ''
            for pg, offset, text in doc_map:
                return_text += text + '\n'
                return_text += 'page {0}'.format(pg + 1) + '\n''\n'
            yield return_text, state
        else:
            with open(file.name, encoding="utf-8") as f:
                content = f.read()
                print(content)
            yield content, state