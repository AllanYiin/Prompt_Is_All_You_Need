import logging
import sys
from io import StringIO
from typing import Any, BinaryIO, Container, Iterator, Optional, cast
from pdfminer.converter import (
    XMLConverter,
    HTMLConverter,
    TextConverter,
    PDFPageAggregator,
    HOCRConverter,
)
from pdfminer.image import ImageWriter
from pdfminer.layout import LAParams, LTPage
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename, FileOrName, AnyIO

from pdfminer.layout import LTTextContainer

__all__ = ["get_document_text","extract_pages"]



def extract_pages(
    pdf_file: FileOrName,
    password: str = "",
    page_numbers: Optional[Container[int]] = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: Optional[LAParams] = None,
) -> Iterator[LTPage]:
    """Extract and yield LTPage objects

    :param pdf_file: Either a file path or a file-like object for the PDF file
        to be worked on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param page_numbers: List of zero-indexed page numbers to extract.
    :param maxpages: The maximum number of pages to parse
    :param caching: If resources should be cached
    :param laparams: An LAParams object from pdfminer.layout. If None, uses
        some default settings that often work well.
    :return: LTPage objects
    """
    if laparams is None:
        laparams = LAParams()

    with open_filename(pdf_file, "rb") as fp:
        fp = cast(BinaryIO, fp)  # we opened in binary mode
        resource_manager = PDFResourceManager(caching=caching)
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        for page in PDFPage.get_pages(
            fp, page_numbers, maxpages=maxpages, password=password, caching=caching
        ):
            interpreter.process_page(page)
            layout = device.get_result()
            yield layout


def get_document_text(filename):
    page_map = []
    offset = 0

    for page_num, page in enumerate(extract_pages(filename)):
        page_text = ''
        for element in page:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()

        page_map.append((page_num, offset, page_text))
        offset += len(page_text)

    return page_map