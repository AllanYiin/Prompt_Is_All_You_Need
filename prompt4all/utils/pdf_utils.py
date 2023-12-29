import logging
import sys
import os
import io
from builtins import *
from lxml import html
from binascii import b2a_hex
import numpy as np
import copy
from PIL import Image
from io import StringIO
from typing import Any, BinaryIO, Container, Iterator, Optional, cast, List
import pdfplumber
import pdfminer
from pdfminer.converter import TextConverter, XMLConverter, HTMLConverter, PDFPageAggregator, HOCRConverter
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.image import ImageWriter
from pdfminer.layout import LAParams, LTPage, LTTextBox, LTTextLine, LTFigure, LTImage, LTChar, LTTextContainer, \
    LTTextBoxHorizontal
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, resolve1
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename, FileOrName, AnyIO
from pdfminer.high_level import extract_text_to_fp
from prompt4all.context import *
from tqdm import tqdm

__all__ = ["get_document_text", "extract_pages"]


class Headers(object):
    def __init__(self, cells: List[Optional[LTTextContainer]]):
        self.cells = cells

    def __len__(self):
        return self.cells.__len__()

    @property
    def header_text(self):
        return [t.get_text() for t in self.cells]

    @property
    def snap_lines(self):
        return [(t.x0 + t.x1) / 2 for t in self.cells]

    def check_data(self, row):
        base_lines = self.snap_lines
        row_lines = row.snap_lines
        slots = [None] * len(base_lines)
        for i in range(len(row_lines)):
            idx = np.argmin(np.abs(np.array(base_lines) - row_lines[i]))
            if slots[idx] is None:
                slots[idx] = row.cells[i]
            else:
                print('reduplicate data')
        for i in range(len(base_lines)):
            if slots[i] is None:
                slots[i] = LTTextBoxHorizontal()
                slots[i].set_bbox(self.cells[i].bbox)

        return slots


class Row(object):
    def __init__(self, header, cells: List[Optional[LTTextContainer]]):
        self.header = header
        self.cells = cells
        slots = self.header.check_data(self)
        self.cells = slots

    @property
    def snap_lines(self):
        return [(t.x0 + t.x1) / 2 for t in self.cells]

    def __repr__(self) -> str:
        return "Row :{0}".format(str([t.get_text() for t in self.cells]))


class PDFPagex:
    def __init__(self, page_number: int, page: LTPage, parent=None):
        self.page_number = page_number
        self.base = page
        self.parent = parent
        self._images = {}
        [self.get_image(e) for e in self.elements if isinstance(e, (LTImage, LTFigure))]
        if self.page_number > 8:
            self.extract_tables()
        print(self._images)

    @property
    def page_text(self):
        _page_text = ''
        for element in self.elements:
            if isinstance(element, LTTextContainer):
                _page_text += element.get_text()
        return _page_text

    def get_image(self, image_item):
        """Try to save the image data from this LTImage object, and return the file name, if successful"""
        lt_image = None
        file_ext = None
        image_list = []
        figure_list = []
        if isinstance(image_item, LTImage):
            image_list.append(image_item)
        elif isinstance(image_item, LTFigure):
            figure_list.append(image_item)
        while len(figure_list) > 0:
            temp_list = []
            for i in range(len(figure_list)):
                temp_list.append(figure_list.pop(0))
            for this_item in temp_list:
                if this_item._objs:
                    for item in this_item._objs:
                        if isinstance(item, LTImage):
                            image_list.append(item)
                        elif isinstance(item, LTFigure):
                            figure_list.append(item)
        for lt_image in image_list:
            if lt_image.stream:
                folder, filename, ext = split_path(self.parent.fp_path)
                try:
                    color_space = lt_image.stream.attrs['ColorSpace'].name.replace('Device', '')
                except:
                    color_space = 'RGB'
                try:
                    buffer = io.BytesIO(lt_image.stream.get_data())

                    img = Image.frombytes(mode=color_space, data=lt_image.stream.get_data(),
                                          size=lt_image.srcsize,
                                          decoder_name='raw')
                    arr = np.array(img) if img else None
                    # arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8).reshape(
                    #     (lt_image.srcsize[1], lt_image.srcsize[0], -1))
                    #
                    # img = Image.fromarray(arr)

                    img.save(os.path.join(folder, filename + '_images',
                                          '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                                                               'png' if img.mode == "RGBA" else 'jpg')))
                    self._images[lt_image.name] = arr
                except Exception as e:
                    try:
                        img = Image.open(io.BytesIO(lt_image.stream.get_data()))

                        img.save(os.path.join(folder, filename + '_images',
                                              '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                                                                   'png' if img.mode == "RGBA" else 'jpg')))
                        if img:
                            self._images[lt_image.name] = np.array(img)
                    except:
                        img = Image.frombytes(mode="1", data=lt_image.stream.get_data(),
                                              size=lt_image.srcsize,
                                              decoder_name='raw')
                        img.save(os.path.join(folder, filename + '_images',
                                              '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                                                                   'png' if img.mode == "RGBA" else 'jpg')))
                        if img:
                            self._images[lt_image.name] = np.array(img)

    @property
    def elements(self):
        return self.base._objs

    @property
    def images(self):
        return self._images

    def extract_tables(self):
        text_areas = [e for e in self.elements if isinstance(e, LTTextContainer)]
        prev_item = None
        header_start = False
        header_start_idx = -1
        row_idx = 0
        column_start = False
        column_start_idx = -1
        candidate_header = []
        candidate_header_text = []
        candidate_rows_data = []
        for i in range(len(text_areas)):
            current_item = text_areas[i]
            if prev_item:
                if abs(current_item.y1 - prev_item.y1) < 1.5:
                    if not header_start and not column_start:
                        header_start = True
                        header_start_idx = i - 1
                else:
                    if header_start:
                        header_start = False
                        if i - header_start_idx > 2:

                            candidate_header.append(Headers(text_areas[header_start_idx:i]))

                            column_start = True
                            column_start_idx = i
                            row_idx = 0
                            row_data = [current_item]
                            for k in range(i + 1, len(text_areas)):
                                if abs(text_areas[k].y1 - current_item.y1) < 1.5:
                                    row_data.append(text_areas[k])
                                    # if len(row_data) == len(candidate_header_text[-1]):
                                    #     break
                            if len(candidate_header[-1]) - 2 < len(row_data) <= len(candidate_header[-1]):
                                candidate_rows_data.append(Row(candidate_header[-1], row_data))
                                row_data = []
                                row_idx += 1
                            else:
                                row_data = []
                                column_start = False
                                candidate_header.pop(-1)


                    elif column_start:
                        row_data = [current_item]
                        for k in range(i + 1, len(text_areas)):
                            if abs(text_areas[k].y1 - current_item.y1) < 1.5:
                                row_data.append(text_areas[k])
                                if len(row_data) == len(candidate_header[-1]):
                                    break
                        if len(candidate_header[-1]) - 2 < len(row_data) <= len(candidate_header[-1]):
                            candidate_rows_data.append(Row(candidate_header[-1], row_data))
                            row_data = []
                            row_idx += 1
                        else:
                            row_data = []
                            column_start = False

            prev_item = current_item
        print(candidate_header_text)


class PDF:
    def __init__(self, fp_path, password=None):
        """
        Args:
            fp_path (str): The file path of the PDF file to be processed.
            password (str, optional): The password for the PDF file, if it is password-protected. Defaults to None.
        Examples:
        >>> pdf = PDF('C:/Users/Allan/Downloads/JOItmC-08-00107-v2.pdf')
        >>> print(pdf.pages)

        """
        self.fp_path = fp_path
        folder, filename, ext = split_path(self.fp_path)
        make_dir_if_need(os.path.join(folder, filename + '_images'))
        self.password = password
        self.doc = None
        self.initial_doc()
        print('PDF document is ready, {0} pages found...'.format(resolve1(self.doc.catalog['Pages'])['Count']),
              flush=True)
        self._pages = {page_num: PDFPagex(page_num, page, self) for page_num, page in
                       tqdm(enumerate(self.extract_pages()), total=resolve1(self.doc.catalog['Pages'])['Count'])}

    def initial_doc(self):
        fp = open(self.fp_path, 'rb')
        parser = PDFParser(fp)
        # create a PDFDocument object that stores the document structure
        self.doc = PDFDocument(parser)
        # connect the parser and document objects
        parser.set_document(self.doc)
        # supply the password for initialization
        # self.doc.initialize(self.password)

    def extract_pages(self,
                      page_numbers: Optional[Container[int]] = None,
                      maxpages: int = 0,
                      caching: bool = True,
                      laparams: Optional[LAParams] = None,
                      ) -> Iterator[LTPage]:
        """Extract and yield LTPage objects
        :param page_numbers: List of zero-indexed page numbers to extract.
        :param maxpages: The maximum number of pages to parse
        :param caching: If resources should be cached
        :param laparams: An LAParams object from pdfminer.layout. If None, uses
            some default settings that often work well.
        :return: LTPage objects
        """
        if laparams is None:
            laparams = LAParams()

        with open_filename(self.fp_path, "rb") as fp:
            fp = cast(BinaryIO, fp)  # we opened in binary mode
            resource_manager = PDFResourceManager(caching=caching)
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)
            for page in PDFPage.get_pages(
                    fp, page_numbers, maxpages=maxpages, password=self.password, caching=caching
            ):
                interpreter.process_page(page)
                layout = device.get_result()
                yield layout

    @property
    def page_count(self):
        return len(self._pages)

    @property
    def pages(self):
        return list(self._pages.values())

    def get_page(self, page_number):
        return self._pages[page_number]


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
