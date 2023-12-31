import logging
import sys
import os
import io
from collections import OrderedDict, Counter
import builtins
import numpy as np
import pandas as pd
from builtins import *
from lxml import html
from binascii import b2a_hex

from itertools import chain
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
    LTTextBoxHorizontal, LTText, LTAnno
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, resolve1
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename, FileOrName, AnyIO
from pdfminer.high_level import extract_text_to_fp
from prompt4all.context import *
from prompt4all.utils.text_utils import *
from tqdm import tqdm

__all__ = ["get_document_text", "PDF", "PDFPagex", "Table", "Row", "Headers"]


class Headers(object):
    def __init__(self, cells: List[Optional[LTTextContainer]]):
        self.cells = cells
        self.data_types = []

    def __len__(self):
        return self.cells.__len__()

    @property
    def header_text(self):
        return [t.get_text().strip() for t in self.cells]

    @property
    def snap_lines(self):
        return [(t.x0 + t.x1) / 2 for t in self.cells]

    @property
    def indexes(self):
        return [t.index for t in self.cells if t.index >= 0]

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

    @property
    def indexes(self):
        return [t.index for t in self.cells if t.index >= 0]

    @property
    def data(self):
        return [convert_data(t.get_text().strip()) for t in self.cells]

    def __repr__(self) -> str:
        return "Row :{0}".format(str([t.get_text().strip() for t in self.cells]))


class Table(object):
    def __init__(self, header, rows: List[Optional[Row]]):
        self.header = header
        self.rows = []
        self.rows.extend(rows)
        self._df = None
        self.placeholder = builtins.min(self.indexes)

    @property
    def indexes(self):
        indexes = []
        indexes.extend(self.header.indexes)
        indexes.extend(list(chain.from_iterable([r.indexes for r in self.rows])))
        return [i for i in indexes if i >= 0]

    @property
    def data(self):
        data = [self.header.header_text]
        for r in self.rows:
            data.append(r.data)
        return data

    @property
    def column_cnt(self):
        return len(self.header)

    @property
    def row_cnt(self):
        return len(self.rows)

    def to_dataframe(self):
        if self._df is None:
            self._df = pd.DataFrame([r.data for r in self.rows], columns=self.header.header_text)
        return self._df

    def to_table_string(self):
        return self.to_dataframe().to_string(index=False)

    def __repr__(self) -> str:
        return "Table :{0}".format(self.header.header_text)


def get_chats(item: LTTextContainer):
    results = []
    for e in item._objs:
        if isinstance(e, LTTextLine):
            for char in e._objs:
                if isinstance(char, LTChar):
                    results.append(char)
        elif isinstance(e, LTTextBox):
            results.append(get_chats(e))
    return results


def get_most_font_name(item: LTTextContainer):
    size_list = [c.fontname for c in get_chats(item)]
    most_common = Counter(size_list).most_common(1)
    return most_common[0][0] if most_common else ''


def get_most_font_size(item: LTTextContainer):
    size_list = [c.height for c in get_chats(item)]
    most_common = Counter(size_list).most_common(1)
    return most_common[0][0] if most_common else 8


class PDFPagex:
    def __init__(self, page_number: int, page: LTPage, parent=None):
        self.page_number = page_number
        if self.page_number == 14:
            print('')
        self.base = page
        self.parent = parent

        self._images = {}
        self._tables = []
        self.reference_mapping = {}
        self.extract_tables()
        [self.get_image(e) for e in self.elements if isinstance(e, (LTImage, LTFigure))]

        print(self.page_text)
        print('')

    def find_closest_text(self, point1):
        text_areas = [e for e in self.elements if
                      isinstance(e, LTTextContainer) and hasattr(e, 'index') and e.index not in self._table_text_areas]
        layout_arr = np.array([[[t.x0, t.y0], [t.x0, t.y1]] for t in text_areas])
        point_array = np.array(list(point1)).reshape((1, 1, 2))
        distance_arr = np.sqrt(((layout_arr - point_array) ** 2).sum(axis=-1))
        closest_idx = distance_arr.min(axis=-1).argmin()
        return text_areas[closest_idx], distance_arr[closest_idx, 0]

    def check_end_state(self, prev_text_area: LTTextContainer, current_text_area: LTTextContainer):
        frequent_font_name = get_most_font_name(current_text_area)
        prev_font_name = get_most_font_name(prev_text_area)
        chars = get_chats(prev_text_area)
        char = chars[-1]
        is_end = True
        full_rate = (char.x1 - prev_text_area.x0) / (prev_text_area.x1 - prev_text_area.x0)
        overlapping_rate = (prev_text_area.x1 - prev_text_area.x0) / (current_text_area.x1 - current_text_area.x0)
        if overlapping_rate < 0.5 or overlapping_rate > 2:
            full_rate = 0.5
        if ((char.x1 - prev_text_area.x0) / (prev_text_area.x1 - prev_text_area.x0) >= 0.9) and (
                frequent_font_name == prev_font_name) and prev_text_area.get_text().strip()[-1] not in ['.', '。']:
            return False
        elif prev_text_area.get_text().strip()[-1] in ['-', ',', ':', ';']:
            return False
        else:
            return True

    @property
    def page_text(self):
        table_placeholders = [t.placeholder for t in self.tables] if len(self.tables) > 0 else []
        _page_text = ''
        prev_item = None
        for element in self.elements:
            if isinstance(element, LTTextContainer):
                page_width = self.base.width
                wh_ratio = builtins.abs((element.x1 - element.x0) / (element.y1 - element.y0))
                frequent_font_height = get_most_font_size(element)

                txt = element.get_text().strip()
                is_end = self.check_end_state(prev_item, element) if prev_item else False

                if hasattr(element, 'index') and element.index in table_placeholders:
                    _page_text += ('\n' if is_end else '') + self.tables[
                        table_placeholders.index(element.index)].to_table_string()
                elif len(txt) < 1:
                    sys.stdout.write(' page {0}: text skip!  len(txt) < 1  {1}\n'.format(self.page_number, txt))
                    pass
                elif frequent_font_height < 6.5:
                    sys.stdout.write(
                        ' page {0}: text skip! frequent_font_height < 6.5  {1} {2}\n'.format(self.page_number,
                                                                                             frequent_font_heighttxt,
                                                                                             txt))
                    pass
                elif len(txt.split('\n')) / len(txt) > 0.3 and wh_ratio < 0.8:  # 直式的側邊修式字
                    sys.stdout.write(
                        ' page {0}: text skip!  len(txt.split(\n)) / len(txt) > 0.3 and wh_ratio < 0.8  {1}\n'.format(
                            self.page_number, txt))
                    pass
                elif hasattr(element, 'index') and element.index not in self._table_text_areas:
                    _page_text += ('\n' if is_end else '') + txt
                else:
                    sys.stdout.write(
                        ' page {0}: text skip!  else  {1}\n'.format(
                            self.page_number, txt))
                prev_item = element

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
            item1, dis1 = self.find_closest_text((lt_image.x0, lt_image.y0))
            item2, dis2 = self.find_closest_text((lt_image.x0, lt_image.y1))
            txt1 = item1.get_text() if hasattr(item1, 'get_text') else ''
            txt2 = item2.get_text() if hasattr(item2, 'get_text') else ''
            if item1 == item2:
                self.reference_mapping['image {0}'.format(lt_image.name)] = item1
            elif dis1 < dis2 * 0.7:
                self.reference_mapping['image {0}'.format(lt_image.name)] = item1
            elif dis2 < dis1 * 0.7:
                self.reference_mapping['image {0}'.format(lt_image.name)] = item2

            if lt_image.stream:
                folder, filename, ext = split_path(self.parent.fp_path)
                filename = filename.replace('.', '_')
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

    @property
    def tables(self):
        return self._tables

    def extract_tables(self):
        self._table_text_areas = []
        text_areas = [e for e in self.elements if isinstance(e, LTTextContainer) and hasattr(e, 'index')]
        prev_item = None
        header_start = False
        header_start_idx = -1
        row_idx = 0
        column_start = False
        column_start_idx = -1
        candidate_tables = []
        candidate_header = []
        candidate_rows_data = []
        row_data = []
        for i in range(len(text_areas)):
            current_item = text_areas[i]
            if current_item.index not in self._table_text_areas:
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
                                candidate_tables.append(Table(candidate_header[-1], candidate_rows_data))
                                candidate_header = []
                                candidate_rows_data = []
            else:
                if column_start:
                    row_data = []
                    column_start = False
                    candidate_tables.append(Table(candidate_header[-1], candidate_rows_data))
                    candidate_header = []
                    candidate_rows_data = []

            prev_item = current_item

            tt1 = list(chain.from_iterable([c.indexes for c in candidate_header]))
            tt2 = list(chain.from_iterable([c.indexes for c in candidate_rows_data]))
            tt3 = list(chain.from_iterable([c.indexes for c in candidate_tables]))
            tt4 = list(chain.from_iterable([c.indexes for c in row_data]))

            self._table_text_areas = list(chain.from_iterable([tt1, tt2, tt3, tt4]))

        if len(candidate_tables) > 0:
            self._tables.extend(candidate_tables)


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
        self.title = ''
        folder, filename, ext = split_path(self.fp_path)
        make_dir_if_need(os.path.join(folder, filename.replace('.', '_') + '_images'))
        self.password = password
        self.doc = None
        self.initial_doc()
        print('PDF document is ready, {0} pages found...'.format(resolve1(self.doc.catalog['Pages'])['Count']),
              flush=True)
        self._pages = {page_num: PDFPagex(page_num, page, self) for page_num, page in
                       tqdm(enumerate(self.extract_pages()), total=resolve1(self.doc.catalog['Pages'])['Count'])}
        self._tables = OrderedDict()
        self._images = OrderedDict()
        for p in self._pages.values():
            if len(p.tables) > 0:
                for t in p.tables:
                    self._tables[(p.page_number, t.placeholder)] = t
            if len(p.images) > 0:
                for k, img in p.images.items():
                    self._images[(p.page_number, k)] = img
        print(self.pages_text)
        print('')

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

    @property
    def doc_text(self):
        return '\n'.join(['page {0}:'.format(p.page_number) + '\n' + p.page_text for p in self.pages])

    @property
    def pages_text(self):
        return {p.page_number: p.page_text for p in self.pages}

    @property
    def tables(self):
        return self._tables

    @property
    def images(self):
        return self._images


#
# def extract_pages(
#         pdf_file: FileOrName,
#         password: str = "",
#         page_numbers: Optional[Container[int]] = None,
#         maxpages: int = 0,
#         caching: bool = True,
#         laparams: Optional[LAParams] = None,
# ) -> Iterator[LTPage]:
#     """Extract and yield LTPage objects
#
#     :param pdf_file: Either a file path or a file-like object for the PDF file
#         to be worked on.
#     :param password: For encrypted PDFs, the password to decrypt.
#     :param page_numbers: List of zero-indexed page numbers to extract.
#     :param maxpages: The maximum number of pages to parse
#     :param caching: If resources should be cached
#     :param laparams: An LAParams object from pdfminer.layout. If None, uses
#         some default settings that often work well.
#     :return: LTPage objects
#     """
#     if laparams is None:
#         laparams = LAParams()
#
#     with open_filename(pdf_file, "rb") as fp:
#         fp = cast(BinaryIO, fp)  # we opened in binary mode
#         resource_manager = PDFResourceManager(caching=caching)
#         device = PDFPageAggregator(resource_manager, laparams=laparams)
#         interpreter = PDFPageInterpreter(resource_manager, device)
#         for page in PDFPage.get_pages(
#                 fp, page_numbers, maxpages=maxpages, password=password, caching=caching
#         ):
#             interpreter.process_page(page)
#             layout = device.get_result()
#             yield layout


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
