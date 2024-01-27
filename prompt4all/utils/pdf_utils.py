import logging
import sys
import os
import io
import time

from tqdm import tqdm
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
    LTTextBoxHorizontal, LTText, LTAnno, LTTextLineHorizontal, LTTextBoxVertical
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, resolve1
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename, FileOrName, AnyIO
from pdfminer.high_level import extract_text_to_fp
from prompt4all.context import *
from prompt4all.utils.text_utils import *
from prompt4all.api.memories import *
from prompt4all.utils.vector_utils import *
from prompt4all.utils.io_utils import *
from prompt4all.common import *
from prompt4all.api import memories
from tqdm import tqdm
from urllib.parse import urlparse, unquote
import requests
from openai import OpenAI

client = OpenAI()
client._custom_headers['Accept-Language'] = 'zh-TW'
__all__ = ["get_document_text", "PDF", "PDFPagex", "Table", "Row", "Headers"]

make_dir_if_need('./download_pdfs')


def get_chars(item: LTTextContainer):
    results = []
    for e in item._objs:
        if isinstance(e, LTTextLine):
            for char in e._objs:
                if isinstance(char, LTChar):
                    results.append(char)
        elif isinstance(e, LTTextBox):
            results.append(get_chars(e))
    return results


def get_major_font_name(item: LTTextContainer):
    size_list = [c.fontname for c in get_chars(item)]
    most_common = Counter(size_list).most_common(1)
    return most_common[0][0] if most_common else ''


def get_major_font_size(item: LTTextContainer):
    size_list = [c.height for c in get_chars(item)]
    most_common = Counter(size_list).most_common(1)
    return most_common[0][0] if most_common else 8


def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme in ['http', 'https', 'ftp', 'ftps']:
        return False
    elif url_parsed.scheme in ('file', ''):  # Possibly a local file
        if os.path.exists(url_parsed.path):
            return True
    else:
        return None


class Headers(object):
    def __init__(self, cells: List[Optional[LTTextContainer]]):
        self.cells = list(sorted(cells, key=lambda e: e.x0, reverse=False))
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
        slots = [None] * len(base_lines)  # builtins.max(,len(row_lines))
        is_add_slut = False
        row.cells.sort(key=lambda e: e.x0, reverse=False)
        for i in range(len(row.cells)):
            is_add_slut = False
            _size = get_major_font_size(row.cells[i]) * 2
            candidate = [cidx for (cidx, c) in enumerate(self.cells) if abs(row.cells[i].x0 - c.x0) < _size
                         or abs(row.cells[i].x1 - c.x1) < _size
                         or abs((row.cells[i].x0 + row.cells[i].x1) / 2 - (c.x0 + c.x1) / 2) < _size]
            if len(candidate) == 0:
                candidate = [cidx for (cidx, c) in enumerate(self.cells) if
                             i + 1 < len(row.cells) and row.cells[i].x1 < c.x0 and row.cells[
                                 i + 1].x0 + _size > c.x1]
                if len(candidate) == 1:
                    insert_idx = candidate[0] + 1
                    slots.insert(insert_idx, row.cells[i])
                    new_h = LTTextBoxHorizontal()
                    new_h.set_bbox(row.cells[i].bbox)
                    self.cells.insert(insert_idx, new_h)
                    is_add_slut = True
            if len(candidate) == 0:
                candidate = [cidx for (cidx, c) in enumerate(self.cells) if
                             row.cells[i].x0 > c.x0 and row.cells[i].x1 <= c.x1]
            if len(candidate) == 0:
                candidate = [cidx for (cidx, c) in enumerate(self.cells) if
                             row.cells[i].x0 < c.x0 and row.cells[i].x1 > c.x1]

            if not is_add_slut and len(candidate) == 1:
                cidx = candidate[0]
                if slots[cidx] is None:
                    slots[cidx] = row.cells[i]
                else:
                    # 若已經有數據，新的數據在外側
                    if row.cells[i].x0 - slots[cidx].x1 > - get_major_font_size(row.cells[i]):
                        insert_idx = cidx + 1
                        slots.insert(insert_idx, row.cells[i])
                        new_h = LTTextBoxHorizontal()
                        new_h.set_bbox(row.cells[i].bbox)
                        self.cells.insert(insert_idx, new_h)
                    else:
                        if row.cells[i].x0 > slots[cidx].x0:
                            insert_idx = cidx + 1
                        else:
                            insert_idx = cidx

                        slots.insert(insert_idx, row.cells[i])
                        new_h = LTTextBoxHorizontal()
                        new_h.set_bbox(row.cells[i].bbox)
                        self.cells.insert(insert_idx, new_h)
                        if row.cells[i].x0 > slots[cidx].x0:
                            self.cells[cidx].set_bbox(
                                (self.cells[cidx].bbox[0], self.cells[cidx].bbox[1], row.cells[i].bbox[0],
                                 self.cells[cidx].bbox[3]))
                            self.cells[cidx + 1].set_bbox(
                                (row.cells[i].bbox[0], self.cells[cidx + 1].bbox[1], row.cells[i].bbox[2],
                                 self.cells[cidx + 1].bbox[3]))
                        else:
                            self.cells[cidx].set_bbox(
                                (self.cells[cidx].bbox[0], self.cells[cidx].bbox[1], row.cells[i].bbox[0],
                                 self.cells[cidx].bbox[3]))
                            self.cells[cidx + 1].set_bbox(
                                (row.cells[i].bbox[0], self.cells[cidx + 1].bbox[1], row.cells[i].bbox[2],
                                 self.cells[cidx + 1].bbox[3]))
        for i in range(len(slots)):
            if slots[i] is None:
                slots[i] = LTTextBoxHorizontal()
        return slots


class Row(object):
    def __init__(self, header, cells: List[Optional[LTTextContainer]]):
        self.header = header

        self.cells = list(sorted(cells, key=lambda e: e.x0, reverse=False))
        slots = self.header.check_data(self)
        self.cells = slots

    @property
    def snap_lines(self):
        return [(self.header.cells[idx].x0 + self.header.cells[idx].x1) / 2 if self.cells[idx] is None else (self.cells[
                                                                                                                 idx].x0 +
                                                                                                             self.cells[
                                                                                                                 idx].x1) / 2
                for idx in range(len(self.cells))]

    @property
    def indexes(self):
        return [None if t is None else t.index for t in self.cells if t.index >= 0]

    @property
    def data(self):
        return [None if t is None else convert_data(t.get_text().strip()) for t in self.cells]

    def __repr__(self) -> str:
        return "Row :{0}".format(str(['' if t is None else t.get_text().strip() for t in self.cells]))

    def __len__(self):
        return self.cells.__len__()


class Table(object):
    def __init__(self, header, rows: List[Optional[Row]], placeholder=None, title=None):
        self.header = header
        self.rows = []
        self.title = title
        self.rows.extend(rows)
        for i in range(len(self.rows)):
            r = self.rows[i]
            if len(r) != len(self.header):
                self.rows[i].cells = self.header.check_data(r)
        self._df = None
        if placeholder is None:
            self.placeholder = builtins.min(self.indexes)
        else:
            self.placeholder = placeholder
        slots_list = []

        print('')

    @property
    def columns(self):
        for r in self.rows:
            if len(r) != len(self.header):
                self.header.check_data(t)
        return {i: [r[i] for r in self.data] for i in range(len(self.header))}

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


def splice_text(row_texts):
    text_output = ""
    last_fontsize = None
    text_lengths = []
    for row_text in row_texts:
        for element in row_text:
            if isinstance(element, LTTextBoxHorizontal):
                current_text = element.get_text().strip()
                current_fontsize = get_major_font_size(element)
                text_lengths.append(len(current_text))

                # 判斷段落是否應該換行
                if len(current_text) < (sum(text_lengths) / len(text_lengths)) * 0.8 or (
                        last_fontsize and current_fontsize != last_fontsize):
                    text_output += "\n"

                text_output += current_text
                last_fontsize = current_fontsize
    return text_output


def arrange_LTChar_extract_table(item: LTTextContainer, margin=0.5):
    # rows = sorted(list(set(c.bbox[3] for c in item)), reverse=True)
    item.sort(key=lambda e: (-e.y1, e.x0), reverse=False)
    row_texts = []
    row_text = []
    col_text = LTTextBoxHorizontal()
    while len(item) > 0:
        _char = item[0]
        row_candidate = [item[idx] for idx in range(len(item)) if abs(_char.y1 - item[idx].y1) <= 0.5 * _char.size]
        x0s = np.argsort(np.array([t.x0 for t in row_candidate]))
        prev_element = None
        for idx in x0s:
            _char = row_candidate[idx]
            if idx == 0:
                col_text.add(_char)
            elif col_text.get_text().endswith('   '):
                if len(col_text._objs) > 0:
                    row_text.append(col_text)
                col_text = LTTextBoxHorizontal()
                col_text.add(_char)
            elif prev_element and _char.x0 - prev_element.x1 > margin * _char.size:
                if len(col_text._objs) > 0:
                    row_text.append(col_text)
                col_text = LTTextBoxHorizontal()
                col_text.add(_char)
            elif prev_element and _char.x0 - prev_element.x1 <= margin * _char.size:

                col_text.add(_char)
            else:
                if len(col_text._objs) > 0:
                    row_text.append(col_text)
                col_text = LTTextBoxHorizontal()
                col_text.add(_char)
            prev_element = _char
        if len(col_text.get_text()) > 0:
            if len(col_text._objs) > 0:
                row_text.append(col_text)
        row_texts.append(row_text)
        col_text = LTTextBoxHorizontal()
        row_text = []
        for r in row_candidate:
            item.remove(r)

    return row_texts


def arrange_LTTextbox_horizontal_extract_table(item: LTTextContainer, margin=0.5):
    # rows = sorted(list(set(c.bbox[3] for c in item)), reverse=True)
    item = list(sorted(item, key=lambda e: (-e.y1), reverse=False))
    row_texts = []

    while len(item) > 0:
        _textbox = item[0]
        font_size = get_major_font_size(_textbox)
        row_candidate = [item[idx] for idx in range(len(item)) if abs(_textbox.y1 - item[idx].y1) <= 0.5 * font_size]
        row_candidate = list(sorted(row_candidate, key=lambda e: (e.x0), reverse=False))
        row_texts.append(row_candidate)

        for r in row_candidate:
            item.remove(r)

        page_texts = []
        tables = []
        while len(row_texts) > 0:
            row_text = row_texts.pop(0)
            if len(row_text) == 1:
                page_texts.append(row_text)
            elif len(row_text) >= 3:
                if len(row_texts) == 0:
                    page_texts.append(row_text)
                    break
                next_row_text = row_texts.pop(0)
                if len(row_texts) == 0:
                    page_texts.append(next_row_text)
                    break
                next2_row_text = row_texts.pop(0)
                if len(next_row_text) >= 3 and len(next2_row_text) >= 3:
                    header = Headers(row_text)
                    rows = [Row(header, next_row_text), Row(header, next2_row_text)]
                    for k in range(3, len(row_texts) - idx):
                        if len(row_texts) == 0:
                            break
                        next3_row_text = row_texts.pop(0)
                        if len(next3_row_text) >= 3 and len(next3_row_text) >= len(row_text) - 2:
                            rows.append(Row(header, next3_row_text))
                        else:
                            tb = Table(header, rows, placeholder=header.cells[0].index)
                            self._tables.append(tb)
                            tables.append(tb)
                            page_texts.append(next3_row_text)
                            break
                else:
                    page_texts.append(row_text)
                    page_texts.append(next_row_text)
                    page_texts.append(next2_row_text)
            else:
                page_texts.append(row_text)
    return splice_text(page_texts), tables


def arrange_LTTextbox_vertical_extract_table(item: LTTextContainer, margin=0.5):
    # rows = sorted(list(set(c.bbox[3] for c in item)), reverse=True)
    item = list(sorted(item, key=lambda e: (e.x0), reverse=False))
    col_texts = []

    while len(item) > 0:
        _textbox = item[0]
        font_size = get_major_font_size(_textbox)
        col_candidate = [item[idx] for idx in range(len(item)) if abs(_textbox.x0 - item[idx].x0) <= 0.5 * font_size]
        col_candidate = list(sorted(col_candidate, key=lambda e: (-e.y1), reverse=False))
        col_texts.append(col_candidate)

        for r in col_candidate:
            item.remove(r)
    return col_texts


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
        if len(layout_arr) > 0:
            distance_arr = np.sqrt(((layout_arr - point_array) ** 2).sum(axis=-1))
            closest_idx = distance_arr.min(axis=-1).argmin()
            return text_areas[closest_idx], distance_arr[closest_idx, 0]
        return None, None

    def check_end_state(self, prev_text_area: LTTextContainer, current_text_area: LTTextContainer):
        frequent_font_name = get_major_font_name(current_text_area)
        prev_font_name = get_major_font_name(prev_text_area)
        chars = get_chars(prev_text_area)
        if chars and len(chars) > 0:
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
                frequent_font_height = get_major_font_size(element)

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
                                                                                             frequent_font_height, txt))
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

    @property
    def page_split_text(self):
        splited_text = []
        table_placeholders = [t.placeholder for t in self.tables] if len(self.tables) > 0 else []
        _page_text = ''
        prev_item = None
        for element in self.elements:
            if isinstance(element, LTTextContainer):
                page_width = self.base.width
                wh_ratio = builtins.abs((element.x1 - element.x0) / (element.y1 - element.y0))
                frequent_font_height = get_major_font_size(element)

                txt = element.get_text().strip()
                is_end = self.check_end_state(prev_item, element) if prev_item else False

                if hasattr(element, 'index') and element.index in table_placeholders:
                    if is_end:
                        splited_text.append(_page_text)
                        _page_text = ''
                    else:
                        _page_text += self.tables[
                            table_placeholders.index(element.index)].to_table_string()
                        splited_text.append(self.tables[
                                                table_placeholders.index(element.index)].to_table_string())
                elif len(txt) < 1:
                    sys.stdout.write(' page {0}: text skip!  len(txt) < 1  {1}\n'.format(self.page_number, txt))
                    pass
                elif frequent_font_height < 6.5:
                    sys.stdout.write(
                        ' page {0}: text skip! frequent_font_height < 6.5  {1} {2}\n'.format(self.page_number,
                                                                                             frequent_font_height, txt))
                    pass
                elif len(txt.split('\n')) / len(txt) > 0.3 and wh_ratio < 0.8:  # 直式的側邊修式字
                    sys.stdout.write(
                        ' page {0}: text skip!  len(txt.split(\n)) / len(txt) > 0.3 and wh_ratio < 0.8  {1}\n'.format(
                            self.page_number, txt))
                    pass
                elif hasattr(element, 'index') and element.index not in self._table_text_areas:

                    if is_end:
                        if len(_page_text + txt) < 200:
                            _page_text += txt
                        else:
                            splited_text.append(_page_text)
                            _page_text = txt
                    else:
                        _page_text += txt
                else:
                    sys.stdout.write(
                        ' page {0}: text skip!  else  {1}\n'.format(
                            self.page_number, txt))
                prev_item = element
        splited_text.append(_page_text)
        return splited_text

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

                    # img.save(os.path.join(folder, filename + '_parsing',
                    #                       '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                    #                                            'png' if img.mode == "RGBA" else 'jpg')))
                    self._images[lt_image.name] = arr
                except Exception as e:
                    try:
                        img = Image.open(io.BytesIO(lt_image.stream.get_data()))

                        # img.save(os.path.join(folder, filename + '_parsing',
                        #                       '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                        #                                            'png' if img.mode == "RGBA" else 'jpg')))
                        if img:
                            self._images[lt_image.name] = np.array(img)
                    except:
                        img = Image.frombytes(mode="1", data=lt_image.stream.get_data(),
                                              size=lt_image.srcsize,
                                              decoder_name='raw')
                        # img.save(os.path.join(folder, filename + '_parsing',
                        #                       '{0}_{1}.{2}'.format(self.page_number, lt_image.name,
                        #                                            'png' if img.mode == "RGBA" else 'jpg')))
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
        prev_text = None
        for (i, e) in enumerate(self.elements):
            if isinstance(e, LTFigure):
                tb = self._extract_table_from_figure(e, i)
                if len(tb) > 0 and prev_text:
                    tb[0].title = prev_text.get_text()
        page_texts, tables = arrange_LTTextbox_horizontal_extract_table(
            [e for e in self.elements if isinstance(e, LTTextContainer)], 1)
        colun_text = arrange_LTTextbox_vertical_extract_table(
            [e for e in self.elements if isinstance(e, LTTextContainer)], 1)
        print('')

        # text_areas = [e for e in self.elements if isinstance(e, LTTextContainer) and hasattr(e, 'index')]
        # prev_item = None
        # header_start = False
        # header_start_idx = -1
        # row_idx = 0
        # column_start = False
        # column_start_idx = -1
        # candidate_tables = []
        # candidate_header = []
        # candidate_rows_data = []
        # row_data = []
        # for i in range(len(text_areas)):
        #     current_item = text_areas[i]
        #     if current_item.index not in self._table_text_areas:
        #         if prev_item:
        #             if abs(current_item.y1 - prev_item.y1) < 1.5:
        #                 if not header_start and not column_start:
        #                     header_start = True
        #                     header_start_idx = i - 1
        #             else:
        #                 if header_start:
        #                     header_start = False
        #                     if i - header_start_idx > 2:
        #
        #                         candidate_header.append(Headers(text_areas[header_start_idx:i]))
        #
        #                         column_start = True
        #                         column_start_idx = i
        #                         row_idx = 0
        #                         row_data = [current_item]
        #                         for k in range(i + 1, len(text_areas)):
        #                             if abs(text_areas[k].y1 - current_item.y1) < 1.5:
        #                                 row_data.append(text_areas[k])
        #                                 # if len(row_data) == len(candidate_header_text[-1]):
        #                                 #     break
        #                         if len(candidate_header[-1]) - 2 < len(row_data) <= len(candidate_header[-1]):
        #                             candidate_rows_data.append(Row(candidate_header[-1], row_data))
        #                             row_data = []
        #                             row_idx += 1
        #                         else:
        #                             row_data = []
        #                             column_start = False
        #                             candidate_header.pop(-1)
        #
        #
        #                 elif column_start:
        #                     row_data = [current_item]
        #                     for k in range(i + 1, len(text_areas)):
        #                         if abs(text_areas[k].y1 - current_item.y1) < 1.5:
        #                             row_data.append(text_areas[k])
        #                             if len(row_data) == len(candidate_header[-1]):
        #                                 break
        #                     if len(candidate_header[-1]) - 2 < len(row_data) <= len(candidate_header[-1]):
        #                         candidate_rows_data.append(Row(candidate_header[-1], row_data))
        #
        #                         row_data = []
        #                         row_idx += 1
        #                     else:
        #                         row_data = []
        #                         column_start = False
        #                         candidate_tables.append(Table(candidate_header[-1], candidate_rows_data))
        #                         candidate_header = []
        #                         candidate_rows_data = []
        #     else:
        #         if column_start:
        #             row_data = []
        #             column_start = False
        #             candidate_tables.append(Table(candidate_header[-1], candidate_rows_data))
        #             candidate_header = []
        #             candidate_rows_data = []
        #
        #     prev_item = current_item
        #
        #     tt1 = list(chain.from_iterable([c.indexes for c in candidate_header]))
        #     tt2 = list(chain.from_iterable([c.indexes for c in candidate_rows_data]))
        #     tt3 = list(chain.from_iterable([c.indexes for c in candidate_tables]))
        #     tt4 = list(chain.from_iterable([c.indexes for c in row_data]))
        #
        #     self._table_text_areas = list(chain.from_iterable([tt1, tt2, tt3, tt4]))
        #
        # if len(candidate_tables) > 0:
        #     self._tables.extend(candidate_tables)

    def _extract_table_from_figure(self, figure: LTFigure, placeholder_index):
        """Extract a table from a figure, if possible. Returns a Table object, or None if no table is found."""
        # First, check if any of the child elements are tables
        rows = []
        prev_item = None

        row_text = ''
        result = ''

        text_areas = [e for e in figure._objs if isinstance(e, LTChar)]
        font_size = get_major_font_size(figure)
        textboxes = []
        row_data = []
        if text_areas and len(text_areas) > 0:
            text_areas.sort(key=lambda e: (-e.y1 // (0.5 * font_size), e.x0), reverse=False)
            current_textbox = LTTextBoxHorizontal()
            results = arrange_LTChar_extract_table(text_areas, 1.1)

            header = Headers(results[0])
            rows = []
            for idx in range(1, len(results)):
                items = results[idx]
                items.sort(key=lambda e: (e.x0), reverse=False)
                rows.append(Row(header, items))

            tb = Table(header, rows, placeholder=placeholder_index)
            self._tables.append(tb)
            return [tb]
            # if len(rows) > 2:
            #     response = client.chat.completions.create(
            #         model="gpt-3.5-turbo-1106",
            #         messages=[
            #             {'role': 'user',
            #              'content': '請以表格的形式來為我整理以下數據，資料行的單位可以註記於資料行header中，最後再將此表格轉換成tab分隔文字:\n\n{0}'.format(
            #                  '\n'.join(rows))}
            #         ],
            #         temperature=0.3,
            #         n=1,
            #         presence_penalty=0,
            #         stream=False
            #     )
            #     result = response.choices[0].message.content

        return []


class PDF(memories.BaseCache):
    def __init__(self, fp_path=None, password=None):
        """
        Args:
            fp_path (str): The file path of the PDF file to be processed.
            password (str, optional): The password for the PDF file, if it is password-protected. Defaults to None.
        Examples:
        >>> _pdf = PDF('../download_pdfs/4q22%20analyst%20meeting%20cn%20website.pdf')
        >>> _pdf.parsing_save()
        >>> _pdf.vectorization()
        >>> _pdf. query('光寶本期光電及雲端物聯網業務狀況如何?',top_k=3)
        >>> _pdf. query('光寶研發投資主要聚焦在哪些領域?',top_k=3)
        >>> print(_pdf.pages)

        """
        self.source = fp_path
        if is_local(fp_path) == False:
            import prompt4all.utils.web_utils as web_utils
            r = requests.get(fp_path, stream=True)
            with open(os.path.join('./download_pdfs', unquote(fp_path.split('/')[-1])), 'wb') as fd:
                chunk_size = 4 * 1024
                for chunk in r.iter_content(chunk_size):
                    fd.write(chunk)
            self.fp_path = os.path.join('./download_pdfs', unquote(fp_path.split('/')[-1]))
        else:
            self.fp_path = fp_path

        self.title = ''
        folder, filename, ext = split_path(self.fp_path)
        make_dir_if_need(os.path.join(folder, filename.replace('.', '_') + '_parsing'))
        self._cache: Dict[str, List[TextFragment]] = {}
        self._vector = None
        self._fragments = None

        self.password = password
        self.doc = None
        self.initial_doc()
        print('PDF document is ready, {0} pages found...'.format(resolve1(self.doc.catalog['Pages'])['Count']),
              flush=True)
        self._pages = {page_num: PDFPagex(page_num, page, self) for page_num, page in
                       tqdm(enumerate(self.extract_pages()), total=resolve1(self.doc.catalog['Pages'])['Count'])}
        self._tables = OrderedDict()
        self._images = OrderedDict()
        # self.vector_dict = OrderedDict()
        # self.corpus_dict = OrderedDict()
        # self.query_vector = None
        for p in self._pages.values():
            if len(p.tables) > 0:
                for t in p.tables:
                    self._tables[(p.page_number, t.placeholder)] = t
            if len(p.images) > 0:
                for k, img in p.images.items():
                    self._images[(p.page_number, k)] = img

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

    def paged_content(self):
        for p in self.pages:
            yield p.page_number, p.page_text, p.images, p.tables

    @property
    def tables(self):
        return self._tables

    @property
    def images(self):
        return self._images

    def parsing_save(self):
        folder, filename, ext = split_path(self.fp_path)
        filename = filename.replace('.', '_')
        make_dir_if_need(os.path.join(folder, filename + '_parsing'))
        for idx in range(len(self._pages)):
            page = self._pages[idx]

            for k, img in page.images.items():
                img = Image.fromarray(img)
                img.save(os.path.join(folder, filename + '_parsing',
                                      '{0}_{1}.{2}'.format(page.page_number, k,
                                                           'png' if img.mode == "RGBA" else 'jpg')))
            for _table in page.tables:
                with open(os.path.join(folder, filename + '_parsing',
                                       'Table_{0}_{1}.txt'.format(page.page_number, _table.placeholder)), 'w',
                          encoding='utf-8') as f:
                    f.write(_table.to_table_string())
            with open(os.path.join(folder, filename + '_parsing', 'Text_page_{0}.txt'.format(page.page_number)), 'w',
                      encoding='utf-8') as f:
                f.write(page.page_text)

    def vectorization(self):
        self.vector_dict = OrderedDict()
        self.corpus_dict = OrderedDict()

        folder, filename, ext = split_path(self.fp_path)
        filename = filename.replace('.', '_')
        make_dir_if_need(os.path.join(folder, filename + '_parsing'))
        if os.path.exists(os.path.join(folder, filename + '_parsing', 'vectors.pkl')):
            recovery = unpickle(os.path.join(folder, filename + '_parsing', 'vectors.pkl'))
            self.vector_dict = recovery
        else:
            n = 0
            for pg_num, pg in self._pages.items():

                textes = pg.page_split_text
                for txt in tqdm(textes):
                    self.corpus_dict[n] = txt
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=txt,
                        encoding_format="float"
                    )

                    tf = TextFragment(source=self.fp_path, page_num=pg_num, paragraph_num=n + 1, text=txt,
                                      embeddings=np.array(response.data[0].embedding), time=time.time())

                    self.vector_dict[n] = tf
                    n += 1
            pickle_it(os.path.join(folder, filename + '_parsing', 'vectors.pkl'), self.vector_dict)
        self.query_vector = np.stack([tf.embeddings for tf in self.vector_dict.values()], axis=0)

    def lookup(self, query_text: str, top_k: int = 5, min_similarity=0.88):
        """Look up based on prompt and top_k.
        Examples:
            >>> cxt=context._context()
            >>> cxt.memory=InMemoryCache()
            >>> cxt.memory.load()
            >>> cxt.memory.lookup('目前歐盟對於電動車以及充電裝有哪些重要、影響深遠的法案或政策呢?',5)
            '{"_cache": {"http:www.wikipedia.org": [{"text": "some text", "embeddings": {"__ndarray__": true, "embeddings": "[1.13222,2.64678,3.4567 ]", "dtype": "float64", "shape": [3]}}]}}'

        """
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text,
            encoding_format="float"
        )
        question_vector = np.expand_dims(np.array(response.data[0].embedding), 0)
        results = element_cosine_distance(question_vector, self._vector)[0]
        ranking = np.argsort(-results)
        ranked_results = results[ranking]
        filted_results = ranked_results[ranked_results > min_similarity]
        filted_ranking = ranking[ranked_results > min_similarity]
        if top_k > len(ranking):
            top_k = len(ranking)

        if len(filted_results) < top_k:
            filted_ranking = ranking[:top_k]
            filted_results = ranked_results[filted_ranking]

        filted_texts = [self._fragments[i].text for i in filted_ranking.tolist()]
        filted_vector = self._vector[filted_ranking]
        cross_corelation = element_cosine_distance(filted_vector, filted_vector)
        remove_list = []
        for i in range(len(filted_vector)):
            for j in range(i + 1, len(filted_vector)):
                if cross_corelation[i][j] > 0.97:
                    remove_list.append(j)
        filted_ranking = [item for i, item in enumerate(filted_ranking) if i not in remove_list]
        filted_results = [item for i, item in enumerate(filted_results) if i not in remove_list]

        query_results = OrderedDict()
        for k, (idx, prob) in enumerate(zip(filted_ranking, filted_results)):
            query_results[k] = {"similarity": '{0:%}'.format(prob), "text": self._fragments[idx].text,
                                "page": int(self._fragments[idx].source) + 1,
                                "paragraph_num": int(self._fragments[idx].paragraph_num) + 1}

        return query_results

    @property
    def vector(self):
        """Return the vector of the query."""
        return self._vector

    @property
    def cache(self):
        """Return the cache."""
        return self._cache

    @property
    def fragments(self):
        """Return the fragments."""
        return self._fragments

    def _sync(self):
        self._fragments = [tfrag for tfrag in list(chain.from_iterable([v for v in self._cache.values()])) if
                           tfrag is not None and
                           tfrag.embeddings is not None and len(tfrag.embeddings) == 1536]
        if self._fragments is not None and len(self._fragments) > 0:
            self._vector = np.stack([f.embeddings for f in self._fragments])

    def update(self, text_fragment: TextFragment) -> None:
        """Update cache based on source and text_fragment."""
        if text_fragment is None or text_fragment.source is None:
            pass
        else:
            if text_fragment.source not in self._cache:
                self._cache[text_fragment.source] = []
            self._cache[text_fragment.source].append(text_fragment)
            self._sync()

    def bulk_update(self, source: str, text_fragments: List[TextFragment]) -> None:
        """Bulk update cache based on source and list of text_fragment."""
        if source is None or len(text_fragments) == 0:
            pass
        else:
            try:
                if source not in self._cache:
                    self._cache[source] = []

                for i in range(len(text_fragments)):
                    text_fragments[i].source = source

                self._cache[source].extend(text_fragments)

                self._sync()
                print(source, 'is update in cache Success!!', flush=True)
            except Exception as e:
                print(source, 'is update in cache Fail!!', flush=True)
                PrintException()


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
