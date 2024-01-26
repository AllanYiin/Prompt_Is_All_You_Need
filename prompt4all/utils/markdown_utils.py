import codecs
import html.entities as htmlentitydefs
import html.parser as HTMLParser
import optparse
import re
import sys
import urllib.parse as urlparse
import urllib.request as urllib
from textwrap import wrap

from bs4 import BeautifulSoup

__all__ = ["escape_markdown_characters", "HTML2Text", "htmltable2markdown"]

# Refactored dictionary with markdown escape characters
escape_chars = {
    "#": "\#",
    "*": "\*",
    "_": "\_",
    "[": "\[",
    "]": "\]",
    "(": "\(",
    ")": "\)",
    "`": "\`",
    ">": "\>",
    "+": "\+",
    "-": "\-",
    ".": "\.",
    "!": "\!",
    "|": "\|",
    "{": "\{",
    "}": "\}",
    "$": "\$",
    "\\": "\\\\"}


def escape_markdown_characters(input_string):
    """
    Escapes markdown characters in a given string.
    Args:
        input_string (str): The string to escape markdown characters in.
    Returns:
        str: String with markdown characters escaped.
    """
    return ''.join(escape_chars[char] if char in escape_chars else char for char in input_string)


# def convert_newlines_to_markdown(text):
#     """
#     Converts newlines in a text to markdown format.
#     Args:
#         text (str): The text to convert.
#     Returns:
#         str: Text with newlines converted to markdown format.
#     """
#     text = text.replace('
# ', '
# ').replace('
#
#
# ', '
#
# ')
#     return text

def key_exists(obj, key):
    """
    Checks if a key exists in a dictionary or object.
    Args:
        obj: The dictionary or object to check in.
        key: The key to check for.
    Returns:
        bool: True if key exists, False otherwise.
    """
    return key in obj


def convert_newlines_to_markdown(text):
    text = text.replace('\n', '  \n')
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text


# Configuration variables
UNICODE_SNOB = True
ESCAPE_SNOB = True
LINKS_EACH_PARAGRAPH = False
BODY_WIDTH = 78
SKIP_INTERNAL_LINKS = True
INLINE_LINKS = False
GOOGLE_LIST_INDENT = 36
IGNORE_ANCHORS = True
IGNORE_IMAGES = True
IGNORE_EMPHASIS = False


# Entity conversion functions and dictionaries
def name_to_codepoint(name):
    """
    Convert an HTML entity name to a codepoint.
    Args:
        name (str): HTML entity name.
    Returns:
        int: Unicode codepoint corresponding to the entity.
    """
    if name == 'apos':
        return ord("'")
    if hasattr(htmlentitydefs, "name2codepoint"):
        return htmlentitydefs.name2codepoint[name]
    else:
        entity = htmlentitydefs.entitydefs[name]
        if entity.startswith("&#") and entity.endswith(";"):
            return int(entity[2:-1])  # not in latin-1
        return ord(codecs.latin_1_decode(entity)[0])


unifiable = {'rsquo': "'", 'lsquo': "'", 'rdquo': '"', 'ldquo': '"',
             'copy': '(C)', 'mdash': '--', 'nbsp': ' ', 'rarr': '->', 'larr': '<-', 'middot': '*',
             'ndash': '-', 'oelig': 'oe', 'aelig': 'ae',
             'agrave': 'a', 'aacute': 'a', 'acirc': 'a', 'atilde': 'a', 'auml': 'a', 'aring': 'a',
             'egrave': 'e', 'eacute': 'e', 'ecirc': 'e', 'euml': 'e',
             'igrave': 'i', 'iacute': 'i', 'icirc': 'i', 'iuml': 'i',
             'ograve': 'o', 'oacute': 'o', 'ocirc': 'o', 'otilde': 'o', 'ouml': 'o',
             'ugrave': 'u', 'uacute': 'u', 'ucirc': 'u', 'uuml': 'u',
             'lrm': '', 'rlm': ''}

unifiable_n = {name_to_codepoint(k): v for k, v in unifiable.items()}


def is_only_whitespace(line):
    """
    Checks if a line consists only of whitespace characters.
    Args:
        line (str): The line to check.
    Returns:
        bool: True if the line contains only whitespace, False otherwise.
    """
    return line.strip() == ""


def header_level(tag):
    """
    Determines the level of a header tag (e.g., h1, h2).
    Args:
        tag (str): The HTML tag to check.
    Returns:
        int: The level of the header (1-9), or 0 if not a valid header.
    """
    if tag[0] == 'h' and len(tag) == 2:
        try:
            level = int(tag[1])
            return level if 1 <= level <= 9 else 0
        except ValueError:
            return 0


def parse_style_attribute(style):
    """
    Parses a CSS style attribute into a dictionary.
    Args:
        style (str): The CSS style attribute.
    Returns:
        dict: A dictionary of CSS properties and their values.
    """
    return dict(pair.strip().split(':', 1) for pair in style.split(';') if ':' in pair)


def parse_css(data):
    """
    Parses CSS data into a dictionary of selectors and their properties.
    Args:
        data (str): The CSS data to parse.
    Returns:
        dict: A dictionary of CSS selectors and properties.
    """
    # Removing @import sentences
    data = data.replace('@import', '')
    # Parsing CSS
    elements = dict(pair.strip().split('{', 1) for pair in data.split('}') if '{' in pair)
    return {selector: parse_style_attribute(styles) for selector, styles in elements.items()}


def compute_element_style(attrs, style_def, parent_style):
    """
    Computes the final style attributes of an element based on its class and parent style.
    Args:
        attrs (dict): The element's attributes.
        style_def (dict): The style definitions.
        parent_style (dict): The parent element's style.
    Returns:
        dict: The computed style attributes.
    """
    style = parent_style.copy()
    for css_class in attrs.get('class', '').split():
        style.update(style_def.get('.' + css_class, {}))
    return style


def google_list_style(style):
    """finds out whether this is an ordered or unordered list"""
    if 'list-style-type' in style:
        list_style = style['list-style-type']
        if list_style in ['disc', 'circle', 'square', 'none']:
            return 'ul'
    return 'ol'


def google_has_height(style):
    """check if the style of the element has the 'height' attribute explicitly defined"""
    if 'height' in style:
        return True
    return False


def google_text_emphasis(style):
    """return a list of all emphasis modifiers of the element"""
    emphasis = []
    if 'text-decoration' in style:
        emphasis.append(style['text-decoration'])
    if 'font-style' in style:
        emphasis.append(style['font-style'])
    if 'font-weight' in style:
        emphasis.append(style['font-weight'])
    return emphasis


def google_fixed_width_font(style):
    """check if the css of the current element defines a fixed width font"""
    font_family = ''
    if 'font-family' in style:
        font_family = style['font-family']
    if 'Courier New' == font_family or 'Consolas' == font_family:
        return True
    return False


def list_numbering_start(attrs):
    """extract numbering from list element attributes"""
    if 'start' in attrs:
        return int(attrs['start']) - 1
    else:
        return 0


class HTML2Text(HTMLParser.HTMLParser):
    def __init__(self, out=None, baseurl=''):
        HTMLParser.HTMLParser.__init__(self)

        # Config options
        self.unicode_snob = UNICODE_SNOB
        self.escape_snob = ESCAPE_SNOB
        self.links_each_paragraph = LINKS_EACH_PARAGRAPH
        self.body_width = BODY_WIDTH
        self.skip_internal_links = SKIP_INTERNAL_LINKS
        self.inline_links = INLINE_LINKS
        self.google_list_indent = GOOGLE_LIST_INDENT
        self.ignore_links = IGNORE_ANCHORS
        self.ignore_images = IGNORE_IMAGES
        self.ignore_emphasis = IGNORE_EMPHASIS
        self.google_doc = False
        self.ul_item_mark = '*'
        self.emphasis_mark = '_'
        self.strong_mark = '**'

        if out is None:
            self.out = self.outtextf
        else:
            self.out = out

        self.outtextlist = []  # empty list to store output characters before they are "joined"

        self.outtext = str()

        self.quiet = 0
        self.p_p = 0  # number of newline character to print before next output
        self.outcount = 0
        self.start = 1
        self.space = 0
        self.a = []
        self.astack = []
        self.maybe_automatic_link = None
        self.absolute_url_matcher = re.compile(r'^[a-zA-Z+]+://')
        self.acount = 0
        self.list = []
        self.blockquote = 0
        self.pre = 0
        self.startpre = 0
        self.code = False
        self.br_toggle = ''
        self.lastWasNL = 0
        self.lastWasList = False
        self.style = 0
        self.style_def = {}
        self.tag_stack = []
        self.emphasis = 0
        self.drop_white_space = 0
        self.inheader = False
        self.abbr_title = None  # current abbreviation definition
        self.abbr_data = None  # last inner HTML (for abbr being defined)
        self.abbr_list = {}  # stack of abbreviations to write later
        self.baseurl = baseurl

        try:
            del unifiable_n[name_to_codepoint('nbsp')]
        except KeyError:
            pass
        unifiable['nbsp'] = '&nbsp_place_holder;'

    def feed(self, data):
        data = data.replace("</' + 'script>", "</ignore>")
        HTMLParser.HTMLParser.feed(self, data)

    def handle(self, data):
        self.feed(data)
        self.feed("")
        return self.optwrap(self.close())

    def outtextf(self, s):
        self.outtextlist.append(s)
        if s: self.lastWasNL = s[-1] == '\n'

    def close(self):
        HTMLParser.HTMLParser.close(self)

        self.pbr()
        self.o('', 0, 'end')

        self.outtext = self.outtext.join(self.outtextlist)
        if self.unicode_snob:
            nbsp = chr(name_to_codepoint('nbsp'))
        else:
            nbsp = u' '
        self.outtext = self.outtext.replace(u'&nbsp_place_holder;', nbsp)

        return self.outtext

    def handle_charref(self, c):
        self.o(self.charref(c), 1)

    def handle_entityref(self, c):
        self.o(self.entityref(c), 1)

    def handle_starttag(self, tag, attrs):
        self.handle_tag(tag, attrs, 1)

    def handle_endtag(self, tag):
        self.handle_tag(tag, None, 0)

    def previous_index(self, attrs):
        """ returns the index of certain set of attributes (of a link) in the
            self.a list

            If the set of attributes is not found, returns None
        """
        if 'href' not in attrs: return None

        i = -1
        for a in self.a:
            i += 1
            match = 0

            if 'href' in a and a['href'] == attrs['href']:
                if 'title' in a or 'title' in attrs:
                    if ('title' in a and 'title' in attrs and
                            a['title'] == attrs['title']):
                        match = True
                else:
                    match = True

            if match: return i

    def drop_last(self, nLetters):
        if not self.quiet:
            self.outtext = self.outtext[:-nLetters]

    def handle_emphasis(self, start, tag_style, parent_style):
        """handles various text emphases"""
        tag_emphasis = google_text_emphasis(tag_style)
        parent_emphasis = google_text_emphasis(parent_style)

        # handle Google's text emphasis
        strikethrough = 'line-through' in tag_emphasis and self.hide_strikethrough
        bold = 'bold' in tag_emphasis and not 'bold' in parent_emphasis
        italic = 'italic' in tag_emphasis and not 'italic' in parent_emphasis
        fixed = google_fixed_width_font(tag_style) and not \
            google_fixed_width_font(parent_style) and not self.pre

        if start:
            # crossed-out text must be handled before other attributes
            # in order not to output qualifiers unnecessarily
            if bold or italic or fixed:
                self.emphasis += 1
            if strikethrough:
                self.quiet += 1
            if italic:
                self.o(self.emphasis_mark)
                self.drop_white_space += 1
            if bold:
                self.o(self.strong_mark)
                self.drop_white_space += 1
            if fixed:
                self.o('`')
                self.drop_white_space += 1
                self.code = True
        else:
            if bold or italic or fixed:
                # there must not be whitespace before closing emphasis mark
                self.emphasis -= 1
                self.space = 0
                self.outtext = self.outtext.rstrip()
            if fixed:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(1)
                    self.drop_white_space -= 1
                else:
                    self.o('`')
                self.code = False
            if bold:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(2)
                    self.drop_white_space -= 1
                else:
                    self.o(self.strong_mark)
            if italic:
                if self.drop_white_space:
                    # empty emphasis, drop it
                    self.drop_last(1)
                    self.drop_white_space -= 1
                else:
                    self.o(self.emphasis_mark)
            # space is only allowed after *all* emphasis marks
            if (bold or italic) and not self.emphasis:
                self.o(" ")
            if strikethrough:
                self.quiet -= 1

    def handle_tag(self, tag, attrs, start):
        # attrs = fixattrs(attrs)
        if attrs is None:
            attrs = {}
        else:
            attrs = dict(attrs)

        if self.google_doc:
            # the attrs parameter is empty for a closing tag. in addition, we
            # need the attributes of the parent nodes in order to get a
            # complete style description for the current element. we assume
            # that google docs export well formed html.
            parent_style = {}
            if start:
                if self.tag_stack:
                    parent_style = self.tag_stack[-1][2]
                tag_style = compute_element_style(attrs, self.style_def, parent_style)
                self.tag_stack.append((tag, attrs, tag_style))
            else:
                dummy, attrs, tag_style = self.tag_stack.pop()
                if self.tag_stack:
                    parent_style = self.tag_stack[-1][2]

        if header_level(tag):
            self.p()
            if start:
                self.inheader = True
                self.o(header_level(tag) * "#" + ' ')
            else:
                self.inheader = False
                return  # prevent redundant emphasis marks on headers

        if tag in ['p', 'div']:
            if self.google_doc:
                if start and google_has_height(tag_style):
                    self.p()
                else:
                    self.soft_br()
            else:
                self.p()

        if tag == "br" and start: self.o("  \n")

        if tag == "hr" and start:
            self.p()
            self.o("* * *")
            self.p()

        if tag in ["head", "style", 'script']:
            if start:
                self.quiet += 1
            else:
                self.quiet -= 1

        if tag == "style":
            if start:
                self.style += 1
            else:
                self.style -= 1

        if tag in ["body"]:
            self.quiet = 0  # sites like 9rules.com never close <head>

        if tag == "blockquote":
            if start:
                self.p()
                self.o('> ', 0, 1)
                self.start = 1
                self.blockquote += 1
            else:
                self.blockquote -= 1
                self.p()

        if tag in ['em', 'i', 'u'] and not self.ignore_emphasis: self.o(self.emphasis_mark)
        if tag in ['strong', 'b'] and not self.ignore_emphasis: self.o(self.strong_mark)
        if tag in ['del', 'strike', 's']:
            if start:
                self.o("<" + tag + ">")
            else:
                self.o("</" + tag + ">")

        if self.google_doc:
            if not self.inheader:
                # handle some font attributes, but leave headers clean
                self.handle_emphasis(start, tag_style, parent_style)

        if tag in ["code", "tt"] and not self.pre: self.o('`')  # TODO: `` `this` ``
        if tag == "abbr":
            if start:
                self.abbr_title = None
                self.abbr_data = ''
                if 'title' in attrs:
                    self.abbr_title = attrs['title']
            else:
                if self.abbr_title != None:
                    self.abbr_list[self.abbr_data] = self.abbr_title
                    self.abbr_title = None
                self.abbr_data = ''

        if tag == "a" and not self.ignore_links:
            if start:
                if 'href' in attrs and not (self.skip_internal_links and attrs['href'].startswith('#')):
                    self.astack.append(attrs)
                    self.maybe_automatic_link = attrs['href']
                else:
                    self.astack.append(None)
            else:
                if self.astack:
                    a = self.astack.pop()
                    if self.maybe_automatic_link:
                        self.maybe_automatic_link = None
                    elif a:
                        if self.inline_links:
                            self.o("](" + escape_md(a['href']) + ")")
                        else:
                            i = self.previous_index(a)
                            if i is not None:
                                a = self.a[i]
                            else:
                                self.acount += 1
                                a['count'] = self.acount
                                a['outcount'] = self.outcount
                                self.a.append(a)
                            self.o("][" + str(a['count']) + "]")

        if tag == "img" and start and not self.ignore_images:
            if 'src' in attrs:
                attrs['href'] = attrs['src']
                alt = attrs.get('alt', '')
                self.o("![" + escape_md(alt) + "]")

                if self.inline_links:
                    self.o("(" + escape_md(attrs['href']) + ")")
                else:
                    i = self.previous_index(attrs)
                    if i is not None:
                        attrs = self.a[i]
                    else:
                        self.acount += 1
                        attrs['count'] = self.acount
                        attrs['outcount'] = self.outcount
                        self.a.append(attrs)
                    self.o("[" + str(attrs['count']) + "]")

        if tag == 'dl' and start: self.p()
        if tag == 'dt' and not start: self.pbr()
        if tag == 'dd' and start: self.o('    ')
        if tag == 'dd' and not start: self.pbr()

        if tag in ["ol", "ul"]:
            # Google Docs create sub lists as top level lists
            if (not self.list) and (not self.lastWasList):
                self.p()
            if start:
                if self.google_doc:
                    list_style = google_list_style(tag_style)
                else:
                    list_style = tag
                numbering_start = list_numbering_start(attrs)
                self.list.append({'name': list_style, 'num': numbering_start})
            else:
                if self.list: self.list.pop()
            self.lastWasList = True
        else:
            self.lastWasList = False

        if tag == 'li':
            self.pbr()
            if start:
                if self.list:
                    li = self.list[-1]
                else:
                    li = {'name': 'ul', 'num': 0}
                if self.google_doc:
                    nest_count = self.google_nest_count(tag_style)
                else:
                    nest_count = len(self.list)
                self.o("  " * nest_count)  # TODO: line up <ol><li>s > 9 correctly.
                if li['name'] == "ul":
                    self.o(self.ul_item_mark + " ")
                elif li['name'] == "ol":
                    li['num'] += 1
                    self.o(str(li['num']) + ". ")
                self.start = 1

        if tag in ["table", "tr"] and start: self.p()
        if tag == 'td': self.pbr()

        if tag == "pre":
            if start:
                self.startpre = 1
                self.pre = 1
            else:
                self.pre = 0
            self.p()

    def pbr(self):
        if self.p_p == 0:
            self.p_p = 1

    def p(self):
        self.p_p = 2

    def soft_br(self):
        self.pbr()
        self.br_toggle = '  '

    def o(self, data, puredata=0, force=0):
        if self.abbr_data is not None:
            self.abbr_data += data

        if not self.quiet:
            if self.google_doc:
                # prevent white space immediately after 'begin emphasis' marks ('**' and '_')
                lstripped_data = data.lstrip()
                if self.drop_white_space and not (self.pre or self.code):
                    data = lstripped_data
                if lstripped_data != '':
                    self.drop_white_space = 0

            if puredata and not self.pre:
                data = re.sub('\s+', ' ', data)
                if data and data[0] == ' ':
                    self.space = 1
                    data = data[1:]
            if not data and not force: return

            if self.startpre:
                # self.out(" :") #TODO: not output when already one there
                if not data.startswith("\n"):  # <pre>stuff...
                    data = "\n" + data

            bq = (">" * self.blockquote)
            if not (force and data and data[0] == ">") and self.blockquote: bq += " "

            if self.pre:
                if not self.list:
                    bq += "    "
                # else: list content is already partially indented
                for i in range(len(self.list)):
                    bq += "    "
                data = data.replace("\n", "\n" + bq)

            if self.startpre:
                self.startpre = 0
                if self.list:
                    data = data.lstrip("\n")  # use existing initial indentation

            if self.start:
                self.space = 0
                self.p_p = 0
                self.start = 0

            if force == 'end':
                # It's the end.
                self.p_p = 0
                self.out("\n")
                self.space = 0

            if self.p_p:
                self.out((self.br_toggle + '\n' + bq) * self.p_p)
                self.space = 0
                self.br_toggle = ''

            if self.space:
                if not self.lastWasNL: self.out(' ')
                self.space = 0

            if self.a and ((self.p_p == 2 and self.links_each_paragraph) or force == "end"):
                if force == "end": self.out("\n")

                newa = []
                for link in self.a:
                    if self.outcount > link['outcount']:
                        self.out("   [" + str(link['count']) + "]: " + urlparse.urljoin(self.baseurl, link['href']))
                        if 'title' in link: self.out(" (" + link['title'] + ")")
                        self.out("\n")
                    else:
                        newa.append(link)

                if self.a != newa: self.out("\n")  # Don't need an extra line when nothing was done.

                self.a = newa

            if self.abbr_list and force == "end":
                for abbr, definition in self.abbr_list.items():
                    self.out("  *[" + abbr + "]: " + definition + "\n")

            self.p_p = 0
            self.out(data)
            self.outcount += 1

    def handle_data(self, data):
        if r'\/script>' in data: self.quiet -= 1

        if self.style:
            self.style_def.update(parse_css(data))

        if not self.maybe_automatic_link is None:
            href = self.maybe_automatic_link
            if href == data and self.absolute_url_matcher.match(href):
                self.o("<" + data + ">")
                return
            else:
                self.o("[")
                self.maybe_automatic_link = None

        if not self.code and not self.pre:
            data = escape_md_section(data, snob=self.escape_snob)
        self.o(data, 1)

    def unknown_decl(self, data):
        pass

    def charref(self, name):
        if name[0] in ['x', 'X']:
            c = int(name[1:], 16)
        else:
            c = int(name)

        if not self.unicode_snob and c in unifiable_n.keys():
            return unifiable_n[c]
        else:
            return chr(c)

    def entityref(self, c):
        if not self.unicode_snob and c in unifiable.keys():
            return unifiable[c]
        else:
            try:
                name_to_codepoint(c)
            except KeyError:
                return "&" + c + ';'
            else:
                return chr(name_to_codepoint(c))

    def replaceEntities(self, s):
        s = s.group(1)
        if s[0] == "#":
            return self.charref(s[1:])
        else:
            return self.entityref(s)

    r_unescape = re.compile(r"&(#?[xX]?(?:[0-9a-fA-F]+|\w{1,8}));")

    def unescape(self, s):
        return self.r_unescape.sub(self.replaceEntities, s)

    def google_nest_count(self, style):
        """calculate the nesting count of google doc lists"""
        nest_count = 0
        if 'margin-left' in style:
            nest_count = int(style['margin-left'][:-2]) / self.google_list_indent
        return nest_count

    def optwrap(self, text):
        """Wrap all paragraphs in the provided text."""
        if not self.body_width:
            return text

        assert wrap, "Requires Python 2.3."
        result = ''
        newlines = 0
        for para in text.split("\n"):
            if len(para) > 0:
                if not skipwrap(para):
                    result += "\n".join(wrap(para, self.body_width))
                    if para.endswith('  '):
                        result += "  \n"
                        newlines = 1
                    else:
                        result += "\n\n"
                        newlines = 2
                else:
                    if not is_only_whitespace(para):
                        result += para + "\n"
                        newlines = 1
            else:
                if newlines < 2:
                    result += "\n"
                    newlines += 1
        return result


ordered_list_matcher = re.compile(r'\d+\.\s')
unordered_list_matcher = re.compile(r'[-\*\+]\s')
md_chars_matcher = re.compile(r"([\\\[\]\(\)])")
md_chars_matcher_all = re.compile(r"([`\*_{}\[\]\(\)#!])")
md_dot_matcher = re.compile(r"""
    ^             # start of line
    (\s*\d+)      # optional whitespace and a number
    (\.)          # dot
    (?=\s)        # lookahead assert whitespace
    """, re.MULTILINE | re.VERBOSE)
md_plus_matcher = re.compile(r"""
    ^
    (\s*)
    (\+)
    (?=\s)
    """, flags=re.MULTILINE | re.VERBOSE)
md_dash_matcher = re.compile(r"""
    ^
    (\s*)
    (-)
    (?=\s|\-)     # followed by whitespace (bullet list, or spaced out hr)
                  # or another dash (header or hr)
    """, flags=re.MULTILINE | re.VERBOSE)
slash_chars = r'\`*_{}[]()#+-.!'
md_backslash_matcher = re.compile(r'''
    (\\)          # match one slash
    (?=[%s])      # followed by a char that requires escaping
    ''' % re.escape(slash_chars),
                                  flags=re.VERBOSE)


def skipwrap(para):
    # If the text begins with four spaces or one tab, it's a code block; don't wrap
    if para[0:4] == '    ' or para[0] == '\t':
        return True
    # If the text begins with only two "--", possibly preceded by whitespace, that's
    # an emdash; so wrap.
    stripped = para.lstrip()
    if stripped[0:2] == "--" and len(stripped) > 2 and stripped[2] != "-":
        return False
    # I'm not sure what this is for; I thought it was to detect lists, but there's
    # a <br>-inside-<span> case in one of the tests that also depends upon it.
    if stripped[0:1] == '-' or stripped[0:1] == '*':
        return True
    # If the text begins with a single -, *, or +, followed by a space, or an integer,
    # followed by a ., followed by a space (in either case optionally preceeded by
    # whitespace), it's a list; don't wrap.
    if ordered_list_matcher.match(stripped) or unordered_list_matcher.match(stripped):
        return True
    return False


def wrapwrite(text):
    text = text.encode('utf-8')
    sys.stdout.buffer.write(text)


def html2text(html, baseurl=''):
    h = HTML2Text(baseurl=baseurl)
    return h.handle(html)


def unescape(s, unicode_snob=False):
    h = HTML2Text()
    h.unicode_snob = unicode_snob
    return h.unescape(s)


def escape_md(text):
    """Escapes markdown-sensitive characters within other markdown constructs."""
    return md_chars_matcher.sub(r"\\\1", text)


def escape_md_section(text, snob=False):
    """Escapes markdown-sensitive characters across whole document sections."""
    text = md_backslash_matcher.sub(r"\\\1", text)
    if snob:
        text = md_chars_matcher_all.sub(r"\\\1", text)
    text = md_dot_matcher.sub(r"\1\\\2", text)
    text = md_plus_matcher.sub(r"\1\\\2", text)
    text = md_dash_matcher.sub(r"\1\\\2", text)
    return text


def url2markdown(baseurl):
    p = optparse.OptionParser('%prog [(filename|url) [encoding]]',
                              version='%prog ' + __version__)
    p.add_option("--ignore-emphasis", dest="ignore_emphasis", action="store_true",
                 default=IGNORE_EMPHASIS, help="don't include any formatting for emphasis")
    p.add_option("--ignore-links", dest="ignore_links", action="store_true",
                 default=IGNORE_ANCHORS, help="don't include any formatting for links")
    p.add_option("--ignore-images", dest="ignore_images", action="store_true",
                 default=IGNORE_IMAGES, help="don't include any formatting for images")
    p.add_option("-g", "--google-doc", action="store_true", dest="google_doc",
                 default=False, help="convert an html-exported Google Document")
    p.add_option("-d", "--dash-unordered-list", action="store_true", dest="ul_style_dash",
                 default=False, help="use a dash rather than a star for unordered list items")
    p.add_option("-e", "--asterisk-emphasis", action="store_true", dest="em_style_asterisk",
                 default=False, help="use an asterisk rather than an underscore for emphasized text")
    p.add_option("-b", "--body-width", dest="body_width", action="store", type="int",
                 default=BODY_WIDTH, help="number of characters per output line, 0 for no wrap")
    p.add_option("-i", "--google-list-indent", dest="list_indent", action="store", type="int",
                 default=GOOGLE_LIST_INDENT, help="number of pixels Google indents nested lists")
    p.add_option("-s", "--hide-strikethrough", action="store_true", dest="hide_strikethrough",
                 default=False, help="hide strike-through text. only relevant when -g is specified as well")
    p.add_option("--escape-all", action="store_true", dest="escape_snob",
                 default=False,
                 help="Escape all special characters.  Output is less readable, but avoids corner case formatting issues.")
    (options, args) = p.parse_args()

    # process input
    encoding = "utf-8"
    if len(args) > 0:
        file_ = args[0]
        if len(args) == 2:
            encoding = args[1]
        if len(args) > 2:
            p.error('Too many arguments')

        if file_.startswith('http://') or file_.startswith('https://'):
            baseurl = file_
            j = urllib.urlopen(baseurl)
            data = j.read()
            if encoding is None:
                try:
                    from feedparser import _getCharacterEncoding as enc
                except ImportError:
                    enc = lambda x, y: ('utf-8', 1)
                encoding = enc(j.headers, data)[0]
                if encoding == 'us-ascii':
                    encoding = 'utf-8'
        else:
            data = open(file_, 'rb').read()
            if encoding is None:
                try:
                    from chardet import detect
                except ImportError:
                    detect = lambda x: {'encoding': 'utf-8'}
                encoding = detect(data)['encoding']
    else:
        data = sys.stdin.read()

    data = data.decode(encoding)
    h = HTML2Text(baseurl=baseurl)
    # handle options
    if options.ul_style_dash: h.ul_item_mark = '-'
    if options.em_style_asterisk:
        h.emphasis_mark = '*'
        h.strong_mark = '__'

    h.body_width = options.body_width
    h.list_indent = options.list_indent
    h.ignore_emphasis = options.ignore_emphasis
    h.ignore_links = options.ignore_links
    h.ignore_images = options.ignore_images
    h.google_doc = options.google_doc
    h.hide_strikethrough = options.hide_strikethrough
    h.escape_snob = options.escape_snob

    wrapwrite(h.handle(data))


REMOVE_ATTRIBUTES = [
    'lang', 'language', 'onmouseover', 'onmouseout', 'script', 'style', 'font',
    'dir', 'face', 'size', 'color', 'class', 'width', 'height', 'hspace',
    'border', 'valign', 'align', 'background', 'bgcolor', 'link', 'vlink',
    'alink', 'cellpadding', 'cellspacing']


def htmltable2markdown(
        html: str, content_conversion_ind: bool = False, all_cols_alignment: str = "left"
) -> str:
    def _transform_cell_content(value: str) -> str:
        chars = {"|": "&#124;", "\n": "<br>"}
        for char, replacement in chars.items():
            value = value.replace(char, replacement)
        return value

    def remove_attr(tag):
        tag.attrs = {key: value for key, value in tag.attrs.items() if key not in REMOVE_ATTRIBUTES}
        return tag

    alignment_options = {"left": " :--- ", "center": " :---: ", "right": " ---: "}
    if all_cols_alignment not in alignment_options.keys():
        raise ValueError(
            "Invalid alignment option for {!r} arg. "
            "Expected one of: {}".format(
                "all_cols_alignment", list(alignment_options.keys())
            )
        )

    soup = BeautifulSoup(html, "html.parser")
    [data.decompose() for data in
     soup(['style', 'script', 'nav', 'button', 'input', 'select', 'option', 'dd', 'dt', 'dl', 'abbr'])]
    if not soup.find():
        return html

    table = []
    table_headings = []
    table_body = []
    html_table = soup.find_all("table")

    md_tables = []

    for _table in html_table:
        table_tr = _table.find_all("tr")
        table_headings = [
            " "
            # + _transform_cell_content(
            #     remove_attr(th).renderContents().decode("utf-8"))
            + th.text.replace('\n', '').strip()
            + " "
            for th in _table.find("tr").find_all("th")]

        if table_headings:
            table.append(table_headings)
            table_tr = table_tr[1:]

        for tr in table_tr:
            td_list = []
            for td in tr.find_all("td"):
                td_list.append(
                    " "
                    # + _transform_cell_content(
                    #     remove_attr(td).renderContents().decode("utf-8"))
                    + td.text.replace('\n', '').strip()
                    + " "
                )
            table_body.append(td_list)

        table += table_body
        md_table_header = "|".join(
            [""]
            + ([" "] * len(table[0]) if not table_headings else table_headings)
            + ["\n"]
            + [alignment_options[all_cols_alignment]] * len(table[0])
            + ["\n"]
        )

        md_table = md_table_header + "".join(
            "|".join([""] + row + ["\n"]) for row in table_body
        )
        md_tables.append(md_table)
    return md_tables
