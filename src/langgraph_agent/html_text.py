"""HTML to text, ours rather than a library's.

Navigation, banners and footers are most of a page's markup, and embedded they
compete with the passage that answers the question. A block is dropped as
chrome when it is *both* short and link-dominated -- neither test alone works:
a heading is short prose, and a paragraph citing sources is link-heavy -- and
headings are exempt from the length test.

Written here rather than taken from a library because it is a measurement
surface: `ExtractedPage` says how many blocks it kept and dropped and whether
it fell back, so a thin page says so and can be graded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Never prose, whatever they contain -- `head` included, or a page's metadata
# and JSON-LD arrive as text.
SKIP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "head", "form",
        "select", "option", "button", "iframe", "canvas", "map", "object",
    }
)

# Chrome by role. Not certainties like SKIP_TAGS -- some pages wrap their
# article in <aside> -- so their text comes back if the fallback fires.
CHROME_TAGS = frozenset({"nav", "header", "footer", "aside", "menu"})

# Tags that end a block, the unit the density test judges: too few and a page is
# one block that always passes, too many and every sentence is judged alone.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "tr", "td", "th", "section", "article", "blockquote",
        "pre", "figcaption", "dt", "dd", "br", "hr", "table", "ul", "ol",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# A block shorter than this *and* link-dominated is chrome.
MIN_BLOCK_WORDS = 8
MAX_LINK_DENSITY = 0.5

# Below this the filtered pass is assumed to have eaten the article -- markup
# broken enough that the skip stack never unwound, say -- and the unfiltered
# text is returned instead, with `fell_back` saying so.
MIN_DOCUMENT_WORDS = 40

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class _Block:
    """One candidate block: its text, and how much of it was anchor text."""

    parts: list[str] = field(default_factory=list)
    words: int = 0
    link_words: int = 0
    heading: int = 0
    preformatted: bool = False

    def text(self) -> str:
        joined = "".join(self.parts)
        if self.preformatted:
            return joined.strip("\n")
        return _WHITESPACE.sub(" ", joined).strip()

    @property
    def link_density(self) -> float:
        return self.link_words / self.words if self.words else 0.0

    def is_chrome(self) -> bool:
        """Short *and* link-dominated. A heading is never judged on length."""
        if self.heading or self.preformatted:
            return False
        return self.words < MIN_BLOCK_WORDS and self.link_density > MAX_LINK_DENSITY


@dataclass
class ExtractedPage:
    """What came out, with `kept`, `dropped` and `fell_back` to grade it by."""

    title: str
    text: str
    kept: int
    dropped: int
    fell_back: bool

    @property
    def words(self) -> int:
        return len(self.text.split())


class _Extractor(HTMLParser):
    """Collect text into blocks, tracking what is skipped and what is a link."""

    def __init__(self) -> None:
        # Entities become text before a word is counted.
        super().__init__(convert_charrefs=True)
        self.blocks: list[_Block] = []
        self.chrome_blocks: list[_Block] = []
        self.title = ""
        self._block = _Block()
        # Counters, so nested tags unwind exactly. Markup that never closes a
        # tag can still leave one stuck, which MIN_DOCUMENT_WORDS catches.
        self._skip = 0
        self._chrome = 0
        self._anchor = 0
        self._pre = 0
        self._in_title = False
        # Only the first <title> names the page: an inline <svg> carries its own
        # ("Search", "Close").
        self._title_done = False

    # -- block bookkeeping --------------------------------------------------

    def _flush(self) -> None:
        if self._block.text():
            (self.chrome_blocks if self._chrome else self.blocks).append(self._block)
        self._block = _Block(preformatted=bool(self._pre))

    # -- HTMLParser hooks ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Before the skip stack: `title` lives inside the skipped `head`.
        if tag == "title":
            self._in_title = not self._title_done
            return
        if tag in SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in CHROME_TAGS:
            self._flush()
            self._chrome += 1
        if tag == "a":
            self._anchor += 1
        if tag == "pre":
            self._pre += 1
        if tag in BLOCK_TAGS:
            self._flush()
            if tag in HEADING_TAGS:
                self._block.heading = int(tag[1])
            self._block.preformatted = bool(self._pre)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            if self._in_title:
                self._title_done = True
            self._in_title = False
            return
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "a":
            self._anchor = max(0, self._anchor - 1)
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
        if tag in BLOCK_TAGS:
            self._flush()
        if tag in CHROME_TAGS:
            self._flush()
            self._chrome = max(0, self._chrome - 1)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        if not data.strip():
            # Whitespace still separates words across inline tags.
            self._block.parts.append(" ")
            return
        self._block.parts.append(data)
        count = len(data.split())
        self._block.words += count
        if self._anchor:
            self._block.link_words += count

    def close(self) -> None:
        super().close()
        self._flush()


def _render(blocks: list[_Block]) -> str:
    """Blocks to markdown. Headings and list items keep their shape."""
    lines: list[str] = []
    for block in blocks:
        text = block.text()
        if not text:
            continue
        if block.heading:
            lines.append(f"{'#' * block.heading} {text}")
        elif block.preformatted:
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text)
    return _BLANK_LINES.sub("\n\n", "\n\n".join(lines)).strip()


def extract(html: str) -> ExtractedPage:
    """Pull the readable article out of a page.

    Never raises on malformed markup: a bad page comes back thin, with the
    counters saying so, rather than lost.
    """
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # The blocks collected before the malformed byte are still the page.
        pass

    kept = [block for block in parser.blocks if not block.is_chrome()]
    dropped = len(parser.blocks) - len(kept) + len(parser.chrome_blocks)
    text = _render(kept)

    fell_back = False
    if len(text.split()) < MIN_DOCUMENT_WORDS:
        # The filter ate the page: a noisy document beats an empty one.
        everything = parser.blocks + parser.chrome_blocks
        fallback = _render(everything)
        if len(fallback.split()) > len(text.split()):
            text, fell_back = fallback, True
            kept, dropped = everything, 0

    return ExtractedPage(
        title=_WHITESPACE.sub(" ", parser.title).strip(),
        text=text,
        kept=len(kept),
        dropped=dropped,
        fell_back=fell_back,
    )
