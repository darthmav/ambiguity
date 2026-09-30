"""HTML to text, ours rather than a library's.

A fetched page is mostly not the page. Navigation, cookie banners, related-post
rails and footers are the bulk of the markup, and embedding them is not a
cosmetic problem here: a chunk of link text scores against queries the way any
other chunk does, so boilerplate does not sit inertly in the corpus, it
*competes* with the passage that answers the question. The corpus already
learned this once from the other direction -- a document embedded in one
`encode()` call was represented by its preamble, and the file that answered the
query lost to a shorter one that merely mentioned it.

The technique is **link density plus block length**, which is what separates
prose from chrome without knowing anything about the site. A navigation strip
is short blocks that are almost entirely anchor text; an article is long blocks
that are almost entirely not. Neither test works alone -- a heading is short
prose, and a paragraph citing three sources is long and link-heavy -- so a
block is dropped only when it is *both* short and link-dominated, and headings
are exempt from the length test entirely because a heading is short by nature
and is the one short thing worth keeping.

Written here rather than taken from a library for two reasons. The obvious one
is that it costs nothing and adds no dependency tree. The one that matters more
is that this is a *measurement surface*: `ExtractedPage` reports how many blocks
it kept, how many it dropped and whether it fell back, so a page that came back
thin says so and can be graded, rather than being an empty string out of a
black box. Everything else in this project that decides what enters the corpus
-- the relevance floor, the chunker, the duplicate scan -- was tuned against
numbers it printed, and an extractor that cannot be tuned the same way would be
the one unmeasured link in that chain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Everything inside these is never prose, whatever it contains. `head` matters
# as much as `script`: a page's metadata and JSON-LD would otherwise arrive as
# a wall of tokens attached to a real URL.
SKIP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "head", "form",
        "select", "option", "button", "iframe", "canvas", "map", "object",
    }
)

# Chrome by role rather than by content. Kept apart from SKIP_TAGS because a
# page that wraps its article in <aside> loses everything if these are treated
# as certainties -- so their text is dropped from the *filtered* pass and comes
# back if the fallback fires.
CHROME_TAGS = frozenset({"nav", "header", "footer", "aside", "menu"})

# Tags that end a block of text. A block is the unit the density test judges,
# so this list decides how coarse that judgement is: too few and an entire page
# is one block that always passes, too many and every sentence is judged alone.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "tr", "td", "th", "section", "article", "blockquote",
        "pre", "figcaption", "dt", "dd", "br", "hr", "table", "ul", "ol",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# A block shorter than this, and link-dominated, is chrome. Both tests have to
# fail it -- see the module docstring for why neither is sufficient alone.
MIN_BLOCK_WORDS = 8
MAX_LINK_DENSITY = 0.5

# Below this the filtered pass is assumed to have eaten the article -- a page
# built entirely from <div>s inside an <aside>, or markup broken enough that
# the skip stack never unwound. The unfiltered text is returned instead, and
# `fell_back` says so, because a silently empty extraction is exactly the
# failure this module exists to make visible.
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
    """What came out, and enough about how to grade it.

    `kept`/`dropped`/`fell_back` are the point: they turn "this page came back
    thin" from a guess into a reading. See the module docstring.
    """

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
        # convert_charrefs is the default and is wanted: entities become text
        # before we ever count a word, so `&amp;` is one token and not three.
        super().__init__(convert_charrefs=True)
        self.blocks: list[_Block] = []
        self.chrome_blocks: list[_Block] = []
        self.title = ""
        self._block = _Block()
        # Counters rather than a boolean: nested <div> inside <nav> inside
        # <aside> has to unwind exactly, and malformed markup that never closes
        # a tag is why this can get stuck -- which is what MIN_DOCUMENT_WORDS
        # catches downstream.
        self._skip = 0
        self._chrome = 0
        self._anchor = 0
        self._pre = 0
        self._in_title = False
        # Only the first <title> names the page. An inline <svg> carries its
        # own for accessibility -- "Search", "Close", "Menu" -- and each one
        # used to be appended to the page's, so the provenance header of a
        # stored page read "Real Title Search Menu Close".
        self._title_done = False

    # -- block bookkeeping --------------------------------------------------

    def _flush(self) -> None:
        if self._block.text():
            (self.chrome_blocks if self._chrome else self.blocks).append(self._block)
        self._block = _Block(preformatted=bool(self._pre))

    # -- HTMLParser hooks ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # `title` is checked before the skip stack, never after: it lives inside
        # `head`, `head` is skipped wholesale, and a check below the skip guard
        # can therefore never fire. It is the one thing worth taking from there,
        # and it is what names the document when a page has no usable <h1>.
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
        # `head` is skipped wholesale, so this is the only way title text
        # arrives; everything else inside a skipped tag is discarded.
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

    def close(self) -> None:  # noqa: D102 - inherited contract
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

    Never raises on malformed markup -- `HTMLParser` is lenient by design, and
    a page that breaks the skip stack is caught by the fallback rather than by
    an exception. A caller gets thin text and the counters saying it is thin,
    which is a thing it can act on; an exception here would only turn a bad
    page into a lost one.
    """
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Nothing HTMLParser raises is worth losing the partial parse over:
        # whatever blocks it had collected before the malformed byte are still
        # the page, and the fallback below decides whether they are enough.
        pass

    kept = [block for block in parser.blocks if not block.is_chrome()]
    dropped = len(parser.blocks) - len(kept) + len(parser.chrome_blocks)
    text = _render(kept)

    fell_back = False
    if len(text.split()) < MIN_DOCUMENT_WORDS:
        # The filter ate the page. Take everything that was not inside a
        # SKIP_TAG, chrome included -- a noisy document is worth more than an
        # empty one, and `fell_back` is what tells the caller which it got.
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
