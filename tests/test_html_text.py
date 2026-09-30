"""Tests for the from-scratch HTML reader.

What is under test is the *judgement*, not the parsing: any parser can turn
tags into text, and the reason this module exists is that most of a page is not
the page. So the assertions are about what gets thrown away — a nav strip, a
footer, a script — and about the two ways that judgement can go wrong: keeping
chrome, and eating the article. The second is the dangerous one, because an
empty extraction is silent, so the fallback has its own test.
"""

from __future__ import annotations

from langgraph_agent.html_text import (
    MIN_DOCUMENT_WORDS,
    extract,
)

ARTICLE = " ".join(f"sentence number {n} about hybrid retrieval and ranking." for n in range(20))

PAGE = f"""
<html>
  <head>
    <title>Hybrid Retrieval — Notes</title>
    <script>var tracking = {{"id": 1}}; console.log("should never be text");</script>
    <style>body {{ color: red; }}</style>
  </head>
  <body>
    <nav><a href="/">Home</a> <a href="/blog">Blog</a> <a href="/about">About</a></nav>
    <header><a href="/login">Sign in</a></header>
    <article>
      <h1>Hybrid Retrieval</h1>
      <p>{ARTICLE}</p>
      <h2>Why BM25</h2>
      <p>{ARTICLE}</p>
      <pre>index = BM25Index(ids, texts)</pre>
    </article>
    <footer><a href="/tos">Terms</a> <a href="/privacy">Privacy</a></footer>
  </body>
</html>
"""


def test_the_article_survives_and_the_chrome_does_not():
    page = extract(PAGE)

    assert "hybrid retrieval and ranking" in page.text
    assert "Sign in" not in page.text
    assert "Privacy" not in page.text
    assert page.dropped > 0


def test_script_and_style_never_become_text():
    """They are not prose in any page, whatever the markup around them says."""
    page = extract(PAGE)

    assert "tracking" not in page.text
    assert "console.log" not in page.text
    assert "color: red" not in page.text


def test_the_title_is_read_out_of_the_head():
    """`head` is skipped wholesale, so the title needs its own path out.

    This regressed once already: the title check sat below the skip guard and
    could therefore never fire, and every stored document was named after its
    URL instead.
    """
    assert extract(PAGE).title == "Hybrid Retrieval — Notes"


def test_headings_are_kept_although_they_are_short():
    """A heading is short by nature; length alone must not condemn it."""
    page = extract(PAGE)

    assert "# Hybrid Retrieval" in page.text
    assert "## Why BM25" in page.text


def test_a_short_link_dominated_block_is_dropped_and_a_long_one_is_not():
    """Both tests have to fail a block, because neither is sufficient alone.

    A paragraph that cites several sources is long and link-heavy and is still
    prose; a nav strip is short and link-heavy and is not.
    """
    # A link strip built from plain <div>s rather than <nav>, which is how most
    # sites actually write one — so this is the density rule doing the work,
    # not CHROME_TAGS. Judged next to real content, because a page consisting
    # of nothing but a nav strip trips the fallback instead (its own test).
    page = extract(
        "<body>"
        "<div><a href='/a'>Widgets</a> <a href='/b'>Gadgets</a> <a href='/c'>Sprockets</a></div>"
        f"<div>{ARTICLE}</div>"
        f"<div>{ARTICLE} see <a href='/d'>one</a> and <a href='/e'>two</a></div>"
        "</body>"
    )

    assert not page.fell_back
    # Short and link-dominated: gone.
    assert "Widgets" not in page.text
    # Long, even though it is link-heavy: kept.
    assert "hybrid retrieval" in page.text
    assert page.dropped == 1


def test_preformatted_text_is_kept_verbatim_in_a_fence():
    assert "```" in extract(PAGE).text
    assert "BM25Index(ids, texts)" in extract(PAGE).text


def test_the_fallback_fires_rather_than_returning_an_empty_page():
    """The filter eating the article must not be silent.

    A page whose whole body is inside <aside> is the ordinary shape of this:
    the chrome rule would drop everything and the caller would get an empty
    string with nothing saying why.
    """
    buried = f"<body><aside><p>{ARTICLE}</p></aside></body>"

    page = extract(buried)

    assert page.fell_back
    assert page.words > MIN_DOCUMENT_WORDS
    assert "hybrid retrieval" in page.text


def test_a_clean_page_does_not_report_a_fallback():
    """Otherwise the flag says nothing — it has to distinguish something."""
    assert not extract(PAGE).fell_back


def test_malformed_markup_returns_what_it_managed_rather_than_raising():
    """A broken page is worth less than a good one and much more than an error."""
    broken = f"<body><div><p>{ARTICLE}<span></div></body></html></p>"

    page = extract(broken)

    assert "hybrid retrieval" in page.text


def test_an_empty_document_is_empty_rather_than_an_exception():
    page = extract("")

    assert page.text == ""
    assert page.words == 0


def test_entities_are_decoded_before_words_are_counted():
    """`&amp;` is one token, not three, and the density test counts tokens."""
    page = extract(f"<body><p>Ranking &amp; retrieval &mdash; {ARTICLE}</p></body>")

    assert "&amp;" not in page.text
    assert "Ranking & retrieval" in page.text


def test_an_icons_title_does_not_join_the_pages():
    """Only the first <title> names the page.

    An inline <svg> carries a title of its own for accessibility, and every
    one of them used to be appended to the page's, so the provenance header
    of a stored page read "Hybrid retrieval Search Close".
    """
    page = extract(
        "<html><head><title>Hybrid retrieval</title></head><body>"
        "<svg><title>Search</title><path d='M0'/></svg>"
        f"<p>{ARTICLE}</p><svg><title>Close</title></svg></body></html>"
    )

    assert page.title == "Hybrid retrieval"
    assert "Search" not in page.text and "Close" not in page.text
