"""``speech/phrases.py``: a streaming reply cut into phrases a voice can speak (docs/design/speech-route.md, section 5).

Each rule in the design is pinned here: sentence ends (not decimals, abbreviations or list numbers), line breaks, tool
calls, clause ends after enough words, the 200-character maximum, code blocks never read and never cut, markdown taken
out of what is spoken, the spoken cap, a discarded round, and offsets that point back into the reply's own text.
"""

from __future__ import annotations

import pytest

from prometheus.speech.phrases import CODE_SKIPPED, TOO_LONG, PhraseCutter, speakable


def _texts(phrases) -> list[str]:
    return [p.text for p in phrases]


def _all(text: str, **kw) -> list:
    cutter = PhraseCutter(**kw)
    return cutter.feed(text) + cutter.finish()


def _trickle(text: str, **kw) -> list:
    """Fed one character at a time, as a slow stream would."""
    cutter = PhraseCutter(**kw)
    out = []
    for ch in text:
        out += cutter.feed(ch)
    return out + cutter.finish()


class TestSentenceEnds:
    def test_a_sentence_end_cuts_once_whitespace_follows(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Hello there. How are")) == ["Hello there."]
        assert _texts(cutter.finish()) == ["How are"]

    def test_a_sentence_end_at_the_end_of_the_buffer_waits(self):
        cutter = PhraseCutter()
        assert cutter.feed("It is 3.") == []
        assert _texts(cutter.feed("5 now. ")) == ["It is 3.5 now."]

    def test_a_decimal_never_cuts(self):
        assert _texts(_all("It costs 3.5 dollars. Fine.")) == ["It costs 3.5 dollars.", "Fine."]

    @pytest.mark.parametrize("abbr", ["e.g.", "i.e.", "Mr.", "Mrs.", "Ms.", "Dr.", "vs.", "St."])
    def test_common_abbreviations_do_not_cut(self, abbr):
        assert _texts(_all(f"Ask {abbr} Smith about it. Then go.")) == [f"Ask {abbr} Smith about it.", "Then go."]

    def test_question_exclamation_ellipsis_and_closing_quotes(self):
        text = 'Really?! He said "go." Wait… Fine.'
        assert _texts(_all(text)) == ["Really?!", 'He said "go."', "Wait…", "Fine."]

    def test_a_list_number_at_line_start_does_not_cut(self):
        assert _texts(_all("1. Apples are red\n2. Pears are green\n")) == ["Apples are red", "Pears are green"]


class TestLineBreaksAndToolCalls:
    def test_a_line_break_after_text_cuts(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Title\nBody text")) == ["Title"]
        assert _texts(cutter.finish()) == ["Body text"]

    def test_blank_lines_are_not_phrases(self):
        assert _texts(_all("One\n\n\nTwo")) == ["One", "Two"]

    def test_a_tool_call_releases_what_is_pending(self):
        cutter = PhraseCutter()
        assert cutter.feed("Let me check") == []
        assert _texts(cutter.boundary()) == ["Let me check"]
        assert _texts(cutter.feed("Done. ")) == ["Done."]
        assert cutter.finish() == []

    def test_a_tool_call_inside_an_open_fence_releases_nothing(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Look:\n```\nx = 1")) == ["Look:"]
        assert cutter.boundary() == []
        assert _texts(cutter.finish()) == [CODE_SKIPPED]


class TestClauses:
    def test_the_first_phrase_cuts_at_a_clause_after_three_words(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Well, I think, that this")) == ["Well, I think,"]

    def test_later_phrases_need_six_words(self):
        cutter = PhraseCutter()
        cutter.feed("Okay. ")
        assert cutter.feed("Yes, one two, and") == []
        assert _texts(cutter.feed(" more words here. ")) == ["Yes, one two, and more words here."]

    def test_a_short_later_clause_waits_for_more(self):
        cutter = PhraseCutter()
        cutter.feed("Okay. ")
        assert _texts(cutter.feed("one two three four five six seven, and ")) == [
            "one two three four five six seven,"]


class TestMaximumLength:
    def test_no_phrase_is_longer_than_the_maximum(self):
        words = " ".join(["word"] * 200)
        phrases = _all(words)
        assert all(len(p.text) <= 200 for p in phrases)
        assert " ".join(_texts(phrases)) == words

    def test_the_cut_is_at_whitespace(self):
        phrases = _all(" ".join(["abcdefghij"] * 40))
        assert all(not p.text.endswith(" ") and "abcdefghij" in p.text for p in phrases)
        assert all(len(w) == 10 for p in phrases for w in p.text.split())

    def test_one_token_longer_than_the_maximum_is_cut_at_the_maximum(self):
        phrases = _all("x" * 450)
        assert [len(p.text) for p in phrases] == [200, 200, 50]

    def test_the_maximum_can_be_set(self):
        assert all(len(p.text) <= 40 for p in _all(" ".join(["word"] * 50), max_chars=40))

    def test_thresholds_out_of_range_are_refused(self):
        with pytest.raises(ValueError):
            PhraseCutter(max_chars=10)
        with pytest.raises(ValueError):
            PhraseCutter(first_min_words=0)


class TestCodeBlocks:
    def test_a_code_block_is_skipped_and_said_so(self):
        text = "Here:\n```py\nx = 1. y = 2\n```\nDone."
        assert _texts(_all(text)) == ["Here:", CODE_SKIPPED, "Done."]

    def test_tilde_fences_too(self):
        assert _texts(_all("A.\n~~~\nrm -rf. x\n~~~\nB.")) == ["A.", CODE_SKIPPED, "B."]

    def test_blocks_in_a_row_are_one(self):
        text = "```\na\n```\n\n```\nb\n```\nAfter."
        assert _texts(_all(text)) == [CODE_SKIPPED, "After."]

    def test_blocks_with_text_between_are_each_said(self):
        text = "```\na\n```\nThen.\n```\nb\n```\n"
        assert _texts(_all(text)) == [CODE_SKIPPED, "Then.", CODE_SKIPPED]

    def test_an_unclosed_fence_holds_until_it_closes(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Look:\n```\ncode. more. ")) == ["Look:"]
        assert _texts(cutter.feed("\n```\nEnd. ")) == [CODE_SKIPPED, "End."]

    def test_an_unclosed_fence_at_the_end_is_skipped(self):
        assert _texts(_all("Look:\n```\nnever closed. at all")) == ["Look:", CODE_SKIPPED]

    def test_no_cut_lands_inside_a_long_fence(self):
        code = "```" + "a" * 1000 + ". b, c\n" * 3 + "```"
        phrases = _all("Before it " + code + " after it.")
        assert _texts(phrases) == ["Before it", CODE_SKIPPED, "after it."]

    def test_text_before_a_fence_on_the_same_line_is_its_own_phrase(self):
        assert _texts(_all("Run this ```ls``` now.")) == ["Run this", CODE_SKIPPED, "now."]

    def test_a_partial_fence_marker_at_the_end_is_held(self):
        cutter = PhraseCutter()
        cutter.feed("Okay. ")
        out = cutter.feed("x" * 199 + " ``")
        assert all("`" not in p.text for p in out)
        out += cutter.feed("`\ncode\n```\n")
        assert _texts(out)[-1] == CODE_SKIPPED

    def test_the_same_phrases_when_trickled(self):
        text = "Here it is.\n```\nx = 1. y\n```\nThat is all. Bye now.\n"
        assert _texts(_trickle(text)) == _texts(_all(text)) == ["Here it is.", CODE_SKIPPED, "That is all.",
                                                                  "Bye now."]


class TestSpoken:
    def test_markdown_is_taken_out(self):
        text = "## Heading\n**Bold** and `code` and [docs](https://docs.example/a).\n> Quoted\n- item one\n"
        assert _texts(_all(text)) == ["Heading", "Bold and code and docs.", "Quoted", "item one"]

    def test_a_bare_url_is_link(self):
        assert _texts(_all("See https://example.org/some/long/path?x=1 now.")) == ["See link now."]

    def test_snake_case_is_kept(self):
        assert _texts(_all("Call my_func_name now.")) == ["Call my_func_name now."]

    def test_a_rule_line_is_not_spoken(self):
        assert _texts(_all("Above.\n---\nBelow.")) == ["Above.", "Below."]

    def test_speakable_alone(self):
        assert speakable("*Hi* __there__") == "Hi there"
        assert speakable("   ") == ""


class TestOffsets:
    def test_offsets_point_into_the_text(self):
        text = "Hi there. Bye now."
        phrases = _all(text)
        assert [text[p.start:p.end] for p in phrases] == ["Hi there.", "Bye now."]

    def test_offsets_hold_when_trickled_and_for_code(self):
        text = "Intro line\n```\ncode\n```\nOutro."
        phrases = _trickle(text)
        assert [text[p.start:p.end] for p in phrases] == ["Intro line", "```\ncode\n```", "Outro."]

    def test_offsets_count_code_points(self):
        text = "Café ☕ ok. Next one."
        phrases = _all(text)
        assert [text[p.start:p.end] for p in phrases] == ["Café ☕ ok.", "Next one."]

    def test_released_is_where_unspoken_text_starts(self):
        cutter = PhraseCutter()
        cutter.feed("One. Two")
        assert cutter.released == len("One. ")


class TestCap:
    def test_the_spoken_cap_ends_with_one_sentence(self):
        cutter = PhraseCutter(spoken_cap=30)
        out = cutter.feed("This is the first sentence. This is the second one. Third. ")
        assert _texts(out) == ["This is the first sentence.", TOO_LONG]
        assert cutter.truncated
        assert cutter.feed("More text. ") == []
        assert cutter.finish() == []

    def test_under_the_cap_nothing_is_truncated(self):
        cutter = PhraseCutter()
        cutter.feed("Short. ")
        cutter.finish()
        assert not cutter.truncated


class TestDiscard:
    def test_a_discard_inside_the_pending_text_cuts_it(self):
        cutter = PhraseCutter()
        cutter.feed("First round text")
        cutter.discard(5)
        assert _texts(cutter.finish()) == ["First"]

    def test_a_discard_before_the_pending_text_drops_it_and_new_text_follows(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("One. Two. garbage garbage")) == ["One.", "Two."]
        cutter.discard(5)
        phrases = cutter.feed("New text. ")
        assert _texts(phrases) == ["New text."]
        assert phrases[0].start == 5

    def test_a_discard_past_the_end_changes_nothing(self):
        cutter = PhraseCutter()
        cutter.feed("Hello")
        cutter.discard(100)
        assert _texts(cutter.finish()) == ["Hello"]


class TestHostileAndLargeInput:
    """Review of #923: a single large feed (``speak`` takes 20,000 characters) must keep every rule and stay fast."""

    def _fast(self, fn, limit: float = 1.0):
        import time

        t0 = time.perf_counter()
        out = fn()
        assert time.perf_counter() - t0 < limit
        return out

    @pytest.mark.parametrize("text", [
        "word " * 300 + "end. Next.",
        "word " * 300 + "end, next",
        "word " * 300 + "```x```",
    ])
    def test_one_large_feed_never_makes_a_long_phrase(self, text):
        assert all(len(p.text) <= 200 for p in _all(text, spoken_cap=100_000))

    @pytest.mark.parametrize("text", ["." * 20_000, "e.g. " * 4_000, "a, " * 6_000, "x" * 20_000,
                                      "`" * 20_000, "\n" * 20_000])
    def test_hostile_input_is_fast(self, text):
        phrases = self._fast(lambda: _all(text, spoken_cap=100_000))
        assert all(len(p.text) <= 200 for p in phrases)

    def test_a_huge_open_fence_streamed_is_fast_and_skipped(self):
        def run():
            cutter = PhraseCutter()
            out = cutter.feed("Here:\n```\n")
            chunk = "abcd"
            for _ in range(100_000):
                out += cutter.feed(chunk)
            return out + cutter.feed("\n```\nDone. ")
        assert _texts(self._fast(run, 3.0)) == ["Here:", CODE_SKIPPED, "Done."]

    def test_released_inside_a_fence_is_the_fence_start(self):
        cutter = PhraseCutter()
        cutter.feed("Hi.\n```\n" + "x" * 500)
        assert cutter.released == len("Hi.\n")

    def test_a_long_url_is_one_phrase(self):
        url = "https://bucket.example/obj?sig=" + "4a1b2c3d" * 45
        assert _texts(_all(f"Get it at {url} today.")) == ["Get it at", "link", "today."]


class TestReviewFixes:
    def test_a_shorter_run_inside_a_longer_fence_does_not_close_it(self):
        text = "Example:\n````md\n```py\nsecret_code(1)\n```\nmore\n````\nDone. "
        assert _texts(_all(text)) == ["Example:", CODE_SKIPPED, "Done."]

    def test_a_tool_call_keeps_a_half_arrived_fence(self):
        cutter = PhraseCutter()
        out = cutter.feed("Let me run this:\n``")
        out += cutter.boundary()
        out += cutter.feed("`sh\nrm -rf build\n```\nDone. ")
        assert _texts(out) == ["Let me run this:", CODE_SKIPPED, "Done."]

    def test_a_discard_before_a_code_phrase_lets_the_next_block_be_said(self):
        cutter = PhraseCutter()
        assert _texts(cutter.feed("Hi.\n```\nx\n```\n")) == ["Hi.", CODE_SKIPPED]
        cutter.discard(4)
        assert _texts(cutter.feed("```\ny\n```\nOk. ")) == [CODE_SKIPPED, "Ok."]

    def test_a_discard_before_the_cap_lifts_it(self):
        cutter = PhraseCutter(spoken_cap=20)
        out = cutter.feed("Hello there. More words here. ")
        assert _texts(out) == ["Hello there.", TOO_LONG]
        cutter.discard(0)
        assert not cutter.truncated
        assert _texts(cutter.feed("Ok now. ")) == ["Ok now."]

    def test_a_year_at_line_start_is_spoken(self):
        assert _texts(_all("1999. It was a year.")) == ["1999.", "It was a year."]

    def test_a_lone_star_and_dunder_names(self):
        assert speakable("5 * 3 = 15") == "5 * 3 = 15"
        assert speakable("a *very* good") == "a very good"


class TestReviewRoundTwo:
    def test_a_growing_backtick_run_stays_fast(self):
        import time

        for ch in "`~":
            cutter = PhraseCutter()
            t0 = time.perf_counter()
            for _ in range(25_000):
                cutter.feed(ch * 4)
            cutter.finish()
            assert time.perf_counter() - t0 < 2.0

    def test_text_after_a_long_closer_split_across_deltas_is_spoken(self):
        text = "```\ncode\n``````\nImportant words here. More."
        assert _texts(_trickle(text)) == _texts(_all(text)) == [CODE_SKIPPED, "Important words here.", "More."]

    def test_code_offsets_match_when_trickled(self):
        text = "Before.\n````\nx ``` y\n````\nAfter."
        assert [(p.start, p.end) for p in _trickle(text)] == [(p.start, p.end) for p in _all(text)]

    def test_a_discard_through_a_phrase_gives_back_its_kept_part(self):
        cutter = PhraseCutter()
        first = cutter.feed("Alpha beta gamma. ")
        assert [(p.start, p.end) for p in first] == [(0, 17)]
        cutter.discard(6)
        phrases = cutter.finish()
        assert [(p.start, p.text) for p in phrases] == [(0, "Alpha")]
