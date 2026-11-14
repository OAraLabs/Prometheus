"""A reply, or any text, as the speech stream's lines (docs/design/speech-route.md, sections 2.5 and 5).

:func:`speak` takes a source of events (a :class:`~prometheus.speech.replies.Reply`'s ``follow()``, or
:func:`text_source`) and yields the stream's lines as dicts, for the route to write one JSON object per line:

* ``start``, always first;
* ``reply`` when the reply binds to its ``message_id``;
* ``phrase`` for each phrase, with its base64 WAV, as soon as the voice box has made it;
* ``discard`` when a round was dropped;
* ``keepalive`` when nothing else was sent for *keepalive* seconds;
* ``done``, exactly once, last.

Two tasks run per stream: one cuts phrases from the source as it arrives (:class:`PhraseCutter`), one synthesises them
one at a time, in order. So phrase 1 is on the phone while phrase 2 is on the voice box and the model writes phrase 3.

How it ends: ``complete`` once everything was spoken (``truncated`` when the cap stopped it); the reply's own reason
(``interrupted``, ``failed``, ``ended``, ``no_reply``) at once, without speaking what is still pending; ``timeout``
when the source sends nothing for *idle_timeout* seconds; ``unavailable`` when the voice box fails, with ``voice``,
``message`` and ``resume_at`` (where the unspoken text starts), and nothing is retried; or a reason the route sets
through *stop* (``replaced``, ``revoked``). When the client goes away, the route stops iterating and both tasks are
cancelled: nothing more is sent to the voice box.

A dropped round (``discard``): no phrase cut before it whose ``end`` is past ``keep_chars`` is sent after it.

Source: novel code for Prometheus, 2026-11-13, after OAra Voice's speech pipeline (one synthesis worker, in order,
fed by a queue; a failure ends synthesis and is announced, never retried quietly).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from prometheus.speech.phrases import Phrase, PhraseCutter
from prometheus.speech.voice import SENTENCES, VoiceUnavailable

logger = logging.getLogger(__name__)

VERSION = 1
MIME = "audio/wav"
#: A reply that sends no new text for this long ends the stream.
IDLE_TIMEOUT = 600.0
#: A line is sent at least this often.
KEEPALIVE = 15.0


class _Synth(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


async def text_source(text: str) -> AsyncIterator[tuple[str, Any]]:
    """The events of a given text: all of it, then ``complete``."""
    yield ("text", text)
    yield ("end", "complete")


async def speak(
    source: AsyncIterator[tuple[str, Any]],
    voice: _Synth,
    *,
    cutter: PhraseCutter | None = None,
    idle_timeout: float = IDLE_TIMEOUT,
    keepalive: float = KEEPALIVE,
    stop: asyncio.Future[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """The stream's lines for *source*, spoken by *voice* (see the module text)."""
    cutter = cutter or PhraseCutter()
    out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    work: asyncio.Queue[tuple[Phrase, int] | None] = asyncio.Queue()
    discards: list[tuple[int, int]] = []     # (generation, keep_chars), in order
    generation = 0

    def dropped(phrase: Phrase, gen: int) -> bool:
        return any(g > gen and phrase.end > keep for g, keep in discards)

    def enqueue(phrases: list[Phrase]) -> None:
        for phrase in phrases:
            work.put_nowait((phrase, generation))

    async def cut() -> None:
        try:
            await _cut()
        except Exception:  # noqa: BLE001 - a broken source must still end the stream
            logger.warning("speech: the reply's source failed", exc_info=True)
            out.put_nowait({"type": "done", "reason": "ended"})

    async def _cut() -> None:
        nonlocal generation
        events = source.__aiter__()
        while True:
            try:
                kind, value = await asyncio.wait_for(events.__anext__(), idle_timeout)
            except StopAsyncIteration:
                kind, value = "end", "ended"
            except TimeoutError:
                out.put_nowait({"type": "done", "reason": "timeout"})
                return
            if kind == "reply":
                out.put_nowait({"type": "reply", "message_id": value})
            elif kind == "text":
                enqueue(cutter.feed(value))
            elif kind == "boundary":
                enqueue(cutter.boundary())
            elif kind == "discard":
                generation += 1
                discards.append((generation, value))
                cutter.discard(value)
                out.put_nowait({"type": "discard", "keep_chars": value})
            elif kind == "end":
                if value == "complete":
                    enqueue(cutter.finish())
                    work.put_nowait(None)  # the synthesis task says done once everything before it is spoken
                else:
                    out.put_nowait({"type": "done", "reason": value})
                return

    async def synthesize() -> None:
        try:
            await _synthesize()
        except Exception:  # noqa: BLE001 - an unexpected voice error must still end the stream, said as unusable
            logger.warning("speech: synthesis failed unexpectedly", exc_info=True)
            out.put_nowait({"type": "done", "reason": "unavailable", "voice": "failed",
                            "message": SENTENCES["failed"], "resume_at": current["start"]})

    current = {"start": 0}

    async def _synthesize() -> None:
        index = 0
        while True:
            item = await work.get()
            if item is None:
                done: dict[str, Any] = {"type": "done", "reason": "complete"}
                if cutter.truncated:
                    done["truncated"] = True
                out.put_nowait(done)
                return
            phrase, gen = item
            if dropped(phrase, gen):
                continue
            current["start"] = phrase.start
            try:
                audio = await voice.synthesize(phrase.text)
            except VoiceUnavailable as exc:
                out.put_nowait({"type": "done", "reason": "unavailable", "voice": exc.reason,
                                "message": exc.message, "resume_at": phrase.start})
                return
            if dropped(phrase, gen):
                continue
            out.put_nowait({"type": "phrase", "index": index, "start": phrase.start, "end": phrase.end,
                            "text": phrase.text, "audio": base64.b64encode(audio).decode("ascii")})
            index += 1

    yield {"type": "start", "version": VERSION, "mime": MIME}
    tasks = [asyncio.ensure_future(cut()), asyncio.ensure_future(synthesize())]
    getter: asyncio.Future[dict[str, Any]] | None = None
    try:
        while True:
            if getter is None:
                getter = asyncio.ensure_future(out.get())
            waiting: set[asyncio.Future[Any]] = {getter}
            if stop is not None:
                waiting.add(stop)
            finished, _ = await asyncio.wait(waiting, timeout=keepalive, return_when=asyncio.FIRST_COMPLETED)
            if getter in finished:
                line = getter.result()
                getter = None
                yield line
                if line["type"] == "done":
                    return
            elif stop is not None and stop in finished:
                yield {"type": "done", "reason": "ended" if stop.cancelled() else stop.result()}
                return
            else:
                yield {"type": "keepalive"}
    finally:
        if getter is not None:
            getter.cancel()
        for task in tasks:
            task.cancel()
        # gather collects every task's ending; a cancel aimed at this generator's own task still propagates.
        await asyncio.gather(*tasks, return_exceptions=True)
