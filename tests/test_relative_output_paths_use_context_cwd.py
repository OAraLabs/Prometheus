"""A relative output path lands under the tool context's cwd, not the daemon's.

THE DEFECT
----------
``engine/agent_loop.py`` sets ``ToolExecutionContext.cwd`` to the session
workspace in a workspace session (``_run_paths``), and write_file / edit_file /
notebook_edit / read_file all resolve a relative path against it. Three write
tools did not:

* ``download_file`` — ``_resolve_destination`` used ``Path.cwd() / candidate``,
  the daemon PROCESS's directory;
* ``tts`` — handed ``output_path`` to espeak/piper as given, so the engine
  subprocess resolved it against the directory it inherited (the daemon's);
* ``image_generate`` — ``assert_path_under_roots`` and the save sink both
  resolved a relative ``output_path`` against the process cwd.

So in a session bound to ``/workspace``, an approved
``download_file(destination="x.bin")`` wrote ``x.bin`` beside the daemon.

Each test RUNS the tool with the process cwd moved somewhere else, so a
regression to the process cwd writes into ``elsewhere`` and fails.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from prometheus.tools.base import ToolExecutionContext, ToolResult
from prometheus.tools.builtin import download_file as dl
from prometheus.tools.builtin import image_generate as ig
from prometheus.tools.builtin import tts
from prometheus.tools.builtin.download_file import DownloadFileInput, DownloadFileTool
from prometheus.tools.builtin.image_generate import ImageGenerateInput, ImageGenerateTool
from prometheus.tools.builtin.tts import TTSInput, TTSTool


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """(workspace, elsewhere): the context cwd, and the process cwd."""
    workspace = tmp_path / "workspace"
    elsewhere = tmp_path / "daemon-cwd"
    workspace.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return workspace, elsewhere


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


@pytest.fixture
def offline_download(monkeypatch):
    """Serve every GET from memory: no DNS, no network."""
    real_client = httpx.AsyncClient

    def client(**kwargs):
        kwargs.pop("event_hooks", None)  # the SSRF hop guard resolves DNS
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"payload")),
            **kwargs,
        )

    monkeypatch.setattr(dl, "is_safe_url", lambda url: True)
    monkeypatch.setattr(dl.httpx, "AsyncClient", client)


async def test_download_relative_destination_lands_in_the_context_cwd(
    dirs, offline_download,
):
    workspace, elsewhere = dirs
    result = await DownloadFileTool().execute(
        DownloadFileInput(url="https://example.com/x.bin", destination="sub/x.bin"),
        ToolExecutionContext(cwd=workspace),
    )
    assert not result.is_error, result.output
    assert (workspace / "sub" / "x.bin").read_bytes() == b"payload"
    assert result.metadata["path"] == str((workspace / "sub" / "x.bin").resolve())
    assert not (elsewhere / "sub").exists()


def test_resolve_destination_takes_no_process_cwd_default():
    """``base`` is required, so no caller can fall back to the process cwd."""
    with pytest.raises(TypeError):
        dl._resolve_destination("https://example.com/x.bin", "x.bin")  # type: ignore[call-arg]


async def test_download_relative_destination_still_hits_the_protected_floor(
    tmp_path, offline_download,
):
    """Resolving against the context cwd keeps the protected-path check: a
    relative path that climbs into /etc is refused, wherever the cwd is."""
    depth = len(tmp_path.resolve().parts) - 1
    climb = "/".join([".."] * depth) + "/etc/prometheus-probe.bin"
    result = await DownloadFileTool().execute(
        DownloadFileInput(url="https://example.com/x.bin", destination=climb),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert result.is_error
    assert "protected path" in result.output


# ---------------------------------------------------------------------------
# tts
# ---------------------------------------------------------------------------


class _FakeProc:
    returncode = 0

    async def communicate(self, _input):
        return b"", b""


@pytest.fixture
def fake_engine(monkeypatch):
    """Stand in for espeak: write the -w path RELATIVE TO THE PROCESS CWD, as
    the real child (which inherits the daemon's cwd) would. No child runs."""
    calls: list[tuple[str, ...]] = []

    async def exec_(*args, **_kwargs):
        calls.append(args)
        out = args[args.index("-w") + 1]
        with open(out, "wb") as fh:  # relative -> os.getcwd(), like the child
            fh.write(b"RIFF-wav")
        return _FakeProc()

    monkeypatch.setattr(tts.asyncio, "create_subprocess_exec", exec_)
    return calls


async def test_tts_relative_output_path_lands_in_the_context_cwd(dirs, fake_engine):
    workspace, elsewhere = dirs
    result = await TTSTool().execute(
        TTSInput(text="hello", engine="espeak", output_path="speech.wav"),
        ToolExecutionContext(cwd=workspace),
    )
    assert not result.is_error, result.output
    assert (workspace / "speech.wav").read_bytes() == b"RIFF-wav"
    assert not (elsewhere / "speech.wav").exists()
    assert str((workspace / "speech.wav").resolve()) in result.output
    assert os.path.isabs(fake_engine[0][fake_engine[0].index("-w") + 1])


# ---------------------------------------------------------------------------
# image_generate
# ---------------------------------------------------------------------------


@pytest.fixture
def through_the_real_sink(monkeypatch):
    """Skip generation but keep the write: the fake backend hands whatever
    output_path execute() passes it to the real ``_save_image_bytes``."""
    async def backend(_requested, _cfg):
        return "pollinations"

    async def pollinations(arguments):
        path = ig._save_image_bytes(
            b"\x89PNG-fake", ext=".png", override_path=arguments.output_path)
        return ToolResult(output=f"Saved {path}", metadata={"path": path})

    monkeypatch.setattr(ig, "_resolve_backend", backend)
    monkeypatch.setattr(ig, "_generate_via_pollinations", pollinations)


async def test_image_relative_output_path_lands_in_the_context_cwd(
    dirs, monkeypatch, through_the_real_sink,
):
    workspace, elsewhere = dirs
    roots = [workspace / "images", elsewhere / "images"]
    monkeypatch.setattr(ig, "_allowed_image_roots", lambda: roots)
    result = await ImageGenerateTool().execute(
        ImageGenerateInput(prompt="a fox", output_path="images/fox.png"),
        ToolExecutionContext(cwd=workspace),
    )
    assert not result.is_error, result.output
    assert (workspace / "images" / "fox.png").read_bytes() == b"\x89PNG-fake"
    assert not (elsewhere / "images").exists()


async def test_image_relative_output_path_is_ruled_on_from_the_context_cwd(
    dirs, monkeypatch, through_the_real_sink,
):
    """The allow-list rules on the path the context names. Under the process
    cwd, ``images/fox.png`` is inside the allowed root; under the context cwd it
    is not — so it is refused, and nothing is written in either place."""
    workspace, elsewhere = dirs
    monkeypatch.setattr(ig, "_allowed_image_roots", lambda: [elsewhere / "images"])
    result = await ImageGenerateTool().execute(
        ImageGenerateInput(prompt="a fox", output_path="images/fox.png"),
        ToolExecutionContext(cwd=workspace),
    )
    assert result.is_error
    assert "allowed image directories" in result.output
    assert not (elsewhere / "images").exists()
    assert not (workspace / "images").exists()
