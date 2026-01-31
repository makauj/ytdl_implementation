import builtins
import sys
import logging
from pathlib import Path

import pytest

import downloader


def test_parse_headers():
    headers = ["Authorization: Bearer TOKEN", "X-Test: value", "badheader", "Empty:", " "]
    out = downloader.parse_headers(headers)
    assert out["Authorization"] == "Bearer TOKEN"
    assert out["X-Test"] == "value"
    assert "badheader" not in out


def test_format_bytes():
    assert downloader._format_bytes(None) == "0B"
    assert downloader._format_bytes(0) == "0B"
    assert downloader._format_bytes(512) == "512.0B"
    assert downloader._format_bytes(1024) == "1.0KB"
    assert downloader._format_bytes(1024 * 1024 * 3.5).startswith("3.5MB")


def test_ytdlprogress_printing(capsys):
    p = downloader.YTDLProgress(quiet=False)
    # simulate a downloading event
    p({
        "status": "downloading",
        "downloaded_bytes": 512,
        "total_bytes": 1024,
        "speed": 256,
        "eta": 2,
        "filename": "file.mp3",
    })
    # simulate finished
    p({"status": "finished", "filename": "file.mp3"})
    out = capsys.readouterr().out
    assert "Downloading" in out
    assert "Finished: file.mp3" in out


def test_run_download_with_retries_all_fail(monkeypatch, tmp_path: Path):
    # make both download functions fail
    calls = {"audio": 0, "video": 0}

    def fail_audio(*args, **kwargs):
        calls["audio"] += 1
        return 1

    def fail_video(*args, **kwargs):
        calls["video"] += 1
        return 1

    monkeypatch.setattr(downloader, "download_audio", fail_audio)
    monkeypatch.setattr(downloader, "download_video", fail_video)

    class Args:
        type = "audio"
        retries = 2
        no_resume = False
        quiet = True
        format = None
        audio_format = "mp3"
        audio_quality = "192"
        output_template = None
        proxy = None
        rate_limit = None
        headers = []

    rc = downloader.run_download_with_retries("http://x", Args(), None, tmp_path)
    assert rc == 1
    assert calls["audio"] == 2


def test_download_audio_no_yt_dlp(monkeypatch, tmp_path: Path):
    # force import of yt_dlp to raise
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yt_dlp":
            raise ImportError("simulated missing package")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = downloader.download_audio("http://x", tmp_path)
    assert rc == 1


def test_ytdllogger_forwards_messages(caplog):
    logger = logging.getLogger("testlogger")
    logger.setLevel(logging.DEBUG)
    yg = downloader.YTDLLogger(logger)
    with caplog.at_level(logging.DEBUG):
        yg.debug("d")
        yg.info("i")
        yg.warning("w")
        yg.error("e")
    messages = [r.message for r in caplog.records]
    assert "d" in messages
    assert "i" in messages
    assert "w" in messages
    assert "e" in messages
