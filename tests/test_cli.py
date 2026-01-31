import argparse
from types import SimpleNamespace
from pathlib import Path

import pytest

import downloader


def test_build_parser_audio():
    parser = downloader.build_parser()
    args = parser.parse_args(["--type", "audio", "http://example.com/audio"])
    assert args.type == "audio"
    assert args.url == "http://example.com/audio"


def test_load_urls_from_file(tmp_path: Path):
    f = tmp_path / "urls.txt"
    f.write_text("# comment\nhttp://a.example\n\nhttp://b.example\n")
    urls = downloader.load_urls_from_file(str(f))
    assert urls == ["http://a.example", "http://b.example"]


def make_args_for_type(t: str) -> argparse.Namespace:
    parser = downloader.build_parser()
    # supply a dummy url so parser succeeds
    args = parser.parse_args(["--type", t, "http://x.example"])
    # make retries small for tests
    args.retries = 2
    args.no_resume = False
    return args


def test_run_download_with_retries_audio_success(monkeypatch, tmp_path: Path):
    args = make_args_for_type("audio")
    called = {"count": 0}

    def fake_download_audio(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(downloader, "download_audio", fake_download_audio)

    rc = downloader.run_download_with_retries("http://a.example", args, None, tmp_path)
    assert rc == 0
    assert called["count"] == 1


def test_run_download_with_retries_audio_retry(monkeypatch, tmp_path: Path):
    args = make_args_for_type("audio")
    seq = {"calls": 0}

    def fake_download_audio(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        seq["calls"] += 1
        # fail first time, succeed second time
        return 1 if seq["calls"] == 1 else 0

    monkeypatch.setattr(downloader, "download_audio", fake_download_audio)

    rc = downloader.run_download_with_retries("http://a.example", args, None, tmp_path)
    assert rc == 0
    assert seq["calls"] == 2


def test_run_download_with_retries_video(monkeypatch, tmp_path: Path):
    args = make_args_for_type("video")
    args.type = "video"
    called = {"count": 0}

    def fake_download_video(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(downloader, "download_video", fake_download_video)

    rc = downloader.run_download_with_retries("http://v.example", args, None, tmp_path)
    assert rc == 0
    assert called["count"] == 1
