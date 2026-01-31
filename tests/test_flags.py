import argparse
from pathlib import Path

import pytest

import downloader


def test_parse_extractor_args():
    inp = [
        "example:auth=token&v=1",
        "other:foo=bar",
        "example:extra=3",
        "bad",
        "empty:",
    ]
    out = downloader.parse_extractor_args(inp)
    assert out["example"] == "auth=token&v=1&extra=3"
    assert out["other"] == "foo=bar"
    assert "bad" not in out


def make_args(argv: list[str]) -> argparse.Namespace:
    parser = downloader.build_parser()
    return parser.parse_args(argv)


def test_run_download_with_retries_forwards_extractor_args_and_cookies_audio(monkeypatch, tmp_path: Path):
    args = make_args(["--type", "audio", "--extractor-args", "ex:foo=1&bar=2", "--cookies", str(tmp_path / "c.txt"), "http://a.example"])
    called = {}

    def fake_audio(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        called["url"] = url
        called["extractor_args"] = kwargs.get("extractor_args")
        called["cookies"] = kwargs.get("cookies")
        called["force_generic"] = kwargs.get("force_generic")
        return 0

    monkeypatch.setattr(downloader, "download_audio", fake_audio)

    rc = downloader.run_download_with_retries("http://a.example", args, None, tmp_path)
    assert rc == 0
    assert called["extractor_args"] == {"ex": "foo=1&bar=2"}
    assert called["cookies"] == str(tmp_path / "c.txt")


def test_run_download_with_retries_forwards_extractor_args_and_force_generic_video(monkeypatch, tmp_path: Path):
    args = make_args(["--type", "video", "--extractor-args", "site:token=abc", "--force-generic", "http://v.example"])
    called = {}

    def fake_video(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        called["url"] = url
        called["extractor_args"] = kwargs.get("extractor_args")
        called["force_generic"] = kwargs.get("force_generic")
        return 0

    monkeypatch.setattr(downloader, "download_video", fake_video)

    rc = downloader.run_download_with_retries("http://v.example", args, None, tmp_path)
    assert rc == 0
    assert called["extractor_args"] == {"site": "token=abc"}
    assert called["force_generic"] is True
