import sys
from pathlib import Path

import downloader


def test_batch_processing_concurrent(monkeypatch, tmp_path: Path):
    # prepare urls file
    f = tmp_path / "urls.txt"
    f.write_text("http://a.example\nhttp://b.example\n")

    # monkeypatch download functions to avoid network
    calls = {"audio": 0, "video": 0}

    def fake_audio(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        calls["audio"] += 1
        return 0

    def fake_video(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        calls["video"] += 1
        return 0

    monkeypatch.setattr(downloader, "download_audio", fake_audio)
    monkeypatch.setattr(downloader, "download_video", fake_video)

    # ensure sys.argv contains flags so load_config detection treats CLI as provided
    argv = ["downloader", "--input-file", str(f), "--type", "audio", "--workers", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    rc = downloader.main(["--input-file", str(f), "--type", "audio", "--workers", "2"])
    assert rc == 0
    assert calls["audio"] == 2


def test_single_url_integration(monkeypatch, tmp_path: Path):
    calls = {"video": 0}

    def fake_video(url, output_dir, ytdl_logger, quiet, resume, **kwargs):
        calls["video"] += 1
        return 0

    monkeypatch.setattr(downloader, "download_video", fake_video)
    monkeypatch.setattr(sys, "argv", ["downloader", "--type", "video", "http://v.example"])

    rc = downloader.main(["--type", "video", "http://v.example"])
    assert rc == 0
    assert calls["video"] == 1
