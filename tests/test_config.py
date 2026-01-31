from pathlib import Path

import downloader


def test_load_config_from_cwd(monkeypatch, tmp_path: Path):
    # create a downloaderrc in the working directory
    cfg = tmp_path / "downloaderrc"
    cfg.write_text("[downloader]\nworkers = 3\nquiet = true\n")
    monkeypatch.chdir(tmp_path)

    out = downloader.load_config()
    assert out.get("workers") == "3"
    assert out.get("quiet").lower() == "true"
