#!/usr/bin/env python
"""
Simple CLI audio/video downloader using yt-dlp.

Usage:
  python downloader.py --type audio <url>
  python downloader.py --type video <url>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import os
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

from yt_dlp import YoutubeDL


def build_parser() -> argparse.ArgumentParser:
    """
    build_parser - function to build the argument parser for the CLI.

    Returns:
        purser.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="downloader",
        description="Download audio or video from a URL using yt-dlp.",
    )
    parser.add_argument(
        "--type",
        choices=["audio", "video"],
        required=True,
        help="Download type: audio or video.",
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Audio/Video URL to download (optional if --input-file is used).",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        default=None,
        help="Path to a file containing URLs (one per line) to download.",
    )
    parser.add_argument(
        "--output",
        default="downloads",
        help="Output folder (default: downloads).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose debug output.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write a log file.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts on transient failure (default: 3).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming partial downloads.",
    )
    parser.add_argument(
        "--format",
        dest="format",
        default=None,
        help="Custom yt-dlp format string (overrides defaults).",
    )
    parser.add_argument(
        "--audio-format",
        dest="audio_format",
        choices=["mp3", "aac", "opus", "m4a", "wav"],
        default="mp3",
        help="Preferred audio codec for extraction (default: mp3).",
    )
    parser.add_argument(
        "--audio-quality",
        dest="audio_quality",
        default="192",
        help="Preferred audio bitrate/quality for extraction (e.g., 192).",
    )
    parser.add_argument(
        "--output-template",
        dest="output_template",
        default=None,
        help="Output template for filenames (yt-dlp template).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent downloads when using --input-file (default: 1).",
    )
    parser.add_argument(
        "--proxy",
        dest="proxy",
        default=None,
        help="Proxy URL to use for downloads (e.g. http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--rate-limit",
        dest="rate_limit",
        type=int,
        default=None,
        help="Download rate limit in bytes/sec (integer).",
    )
    parser.add_argument(
        "--header",
        dest="headers",
        action="append",
        default=[],
        help="Additional HTTP header to send (can be repeated): 'Key: Value'",
    )
    return parser


def load_urls_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    urls: List[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            urls.append(s)
    return urls


def load_config() -> dict:
    """Load configuration from local `downloaderrc` or user's home `~/.downloaderrc`.

    Returns a flat dict of options (strings); CLI arguments override these.
    """
    cfg = configparser.ConfigParser()
    candidates = [
        Path("downloaderrc"),
        Path("downloaderrc.ini"),
        Path.home() / ".downloaderrc",
        Path.home() / "downloaderrc.ini",
    ]
    found = []
    for p in candidates:
        if p.exists():
            found.append(str(p))
    if not found:
        return {}
    cfg.read(found)
    section = "downloader"
    if section not in cfg:
        # allow top-level options too
        section = cfg.sections()[0] if cfg.sections() else None
    out: dict = {}
    if section:
        for k, v in cfg[section].items():
            out[k] = v
    return out


def _flag_provided(*names: str) -> bool:
    """Return True if any of the flag names were provided on the command line.

    Example: _flag_provided('--quiet')
    """
    for n in names:
        if n in sys.argv:
            return True
    return False


def parse_headers(header_list: List[str]) -> dict:
    """Parse a list of 'Key: Value' strings into a dict for yt-dlp's `http_headers` option."""
    headers: dict = {}
    for h in header_list or []:
        if not h or ":" not in h:
            continue
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def run_download_with_retries(
    url: str,
    args: argparse.Namespace,
    ytdl_logger: object,
    output_dir: Path,
) -> int:
    logger = logging.getLogger("downloader")
    resume = not bool(args.no_resume)
    retries = max(1, int(args.retries or 1))
    for attempt in range(1, retries + 1):
        if args.type == "audio":
            rc = download_audio(
                url,
                output_dir,
                ytdl_logger,
                args.quiet,
                resume,
                fmt=args.format,
                audio_format=args.audio_format,
                audio_quality=args.audio_quality,
                output_template=args.output_template,
                proxy=getattr(args, "proxy", None),
                rate_limit=getattr(args, "rate_limit", None),
                headers=parse_headers(getattr(args, "headers", [])),
            )
        else:
            rc = download_video(
                url,
                output_dir,
                ytdl_logger,
                args.quiet,
                resume,
                fmt=args.format,
                output_template=args.output_template,
                proxy=getattr(args, "proxy", None),
                rate_limit=getattr(args, "rate_limit", None),
                headers=parse_headers(getattr(args, "headers", [])),
            )
        if rc == 0:
            return 0
        logger.warning("Attempt %d/%d failed for %s", attempt, retries, url)
        if attempt < retries:
            backoff = min(60, 2 ** attempt)
            logger.info("Retrying %s in %s seconds...", url, backoff)
            time.sleep(backoff)
    logger.error("All %d attempts failed for %s", retries, url)
    return 1


def download_audio(
    url: str,
    output_dir: Path,
    ytdl_logger: Optional[object] = None,
    quiet: bool = False,
    resume: bool = True,
    fmt: Optional[str] = None,
    audio_format: str = "mp3",
    audio_quality: str = "320",
    output_template: Optional[str] = None,
    proxy: Optional[str] = None,
    rate_limit: Optional[int] = None,
    headers: Optional[dict] = None,
) -> int:
    """
    download_audio - this function downloads audio from a given URL.

    Args:
        url (str): The URL to download audio from.
        output_dir (Path): The directory to save the downloaded audio.

    Returns:
        int: Exit code of the download process.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # base options
    outtmpl = str(output_dir / (output_template or "%(title)s.%(ext)s"))
    ydl_opts = {
        "format": fmt or "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": str(audio_quality),
            }
        ],
    }
    if ytdl_logger is not None:
        ydl_opts["logger"] = ytdl_logger
    if quiet:
        ydl_opts["quiet"] = True
    if proxy:
        ydl_opts["proxy"] = proxy
    if rate_limit:
        ydl_opts["ratelimit"] = int(rate_limit)
    if headers:
        ydl_opts["http_headers"] = headers
    # resume partial downloads when possible
    ydl_opts["continuedl"] = bool(resume)
    # show progress and ETA
    progress = YTDLProgress(quiet=quiet)
    ydl_opts["progress_hooks"] = [progress]
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:  # ImportError or other
        logging.getLogger("downloader").error(
            "yt-dlp not available: install yt-dlp to download media (%s)", exc
        )
        return 1
    with YoutubeDL(ydl_opts) as ydl:
        try:
            return ydl.download([url])
        except Exception as exc:
            logging.getLogger("downloader").exception("Download failed: %s", exc)
            return 1


def download_video(
    url: str,
    output_dir: Path,
    ytdl_logger: Optional[object] = None,
    quiet: bool = False,
    resume: bool = True,
    fmt: Optional[str] = None,
    output_template: Optional[str] = None,
    proxy: Optional[str] = None,
    rate_limit: Optional[int] = None,
    headers: Optional[dict] = None,
) -> int:
    """
    download_video - this function downloads video from a given URL.

    Args:
        url (str): The URL to download video from.
        output_dir (Path): The directory to save the downloaded video.

    Returns:
        int: Exit code of the download process.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / (output_template or "%(title)s.%(ext)s"))
    ydl_opts = {
        "format": fmt or "bestvideo+bestaudio/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
    }
    if ytdl_logger is not None:
        ydl_opts["logger"] = ytdl_logger
    if quiet:
        ydl_opts["quiet"] = True
    if proxy:
        ydl_opts["proxy"] = proxy
    if rate_limit:
        ydl_opts["ratelimit"] = int(rate_limit)
    if headers:
        ydl_opts["http_headers"] = headers
    # resume partial downloads when possible
    ydl_opts["continuedl"] = bool(resume)
    # show progress and ETA
    progress = YTDLProgress(quiet=quiet)
    ydl_opts["progress_hooks"] = [progress]
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        logging.getLogger("downloader").error(
            "yt-dlp not available: install yt-dlp to download media (%s)", exc
        )
        return 1
    with YoutubeDL(ydl_opts) as ydl:
        try:
            return ydl.download([url])
        except Exception as exc:
            logging.getLogger("downloader").exception("Download failed: %s", exc)
            return 1


def _format_bytes(num: Optional[float]) -> str:
    if not num or num <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while num >= 1024 and idx < len(units) - 1:
        num /= 1024.0
        idx += 1
    return f"{num:3.1f}{units[idx]}"


class YTDLProgress:
    """Simple progress hook for yt-dlp showing percent, speed and ETA."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._last_len = 0

    def __call__(self, d: dict) -> None:
        if self.quiet:
            return
        status = d.get("status")
        if status == "downloading":
            downloaded = d.get("downloaded_bytes", 0) or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            speed = d.get("speed") or 0
            eta = d.get("eta")
            if total:
                percent = downloaded / total * 100
                pct = f"{percent:5.1f}%"
            else:
                pct = "  ?.?%"
            speed_s = _format_bytes(speed) + "/s"
            eta_s = f"{int(eta)}s" if eta is not None else "?s"
            fname = d.get("filename") or d.get("tmpfilename") or "(unknown)"
            line = f"Downloading {fname}: {pct} {speed_s} ETA {eta_s}"
            print("\r" + line.ljust(self._last_len), end="", flush=True)
            self._last_len = max(self._last_len, len(line))
        elif status == "finished":
            # finish line and newline
            fname = d.get("filename") or d.get("tmpfilename") or "(unknown)"
            print()
            print(f"Finished: {fname}")


class YTDLLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)


def main(argv: list[str] | None = None) -> int:
    """
    main - main function to parse arguments and initiate download.
    Args:
        argv (list[str] | None): List of command-line arguments. If None, uses sys.argv.
    Returns:
        int: Exit code of the program.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # load config and apply defaults when flags not provided
    cfg = load_config()
    # helper to set attribute from cfg if flag not given on CLI
    def apply_cfg(key: str, attr: str, cast=None):
        if not _flag_provided(f"--{key}") and key in cfg:
            val = cfg[key]
            try:
                setattr(args, attr, cast(val) if cast else val)
            except Exception:
                setattr(args, attr, val)

    apply_cfg("output", "output")
    apply_cfg("retries", "retries", int)
    apply_cfg("no-resume", "no_resume", lambda v: v.lower() in ("1", "true", "yes", "on"))
    apply_cfg("quiet", "quiet", lambda v: v.lower() in ("1", "true", "yes", "on"))
    apply_cfg("verbose", "verbose", lambda v: v.lower() in ("1", "true", "yes", "on"))
    apply_cfg("audio-format", "audio_format")
    apply_cfg("audio-quality", "audio_quality")
    apply_cfg("format", "format")
    apply_cfg("output-template", "output_template")
    apply_cfg("log-file", "log_file")
    apply_cfg("workers", "workers", int)
    apply_cfg("input-file", "input_file")

    output_dir = Path(args.output)

    # configure logging
    logger = logging.getLogger("downloader")
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.handlers[:] = [handler]
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)

    ytdl_logger = YTDLLogger(logger)

    # collect URLs: either single url or from input file
    urls: List[str] = []
    if args.input_file:
        try:
            urls = load_urls_from_file(args.input_file)
        except Exception as exc:
            logger.error("Failed to read input file: %s", exc)
            return 1
    elif args.url:
        urls = [args.url]
    else:
        parser.error("either a URL or --input-file must be provided")

    # if running concurrent downloads, suppress per-download live progress
    if getattr(args, "workers", 1) > 1:
        args.quiet = True

    failures = 0
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers > 1 and len(urls) > 1:
        logger.info("Starting concurrent downloads: %d workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_download_with_retries, url, args, ytdl_logger, output_dir): url for url in urls}
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    rc = fut.result()
                except Exception as exc:
                    logger.exception("Download raised for %s: %s", url, exc)
                    failures += 1
                else:
                    if rc != 0:
                        failures += 1
    else:
        for url in urls:
            logger.info("Processing: %s", url)
            rc = run_download_with_retries(url, args, ytdl_logger, output_dir)
            if rc != 0:
                failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

