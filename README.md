# Downloader

CLI tool to download audio or video using yt-dlp.

## Setup

```bash
pip install -r requirements.txt
```

> Note: Audio extraction uses FFmpeg. Install FFmpeg and ensure it is on your PATH.

Install locally (optional):

```bash
# Install editable for development
pip install -e .

# Or install normally
pip install .
```

## Usage

```bash
python downloader.py --type audio <url>
python downloader.py --type video <url>
```

Optional output directory:

```bash
python downloader.py --type audio --output downloads <url>
```

Verbosity and logging:

```bash
python downloader.py --type audio --verbose <url>
python downloader.py --type video --quiet --log-file downloader.log <url>
```

Note: `--quiet` and `--verbose` are mutually exclusive.

Resume and retries:

```bash
python downloader.py --type audio --retries 3 <url>
python downloader.py --type video --no-resume --retries 1 <url>
```

`--retries` controls how many attempts the tool will make on failure (default 3). Set `--no-resume` to disable resuming partial downloads.

Format & quality selection:

```bash
# Custom yt-dlp format selection (advanced):
python downloader.py --type video --format "bv*+ba/best" <url>

# Choose audio codec and quality for extracted audio (default mp3 192):
python downloader.py --type audio --audio-format opus --audio-quality 160 <url>

# Control filename template (yt-dlp templating syntax):
python downloader.py --type audio --output-template "%(uploader)s - %(title)s.%(ext)s" <url>
```

Batch / playlist input:

Create a plain text file with one URL per line (lines starting with `#` are ignored). Then run:

```bash
python downloader.py --input-file urls.txt --type audio
```

The tool will process each URL in order and return a non-zero exit code if any downloads fail.

Concurrent downloads:

```bash
# Process a list of URLs concurrently with 4 workers
python downloader.py --input-file urls.txt --type video --workers 4
```

When `--workers` is greater than 1 the live per-download progress line is suppressed to keep the console output readable.

Configuration file:

You can create a simple INI-style config file named `downloaderrc` or `downloaderrc.ini` in the current folder or `~/.downloaderrc` with a `[downloader]` section. Example:

```ini
[downloader]
output = downloads
retries = 3
audio-format = mp3
audio-quality = 192
workers = 2
# Use true/false for boolean flags
quiet = false
```

CLI flags override values in the config file.

Proxy / rate-limit / headers:

```bash
# Use a HTTP proxy
python downloader.py --type video --proxy http://127.0.0.1:8080 <url>

# Rate limit in bytes/sec (e.g. 50000 bytes/sec)
python downloader.py --type audio --rate-limit 50000 <url>

# Send custom headers (repeatable)
python downloader.py --type video --header "Authorization: Bearer TOKEN" --header "X-My: header" <url>
```

Progress display:

The downloader prints a simple progress line with percent, current speed, and ETA while downloading. Use `--quiet` to suppress the progress output.
