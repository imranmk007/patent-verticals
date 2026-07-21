import argparse
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
from pypdf import PdfReader, PdfWriter

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 1536
QUERY_TEMPLATE = "task: search result | query: {content}"

TEXT_CHUNK_WORDS = 3500
TEXT_OVERLAP_WORDS = 300
TEXT_CHUNK_CHAR_CAP = 24000
MAX_WORD_CHARS = 2000

PDF_CHUNK_PAGES = 2
PDF_OVERLAP_PAGES = 1

IMAGES_PER_REQUEST = 3
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048

VIDEO_CHUNK_SECONDS = 60
VIDEO_OVERLAP_SECONDS = 20

AUDIO_CHUNK_SECONDS = 30
AUDIO_OVERLAP_SECONDS = 15

MAX_INLINE_BYTES = 18 * 1024 * 1024
REQUEST_TIMEOUT_MS = 120000
EMBED_ATTEMPTS = 5
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

MAX_REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_MAX_RPM", "90"))
MAX_TOKENS_PER_MINUTE = int(os.environ.get("GEMINI_MAX_TPM", "28000"))
RATE_WINDOW_SECONDS = 60.0

MODEL_TOKEN_LIMIT = 8192
CHARS_PER_TOKEN = 4
IMAGE_TOKENS = 258
PDF_TOKENS_PER_PAGE = 258
AUDIO_TOKENS_PER_SECOND = 25
VIDEO_TOKENS_PER_CHUNK = 66 * 32

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "test-files"
DEFAULT_OUTPUT_DIR = BASE_DIR / "embeddings"

API_KEY_ENV = "GEMINI_API_KEY"
LOCAL_API_KEY = ""

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

CATEGORY_BY_MIME = {
    "application/pdf": "document",
    DOCX_MIME: "document",
    "text/plain": "text",
    "text/html": "text",
    "text/csv": "text",
    "text/rtf": "text",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "image/heic": "image",
    "image/heif": "image",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/m4a": "audio",
    "audio/ogg": "audio",
    "audio/aac": "audio",
    "audio/flac": "audio",
    "video/mp4": "video",
    "video/mpeg": "video",
    "video/quicktime": "video",
    "video/webm": "video",
}

EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".txt": "text/plain",
    ".text": "text/plain",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".rtf": "text/rtf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/m4a",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs"}
HEIF_BRANDS = {b"mif1", b"msf1", b"heif"}
M4A_BRANDS = {b"M4A ", b"M4B ", b"M4P "}
UNSUPPORTED_FTYP_BRANDS = {b"avif", b"avis"}
FTYP_AUDIO_EXTENSIONS = {".m4a", ".m4b", ".m4p", ".aac", ".mp3", ".ogg", ".oga", ".flac", ".wav"}

AUDIO_FFMPEG_ARGS = ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav"]
VIDEO_FFMPEG_ARGS = [
    "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4",
]

DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
PROGRESS_TIME_PATTERN = re.compile(r"time=(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
HTML_MARKER_PATTERN = re.compile(r"<(!doctype\s+html|html|head|body|title)[\s>]", re.IGNORECASE)

class FileProcessingError(Exception):
    pass

@dataclass
class PreparedChunk:
    category: str
    meta: dict
    parts: list
    text: str | None = None
    est_tokens: int = 0
    source: str = ""
    mime: str = ""
    sha: str = ""
    uid: str = ""
    index: int = 0
    total: int = 0
    embedding: list | None = None
    error: str | None = None

def resolve_api_key():
    key = os.environ.get(API_KEY_ENV, "").strip() or LOCAL_API_KEY
    if not key:
        raise SystemExit(f"no API key: set the {API_KEY_ENV} environment variable")
    return key

def windowed_spans(total, size, step, tail_epsilon=0.0):
    spans = []
    start = 0
    while True:
        end = min(start + size, total)
        spans.append((start, end))
        if end >= total - tail_epsilon:
            break
        start += step
    return spans

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def is_docx_zip(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return False

def isobmff_has_video_stream(path):
    try:
        probe = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (FileProcessingError, subprocess.TimeoutExpired, OSError):
        return None
    for line in probe.stderr.splitlines():
        if "Video:" in line and "attached pic" not in line.lower():
            return True
    return False

def isobmff_av_mime(path):
    has_video = isobmff_has_video_stream(path)
    if has_video is None:
        return "audio/m4a" if path.suffix.lower() in FTYP_AUDIO_EXTENSIONS else "video/mp4"
    return "video/mp4" if has_video else "audio/m4a"

def sniff_mime(head, path):
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"{\\rtf"):
        return "text/rtf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and len(head) >= 12:
        if head[8:12] == b"WAVE":
            return "audio/wav"
        if head[8:12] == b"WEBP":
            return "image/webp"
        return None
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head.startswith(b"ID3"):
        return "audio/mp3"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        if (head[1] & 0x06) == 0x00:
            return "audio/aac"
        return "audio/mp3"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in UNSUPPORTED_FTYP_BRANDS:
            return None
        if brand in HEIC_BRANDS:
            return "image/heic"
        if brand in HEIF_BRANDS:
            return "image/heif"
        if brand in M4A_BRANDS:
            return "audio/m4a"
        return isobmff_av_mime(path)
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm" if b"webm" in head[:256] else None
    if head.startswith(b"\x00\x00\x01\xba") or head.startswith(b"\x00\x00\x01\xb3"):
        return "video/mpeg"
    if head.startswith(b"PK\x03\x04"):
        return DOCX_MIME if is_docx_zip(path) else None
    return None

def looks_like_text(head):
    if not head or b"\x00" in head:
        return False
    try:
        sample = head.decode("utf-8")
    except UnicodeDecodeError:
        sample = head.decode("latin-1")
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t\f")
    return printable / len(sample) >= 0.9

def classify_text_mime(head):
    sample = head.decode("utf-8", errors="replace")
    if HTML_MARKER_PATTERN.search(sample):
        return "text/html"
    return "text/plain"

def detect_mime(path):
    try:
        with path.open("rb") as handle:
            head = handle.read(8192)
    except OSError as exc:
        raise FileProcessingError(f"could not read file: {exc}")
    sniffed = sniff_mime(head, path)
    if sniffed:
        return sniffed
    extension_mime = EXTENSION_MIME.get(path.suffix.lower())
    if extension_mime:
        if extension_mime.startswith("text/") and not looks_like_text(head):
            return None
        return extension_mime
    if looks_like_text(head):
        return classify_text_mime(head)
    return None

def decode_text(data):
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")

def html_to_text(markup):
    markup = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", markup)
    markup = re.sub(r"(?s)<!--.*?-->", " ", markup)
    markup = re.sub(r"(?s)<[^>]+>", " ", markup)
    return html.unescape(markup)

def rtf_to_text(markup):
    markup = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|info|pict)[^{}]*\}", " ", markup)
    markup = re.sub(r"\\(?:par|line)\b", "\n", markup)
    markup = re.sub(r"\\'[0-9a-fA-F]{2}", " ", markup)
    markup = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", markup)
    return markup.replace("{", " ").replace("}", " ")

def split_oversized_words(words):
    normalized = []
    for word in words:
        if len(word) <= MAX_WORD_CHARS:
            normalized.append(word)
        else:
            normalized.extend(word[i:i + MAX_WORD_CHARS] for i in range(0, len(word), MAX_WORD_CHARS))
    return normalized

def char_capped_spans(words, window_start, window_end):
    spans = []
    piece_start = window_start
    length = 0
    for i in range(window_start, window_end):
        addition = len(words[i]) + 1
        if length and length + addition > TEXT_CHUNK_CHAR_CAP:
            spans.append((piece_start, i))
            piece_start = i
            length = 0
        length += addition
    spans.append((piece_start, window_end))
    return spans

def chunk_text_file(path, mime):
    text = decode_text(path.read_bytes())
    if mime == "text/html":
        text = html_to_text(text)
    elif mime == "text/rtf":
        text = rtf_to_text(text)
    words = split_oversized_words(text.split())
    if not words:
        raise FileProcessingError("no extractable text")
    chunks = []
    step = TEXT_CHUNK_WORDS - TEXT_OVERLAP_WORDS
    for window_start, window_end in windowed_spans(len(words), TEXT_CHUNK_WORDS, step):
        for start, end in char_capped_spans(words, window_start, window_end):
            body = " ".join(words[start:end])
            chunks.append(PreparedChunk(
                category="text",
                meta={"word_start": start, "word_end": end},
                text=body,
                est_tokens=len(body) // CHARS_PER_TOKEN + 8,
                parts=[types.Part.from_text(text=QUERY_TEMPLATE.format(content=body))],
            ))
    return chunks

def docx_to_pdf_bytes(path):
    with tempfile.TemporaryDirectory() as workspace:
        source = Path(workspace) / "input.docx"
        target = Path(workspace) / "input.pdf"
        shutil.copyfile(path, source)
        script = "import sys; from docx2pdf import convert; convert(sys.argv[1], sys.argv[2])"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(source), str(target)],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            raise FileProcessingError("docx conversion timed out")
        if proc.returncode != 0 or not target.exists():
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            reason = detail[-1] if detail else "unknown error"
            raise FileProcessingError(f"docx conversion failed (requires Microsoft Word): {reason}")
        return target.read_bytes()

def pdf_chunks_from_bytes(data, extra_meta):
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise FileProcessingError("PDF is password protected")
        page_count = len(reader.pages)
    except FileProcessingError:
        raise
    except Exception as exc:
        raise FileProcessingError(f"unreadable PDF: {exc}")
    if page_count == 0:
        raise FileProcessingError("PDF has no pages")
    chunks = []
    step = PDF_CHUNK_PAGES - PDF_OVERLAP_PAGES
    for start, end in windowed_spans(page_count, PDF_CHUNK_PAGES, step):
        writer = PdfWriter()
        for page in reader.pages[start:end]:
            writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        chunks.append(PreparedChunk(
            category="document",
            meta={"page_start": start + 1, "page_end": end, **extra_meta},
            est_tokens=PDF_TOKENS_PER_PAGE * (end - start),
            parts=[types.Part.from_bytes(data=buffer.getvalue(), mime_type="application/pdf")],
        ))
    return chunks

def chunk_document_file(path, mime):
    if mime == DOCX_MIME:
        return pdf_chunks_from_bytes(docx_to_pdf_bytes(path), {"converted_from": "docx"})
    return pdf_chunks_from_bytes(path.read_bytes(), {})

def chunk_image_file(path, mime):
    data = path.read_bytes()
    if mime in ("image/png", "image/jpeg") and len(data) <= MAX_IMAGE_BYTES:
        return [PreparedChunk(
            category="image",
            meta={},
            est_tokens=IMAGE_TOKENS,
            parts=[types.Part.from_bytes(data=data, mime_type=mime)],
        )]
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        hint = ""
        if mime in ("image/heic", "image/heif") and not HEIF_SUPPORTED:
            hint = " (install pillow-heif for HEIC/HEIF support)"
        raise FileProcessingError(f"cannot decode image: {exc}{hint}")
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return [PreparedChunk(
        category="image",
        meta={"transcoded_to": "image/png"},
        est_tokens=IMAGE_TOKENS,
        parts=[types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")],
    )]

_ffmpeg_executable = None

def ffmpeg_path():
    global _ffmpeg_executable
    if _ffmpeg_executable:
        return _ffmpeg_executable
    located = shutil.which("ffmpeg")
    if not located:
        try:
            import imageio_ffmpeg
            located = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise FileProcessingError("ffmpeg not found: install ffmpeg or run `pip install imageio-ffmpeg`")
    _ffmpeg_executable = located
    return located

def parse_timestamp(match):
    hours, minutes, seconds = match
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def media_duration_seconds(path):
    try:
        probe = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        match = DURATION_PATTERN.search(probe.stderr)
        if match:
            return parse_timestamp(match.groups())
        decode = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
        matches = PROGRESS_TIME_PATTERN.findall(decode.stderr)
        if matches:
            return parse_timestamp(matches[-1])
    except subprocess.TimeoutExpired:
        raise FileProcessingError("media probing timed out")
    raise FileProcessingError("could not determine media duration (file may be corrupt)")

def extract_media_chunk(path, start, duration, encode_args, suffix):
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.close()
    out_path = Path(handle.name)
    command = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path),
        *encode_args, "-y", str(out_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or out_path.stat().st_size == 0:
            detail = proc.stderr.strip().splitlines()
            reason = detail[-1] if detail else "unknown error"
            raise FileProcessingError(f"ffmpeg chunk extraction failed: {reason}")
        payload = out_path.read_bytes()
    except subprocess.TimeoutExpired:
        raise FileProcessingError("ffmpeg chunk extraction timed out")
    finally:
        out_path.unlink(missing_ok=True)
    if len(payload) > MAX_INLINE_BYTES:
        raise FileProcessingError(f"encoded chunk exceeds {MAX_INLINE_BYTES // (1024 * 1024)}MB inline limit")
    return payload

def media_chunks(path, size, overlap, encode_args, suffix, out_mime, category):
    duration = media_duration_seconds(path)
    if duration <= 0:
        raise FileProcessingError("media file has no measurable duration")
    chunks = []
    for start, end in windowed_spans(duration, size, size - overlap, tail_epsilon=0.5):
        payload = extract_media_chunk(path, start, end - start, encode_args, suffix)
        if category == "audio":
            est = int(AUDIO_TOKENS_PER_SECOND * (end - start))
        else:
            est = VIDEO_TOKENS_PER_CHUNK
        chunks.append(PreparedChunk(
            category=category,
            meta={"start_seconds": round(start, 2), "end_seconds": round(end, 2)},
            est_tokens=est,
            parts=[types.Part.from_bytes(data=payload, mime_type=out_mime)],
        ))
    return chunks

def chunk_audio_file(path, mime):
    return media_chunks(path, AUDIO_CHUNK_SECONDS, AUDIO_OVERLAP_SECONDS,
                        AUDIO_FFMPEG_ARGS, ".wav", "audio/wav", "audio")

def chunk_video_file(path, mime):
    return media_chunks(path, VIDEO_CHUNK_SECONDS, VIDEO_OVERLAP_SECONDS,
                        VIDEO_FFMPEG_ARGS, ".mp4", "video/mp4", "video")

HANDLERS = {
    "text": chunk_text_file,
    "document": chunk_document_file,
    "image": chunk_image_file,
    "audio": chunk_audio_file,
    "video": chunk_video_file,
}

class RateLimiter:
    def __init__(self, max_rpm, max_tpm, window=RATE_WINDOW_SECONDS):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.window = window
        self.requests = deque()
        self.tokens = deque()
        self.token_sum = 0

    def _prune(self, now):
        cutoff = now - self.window
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        while self.tokens and self.tokens[0][0] <= cutoff:
            self.token_sum -= self.tokens.popleft()[1]

    def acquire(self, tokens):
        tokens = min(tokens, self.max_tpm) if self.max_tpm else tokens
        while True:
            now = time.monotonic()
            self._prune(now)
            wait = 0.0
            if self.max_rpm and len(self.requests) >= self.max_rpm:
                wait = max(wait, self.window - (now - self.requests[0]))
            if self.max_tpm and self.tokens and self.token_sum + tokens > self.max_tpm:
                wait = max(wait, self.window - (now - self.tokens[0][0]))
            if wait <= 0:
                break
            logging.info(f"rate-limit pacing: waiting {wait:.1f}s to stay under "
                         f"{self.max_rpm} req/min and {self.max_tpm} tok/min")
            time.sleep(min(wait, self.window))
        stamp = time.monotonic()
        self.requests.append(stamp)
        self.tokens.append((stamp, tokens))
        self.token_sum += tokens

def embed_group(client, group, limiter):
    contents = [types.Content(parts=chunk.parts) for chunk in group]
    config = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
    limiter.acquire(sum(chunk.est_tokens for chunk in group))
    delay = 2.0
    for attempt in range(1, EMBED_ATTEMPTS + 1):
        try:
            response = client.models.embed_content(model=EMBED_MODEL, contents=contents, config=config)
            embeddings = response.embeddings or []
            if len(embeddings) != len(group):
                raise FileProcessingError(f"expected {len(group)} embeddings, got {len(embeddings)}")
            vectors = []
            for embedding in embeddings:
                if not embedding.values:
                    raise FileProcessingError(
                        "API returned an empty embedding (content may be unsupported or unreadable)")
                vectors.append(list(embedding.values))
            return vectors
        except FileProcessingError:
            raise
        except genai_errors.APIError as exc:
            if exc.code in RETRYABLE_CODES and attempt < EMBED_ATTEMPTS:
                logging.warning(f"embedding request got {exc.code}, retrying in {delay:.0f}s "
                                f"(attempt {attempt}/{EMBED_ATTEMPTS})")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                raise FileProcessingError(f"embedding request failed ({exc.code}): {exc.message}")
        except Exception as exc:
            if attempt < EMBED_ATTEMPTS:
                logging.warning(f"embedding request error, retrying in {delay:.0f}s: {exc}")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                raise FileProcessingError(f"embedding request failed: {exc}")
    raise FileProcessingError("embedding request failed: retries exhausted")

def build_request_groups(chunks):
    groups = []
    pending_images = []
    for chunk in chunks:
        if chunk.category == "image":
            pending_images.append(chunk)
            if len(pending_images) == IMAGES_PER_REQUEST:
                groups.append(pending_images)
                pending_images = []
        else:
            groups.append([chunk])
    if pending_images:
        groups.append(pending_images)
    return groups

def run_embedding(client, chunks):
    groups = build_request_groups(chunks)
    limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
    logging.info(f"embedding {len(chunks)} chunk(s) in {len(groups)} request(s) with {EMBED_MODEL} "
                 f"(paced to {MAX_REQUESTS_PER_MINUTE} req/min, {MAX_TOKENS_PER_MINUTE} tok/min)")
    for position, group in enumerate(groups, 1):
        try:
            vectors = embed_group(client, group, limiter)
            for chunk, vector in zip(group, vectors):
                chunk.embedding = vector
        except FileProcessingError as exc:
            if len(group) > 1:
                for chunk in group:
                    try:
                        chunk.embedding = embed_group(client, [chunk], limiter)[0]
                    except FileProcessingError as single_exc:
                        chunk.error = str(single_exc)
                        logging.warning(f"{chunk.source} chunk {chunk.index}: {single_exc}")
            else:
                group[0].error = str(exc)
                logging.warning(f"{group[0].source} chunk {group[0].index}: {exc}")
        if position % 10 == 0 or position == len(groups):
            logging.info(f"completed {position}/{len(groups)} embedding request(s)")

def discover_files(input_dir):
    files = []
    for candidate in sorted(input_dir.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(input_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(candidate)
    return files

def prepare_file(path, relative):
    if path.stat().st_size == 0:
        raise FileProcessingError("file is empty")
    mime = detect_mime(path)
    if mime is None:
        raise FileProcessingError("unsupported file type")
    category = CATEGORY_BY_MIME[mime]
    chunks = HANDLERS[category](path, mime)
    digest = file_sha256(path)
    for position, chunk in enumerate(chunks):
        chunk.source = relative
        chunk.mime = mime
        chunk.sha = digest
        chunk.uid = f"{digest[:12]}-{position:04d}"
        chunk.index = position
        chunk.total = len(chunks)
    return mime, category, chunks

def chunk_record(chunk):
    return {
        "id": chunk.uid,
        "source_file": chunk.source,
        "file_sha256": chunk.sha,
        "mime_type": chunk.mime,
        "category": chunk.category,
        "chunk_index": chunk.index,
        "chunk_count": chunk.total,
        "chunk_meta": chunk.meta,
        "text": chunk.text,
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "embedding": chunk.embedding,
    }

def finalize_reports(reports, chunks):
    by_source = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)
    for report in reports:
        if report["status"] != "prepared":
            continue
        file_chunks = by_source.get(report["file"], [])
        embedded = [c for c in file_chunks if c.embedding is not None]
        failed = [c for c in file_chunks if c.error]
        if embedded and not failed:
            report["status"] = "embedded"
        elif embedded:
            report["status"] = "partially_embedded"
            report["error"] = failed[0].error
        else:
            report["status"] = "embedding_failed"
            report["error"] = failed[0].error if failed else "unknown embedding failure"

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Detect, chunk, and embed input files with the Gemini embedding API")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    for noisy_logger in ("pypdf", "httpx", "google_genai"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    args = build_arg_parser().parse_args(argv)
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logging.error(f"input folder not found: {input_dir}")
        return 2
    files = discover_files(input_dir)
    logging.info(f"found {len(files)} file(s) in {input_dir}")

    reports = []
    all_chunks = []
    for path in files:
        relative = str(path.relative_to(input_dir))
        report = {"file": relative, "mime_type": None, "category": None,
                  "chunks": 0, "status": "skipped", "error": None}
        reports.append(report)
        try:
            mime, category, chunks = prepare_file(path, relative)
            report.update(mime_type=mime, category=category,
                          chunks=len(chunks), status="prepared")
            all_chunks.extend(chunks)
            logging.info(f"{relative}: {category} ({mime}) -> {len(chunks)} chunk(s)")
        except FileProcessingError as exc:
            report["error"] = str(exc)
            logging.warning(f"skipping {relative}: {exc}")
        except Exception as exc:
            report["error"] = f"unexpected error: {exc}"
            logging.error(f"skipping {relative}: unexpected error: {exc}")

    if all_chunks:
        client = genai.Client(api_key=resolve_api_key(),
                              http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        run_embedding(client, all_chunks)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "input-embeddings.jsonl"
    embedded_count = 0
    with records_path.open("w", encoding="utf-8") as sink:
        for chunk in all_chunks:
            if chunk.embedding is None:
                continue
            sink.write(json.dumps(chunk_record(chunk), ensure_ascii=False) + "\n")
            embedded_count += 1

    finalize_reports(reports, all_chunks)
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "input_dir": str(input_dir),
        "records_file": records_path.name,
        "files_found": len(files),
        "chunks_embedded": embedded_count,
        "files": reports,
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    embedded_files = sum(1 for r in reports if r["status"] == "embedded")
    partial_files = sum(1 for r in reports if r["status"] == "partially_embedded")
    problem_files = sum(1 for r in reports if r["status"] in ("skipped", "embedding_failed"))
    logging.info(f"done: {embedded_files} file(s) fully embedded, {partial_files} partial, "
                 f"{problem_files} skipped/failed")
    logging.info(f"wrote {embedded_count} embedding record(s) to {records_path}")
    return 0 if embedded_count else 1

if __name__ == "__main__":
    sys.exit(main())
