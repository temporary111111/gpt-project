"""Acquire the first real corpus:

- Simple English Wikipedia dump (CC BY-SA)                 -> data/raw/en/
- Tagalog Wikipedia dump (CC BY-SA)                       -> data/raw/tl/
- Tagalog Wikisource dump (CC BY-SA, literary works)      -> data/raw/tl/
- Top public-domain books from Project Gutenberg          -> data/raw/en/

All downloads are streamed to disk with hard disk-space checks:
- stop if free space < 25 GB
- stop if cumulative downloads exceed MAX_TOTAL_BYTES (8 GB limit)
- each download capped at MAX_SOURCE_BYTES

Usage:
    .venv\\Scripts\\python.exe scripts\\acquire_corpus.py [--skip-wiki] [--skip-gutenberg] [--max-books 120]
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.gutenberg import decode_text, extract_gutenberg_body  # noqa: E402
from src.data.manifest import SourceRecord, append_source_record, load_source_records  # noqa: E402
from src.data.mediawiki import extract_mw_text  # noqa: E402

USER_AGENT = "chatgpt-like-local/0.1 (educational non-commercial research; contact: local)"
MIN_FREE_GB = 25.0
MAX_TOTAL_BYTES = 8 * 1024 ** 3      # 8 GB hard cap
MAX_SOURCE_BYTES = 2 * 1024 ** 3     # per-source cap
MANIFEST = os.path.join("data", "manifests", "sources.jsonl")

WIKI_DUMPS = [
    {
        "name": "simplewiki",
        "lang": "en",
        "url": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
        "local": os.path.join("data", "sources", "en", "simplewiki-pages-articles.xml.bz2"),
        "license": "CC BY-SA 4.0",
        "type": "wikipedia dump",
    },
    {
        "name": "tlwiki",
        "lang": "tl",
        "url": "https://dumps.wikimedia.org/tlwiki/latest/tlwiki-latest-pages-articles.xml.bz2",
        "local": os.path.join("data", "sources", "tl", "tlwiki-pages-articles.xml.bz2"),
        "license": "CC BY-SA 4.0",
        "type": "wikipedia dump",
    },
    {
        "name": "tlwikisource",
        "lang": "tl",
        "url": "https://dumps.wikimedia.org/tlwikisource/latest/tlwikisource-latest-pages-articles.xml.bz2",
        "local": os.path.join("data", "sources", "tl", "tlwikisource-pages-articles.xml.bz2"),
        "license": "CC BY-SA 4.0",
        "type": "wikisource dump",
    },
]

GUTENBERG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
GUTENBERG_ROBOTS = "https://www.gutenberg.org/robots.txt"
TXT_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
]


def free_gb() -> float:
    return shutil.disk_usage(os.getcwd()).free / 1024 ** 3


def check_limits(downloaded_total: int) -> None:
    if free_gb() < MIN_FREE_GB:
        print(f"ABORT: free disk {free_gb():.1f} GB < {MIN_FREE_GB} GB limit")
        sys.exit(1)
    if downloaded_total > MAX_TOTAL_BYTES:
        print(f"ABORT: cumulative downloads exceed 8 GB")
        sys.exit(1)


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def manifest_has(records: list, name: str) -> bool:
    return any(r.source_name == name for r in records)


def download_stream(url: str, dest: str, cap_bytes: int) -> int:
    """Streams a download to disk; returns bytes written."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    total = 0
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > cap_bytes:
                raise RuntimeError(f"download exceeds cap of {cap_bytes} bytes: {url}")
            f.write(chunk)
    return total


def fetch_robots() -> Optional[str]:
    try:
        return http_get(GUTENBERG_ROBOTS).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"WARNING: could not fetch Gutenberg robots.txt: {e}")
        return None


def list_gutenberg_books(max_books: int) -> list:
    """English text books from the official catalog (catalog order).

    Robots.txt only disallows /ebooks/search; the catalog feed and book files
    are the documented bulk-access path. Books are filtered by size at
    download time (real books, not short speeches/poems).
    """
    catalog = os.path.join("data", "sources", "en", "pg_catalog.csv.gz")
    if not os.path.exists(catalog):
        os.makedirs(os.path.dirname(catalog), exist_ok=True)
        print("[gutenberg] downloading official catalog ...")
        download_stream(GUTENBERG_CATALOG, catalog, MAX_SOURCE_BYTES)
    import csv
    books = []
    with gzip.open(catalog, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = (row.get("Language") or "").split(",")
            if not any(l.strip() in ("en", "en_US", "en-GB") for l in lang):
                continue
            if (row.get("Type") or "").strip() != "Text":
                continue
            bid = (row.get("Text#") or "").strip()
            if not bid.isdigit():
                continue
            title = (row.get("Title") or f"pg{bid}").replace("\n", " ")[:120]
            books.append((int(bid), f"https://www.gutenberg.org/ebooks/{bid}", title))
    print(f"[gutenberg] catalog: {len(books)} public-domain English text books "
          f"(taking first {max(400, len(books)) if len(books) > 400 else len(books)} as candidates)")
    return books[:400]


def extract_wiki_dump(dump_path: str, out_path: str, source: str, lang: str) -> dict:
    """Streams a pages-articles dump to raw JSONL documents."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    docs = 0
    skipped = 0
    with bz2.BZ2File(dump_path, "r") as raw:
        context = ET.iterparse(raw, events=("end",))
        for event, elem in context:
            tag = elem.tag.split("}")[-1]
            if tag != "page":
                # NOTE: do NOT clear() non-page elements here - ElementTree
                # reuses cleared element storage, corrupting later page text.
                continue
            title = ""
            ns = "0"
            text = ""
            is_redirect = False
            for child in elem:
                ctag = child.tag.split("}")[-1]
                if ctag == "title":
                    title = child.text or ""
                elif ctag == "ns":
                    ns = child.text or "0"
                elif ctag == "redirect":
                    is_redirect = True
                elif ctag == "revision":
                    for gc in child:
                        gtag = gc.tag.split("}")[-1]
                        if gtag == "text":
                            text = gc.text or ""
            elem.clear()
            if ns != "0" or is_redirect or not text.strip():
                skipped += 1
                continue
            if re.match(r"^(File|Image|Category|Template|Media|Wikipedia|Help|Portal|Special|Module):", title):
                skipped += 1
                continue
            body = extract_mw_text(text)
            if len(body) < 50:
                skipped += 1
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "source": source, "doc_id": title, "title": title, "text": body,
                    "lang": lang,
                }, ensure_ascii=False) + "\n")
            docs += 1
    return {"docs": docs, "skipped": skipped}


def acquire_wiki(downloaded_total: int) -> int:
    for spec in WIKI_DUMPS:
        check_limits(downloaded_total)
        local = spec["local"]
        if os.path.exists(local):
            print(f"[wiki] already present: {local}")
            size = os.path.getsize(local)
        else:
            print(f"[wiki] downloading {spec['name']} ...")
            size = download_stream(spec["url"], local, MAX_SOURCE_BYTES)
            print(f"[wiki] downloaded {size / 1024 ** 2:.1f} MiB")
        downloaded_total += size
        out = os.path.join("data", "raw", spec["lang"], spec["name"] + ".jsonl")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[wiki] raw docs already extracted: {out}")
            continue
        print(f"[wiki] extracting {spec['name']} -> {out} ...")
        stats = extract_wiki_dump(local, out, spec["name"], spec["lang"])
        print(f"[wiki] {spec['name']}: {stats['docs']} docs, {stats['skipped']} skipped")
        if not manifest_has(load_source_records(MANIFEST), spec["name"]):
            append_source_record(MANIFEST, SourceRecord(
            source_name=spec["name"], language=spec["lang"], source_type=spec["type"],
            source_url=spec["url"], license=spec["license"],
            retrieval_date=time.strftime("%Y-%m-%d"), local_path=local,
            raw_size_bytes=size, status="downloaded", notes=f"{stats['docs']} docs extracted",
        ))
    return downloaded_total


def acquire_gutenberg(max_books: int, downloaded_total: int) -> int:
    robots = fetch_robots()
    if robots is not None:
        disallowed = [l.split(":", 1)[1].strip() for l in robots.splitlines()
                      if l.lower().startswith("disallow")]
        print(f"[gutenberg] robots.txt fetched; Disallow rules: {disallowed or ['none']}")
        if any("/ebooks/search" in r for r in disallowed):
            pass  # we only use the catalog feed + book files, which are allowed

    books = list_gutenberg_books(max_books)
    append_source_record(MANIFEST, SourceRecord(
        source_name="gutenberg_catalog", language="en", source_type="catalog",
        source_url=GUTENBERG_CATALOG, license="Project Gutenberg catalog",
        retrieval_date=time.strftime("%Y-%m-%d"),
        local_path=os.path.join("data", "sources", "en", "pg_catalog.csv.gz"),
        raw_size_bytes=os.path.getsize(os.path.join("data", "sources", "en", "pg_catalog.csv.gz")),
        status="downloaded", notes="official catalog used to select top public-domain books",
    ))
    print(f"[gutenberg] downloading up to {max_books} books (keeping full-length books)")
    out = os.path.join("data", "raw", "en", "gutenberg_books.jsonl")
    downloaded = 0
    valid = 0
    for i, (bid, url, title) in enumerate(books):
        if valid >= max_books:
            break
        check_limits(downloaded_total + downloaded)
        local = os.path.join("data", "sources", "en", "gutenberg", f"pg{bid}.txt")
        text = None
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            os.makedirs(os.path.dirname(local), exist_ok=True)
            for template in TXT_URLS:
                try:
                    body = http_get(template.format(id=bid), timeout=90)
                    text = decode_text(body)
                    with open(local, "w", encoding="utf-8") as f:
                        f.write(text)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError):
                    continue
            time.sleep(0.4)  # politeness delay
        if text is None:
            print(f"[gutenberg] FAILED to download book {bid}")
            continue
        downloaded += os.path.getsize(local)
        body = extract_gutenberg_body(text)
        doc_title = title[:120]
        if len(body) < 10_000:
            print(f"[gutenberg] book {bid}: too short ({len(body)} chars), skipped")
            continue
        valid += 1
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "source": "gutenberg", "doc_id": f"gutenberg_{bid}", "title": doc_title,
                "text": body, "lang": "en",
            }, ensure_ascii=False) + "\n")
        append_source_record(MANIFEST, SourceRecord(
            source_name=f"gutenberg_{bid}", language="en",
            source_type="public-domain book", source_url=url,
            license="Public Domain (US)", retrieval_date=time.strftime("%Y-%m-%d"),
            local_path=local, raw_size_bytes=os.path.getsize(local),
            status="downloaded", notes=f"{doc_title}",
        )) if not manifest_has(load_source_records(MANIFEST), f"gutenberg_{bid}") else None
        if (i + 1) % 20 == 0:
            print(f"[gutenberg] {i + 1}/{len(books)} books processed")
    return downloaded_total + downloaded


def main():
    p = argparse.ArgumentParser(description="Acquire the first real corpus")
    p.add_argument("--skip-wiki", action="store_true")
    p.add_argument("--skip-gutenberg", action="store_true")
    p.add_argument("--max-books", type=int, default=120)
    args = p.parse_args()

    print(f"free disk: {free_gb():.1f} GB (min required {MIN_FREE_GB} GB)")
    total = 0
    if not args.skip_wiki:
        total = acquire_wiki(total)
    if not args.skip_gutenberg:
        total = acquire_gutenberg(args.max_books, total)
    print(f"\nacquisition complete. cumulative downloads: {total / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()