#!/usr/bin/env python3
"""Download the CosyVoice3 checkpoints that ``convert_weights.py`` consumes.

Downloads only the three files the converter actually reads — ``llm.pt``,
``flow.pt``, ``hift.pt`` — from the pinned Hugging Face revision, and falls back
to the ModelScope mirror (same model, same version) when HF is unreachable.
Each file is verified against a hardcoded SHA-256 that was cross-checked against
both sources before being committed, so a byte that doesn't match the official
release is never accepted.

Standard library only (``urllib`` / ``hashlib``), so it runs under a bare
``python3`` with no third-party packages and no ``PYTHONPATH`` pointing at
CosyVoice:

    python3 tools/fetch_model.py --out-dir models

Behavior:

  - pinned revision + per-file SHA-256 (dual-source cross-verified);
  - resumes an interrupted download (HTTP ``Range``) via a ``<file>.part``
    temporary file, which is renamed into place only after the hash matches;
  - skips a file that is already present with the correct hash (so a re-run is
    a no-op);
  - on hash mismatch the file is deleted and the process exits non-zero.

License note (facts only, no conclusion): the weights come from
``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``, tagged ``apache-2.0`` on Hugging Face;
the model card also carries an "for academic purposes only" statement that the
maintainers have not reconciled (HF discussion #19). See
``data/tokenizer/README.md`` for the tokenizer-side record.
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request

# Pinned source. Both identifiers name the same 0.5B model; the revision fixes
# the exact snapshot the hashes below were computed against.
HF_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
HF_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
MS_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MS_REVISION = "master"

# The three checkpoints convert_weights.py reads. ``size`` is informational
# (progress/debug); the authoritative check is ``sha256``.
FILES = {
    "llm.pt": {
        "size": 2024669519,
        "sha256": "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970",
    },
    "flow.pt": {
        "size": 1329116148,
        "sha256": "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b",
    },
    "hift.pt": {
        "size": 83202622,
        "sha256": "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
    },
}

_USER_AGENT = "velum-fetch/0.1"
_CHUNK = 1 << 20  # 1 MiB


def _hf_urls(name):
    # The official host first, then the community mirror (huggingface.co is
    # sometimes unreachable from certain networks). Same repo + revision.
    return [
        f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{name}",
        f"https://hf-mirror.com/{HF_REPO}/resolve/{HF_REVISION}/{name}",
    ]


def _ms_url(name):
    return (f"https://modelscope.cn/api/v1/models/{MS_REPO}/repo"
            f"?FilePath={name}&Revision={MS_REVISION}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest):
    """Stream ``url`` to ``dest``, resuming an existing ``dest + ".part"``."""
    part = dest + ".part"
    offset = os.path.getsize(part) if os.path.exists(part) else 0

    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        code = resp.getcode()
        # Resume only if the server honoured the Range request (206); if it
        # ignored it and answered 200, restart from scratch.
        resume_from = offset if (offset and code == 206) else 0
        mode = "ab" if resume_from else "wb"
        with open(part, mode) as out:
            shutil.copyfileobj(resp, out, length=_CHUNK)


def _fetch_one(name, meta, out_dir):
    dest = os.path.join(out_dir, name)

    if os.path.exists(dest) and _sha256(dest) == meta["sha256"]:
        print(f"  {name}: already present, hash ok")
        return

    urls = _hf_urls(name) + [_ms_url(name)]
    last_err = None
    for url in urls:
        try:
            print(f"  {name}: downloading {url}")
            _download(url, dest)
            part = dest + ".part"
            got = _sha256(part)
            if got != meta["sha256"]:
                # Corrupted or truncated -> do not keep it, do not continue.
                os.remove(part)
                print(f"  {name}: SHA-256 mismatch\n"
                      f"    expected {meta['sha256']}\n"
                      f"    got      {got}")
                sys.exit(1)
            os.replace(part, dest)
            print(f"  {name}: ok ({meta['size']} bytes)")
            return
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"network: {e.reason}"
        except OSError as e:
            last_err = str(e)
        # Fall through to the next source (mirror / ModelScope).
        print(f"  {name}: {url} failed ({last_err}); trying next source")

    # All sources failed. Remove any partial file so a later run starts clean.
    part = dest + ".part"
    if os.path.exists(part):
        os.remove(part)
    sys.exit(f"error: could not download {name} from any source ({last_err})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="models",
                    help="directory to download into (default: models/)")
    args = ap.parse_args()

    # One line, before any download: weight source + license status (facts only).
    print(f"model weights: {HF_REPO} @ {HF_REVISION[:7]} "
          f"(HF license tag: apache-2.0; model card also states "
          f"'for academic purposes only', not yet clarified)")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, meta in FILES.items():
        _fetch_one(name, meta, args.out_dir)
    print(f"done: {', '.join(FILES)} -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
