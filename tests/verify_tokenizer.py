#!/usr/bin/env python3
"""Bit-exact comparison of the C++ Qwen2Tokenizer against CosyVoice3Tokenizer.

Feeds ground-truth cases through the compiled ``velum_tokenizer_dump`` utility
and asserts every case is *identical* — this is a tokenizer, so there is no
floating-point tolerance: any id difference in any position is a FAIL.

Two ground-truth sources are checked:

  1. ``tests/tokenizer_cases.bin`` (produced by tests/tokenizer_reference.py):
     text + ground-truth ids, a broad corpus covering edge cases. Optional —
     it is regenerated on demand and not committed.
  2. ``tools/tokenizer_golden.json`` (committed): the human-readable golden
     reference mapping sample texts to their expected token ids.

Usage:
    python3 tests/verify_tokenizer.py            # uses build/velum_tokenizer_dump
    VELUM_TOKENIZER_DUMP=./build/velum_tokenizer_dump python3 tests/verify_tokenizer.py
"""

import json
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_cases(path):
    cases = []
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            tl = struct.unpack("<I", f.read(4))[0]
            text = f.read(tl).decode("utf-8")
            il = struct.unpack("<I", f.read(4))[0]
            ids = list(struct.unpack("<%di" % il, f.read(4 * il)))
            cases.append((text, ids))
    return cases


def write_cases(path, cases):
    """Write ``cases`` (list of (text, ids)) in the cases.bin format that
    ``tests/tokenizer_dump.cpp`` consumes."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(cases)))
        for text, ids in cases:
            data = text.encode("utf-8")
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack("<%di" % len(ids), *ids))


def load_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return [(c["text"], c["ids"]) for c in doc["cases"]]


def read_cpp(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        ids_list = []
        for _ in range(n):
            il = struct.unpack("<I", f.read(4))[0]
            ids_list.append(list(struct.unpack("<%di" % il, f.read(4 * il))))
    return ids_list


def verify(dump_bin, data_dir, cases, label, tmp):
    """Run the C++ tokenizer over ``cases`` and return (n_fail, n_total)."""
    cases_bin = os.path.join(tmp, label + ".cases.bin")
    out_bin = os.path.join(tmp, label + ".out.bin")
    write_cases(cases_bin, cases)
    subprocess.run([dump_bin, data_dir, cases_bin, out_bin], check=True)
    cpp = read_cpp(out_bin)

    n_fail = 0
    for i, ((text, ref_ids), cpp_ids) in enumerate(zip(cases, cpp)):
        if ref_ids == cpp_ids:
            print(f"[{label}] case {i:2d}: PASS  ({len(ref_ids)} tokens)  {text!r}")
            continue
        n_fail += 1
        print(f"[{label}] case {i:2d}: FAIL  text={text!r}")
        print(f"   ref {len(ref_ids):3d}: {ref_ids}")
        print(f"   cpp {len(cpp_ids):3d}: {cpp_ids}")
        m = min(len(ref_ids), len(cpp_ids))
        for j in range(m):
            if ref_ids[j] != cpp_ids[j]:
                print(f"   first diff @ {j}: ref={ref_ids[j]}  cpp={cpp_ids[j]}")
                break

    return n_fail, len(cases)


def main():
    dump_bin = os.environ.get("VELUM_TOKENIZER_DUMP",
                              os.path.join(ROOT, "build", "velum_tokenizer_dump"))
    data_dir = os.path.join(ROOT, "build", "tokenizer")
    cases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tokenizer_cases.bin")
    golden_path = os.path.join(ROOT, "tools", "tokenizer_golden.json")

    if not os.path.exists(dump_bin):
        sys.exit(f"velum_tokenizer_dump not found at {dump_bin}; build it first "
                 f"(cmake --build build)")
    if not os.path.exists(os.path.join(data_dir, "vocab.tsv")):
        sys.exit(f"tokenizer data not found in {data_dir}; run tools/export_tokenizer.py")
    if not os.path.exists(golden_path):
        sys.exit(f"golden reference not found at {golden_path}")

    sources = [(golden_path, load_golden, "golden.json")]
    if os.path.exists(cases_path):
        sources.insert(0, (cases_path, read_cases, "cases.bin"))
    else:
        print(f"note: {cases_path} not found; skipping (regenerate with "
              f"tests/tokenizer_reference.py)")

    total_fail = 0
    total_cases = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path, loader, label in sources:
            cases = loader(path)
            n_fail, n = verify(dump_bin, data_dir, cases, label, tmp)
            total_fail += n_fail
            total_cases += n

    print("-" * 70)
    print(f"{total_cases - total_fail}/{total_cases} cases bit-exact")
    if total_fail:
        print("FAIL: tokenizer is not bit-exact vs CosyVoice3Tokenizer.")
        sys.exit(1)
    print("PASS: all cases bit-exact vs CosyVoice3Tokenizer.")


if __name__ == "__main__":
    main()
