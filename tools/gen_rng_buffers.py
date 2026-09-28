#!/usr/bin/env python3
"""Generate the fixed RNG buffers (``flow_noise.bin`` + ``hift_source.bin``).

The reference samples two model-internal buffers from PyTorch's RNG once at
construction time; the C++ side loads the frozen values verbatim instead of
reimplementing PyTorch's RNG (``FlowDecoder::load_noise`` / ``HiftVocoder::load_source``):

  flow_noise.bin
    ``CausalConditionalCFM.__init__`` runs ``set_all_random_seed(0)`` then draws
    ``rand_noise = torch.randn([1, 80, 50 * 300])``. ``set_all_random_seed`` only
    reseeds ``random``/``numpy``/``torch``/``torch.cuda``, and ``rand_noise`` is
    the *very next* draw, so a bare ``torch.manual_seed(0); torch.randn(...)``
    reproduces it **bit-exact** — no CosyVoice import is needed.

  hift_source.bin
    ``SineGen2(causal=True)`` draws ``rand_ini = torch.rand(1, 9)`` (col 0 zeroed)
    and ``sine_waves = torch.rand(1, 300 * 24000, 9)`` from the *unseeded* global
    RNG, so the values are arbitrary: any fixed seed is equally valid. Correctness
    is *consuming the same bytes* on both sides (``tests/verify_e2e.py`` overrides
    the reference's internal buffers with this file), never reproducing CosyVoice's
    RNG. We still seed 0 so the file is deterministic and reproducible.

Because neither buffer depends on the weights or on the CosyVoice package, this
script needs only PyTorch (+ numpy) from ``tools/requirements-convert.txt`` and
is run at build time by the ``models`` target.

File formats (little-endian, identical to the reference exporters
``tests/export_flow_noise.py`` / ``tests/export_hift_source.py``):

    flow_noise.bin:
        u32 magic = 0x45534E46 ("FNSE"); u32 version = 1; u32 mel_dim = 80;
        u64 max_frames = 50 * 300; f32[mel_dim * max_frames]   (row-major [c][t])

    hift_source.bin:
        u32 magic = 0x43525348 ("HSRC"); u32 version = 1; u32 harmonic_dim = 9;
        u64 sine_max_samples = 300 * 24000;
        f32[harmonic_dim]                     rand_ini
        f32[sine_max_samples * harmonic_dim]  sine_waves      (row-major [t][h])

Usage:
    python3 tools/gen_rng_buffers.py --out-dir build/
"""

import argparse
import hashlib
import os
import struct

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise SystemExit(
        "gen_rng_buffers.py: PyTorch (and numpy) are required but not installed.\n"
        "  python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-convert.txt\n"
        f"  ({exc})")

MEL_DIM = 80
MAX_FRAMES = 50 * 300            # 15,000
SINE_MAX_SAMPLES = 300 * 24000   # 7,200,000
HARMONIC_DIM = 9                 # NB_HARMONICS + 1
SEED = 0


def _sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_flow_noise(out_dir):
    torch.manual_seed(SEED)
    rand_noise = torch.randn(1, MEL_DIM, MAX_FRAMES)          # (1, 80, 15000)
    path = os.path.join(out_dir, "flow_noise.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x45534E46))                 # "FNSE"
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", MEL_DIM))
        f.write(struct.pack("<Q", MAX_FRAMES))
        f.write(rand_noise.detach().cpu().numpy().astype(np.float32).tobytes())
    size = os.path.getsize(path)
    print(f"wrote {path}")
    print(f"  rand_noise : {tuple(rand_noise.shape)}  (deterministic, seed {SEED})")
    print(f"  file size  : {size} bytes ({size / 1e6:.2f} MB)  sha256={_sha256(path)}")
    return path


def write_hift_source(out_dir):
    torch.manual_seed(SEED)
    rand_ini = torch.rand(1, HARMONIC_DIM)                    # (1, 9)
    rand_ini[:, 0] = 0                                        # fundamental phase zeroed
    sine_waves = torch.rand(1, SINE_MAX_SAMPLES, HARMONIC_DIM)  # (1, 7200000, 9)
    path = os.path.join(out_dir, "hift_source.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x43525348))                 # "HSRC"
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", HARMONIC_DIM))
        f.write(struct.pack("<Q", SINE_MAX_SAMPLES))
        f.write(rand_ini.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(sine_waves.detach().cpu().numpy().astype(np.float32).tobytes())
    size = os.path.getsize(path)
    print(f"wrote {path}")
    print(f"  rand_ini   : {tuple(rand_ini.shape)}  col0={float(rand_ini[0, 0]):.1f}")
    print(f"  sine_waves : {tuple(sine_waves.shape)}  (full 300 s bank)")
    print(f"  file size  : {size} bytes ({size / 1e6:.1f} MB)  sha256={_sha256(path)}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="directory to write the buffers into")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    write_flow_noise(args.out_dir)
    write_hift_source(args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
