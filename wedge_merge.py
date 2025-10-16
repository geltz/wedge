import argparse
import threading
import time
import sys
import os
from itertools import cycle

import sd_mecha
from wedge import wedge

def _supports_tty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _spinner_frames():
    if os.name == "nt":
        return ["|", "/", "-", "\\"]
    return ["⠋","⠙","⠸","⠴","⠦","⠇"]

def _animate_wedge(stop_event, label="wedge", extra=""):
    if not _supports_tty():
        return
    frames = cycle(_spinner_frames())
    letters = list(label)
    pos = 0
    up = "\x1b[1A"
    down = "\x1b[1B"
    clear = "\x1b[2K\r"
    while not stop_event.is_set():
        ch = next(frames)
        s = letters[:]
        s[pos % len(s)] = ch
        sys.stdout.write(up + clear + f"[{''.join(s)}]{(' ' + extra) if extra else ''}" + down)
        sys.stdout.flush()
        time.sleep(0.08)
        if ch in ("\\", "⠇"):
            pos += 1
    sys.stdout.write(up + clear + f"[{label}] done" + down + "\n")
    sys.stdout.flush()

def main():
    p = argparse.ArgumentParser(description="WEDGE merge with sd-mecha")
    p.add_argument("-a","--modela", required=True, help="path to model A (.safetensors)")
    p.add_argument("-b","--modelb", required=True, help="path to model B (.safetensors)")
    p.add_argument("-o","--output", required=True, help="output path (.safetensors)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    p.add_argument("--tau-lo", type=float, default=0.10, dest="tau_lo")
    p.add_argument("--tau-hi", type=float, default=0.60, dest="tau_hi")
    p.add_argument("--winsor-k", type=float, default=0.0)
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.6)
    p.add_argument("--t0", type=float, default=1.0)
    args = p.parse_args()

    A = sd_mecha.model(args.modela)
    B = sd_mecha.model(args.modelb)

    recipe = wedge(
        A, B,
        alpha=args.alpha,
        tmin=args.tmin, tmax=args.tmax,
        tau_lo=args.tau_lo, tau_hi=args.tau_hi,
        winsor_k=args.winsor_k,
        depth_scale=args.depth_scale,
        lambda_ce=args.lambda_ce,
        t0=args.t0,
    )

    if _supports_tty():
        sys.stdout.write("[wedge]\n")
        sys.stdout.flush()

    stop_event = threading.Event()
    extra = f"α={args.alpha:.2f} ds={args.depth_scale:.2f}"
    spinner_thread = threading.Thread(target=_animate_wedge, args=(stop_event, "wedge", extra), daemon=True)
    spinner_thread.start()

    try:
        sd_mecha.merge(recipe, output=args.output)
    finally:
        stop_event.set()
        spinner_thread.join()
        sys.stdout.write(f"Saved → {args.output}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
