"""Stream a WAV file to the Swar Rakshak 500 ms WebSocket endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets


def _read_mono_int16(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1) if waveform.shape[1] > 1 else waveform[:, 0]
    pcm = np.clip(mono, -1.0, 1.0)
    return np.rint(pcm * 32767.0).astype("<i2"), int(sample_rate)


async def _run(url: str, path: Path) -> None:
    samples, sample_rate = _read_mono_int16(path)
    hop = max(1, int(round(sample_rate * 0.5)))
    async with websockets.connect(url, max_size=None) as socket:
        await socket.send(json.dumps({"type": "config", "sample_rate": sample_rate, "channels": 1, "format": "int16"}))
        for start in range(0, len(samples), hop):
            chunk = samples[start : start + hop]
            if len(chunk) < hop:
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            await socket.send(chunk.tobytes())
            await asyncio.sleep(0.5)
            while True:
                try:
                    message = await asyncio.wait_for(socket.recv(), timeout=0.01)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    return
                payload = json.loads(message)
                if payload.get("type") == "result":
                    print(f"t={payload['t']:.1f} score={payload['score']:.4f} verdict={payload['verdict']}", flush=True)
                elif payload.get("type") == "warmup":
                    print(f"t={payload['seconds']:.1f} score=warmup verdict=warmup", flush=True)
                elif payload.get("type") == "error":
                    print(f"error: {payload.get('message', 'unknown error')}", file=sys.stderr, flush=True)
        await socket.send(json.dumps({"type": "stop"}))
        try:
            while True:
                message = await socket.recv()
                payload = json.loads(message)
                if payload.get("type") == "result":
                    print(f"t={payload['t']:.1f} score={payload['score']:.4f} verdict={payload['verdict']}", flush=True)
                elif payload.get("type") == "error":
                    print(f"error: {payload.get('message', 'unknown error')}", file=sys.stderr, flush=True)
        except websockets.ConnectionClosed:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    args = parser.parse_args()
    asyncio.run(_run(args.url, args.wav))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
