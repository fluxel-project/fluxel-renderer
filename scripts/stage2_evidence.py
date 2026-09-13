#!/usr/bin/env python3
"""Capture browser-backed Stage 2 evidence without external Python packages.

The page under test supplies a deliberately small, application-owned protocol::

    window.__fluxelEvidence = {
      ready: async () => void,
      snapshot: async (label) => ({ ...structured observation... }),
      resize: async (width, height) => void,
      loseContext/loseDevice: async () => void,
      restoreContext/recoverDevice: async () => void,
      setVisibleForTest: async (visible) => void,
      dispose: async () => void,
    };

``snapshot`` must report any renderer diagnostics under ``diagnostics`` (an
array of strings or objects).  For a visible state it must additionally return
``{blocked: false, report: {...}, extent: {width, height}, keyColors: [...]}``;
the four key-colour entries must each have equal ``expected`` and ``actual``
RGBA values.  For a deliberately blocked state (zero size, hidden, or lost
context), it must return ``{blocked: true}`` and no ``report``.  The harness
also asks a WebGL canvas for ``getError()`` when that backend is selected. A
mock or a successful JS call
is *not* GPU evidence: this tool records real Chrome screenshots, browser
diagnostics, the GPU identity, and page observations for a page that actually
uses the browser GPU stack.

It intentionally uses only the Python standard library.  This keeps the
evidence recipe portable and makes the exact Chrome invocation and CDP traffic
durable artifacts instead of hiding them behind a test framework.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote
from urllib.request import Request, urlopen


class EvidenceError(RuntimeError):
    """A prerequisite, protocol invariant, or browser correctness gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode_png_pixels(image: bytes) -> tuple[int, int, int, bytes]:
    """Decode Chrome's non-interlaced 8-bit RGB/RGBA screenshot."""
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise EvidenceError("Chrome screenshot is not a PNG")
    position = 8
    width = height = colour_type = bit_depth = interlace = None
    compressed = bytearray()
    while position < len(image):
        length = struct.unpack(">I", image[position:position + 4])[0]
        kind = image[position + 4:position + 8]
        payload = image[position + 8:position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, colour_type, _, _, interlace = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if not width or not height or bit_depth != 8 or colour_type not in (2, 6) or interlace != 0:
        raise EvidenceError(
            f"unsupported Chrome PNG layout: {width}x{height}, depth={bit_depth}, "
            f"colour={colour_type}, interlace={interlace}"
        )
    channels = 4 if colour_type == 6 else 3
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows = bytearray(height * stride)

    def paeth(left: int, above: int, upper_left: int) -> int:
        estimate = left + above - upper_left
        dl, da, du = abs(estimate - left), abs(estimate - above), abs(estimate - upper_left)
        return left if dl <= da and dl <= du else above if da <= du else upper_left

    source = 0
    previous = bytearray(stride)
    for row_index in range(height):
        filter_kind = raw[source]
        source += 1
        encoded = raw[source:source + stride]
        source += stride
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_kind == 0:
                predictor = 0
            elif filter_kind == 1:
                predictor = left
            elif filter_kind == 2:
                predictor = above
            elif filter_kind == 3:
                predictor = (left + above) // 2
            elif filter_kind == 4:
                predictor = paeth(left, above, upper_left)
            else:
                raise EvidenceError(f"unsupported PNG filter {filter_kind}")
            row[index] = (value + predictor) & 0xff
        rows[row_index * stride:(row_index + 1) * stride] = row
        previous = row
    return width, height, channels, bytes(rows)


def apply_webgpu_screenshot_oracle(snapshot: dict[str, object], image: bytes) -> None:
    """Fill key-colour actuals from the real Chrome compositor screenshot."""
    image_width, image_height, channels, pixels = decode_png_pixels(image)
    rect = snapshot.get("canvasRectCss")
    colours = snapshot.get("keyColors")
    if not isinstance(rect, dict) or not isinstance(colours, list):
        raise EvidenceError("WebGPU snapshot lacks canvasRectCss/keyColors")
    rect_width, rect_height = rect.get("width"), rect.get("height")
    rect_x, rect_y = rect.get("x", 0), rect.get("y", 0)
    pixel_width, pixel_height = snapshot_extent(snapshot)
    if not all(isinstance(value, (int, float)) for value in (rect_x, rect_y, rect_width, rect_height)):
        raise EvidenceError(f"invalid canvas CSS rect: {rect!r}")
    if rect_width <= 0 or rect_height <= 0:
        raise EvidenceError(f"non-drawable canvas CSS rect: {rect!r}")
    scale_x = pixel_width / rect_width
    scale_y = pixel_height / rect_height
    if not math.isclose(scale_x, scale_y, rel_tol=0.01, abs_tol=0.01):
        raise EvidenceError(
            f"canvas backing/CSS scales disagree: x={scale_x}, y={scale_y}"
        )
    canvas_right = (rect_x + rect_width) * scale_x
    canvas_bottom = (rect_y + rect_height) * scale_y
    if rect_x < 0 or rect_y < 0 or canvas_right > image_width + 1 or canvas_bottom > image_height + 1:
        raise EvidenceError(
            "canvas screenshot mapping falls outside the captured compositor image: "
            f"rect={rect!r}, image={[image_width, image_height]!r}, scale={scale_x}"
        )
    stride = image_width * channels
    for colour in colours:
        normalized = colour.get("normalized") if isinstance(colour, dict) else None
        if not isinstance(normalized, list) or len(normalized) != 2:
            raise EvidenceError(f"invalid WebGPU colour coordinate: {colour!r}")
        x = round((rect_x + rect_width * normalized[0]) * scale_x)
        y = round((rect_y + rect_height * normalized[1]) * scale_y)
        x = min(image_width - 1, max(0, x))
        y = min(image_height - 1, max(0, y))
        start = y * stride + x * channels
        actual = list(pixels[start:start + channels])
        if channels == 3:
            actual.append(255)
        colour["actual"] = actual


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a small read-only probe and return one-line output when possible."""
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    output = (completed.stdout or completed.stderr).strip()
    return output if output else f"exit {completed.returncode}"


def git_sha(repository: Path) -> str:
    return run_text(["git", "rev-parse", "HEAD"], cwd=repository)


def git_repository_info(path: Path) -> dict[str, object]:
    root_text = run_text(["git", "rev-parse", "--show-toplevel"], cwd=path)
    root = Path(root_text)
    if not root.is_dir():
        return {"root": root_text, "sha": "unavailable", "status": "unavailable"}
    return {
        "root": str(root.resolve()),
        "sha": git_sha(root),
        "status": run_text(["git", "status", "--short"], cwd=root),
    }


def chrome_default() -> Path | None:
    """Find a locally installed Chrome-family binary without assuming Windows."""
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    for program in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        located = shutil.which(program)
        if located:
            candidates.append(Path(located))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def chrome_version(chrome: Path) -> str:
    """Ask the selected browser for its version without opening a page."""
    try:
        completed = subprocess.run(
            [str(chrome), "--version"], capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    version = (completed.stdout or completed.stderr).strip()
    return version or "unavailable"


def entry_path(value: str) -> PurePosixPath:
    """Accept one relative URL path and reject host-platform path spellings.

    The entry becomes both a URL and a filesystem path below ``--site-root``.
    Rejecting backslashes and traversal keeps those two interpretations identical
    on Windows, macOS, and Linux.
    """
    if not value or "\\" in value:
        raise EvidenceError("--entry must be a non-empty relative URL path using '/' only")
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise EvidenceError("--entry must stay below --site-root without '.' or '..' segments")
    return path


class QuietHandler(SimpleHTTPRequestHandler):
    """Serve a supplied root while preserving each request in a durable log."""

    server_version = "FluxelEvidenceHTTP/1"

    def log_message(self, fmt: str, *args: object) -> None:
        self.server.request_log.put(  # type: ignore[attr-defined]
            {"at": utc_now(), "message": fmt % args}
        )


class EvidenceHttpServer:
    def __init__(self, root: Path) -> None:
        handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(root), **kwargs)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.request_log = queue.SimpleQueue()  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "EvidenceHttpServer":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def drain_log(self) -> list[dict[str, str]]:
        records = []
        while True:
            try:
                records.append(self._server.request_log.get_nowait())  # type: ignore[attr-defined]
            except queue.Empty:
                return records


class WebSocket:
    """Minimal RFC 6455 client for Chrome's local, unencrypted CDP endpoint."""

    def __init__(self, url: str, timeout: float = 20) -> None:
        if not url.startswith("ws://"):
            raise EvidenceError(f"only local ws:// CDP URLs are supported, got {url!r}")
        host_and_path = url[5:]
        host_port, _, path = host_and_path.partition("/")
        host, _, port_text = host_port.partition(":")
        if not host or not port_text:
            raise EvidenceError(f"invalid CDP WebSocket URL: {url!r}")
        self._socket = socket.create_connection((host, int(port_text)), timeout=timeout)
        self._socket.settimeout(timeout)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = self._receive_http_headers()
        if not response.startswith("HTTP/1.1 101"):
            raise EvidenceError(f"Chrome refused CDP WebSocket: {response.splitlines()[0]!r}")

    def _receive_http_headers(self) -> str:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise EvidenceError("CDP closed during WebSocket handshake")
            data.extend(chunk)
        return data.decode("latin-1")

    def send_json(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        mask = secrets.token_bytes(4)
        length = len(data)
        if length < 126:
            header = bytes((0x81, 0x80 | length))
        elif length <= 0xFFFF:
            header = bytes((0x81, 0x80 | 126)) + struct.pack("!H", length)
        else:
            header = bytes((0x81, 0x80 | 127)) + struct.pack("!Q", length)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        self._socket.sendall(header + mask + masked)

    def receive_json(self) -> dict[str, object]:
        first, second = self._read_exact(2)
        opcode = first & 0x0F
        length = second & 0x7F
        masked = bool(second & 0x80)
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length)
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        if opcode == 0x8:
            raise EvidenceError("Chrome closed its CDP connection")
        if opcode == 0x9:
            self._socket.sendall(b"\x8a\x00")
            return self.receive_json()
        if opcode != 0x1:
            raise EvidenceError(f"unexpected CDP WebSocket opcode {opcode}")
        return json.loads(payload.decode("utf-8"))

    def _read_exact(self, amount: int) -> bytes:
        parts = bytearray()
        while len(parts) < amount:
            chunk = self._socket.recv(amount - len(parts))
            if not chunk:
                raise EvidenceError("CDP connection ended unexpectedly")
            parts.extend(chunk)
        return bytes(parts)

    def close(self) -> None:
        try:
            self._socket.sendall(b"\x88\x80\x00\x00\x00\x00")
        except OSError:
            pass
        self._socket.close()


class Cdp:
    def __init__(self, websocket_url: str) -> None:
        self._ws = WebSocket(websocket_url)
        self._next_id = 1
        self.events: list[dict[str, object]] = []

    def call(self, method: str, params: dict[str, object] | None = None) -> dict[str, object]:
        identifier = self._next_id
        self._next_id += 1
        self._ws.send_json({"id": identifier, "method": method, "params": params or {}})
        while True:
            message = self._ws.receive_json()
            if message.get("id") != identifier:
                self.events.append(message)
                continue
            if "error" in message:
                raise EvidenceError(f"CDP {method} failed: {message['error']}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def enable_diagnostics(self) -> None:
        for method in ("Page.enable", "Runtime.enable", "Log.enable", "Performance.enable"):
            self.call(method)

    def evaluate(self, expression: str, *, await_promise: bool = True) -> object:
        result = self.call("Runtime.evaluate", {
            "expression": expression, "awaitPromise": await_promise,
            "returnByValue": True, "userGesture": True,
        })
        details = result.get("exceptionDetails")
        if details:
            raise EvidenceError(f"page evaluation exception: {details}")
        value = result.get("result", {})
        if not isinstance(value, dict):
            return None
        if "value" in value:
            return value["value"]
        return value.get("description")

    def close(self) -> None:
        self._ws.close()


def wait_for_debugger(port: int, process: subprocess.Popen[str], timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{port}/json/version"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvidenceError(f"Chrome exited before CDP became available ({process.returncode})")
        try:
            with urlopen(endpoint, timeout=1) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError:
            time.sleep(0.1)
    raise EvidenceError(f"timed out waiting for Chrome CDP on port {port}")


def page_target(port: int) -> dict[str, object]:
    # Chrome 136+ rejects GET here (405) and explicitly requires PUT.
    request = Request(f"http://127.0.0.1:{port}/json/new", method="PUT")
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def protocol_call(name: str, *args: object) -> str:
    encoded = ",".join(json.dumps(argument) for argument in args)
    return (
        "(async () => { const e = window.__fluxelEvidence;"
        "if (!e || typeof e." + name + " !== 'function')"
        " throw new Error('missing window.__fluxelEvidence." + name + "()');"
        "return await e." + name + "(" + encoded + "); })()"
    )


def renderer_observation(cdp: Cdp, backend: str) -> dict[str, object]:
    """Collect browser-side GPU facts without taking ownership of the app context."""
    expression = r"""(() => {
      const backend = BACKEND;
      const canvases = [...document.querySelectorAll('canvas')];
      const webgl = canvases.map((canvas, index) => {
        if (backend === 'webgpu') return {index, kind: 'webgpu', width: canvas.width, height: canvas.height};
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) return {index, kind: 'non-webgl', width: canvas.width, height: canvas.height};
        const debug = gl.getExtension('WEBGL_debug_renderer_info');
        const error = gl.getError();
        return {index, kind: 'webgl', width: canvas.width, height: canvas.height, error,
          vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : null,
          renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null,
          version: gl.getParameter(gl.VERSION)};
      });
      return {href: location.href, visibility: document.visibilityState,
        user_agent: navigator.userAgent, device_memory: navigator.deviceMemory || null,
        js_heap: performance.memory ? {used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize, limit: performance.memory.jsHeapSizeLimit} : null,
        canvases: webgl};
    })()""".replace("BACKEND", json.dumps(backend))
    observation = cdp.evaluate(expression)
    return observation if isinstance(observation, dict) else {"raw": observation}


def snapshot_extent(snapshot: dict[str, object]) -> tuple[int, int]:
    """Read the protocol's required render extent, rejecting ambiguous shapes."""
    raw_extent = snapshot.get("extent")
    if not isinstance(raw_extent, dict):
        raise EvidenceError("snapshot has no structured extent object")
    width, height = raw_extent.get("width"), raw_extent.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise EvidenceError(f"snapshot extent must contain integer width/height: {raw_extent!r}")
    return width, height


def reject_bad_snapshot(
    snapshot: object, label: str, *, visible: bool, require_colour_actual: bool = True
) -> None:
    if not isinstance(snapshot, dict):
        raise EvidenceError(f"{label}: snapshot must be a structured object, got {type(snapshot).__name__}")
    diagnostics = snapshot.get("diagnostics", [])
    if isinstance(diagnostics, str):
        diagnostics = [diagnostics]
    if diagnostics:
        raise EvidenceError(f"{label}: renderer structured diagnostics are not clean: {diagnostics!r}")
    errors = snapshot.get("errors", [])
    if errors:
        raise EvidenceError(f"{label}: renderer reported errors: {errors!r}")
    if snapshot.get("blockedError") is not None:
        raise EvidenceError(f"{label}: unexpected blocked-state exception: {snapshot.get('blockedError')!r}")
    if visible:
        if snapshot.get("blocked") is not False:
            raise EvidenceError(f"{label}: visible state must explicitly report blocked: false")
        report = snapshot.get("report")
        if not isinstance(report, dict):
            raise EvidenceError(f"{label}: visible state must contain a renderer report")
        if report.get("outcome") != "submitted":
            raise EvidenceError(f"{label}: visible state did not prove a submitted frame: {report!r}")
        marker = report.get("frameMarker")
        if not isinstance(marker, int) or marker <= 0:
            raise EvidenceError(f"{label}: submitted frame has no positive integer marker: {report!r}")
        width, height = snapshot_extent(snapshot)
        if width <= 0 or height <= 0:
            raise EvidenceError(f"{label}: visible state has non-positive extent {width}x{height}")
    else:
        if snapshot.get("blocked") is not True:
            raise EvidenceError(f"{label}: blocked state must explicitly report blocked: true")
        if snapshot.get("report") is not None:
            raise EvidenceError(f"{label}: blocked state must not expose a renderer report")

    colors = snapshot.get("key_colors", snapshot.get("keyColors"))
    if visible and not isinstance(colors, list):
        raise EvidenceError(f"{label}: visible state must provide four keyColors entries")
    if visible and len(colors) != 4:
        raise EvidenceError(f"{label}: visible state must provide exactly four keyColors, got {len(colors)}")
    if isinstance(colors, dict):
        colors = [colors]
    for color in colors if isinstance(colors, list) else []:
        if not isinstance(color, dict):
            raise EvidenceError(f"{label}: keyColors entry is not an object: {color!r}")
        expected, actual = color.get("expected"), color.get("actual")
        if visible and expected is None:
            raise EvidenceError(f"{label}: keyColors entry has no expected value: {color!r}")
        if visible and require_colour_actual and actual is None:
            raise EvidenceError(f"{label}: keyColors entry must contain expected and actual: {color!r}")
        if expected is not None and actual is not None and expected != actual:
            raise EvidenceError(f"{label}: key colour mismatch: {color!r}")


def reject_diagnostics(cdp: Cdp, observation: dict[str, object]) -> None:
    bad: list[object] = []
    for event in cdp.events:
        method = event.get("method")
        params = event.get("params", {})
        if method == "Runtime.exceptionThrown":
            bad.append(event)
        elif method == "Log.entryAdded":
            entry = params.get("entry", {}) if isinstance(params, dict) else {}
            text = entry.get("text", "") if isinstance(entry, dict) else ""
            harness_readback_warning = (
                isinstance(text, str)
                and entry.get("source") == "rendering"
                and entry.get("level") == "warning"
                and text.startswith("[.WebGL-")
                and "]GL Driver Message (OpenGL, Performance, GL_CLOSE_PATH_NV, High): "
                "GPU stall due to ReadPixels" in text
                and (
                    text.endswith("GPU stall due to ReadPixels")
                    or text.endswith("GPU stall due to ReadPixels (this message will no longer repeat)")
                )
            )
            if (
                isinstance(entry, dict)
                and entry.get("level") in {"error", "warning"}
                and not harness_readback_warning
            ):
                bad.append(event)
        elif method == "Runtime.consoleAPICalled":
            if isinstance(params, dict) and params.get("type") in {"error", "assert"}:
                bad.append(event)
    canvas_errors = []
    for canvas in observation.get("canvases", []) if isinstance(observation.get("canvases"), list) else []:
        if isinstance(canvas, dict) and canvas.get("kind") == "webgl" and canvas.get("error") not in (0, None):
            canvas_errors.append(canvas)
    if canvas_errors:
        bad.append({"webgl_errors": canvas_errors})
    if bad:
        raise EvidenceError(f"browser/GPU diagnostics are not clean: {bad!r}")


def capture_screenshot(cdp: Cdp) -> bytes:
    response = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True})
    data = response.get("data")
    if not isinstance(data, str):
        raise EvidenceError("Chrome did not return screenshot data")
    return base64.b64decode(data)


def write_screenshot(cdp: Cdp, destination: Path) -> dict[str, object]:
    image = capture_screenshot(cdp)
    destination.write_bytes(image)
    return {"path": str(destination), "sha256": sha256_bytes(image), "bytes": len(image)}


def capture_state(
    cdp: Cdp, output: Path, label: str, repeats: int, *, visible: bool, backend: str
) -> dict[str, object]:
    records = []
    previous_marker = 0

    def next_snapshot(sample_label: str) -> dict[str, object]:
        nonlocal previous_marker
        for _ in range(12):
            value = cdp.evaluate(protocol_call("snapshot", sample_label))
            reject_bad_snapshot(
                value,
                sample_label,
                visible=visible,
                require_colour_actual=backend != "webgpu",
            )
            if not isinstance(value, dict):
                raise EvidenceError(f"{sample_label}: snapshot is not an object")
            if not visible:
                return value
            report = value["report"]
            marker = report["frameMarker"]
            if marker > previous_marker:
                previous_marker = marker
                return value
            time.sleep(0.02)
        raise EvidenceError(f"{sample_label}: frame marker did not advance beyond {previous_marker}")

    for index in range(repeats):
        time.sleep(0.15)
        try:
            snapshot = next_snapshot(f"{label}-{index}")
        except BaseException:
            # Preserve one visual diagnostic when a structured/readback gate
            # fails; otherwise the most useful clue disappears with Chrome.
            write_screenshot(cdp, output / f"{label}-{index}-failure.png")
            raise
        screenshot = write_screenshot(cdp, output / f"{label}-{index}.png")
        if visible and backend == "webgpu":
            apply_webgpu_screenshot_oracle(snapshot, Path(screenshot["path"]).read_bytes())
            reject_bad_snapshot(snapshot, f"{label}-{index}", visible=True)
        records.append({"snapshot": snapshot, "screenshot": screenshot})
    hashes = {record["screenshot"]["sha256"] for record in records}
    if len(hashes) != 1:
        raise EvidenceError(
            f"{label}: static scene changed across {repeats} separated captures: {sorted(hashes)!r}"
        )
    dense = []
    if visible:
        for index in range(15):
            time.sleep(0.02)
            snapshot = next_snapshot(f"{label}-dense-{index}")
            if backend == "webgpu":
                apply_webgpu_screenshot_oracle(snapshot, capture_screenshot(cdp))
                reject_bad_snapshot(snapshot, f"{label}-dense-{index}", visible=True)
            dense.append(snapshot)
    return {
        "samples": records,
        "dense_frame_samples": dense,
        "unique_screenshot_hashes": len(hashes),
    }


def scoped_measurements(
    states: dict[str, object], startup_ms: float, backend: str
) -> dict[str, object]:
    cpu_samples: list[float] = []
    wasm_memory: list[int] = []
    for state in states.values():
        if not isinstance(state, dict):
            continue
        snapshots = [
            sample.get("snapshot", {})
            for sample in state.get("samples", [])
            if isinstance(sample, dict)
        ]
        snapshots.extend(
            sample for sample in state.get("dense_frame_samples", [])
            if isinstance(sample, dict)
        )
        for snapshot in snapshots:
            report = snapshot.get("report", {}) if isinstance(snapshot, dict) else {}
            if not isinstance(report, dict):
                continue
            cpu = report.get("cpuSubmissionMs")
            memory = report.get("wasmMemoryBytes")
            if isinstance(cpu, (int, float)):
                cpu_samples.append(float(cpu))
            if isinstance(memory, int):
                wasm_memory.append(memory)
    ordered = sorted(cpu_samples)
    if not ordered or not all(math.isfinite(sample) for sample in ordered):
        raise EvidenceError("CPU submission samples must be present and finite")
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    p95 = ordered[p95_index]
    if p95 > 5.0:
        raise EvidenceError(f"CPU submission p95 exceeds 5 ms: {p95}")
    if not wasm_memory or any(sample <= 0 for sample in wasm_memory):
        raise EvidenceError("WASM memory samples must be present and positive")
    strictly_growing = len(wasm_memory) > 1 and all(
        current > previous for previous, current in zip(wasm_memory, wasm_memory[1:])
    )
    if strictly_growing:
        raise EvidenceError("WASM memory grew at every sampled frame")
    return {
        "scope": (
            "retained Stage 1 scene in named Chrome "
            f"{'WebGPU' if backend == 'webgpu' else 'WebGL2'} target"
        ),
        "navigation_to_ready_ms": startup_ms,
        "cpu_submission_ms": {
            "count": len(ordered),
            "min": ordered[0] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "p95": p95,
            "max": ordered[-1] if ordered else None,
        },
        "wasm_memory_bytes": {
            "sample_count": len(wasm_memory),
            "min": min(wasm_memory) if wasm_memory else None,
            "max": max(wasm_memory) if wasm_memory else None,
        },
    }


def validate_webgpu_lifecycle(states: dict[str, object]) -> None:
    def snapshot(state: str) -> dict[str, object]:
        value = states.get(state)
        if not isinstance(value, dict) or not value.get("samples"):
            raise EvidenceError(f"missing WebGPU lifecycle state: {state}")
        result = value["samples"][0].get("snapshot")
        if not isinstance(result, dict):
            raise EvidenceError(f"invalid WebGPU lifecycle snapshot: {state}")
        return result

    stable = snapshot("stable")
    lost = snapshot("device_lost")
    recovered = snapshot("device_recovered")
    disposed = snapshot("disposed")
    stable_backend = stable.get("backend", {})
    lost_backend = lost.get("backend", {})
    recovered_backend = recovered.get("backend", {})
    stable_generation = stable_backend.get("generation")
    lost_generation = lost_backend.get("generation")
    recovered_generation = recovered_backend.get("generation")
    if not all(isinstance(value, int) for value in (
        stable_generation, lost_generation, recovered_generation
    )):
        raise EvidenceError("WebGPU lifecycle snapshots lack integer generations")
    if lost_generation != stable_generation or recovered_generation != stable_generation + 1:
        raise EvidenceError(
            "WebGPU device generation did not remain stable at loss and increment once on recovery: "
            f"{stable_generation} -> {lost_generation} -> {recovered_generation}"
        )
    if lost_backend.get("state") != "Lost" or not lost.get("blocked"):
        raise EvidenceError("controlled device destruction did not produce a blocked Lost state")
    if lost_backend.get("lossReason") != "destroyed":
        raise EvidenceError(
            "controlled device destruction did not report GPUDeviceLostInfo.reason=destroyed"
        )
    if recovered_backend.get("state") != "Active":
        raise EvidenceError("WebGPU recovery did not install an Active replacement generation")
    report = recovered.get("report")
    if not isinstance(report, dict) or report.get("outcome") != "submitted":
        raise EvidenceError("replacement WebGPU generation did not submit a visible frame")
    if disposed.get("backend", {}).get("state") != "Disposed" or not disposed.get("blocked"):
        raise EvidenceError("WebGPU disposal did not reach a blocked Disposed terminal state")


def validate_staged_wasm(stage: Path) -> dict[str, object]:
    """Require real wasm-bindgen glue in the served tree before opening Chrome."""
    if not stage.is_dir():
        raise EvidenceError(f"missing required wasm staging directory: {stage}")
    javascript = sorted(stage.glob("*.js"))
    wasm = sorted(stage.glob("*_bg.wasm"))
    if not javascript or not wasm:
        raise EvidenceError(
            f"{stage} must contain wasm-bindgen JS glue and a *_bg.wasm payload before evidence runs"
        )
    empty = [path for path in [*javascript, *wasm] if path.stat().st_size == 0]
    if empty:
        raise EvidenceError(f"empty wasm-bindgen staging artifact(s): {empty}")
    referenced = {payload.name: False for payload in wasm}
    for glue in javascript:
        source = glue.read_text(encoding="utf-8", errors="replace")
        for payload in wasm:
            if payload.name in source:
                referenced[payload.name] = True
    unreferenced = [name for name, found in referenced.items() if not found]
    if unreferenced:
        raise EvidenceError(f"wasm-bindgen JS glue does not reference staged payload(s): {unreferenced}")
    return {
        "directory": str(stage),
        "javascript": [{"path": str(path), "sha256": sha256_bytes(path.read_bytes())} for path in javascript],
        "wasm": [{"path": str(path), "sha256": sha256_bytes(path.read_bytes())} for path in wasm],
    }


def maybe_bindgen(args: argparse.Namespace, site_root: Path, command_log: list[list[str]]) -> dict[str, object]:
    stage = (site_root.joinpath(*entry_path(args.entry).parts)).parent / "wasm"
    if not args.wasm:
        return validate_staged_wasm(stage)
    wasm = Path(args.wasm).resolve()
    if not wasm.is_file():
        raise EvidenceError(f"--wasm does not exist: {wasm}")
    executable = args.wasm_bindgen or shutil.which("wasm-bindgen")
    if not executable:
        raise EvidenceError(
            "wasm-bindgen-cli is required for --wasm but was not found. Install/pin it for the "
            "project, or pass --wasm-bindgen C:\\path\\to\\wasm-bindgen.exe."
        )
    out_dir = Path(args.bindgen_out).resolve() if args.bindgen_out else stage.resolve()
    if out_dir != stage.resolve():
        raise EvidenceError("wasm-bindgen output must be staged beside the entry at <entry-dir>/wasm")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "--target", "web", "--out-dir", str(out_dir), str(wasm)]
    command_log.append(command)
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise EvidenceError(f"wasm-bindgen failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return validate_staged_wasm(stage)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-root", required=True, help="Static jsbridge demo root served over localhost.")
    parser.add_argument("--entry", default="index.html", help="Relative URL path below --site-root, using '/' (default: index.html).")
    parser.add_argument("--backend", choices=("webgl2", "webgpu"), default="webgl2",
                        help="Browser GPU lifecycle under test (default: webgl2).")
    parser.add_argument("--chrome", help="Exact chrome.exe path; defaults to installed Google Chrome.")
    parser.add_argument("--wasm", help="Optional .wasm input to process through wasm-bindgen before serving.")
    parser.add_argument("--wasm-bindgen", help="Explicit wasm-bindgen executable path.")
    parser.add_argument("--bindgen-out", help="Must equal <entry-dir>/wasm; retained for explicit invocation logs.")
    parser.add_argument("--output", help="Evidence output root (default target/evidence/<sha>/stage2).")
    parser.add_argument("--frames", type=int, default=3, help="Screenshots per stable state (default: 3).")
    parser.add_argument("--timeout", type=float, default=30, help="Chrome/protocol timeout seconds.")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--headless", action="store_true", help="Run Chrome headlessly (for CI smoke only).")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.frames < 2:
        raise EvidenceError("--frames must be at least 2: one screenshot cannot detect flicker")
    repo = Path(__file__).resolve().parents[1]
    site_root = Path(args.site_root).resolve()
    if not site_root.is_dir():
        raise EvidenceError(f"--site-root is not a directory: {site_root}")
    entry = entry_path(args.entry)
    entry_file = site_root.joinpath(*entry.parts)
    if not entry_file.is_file():
        raise EvidenceError(f"entry does not exist below --site-root: {entry}")
    chrome = Path(args.chrome).resolve() if args.chrome else chrome_default()
    if not chrome or not chrome.is_file():
        raise EvidenceError("Chrome was not found; pass --chrome with an exact chrome.exe path")
    sha = git_sha(repo)
    if len(sha) != 40:
        raise EvidenceError(f"cannot determine exact repository SHA: {sha}")
    stage_name = "stage2" if args.backend == "webgl2" else "stage2-webgpu"
    output = Path(args.output).resolve() if args.output else repo / "target/evidence" / sha / stage_name
    output.mkdir(parents=True, exist_ok=True)
    command_log: list[list[str]] = []
    staged_wasm = maybe_bindgen(args, site_root, command_log)
    chrome_log = output / "chrome.stderr.log"
    port_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_socket.bind(("127.0.0.1", 0))
    port = port_socket.getsockname()[1]
    port_socket.close()
    user_data = output / "chrome-profile"
    user_data.mkdir(exist_ok=True)
    chrome_command = [
        str(chrome), f"--remote-debugging-port={port}", f"--user-data-dir={user_data}",
        "--no-first-run", "--no-default-browser-check", "--enable-logging=stderr",
        f"--window-size={args.width},{args.height}", "about:blank",
    ]
    if args.headless:
        chrome_command.insert(-1, "--headless=new")
        if args.backend == "webgpu":
            # Headless Chrome keeps WebGPU behind its explicit test gate even
            # when the selected Windows adapter is hardware-backed.
            chrome_command.insert(-1, "--enable-unsafe-webgpu")
            chrome_command.insert(-1, "--ignore-gpu-blocklist")
        if sys.platform.startswith("linux"):
            # Hosted Linux runners commonly execute Chrome in a container
            # without a usable setuid sandbox and with a small /dev/shm.
            chrome_command.insert(-1, "--no-sandbox")
            chrome_command.insert(-1, "--disable-dev-shm-usage")
            chrome_command.insert(-1, "--enable-unsafe-swiftshader")
            if args.backend == "webgpu":
                # Linux hosted runners use bundled software Vulkan only as an
                # ABI/lifecycle smoke; this does not establish a support target.
                chrome_command.insert(-1, "--enable-features=Vulkan")
                chrome_command.insert(-1, "--use-angle=swiftshader")
                chrome_command.insert(-1, "--use-vulkan=swiftshader")
    command_log.append(chrome_command)
    process: subprocess.Popen[str] | None = None
    cdp: Cdp | None = None
    manifest: dict[str, object] = {"schema": 1, "started_at": utc_now(), "repo_sha": sha,
        "repositories": {"rendering": git_repository_info(repo), "site": git_repository_info(site_root)},
        "site_root": str(site_root), "entry": entry.as_posix(), "backend": args.backend, "commands": command_log,
        "chrome": {"path": str(chrome), "version": chrome_version(chrome)},
        "os": {"name": os.name, "platform": sys.platform}, "wasm_bindgen_staging": staged_wasm,
        "states": {}}
    try:
        with EvidenceHttpServer(site_root) as server, chrome_log.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(chrome_command, stdout=stderr, stderr=stderr, text=True)
            debugger = wait_for_debugger(port, process, args.timeout)
            target = page_target(port)
            websocket = target.get("webSocketDebuggerUrl")
            if not isinstance(websocket, str):
                raise EvidenceError("Chrome /json/new did not provide a page CDP URL")
            cdp = Cdp(websocket)
            cdp.enable_diagnostics()
            url = f"{server.origin}/{quote(entry.as_posix())}"
            ready_started = time.monotonic()
            cdp.call("Page.navigate", {"url": url})
            deadline = time.monotonic() + args.timeout
            last_ready_error: BaseException | None = None
            while time.monotonic() < deadline:
                try:
                    cdp.evaluate(protocol_call("ready"))
                    break
                except EvidenceError as error:
                    last_ready_error = error
                    time.sleep(0.1)
            else:
                raise EvidenceError(
                    "timed out waiting for window.__fluxelEvidence.ready(); "
                    f"last error: {last_ready_error!r}; events: {cdp.events[-8:]!r}"
                )
            navigation_to_ready_ms = (time.monotonic() - ready_started) * 1000.0
            states = manifest["states"]
            assert isinstance(states, dict)
            states["stable"] = capture_state(cdp, output, "stable", args.frames, visible=True, backend=args.backend)
            stable_sample = states["stable"]["samples"][0]["snapshot"]
            stable_rect = stable_sample.get("canvasRectCss", {})
            stable_css_width = max(1, round(stable_rect.get("width", args.width)))
            stable_css_height = max(1, round(stable_rect.get("height", args.height)))
            cdp.evaluate(protocol_call("resize", max(1, stable_css_width // 2), max(1, stable_css_height // 2)))
            states["resized"] = capture_state(cdp, output, "resized", args.frames, visible=True, backend=args.backend)
            cdp.evaluate(protocol_call("resize", 0, 0))
            states["zero_size"] = capture_state(cdp, output, "zero-size", args.frames, visible=False, backend=args.backend)
            cdp.evaluate(protocol_call("resize", stable_css_width, stable_css_height))
            states["restored"] = capture_state(cdp, output, "restored", args.frames, visible=True, backend=args.backend)
            cdp.evaluate(protocol_call("setVisibleForTest", False))
            states["hidden"] = capture_state(cdp, output, "hidden", args.frames, visible=False, backend=args.backend)
            cdp.evaluate(protocol_call("setVisibleForTest", True))
            states["visible"] = capture_state(cdp, output, "visible", args.frames, visible=True, backend=args.backend)
            if args.backend == "webgpu":
                cdp.evaluate(protocol_call("loseDevice"))
                states["device_lost"] = capture_state(cdp, output, "device-lost", args.frames, visible=False, backend=args.backend)
                cdp.evaluate(protocol_call("recoverDevice"))
                states["device_recovered"] = capture_state(cdp, output, "device-recovered", args.frames, visible=True, backend=args.backend)
            else:
                cdp.evaluate(protocol_call("loseContext"))
                states["context_lost"] = capture_state(cdp, output, "context-lost", args.frames, visible=False, backend=args.backend)
                cdp.evaluate(protocol_call("restoreContext"))
                states["context_restored"] = capture_state(cdp, output, "context-restored", args.frames, visible=True, backend=args.backend)
            cdp.evaluate(protocol_call("dispose"))
            if args.backend == "webgpu":
                states["disposed"] = capture_state(
                    cdp, output, "disposed", 1, visible=False, backend=args.backend
                )
                validate_webgpu_lifecycle(states)
            observation = renderer_observation(cdp, args.backend)
            metrics = cdp.call("Performance.getMetrics").get("metrics", [])
            reject_diagnostics(cdp, observation)
            manifest.update({"finished_at": utc_now(), "result": "pass", "browser": debugger,
                             "gpu_and_canvas": observation, "performance_metrics": metrics,
                             "scoped_measurements": scoped_measurements(
                                 states, navigation_to_ready_ms, args.backend
                             ),
                             "http_requests": server.drain_log(), "diagnostic_events": cdp.events,
                             "logs": {"chrome_stderr": str(chrome_log)}})
    except BaseException as error:
        manifest.update({"finished_at": utc_now(), "result": "fail", "error": repr(error),
                         "logs": {"chrome_stderr": str(chrome_log)}})
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
        if cdp:
            cdp.close()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    print(f"Stage 2 {args.backend} browser evidence passed: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        print(f"stage2 evidence failed: {error}", file=sys.stderr)
        raise SystemExit(2)
