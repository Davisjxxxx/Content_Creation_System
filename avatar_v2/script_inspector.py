"""Static AST Inspector for imported Python generation scripts (Sprint 4, Section 4).

Performs pure static analysis via Python's ``ast`` module without ever importing,
executing, ``eval()``-ing, or ``exec()``-ing the inspected script.

Classifications:
- ``compatible_native``: Uses the native Avatar V2 Python SDK with no blocking unsafe patterns.
- ``statically_extractable``: Contains valid declarative prompt libraries / scene recipes that
  can be safely extracted and run through Avatar V2's deterministic prompt engine without
  executing script code.
- ``potentially_compatible_legacy``: Legacy generator (e.g. A1111/SDXL client script) that
  may be statically extracted now or adapted in Sprint 5.
- ``requires_review``: Valid Python script containing network, subprocess, filesystem-write,
  GPU, or dynamic-execution patterns requiring explicit review before sandboxed execution.
- ``unsupported``: Syntax error, non-Python file, missing/blocked dependencies, or unsupported API.

Disclaimer: Static AST inspection cannot prove arbitrary Python code is safe. All script
execution is independently isolated via OS sandboxing (Bubblewrap) and host broker enforcement.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .prompt_extractor import extract_prompt_data_from_source

ScriptClassification = Literal[
    "compatible_native",
    "statically_extractable",
    "potentially_compatible_legacy",
    "requires_review",
    "unsupported",
]

STATIC_SAFETY_DISCLAIMER = (
    "Static AST inspection identifies declared dependencies and known patterns, "
    "but does NOT prove arbitrary Python code is safe. Execution is always isolated "
    "in a network-off Bubblewrap sandbox with host-enforced limits."
)

_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))
_NATIVE_SDK_MODULES: frozenset[str] = frozenset({"avatar_v2_sdk", "avatar_v2.sdk", "avatar_v2.sdk_client"})
_NETWORK_MODULES: frozenset[str] = frozenset(
    {"requests", "httpx", "urllib", "urllib3", "aiohttp", "socket", "http", "ftplib", "smtplib", "websocket", "websockets"}
)
_SUBPROCESS_MODULES: frozenset[str] = frozenset({"subprocess", "pty", "multiprocessing", "popen2"})
_GPU_MODULES: frozenset[str] = frozenset({"torch", "cupy", "pynvml", "nvidia_smi", "diffusers", "accelerate", "xformers"})
_DYNAMIC_EXEC_CALLS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})
_SENSITIVE_ENV_TOKENS: tuple[str, ...] = ("RUNWAY", "API_KEY", "SECRET", "TOKEN", "PASSWORD", "AWS_", "OPENAI")
_CONFIG_EXT_RE = re.compile(r"[^\s'\"<>]+\.(?:json|ya?ml|toml|ini|cfg)$", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_A1111_ENDPOINT_RE = re.compile(r"/sdapi/v\d+/(?:txt2img|img2img|sd-models|options|progress)", re.IGNORECASE)
_COMFY_ENDPOINT_RE = re.compile(r"(?:127\.0\.0\.1:8188|/prompt|/history|/free|/object_info)", re.IGNORECASE)
_RUNWAY_RE = re.compile(r"(?:api\.dev\.runwayml\.com|\brunwayml\b)", re.IGNORECASE)
_FS_WRITE_METHODS: frozenset[str] = frozenset(
    {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "rename", "replace", "rmtree", "remove", "makedirs", "copy", "copy2", "move"}
)


@dataclass(frozen=True)
class InspectionFinding:
    category: str
    severity: Literal["info", "warning", "error"]
    line: int
    message: str


@dataclass
class ScriptInspectionReport:
    ok: bool
    script_path: str
    filename: str
    sha256: str
    size_bytes: int
    line_count: int
    syntax_valid: bool
    classification: ScriptClassification
    expected_backend: str
    imports: list[str] = field(default_factory=list)
    third_party_dependencies: list[str] = field(default_factory=list)
    config_references: list[str] = field(default_factory=list)
    input_output_paths: list[str] = field(default_factory=list)
    network_targets: list[str] = field(default_factory=list)
    generation_loops: list[dict[str, Any]] = field(default_factory=list)
    uses_subprocess: bool = False
    uses_filesystem_writes: bool = False
    uses_network: bool = False
    uses_gpu_ops: bool = False
    uses_dynamic_exec: bool = False
    uses_sensitive_env: bool = False
    uses_native_sdk: bool = False
    has_unbounded_loop: bool = False
    unsupported_apis: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    extraction_summary: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    safety_disclaimer: str = STATIC_SAFETY_DISCLAIMER
    error: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _top_module(name: str) -> str:
    return name.split(".", 1)[0]


def inspect_script_source(
    source: str,
    *,
    script_path: str = "<memory>.py",
    sha256: str | None = None,
) -> ScriptInspectionReport:
    """Statically inspect Python script source via AST without importing or executing it."""
    path_obj = Path(script_path)
    raw_bytes = source.encode("utf-8", errors="replace")
    digest = sha256 or hashlib.sha256(raw_bytes).hexdigest()
    lines = source.splitlines()

    report = ScriptInspectionReport(
        ok=True,
        script_path=str(script_path),
        filename=path_obj.name,
        sha256=digest,
        size_bytes=len(raw_bytes),
        line_count=len(lines),
        syntax_valid=False,
        classification="unsupported",
        expected_backend="unknown",
    )

    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        report.ok = False
        report.syntax_valid = False
        report.error = f"Python syntax error on line {exc.lineno}: {exc.msg}"
        report.findings.append(
            asdict(InspectionFinding("syntax", "error", int(exc.lineno or 0), report.error))
        )
        report.recommended_actions.append("Fix the Python syntax error before importing this script.")
        return report

    report.syntax_valid = True

    imports_seen: list[str] = []
    third_party: list[str] = []
    config_refs: list[str] = []
    io_paths: list[str] = []
    net_targets: list[str] = []
    unsupported_apis: list[str] = []
    findings: list[InspectionFinding] = []

    has_a1111_signals = False
    has_comfy_signals = False
    has_runway_signals = False
    has_diffusers_signals = False

    def _add_unique(lst: list[str], item: str) -> None:
        if item and item not in lst:
            lst.append(item)

    for node in ast.walk(tree):
        lineno = int(getattr(node, "lineno", 0))

        # --- Imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                top = _top_module(mod)
                _add_unique(imports_seen, mod)
                if mod in _NATIVE_SDK_MODULES or top == "avatar_v2_sdk":
                    report.uses_native_sdk = True
                elif top not in _STDLIB_MODULES:
                    _add_unique(third_party, top)
                if top in _NETWORK_MODULES:
                    report.uses_network = True
                    findings.append(InspectionFinding("network", "warning", lineno, f"Imports network module '{mod}'."))
                if top in _SUBPROCESS_MODULES:
                    report.uses_subprocess = True
                    findings.append(InspectionFinding("subprocess", "warning", lineno, f"Imports subprocess module '{mod}'."))
                if top in _GPU_MODULES:
                    report.uses_gpu_ops = True
                    if top == "diffusers":
                        has_diffusers_signals = True
                    findings.append(InspectionFinding("gpu", "warning", lineno, f"Imports direct GPU/ML module '{mod}'."))
                if top in {"ctypes", "cffi"}:
                    report.uses_dynamic_exec = True
                    findings.append(InspectionFinding("unsafe_exec", "error", lineno, f"Imports native FFI module '{mod}'."))
                if "a1111" in mod.lower():
                    has_a1111_signals = True

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = _top_module(mod)
            full_ref = ("." * node.level) + mod
            _add_unique(imports_seen, full_ref or ".")
            if mod in _NATIVE_SDK_MODULES or top == "avatar_v2_sdk":
                report.uses_native_sdk = True
            elif node.level == 0 and top and top not in _STDLIB_MODULES:
                _add_unique(third_party, top)
            if top in _NETWORK_MODULES:
                report.uses_network = True
                findings.append(InspectionFinding("network", "warning", lineno, f"Imports from network module '{mod}'."))
            if top in _SUBPROCESS_MODULES:
                report.uses_subprocess = True
                findings.append(InspectionFinding("subprocess", "warning", lineno, f"Imports from subprocess module '{mod}'."))
            if top in _GPU_MODULES:
                report.uses_gpu_ops = True
                if top == "diffusers":
                    has_diffusers_signals = True
                findings.append(InspectionFinding("gpu", "warning", lineno, f"Imports from GPU/ML module '{mod}'."))
            if "a1111" in mod.lower() or any("a1111" in a.name.lower() for a in node.names):
                has_a1111_signals = True
            if top == "importlib":
                report.uses_dynamic_exec = True
                findings.append(InspectionFinding("unsafe_exec", "warning", lineno, "Uses dynamic import module 'importlib'."))

        # --- Calls ---
        elif isinstance(node, ast.Call):
            callee = _dotted_name(node.func)
            short = callee.rsplit(".", 1)[-1]
            if callee in _DYNAMIC_EXEC_CALLS or short in _DYNAMIC_EXEC_CALLS or callee == "importlib.import_module":
                report.uses_dynamic_exec = True
                findings.append(InspectionFinding("unsafe_exec", "error", lineno, f"Calls dynamic execution primitive '{callee}'."))
            if callee in {"os.system", "os.popen", "os.execv", "os.execve", "os.spawnv", "os.fork"} or callee.startswith("subprocess."):
                report.uses_subprocess = True
                findings.append(InspectionFinding("subprocess", "warning", lineno, f"Invokes external process via '{callee}'."))
            if short in _FS_WRITE_METHODS or callee.startswith("shutil."):
                report.uses_filesystem_writes = True
                findings.append(InspectionFinding("filesystem", "warning", lineno, f"Performs filesystem mutation via '{callee}'."))
            if short == "open":
                mode = ""
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        mode = kw.value.value
                if any(ch in mode for ch in ("w", "a", "x", "+")):
                    report.uses_filesystem_writes = True
                    findings.append(InspectionFinding("filesystem", "warning", lineno, f"Opens file for writing (mode='{mode}')."))
            if "cuda" in callee.lower() or short in {"empty_cache", "ipc_collect"}:
                report.uses_gpu_ops = True
                findings.append(
                    InspectionFinding("gpu", "warning", lineno, f"Directly calls GPU memory op '{callee}' (Avatar V2 manages GPU centrally).")
                )
            if callee in {"os.getenv", "os.environ.get"} and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    key = node.args[0].value.upper()
                    if any(tok in key for tok in _SENSITIVE_ENV_TOKENS):
                        report.uses_sensitive_env = True
                        findings.append(
                            InspectionFinding("credentials", "error", lineno, f"Reads potentially sensitive environment variable '{node.args[0].value}'.")
                        )

        # --- Attribute access (os.environ) ---
        elif isinstance(node, ast.Subscript):
            owner = _dotted_name(node.value)
            if owner == "os.environ":
                key_text = ""
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    key_text = node.slice.value
                if any(tok in key_text.upper() for tok in _SENSITIVE_ENV_TOKENS):
                    report.uses_sensitive_env = True
                    findings.append(
                        InspectionFinding("credentials", "error", lineno, f"Accesses sensitive os.environ['{key_text}'].")
                    )

        # --- Loops ---
        elif isinstance(node, ast.While):
            unbounded = isinstance(node.test, ast.Constant) and bool(node.test.value) is True
            report.generation_loops.append({"kind": "while", "line": lineno, "unbounded": unbounded})
            if unbounded:
                report.has_unbounded_loop = True
                findings.append(
                    InspectionFinding(
                        "loop",
                        "warning",
                        lineno,
                        "Contains unbounded 'while True:' loop; host-enforced generation limit (default 10 items) will cap runs.",
                    )
                )
        elif isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call) and _dotted_name(node.iter.func) == "range":
                bound = None
                if node.iter.args and isinstance(node.iter.args[-1], ast.Constant) and isinstance(node.iter.args[-1].value, int):
                    bound = int(node.iter.args[-1].value)
                report.generation_loops.append({"kind": "for_range", "line": lineno, "bound": bound})

        # --- String literals (configs, URLs, backend signatures, file paths) ---
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if not val or len(val) > 500:
                continue
            if _CONFIG_EXT_RE.match(val):
                _add_unique(config_refs, val)
            for match in _URL_RE.findall(val):
                _add_unique(net_targets, match)
                report.uses_network = True
                if "127.0.0.1:7860" in match or "localhost:7860" in match:
                    has_a1111_signals = True
                if "127.0.0.1:8188" in match or "localhost:8188" in match:
                    has_comfy_signals = True
            if _A1111_ENDPOINT_RE.search(val):
                has_a1111_signals = True
                _add_unique(unsupported_apis, f"Automatic1111 API ({val}) — optional adapter planned for Sprint 5")
            if _COMFY_ENDPOINT_RE.search(val):
                has_comfy_signals = True
                _add_unique(unsupported_apis, "Direct ComfyUI HTTP API access is blocked; use Avatar V2 SDK broker")
            if _RUNWAY_RE.search(val):
                has_runway_signals = True
                _add_unique(unsupported_apis, "Cloud Runway API access is blocked in local-only Script mode")
            if ("/" in val or "\\" in val) and not val.startswith("http") and len(val) <= 180:
                if any(val.startswith(prefix) for prefix in ("/", "./", "../", "~/", "output", "outputs", "data/", "configs/", "prompts/")):
                    _add_unique(io_paths, val)

    # Static structured prompt extraction (always safe, zero execution).
    extraction = extract_prompt_data_from_source(source, source_path=str(script_path))
    report.extraction_summary = {
        "has_extractable_data": extraction.has_extractable_data,
        "categories": list(extraction.categories.keys()),
        "recipes_count": len(extraction.recipes),
        "profiles_count": len(extraction.profiles),
        "templates_count": len(extraction.templates),
        "candidate_pools": list(extraction.candidate_pools.keys()),
        "unextractable_count": len(extraction.unextractable),
        "skipped_legacy_count": len(extraction.skipped_legacy),
        "warnings": extraction.warnings,
    }

    report.imports = imports_seen
    report.third_party_dependencies = third_party
    report.config_references = config_refs
    report.input_output_paths = io_paths[:30]
    report.network_targets = net_targets[:20]
    report.unsupported_apis = unsupported_apis

    # Determine expected backend.
    if report.uses_native_sdk:
        report.expected_backend = "avatar_v2_sdk"
    elif has_a1111_signals:
        report.expected_backend = "a1111"
    elif has_comfy_signals:
        report.expected_backend = "comfyui_direct"
    elif has_runway_signals:
        report.expected_backend = "runway_cloud"
    elif has_diffusers_signals:
        report.expected_backend = "diffusers"
    elif extraction.has_extractable_data:
        report.expected_backend = "declarative_prompt_library"
    else:
        report.expected_backend = "unknown"

    # Classify script into the 5 required categories.
    has_blocking_errors = any(f.severity == "error" for f in findings)
    has_risky_ops = (
        report.uses_subprocess
        or report.uses_network
        or report.uses_filesystem_writes
        or report.uses_gpu_ops
        or report.uses_dynamic_exec
        or report.uses_sensitive_env
    )

    if report.uses_native_sdk and not has_blocking_errors and not unsupported_apis and not report.uses_subprocess and not report.uses_network:
        report.classification = "compatible_native"
        report.recommended_actions.append("Ready to run inside the Bubblewrap sandbox using Avatar V2's local broker.")
    elif has_a1111_signals:
        report.classification = "potentially_compatible_legacy"
        if extraction.has_extractable_data:
            report.recommended_actions.append(
                "Prompt libraries/recipes were safely extracted statically and can run now without executing legacy code."
            )
        report.recommended_actions.append(
            "Direct A1111 HTTP compatibility adapter is scheduled for Sprint 5; migrate to avatar_v2_sdk for native execution."
        )
    elif has_runway_signals or has_diffusers_signals or (has_comfy_signals and not report.uses_native_sdk):
        report.classification = "unsupported"
        if has_runway_signals:
            report.recommended_actions.append("Script mode is local-only and blocks direct Runway cloud access.")
        if has_comfy_signals:
            report.recommended_actions.append("Replace direct ComfyUI HTTP calls with avatar_v2_sdk so ResourcePlanner manages GPU memory.")
        if has_diffusers_signals:
            report.recommended_actions.append("Direct diffusers GPU inference is unsupported inside the sandbox; submit through avatar_v2_sdk.")
    elif extraction.has_extractable_data and not report.uses_native_sdk and not has_risky_ops:
        report.classification = "statically_extractable"
        report.recommended_actions.append(
            "Declarative prompt library/recipes extracted statically; run safely through Avatar V2's deterministic prompt engine."
        )
    elif has_blocking_errors or has_risky_ops:
        report.classification = "requires_review"
        if extraction.has_extractable_data:
            report.recommended_actions.append(
                "Safer alternative: use the statically extracted prompt library without executing script code."
            )
        report.recommended_actions.append(
            "Review flagged operations carefully. Sandbox blocks host network, GPU devices, credentials, and host filesystem writes."
        )
    else:
        report.classification = "unsupported"
        report.recommended_actions.append(
            "Import 'avatar_v2_sdk' to submit requests natively, or define literal prompt lists/recipes for static extraction."
        )

    report.findings = [asdict(f) for f in findings]
    return report


def inspect_script(path: str | Path) -> ScriptInspectionReport:
    """Read and statically inspect a Python script file without executing or modifying it."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return ScriptInspectionReport(
            ok=False,
            script_path=str(file_path),
            filename=file_path.name,
            sha256="",
            size_bytes=0,
            line_count=0,
            syntax_valid=False,
            classification="unsupported",
            expected_backend="unknown",
            error=f"Script file not found: {file_path}",
            recommended_actions=["Select an existing local .py file."],
        )
    if file_path.suffix.lower() != ".py":
        return ScriptInspectionReport(
            ok=False,
            script_path=str(file_path),
            filename=file_path.name,
            sha256="",
            size_bytes=file_path.stat().st_size,
            line_count=0,
            syntax_valid=False,
            classification="unsupported",
            expected_backend="unknown",
            error=f"Unsupported file type '{file_path.suffix}'. Select a Python (.py) file.",
            recommended_actions=["Choose a .py script file."],
        )
    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) > 2 * 1024 * 1024:
        return ScriptInspectionReport(
            ok=False,
            script_path=str(file_path),
            filename=file_path.name,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            size_bytes=len(raw_bytes),
            line_count=0,
            syntax_valid=False,
            classification="unsupported",
            expected_backend="unknown",
            error="Script exceeds 2 MB static inspection limit.",
        )
    source = raw_bytes.decode("utf-8", errors="replace")
    return inspect_script_source(
        source,
        script_path=str(file_path.resolve()),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
