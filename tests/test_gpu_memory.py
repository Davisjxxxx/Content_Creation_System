from avatar_v2 import gpu_memory
from avatar_v2.gpu_memory import GPUMemoryManager, GPUSnapshot, parse_nvidia_smi_memory


def test_parse_multiple_gpu_memory_rows():
    rows = "0, NVIDIA GeForce RTX 4070, 10240, 12282\n1, GeForce GTX 1080 Ti, 2048, 11264\n"
    result = parse_nvidia_smi_memory(rows)
    assert len(result) == 2
    assert result[0].index == 0
    assert result[0].name == "NVIDIA GeForce RTX 4070"
    assert result[0].used_mb == 10240
    assert result[0].total_mb == 12282
    assert result[1].index == 1


def test_memory_manager_threshold_cleanup(monkeypatch):
    snapshots = [GPUSnapshot(index=0, name="RTX 4070", used_mb=11000, total_mb=12000)]
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: snapshots)

    class Provider:
        def __init__(self):
            self.calls = 0

        def free_memory(self, *, unload_models=True, free_memory=True):
            assert unload_models is True
            assert free_memory is True
            self.calls += 1

    provider = Provider()
    manager = GPUMemoryManager(threshold=0.85, check_interval_s=1, enabled=True)
    result = manager.maybe_cleanup(provider, reason="test", wait_s=0)
    assert result["cleanup_requested"] is True
    assert provider.calls == 1


def test_memory_manager_does_not_cleanup_below_threshold(monkeypatch):
    snapshots = [GPUSnapshot(index=0, name="RTX 4070", used_mb=4000, total_mb=12000)]
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: snapshots)

    class Provider:
        def __init__(self):
            self.calls = 0

        def free_memory(self, **kwargs):
            self.calls += 1

    provider = Provider()
    manager = GPUMemoryManager(threshold=0.85, enabled=True)
    result = manager.maybe_cleanup(provider, reason="test", wait_s=0)
    assert result["cleanup_requested"] is False
    assert provider.calls == 0


def test_manual_forced_cleanup_works_when_auto_cleanup_is_disabled(monkeypatch):
    snapshots = [GPUSnapshot(index=0, name="RTX 4070", used_mb=4000, total_mb=12000)]
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: snapshots)

    class Provider:
        calls = 0

        def free_memory(self, **kwargs):
            self.calls += 1

    provider = Provider()
    result = GPUMemoryManager(threshold=0.85, enabled=False).maybe_cleanup(
        provider, reason="manual-gpu-free", force=True, wait_s=0
    )
    assert result["enabled"] is False
    assert result["forced"] is True
    assert result["cleanup_requested"] is True
    assert provider.calls == 1


def test_memory_sensitive_profiles_force_cleanup_below_threshold(monkeypatch):
    from avatar_v2.render_queue import MEMORY_SENSITIVE_PROFILES

    snapshots = [GPUSnapshot(index=0, name="RTX 4070", used_mb=900, total_mb=8188)]
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: snapshots)

    class Provider:
        def __init__(self):
            self.calls = 0

        def free_memory(self, *, unload_models=True, free_memory=True):
            assert unload_models is True
            assert free_memory is True
            self.calls += 1

    assert MEMORY_SENSITIVE_PROFILES == {"int8_12gb", "low_memory", "low_memory_quality"}
    for profile in sorted(MEMORY_SENSITIVE_PROFILES):
        provider = Provider()
        result = GPUMemoryManager(threshold=0.85, enabled=True).maybe_cleanup(
            provider,
            reason=f"pre-render:{profile}",
            force=True,
            wait_s=0,
        )
        assert result["forced"] is True
        assert result["cleanup_requested"] is True
        assert provider.calls == 1


def test_forced_cleanup_waits_for_actual_memory_drop(monkeypatch):
    sequence = iter([
        [GPUSnapshot(index=0, name="RTX 4070", used_mb=6900, total_mb=8188)],
        [GPUSnapshot(index=0, name="RTX 4070", used_mb=6900, total_mb=8188)],
        [GPUSnapshot(index=0, name="RTX 4070", used_mb=980, total_mb=8188)],
        [GPUSnapshot(index=0, name="RTX 4070", used_mb=980, total_mb=8188)],
    ])
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: next(sequence))
    clock = iter([0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr(gpu_memory.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(gpu_memory.time, "sleep", lambda _: None)

    class Provider:
        def free_memory(self, **kwargs):
            pass

    result = GPUMemoryManager(enabled=True).maybe_cleanup(
        Provider(), reason="manual", force=True, wait_s=60
    )
    assert result["after"][0]["used_mb"] == 980
    assert result["released_mb"] == 5920
    assert result["effective"] is True


def test_emergency_terminate_requires_explicit_pid(monkeypatch):
    monkeypatch.setattr(
        gpu_memory,
        "query_gpu_processes",
        lambda: [{"pid": 1234, "name": "python.exe", "memory_mb": 8000}],
    )
    killed = []
    monkeypatch.setattr(gpu_memory.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(gpu_memory.os, "getpid", lambda: 9999)

    manager = GPUMemoryManager()
    result = manager.emergency_terminate([])
    assert killed == []
    assert result["terminated"] == []

    result = manager.emergency_terminate([1234])
    assert killed == [1234]
    assert result["terminated"][0]["pid"] == 1234
