from __future__ import annotations
import importlib

def torch_info() -> dict:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"available": False, "device": "unavailable", "detail": "PyTorch is not installed."}
    if not torch.cuda.is_available():
        return {"available": False, "device": "cpu", "detail": "CUDA is unavailable; CPU fallback will be used where supported."}
    free, total = torch.cuda.mem_get_info()
    return {"available": True, "device": "cuda", "name": torch.cuda.get_device_name(0),
            "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2),
            "detail": "CUDA available."}

def preferred_device(min_free_gib: float = 0) -> str:
    info = torch_info()
    return "cuda" if info["available"] and info.get("free_gib", 0) >= min_free_gib else "cpu"
