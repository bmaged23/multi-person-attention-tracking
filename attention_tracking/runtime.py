"""Load pinned CUDA shared libraries without a training framework or system toolkit."""

import ctypes
from pathlib import Path
import site

_HANDLES = []
_LOADED = False


def preload_cuda():
    global _LOADED
    if _LOADED:
        return
    libraries = [
        "cuda_runtime/lib/libcudart.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cufft/lib/libcufft.so.11",
        "curand/lib/libcurand.so.10",
        "cuda_nvrtc/lib/libnvrtc.so.12",
    ]
    pending = []
    for library in libraries:
        path = next(
            (
                Path(folder) / "nvidia" / library
                for folder in site.getsitepackages()
                if (Path(folder) / "nvidia" / library).is_file()
            ),
            None,
        )
        if path is None:
            raise RuntimeError("Install pinned GPU requirements; missing " + library)
        pending.append(path)
    cudnn = pending[0].parents[2] / "cudnn/lib"
    pending.extend(sorted(cudnn.glob("libcudnn*.so.9")))
    if not list(cudnn.glob("libcudnn.so.9")):
        raise RuntimeError("Install pinned cuDNN requirements")
    while pending:
        failed = []
        for path in pending:
            try:
                _HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            except OSError:
                failed.append(path)
        if len(failed) == len(pending):
            raise RuntimeError(
                "CUDA dependency loading failed: " + ", ".join(p.name for p in failed)
            )
        pending = failed
    _LOADED = True
