import numpy as np

OUT_DTYPE = np.float32


def load_npy(path):
    a = np.load(path)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return np.ascontiguousarray(a, dtype=np.float32)


def save_npy(path, arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0).astype(OUT_DTYPE, copy=False)
    np.save(path, np.ascontiguousarray(arr))
