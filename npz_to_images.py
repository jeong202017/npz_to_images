#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# 기본(가장 흔한) 9개 단일 결함 라벨 이름
DEFAULT_CLASS_NAMES = [
    "Normal", "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-Full", "Scratch", "Random"
]


def pick_array(npz, kind="X"):
    """
    Kaggle npz에서 X(이미지) / Y(라벨)로 보이는 배열을 최대한 자동 추정.
    - X: 3차원이고 (52,52) 같은 2D map이 반복되는 형태를 선호
    - Y: 2차원이고 (N, C) 원핫/멀티핫 형태를 선호
    """
    arrays = {k: npz[k] for k in npz.files}

    if kind == "X":
        # 후보: ndim==3인 것들
        candidates = []
        for k, a in arrays.items():
            if isinstance(a, np.ndarray) and a.ndim == 3:
                candidates.append((k, a))
        if not candidates:
            raise ValueError("npz에서 3차원(X로 보이는) 배열을 찾지 못했습니다. keys=" + str(npz.files))

        # 52x52가 포함된 형태를 우선
        def score(item):
            k, a = item
            shp = a.shape
            # (N,52,52)면 점수 높게
            s = 0
            if len(shp) == 3:
                if shp[1] == 52 and shp[2] == 52:
                    s += 100
                if shp[0] == 52 and shp[1] == 52:
                    s += 80
                # N이 가장 큰 축일 가능성
                s += max(shp) / 1000.0
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0][0], candidates[0][1]

    if kind == "Y":
        candidates = []
        for k, a in arrays.items():
            if isinstance(a, np.ndarray) and a.ndim == 2:
                candidates.append((k, a))
        if not candidates:
            # 라벨이 없는 npz도 있을 수 있음
            return None, None

        # (N, C) 중 C가 5~100 사이 정도면 라벨일 확률 큼
        def score(item):
            k, a = item
            n, c = a.shape
            s = 0
            if 5 <= c <= 200:
                s += 100
            # N이 큰 축이면 보통 (N,C)
            if n > c:
                s += 10
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0][0], candidates[0][1]

    raise ValueError("kind must be 'X' or 'Y'")


def normalize_X_shape(X):
    """
    X가 (N, H, W) 형태가 아니면 최대한 그 형태로 변환.
    흔한 경우:
    - (N,52,52) OK
    - (52,52,N) -> (N,52,52)
    """
    if X.ndim != 3:
        raise ValueError(f"X ndim must be 3, got {X.ndim}")

    # 이미 (N,H,W)로 보이면 그대로
    if X.shape[1] == 52 and X.shape[2] == 52:
        return X

    # (52,52,N) 형태면 transpose
    if X.shape[0] == 52 and X.shape[1] == 52:
        return np.transpose(X, (2, 0, 1))

    # 그 외: 가장 큰 축을 N으로 가정하고 H,W는 나머지
    n_axis = int(np.argmax(X.shape))
    axes = [n_axis] + [ax for ax in range(3) if ax != n_axis]
    X2 = np.transpose(X, axes)

    return X2


def load_class_names(path_or_json, num_classes):
    """
    - 기본: DEFAULT_CLASS_NAMES 사용
    - 또는 --class_names_json 로 JSON 파일/문자열 제공 가능
      예) '["Normal","Center",...]' 또는 /path/to/classes.json
    """
    if path_or_json is None:
        if num_classes == len(DEFAULT_CLASS_NAMES):
            return DEFAULT_CLASS_NAMES
        return [f"class_{i}" for i in range(num_classes)]

    p = Path(path_or_json)
    if p.exists() and p.is_file():
        txt = p.read_text(encoding="utf-8")
        names = json.loads(txt)
    else:
        names = json.loads(path_or_json)

    if len(names) != num_classes:
        raise ValueError(f"class_names 길이({len(names)})가 Y의 클래스 수({num_classes})와 다릅니다.")
    return names


# def label_to_folder(y_vec, class_names, multilabel_join="+", empty_name="Unknown"):
#     """
#     y_vec:
#     - 원핫/멀티핫: 0/1 또는 확률값
#     """
#     if y_vec is None:
#         return empty_name

#     y = np.asarray(y_vec).ravel()
#     # 확률/실수면 0.5 기준으로 멀티핫 판단
#     idx = np.where(y >= 0.5)[0].tolist()

#     if len(idx) == 0:
#         return empty_name

#     names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in idx]
#     # 멀티라벨이면 "A+B" 형태로 폴더 생성
#     return multilabel_join.join(names)

def label_to_folder(y_vec, class_names, empty_name="Unknown"):
    """
    규칙:
    - 라벨 1개만 활성 → 해당 결함명 폴더
    - 라벨 2개 이상 → Mixed 폴더
    """
    if y_vec is None:
        return empty_name

    y = np.asarray(y_vec).ravel()
    idx = np.where(y >= 0.5)[0].tolist()

    if len(idx) == 0:
        return empty_name

    # 🔹 단일 결함
    if len(idx) == 1:
        i = idx[0]
        return class_names[i] if i < len(class_names) else f"class_{i}"

    # 🔹 혼합 결함
    return "Mixed"


def save_one_image(arr2d, out_path, cmap="viridis", dpi=200):
    """
    arr2d: (H,W)
    cmap: 'gray' 추천 가능. 기본은 viridis.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(2, 2))
    plt.imshow(arr2d, interpolation="nearest", cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Convert wafer-map NPZ to PNG/JPG images, optionally grouped by (multi)labels."
    )
    ap.add_argument("--npz", required=True, help="입력 .npz 파일 경로")
    ap.add_argument("--out", required=True, help="출력 폴더 경로")
    ap.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg"], help="저장 포맷")
    ap.add_argument("--limit", type=int, default=-1, help="저장할 최대 샘플 수 (-1이면 전부)")
    ap.add_argument("--cmap", default="gray", help="matplotlib colormap (추천: gray)")
    ap.add_argument("--dpi", type=int, default=200, help="저장 DPI")
    ap.add_argument("--no_label_folders", action="store_true",
                    help="라벨 폴더로 나누지 않고 out 폴더에 전부 저장")
    ap.add_argument("--class_names_json", default=None,
                    help="클래스 이름 JSON 문자열 또는 JSON 파일 경로 (옵션)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(npz_path, allow_pickle=True)

    x_key, X = pick_array(npz, "X")
    y_key, Y = pick_array(npz, "Y")

    X = normalize_X_shape(X)

    N = X.shape[0]
    if args.limit is not None and args.limit > 0:
        N = min(N, args.limit)

    if Y is not None:
        if Y.shape[0] != X.shape[0]:
            # 혹시 transpose가 필요한 경우를 대략 보정
            if Y.shape[1] == X.shape[0]:
                Y = Y.T
            else:
                raise ValueError(f"X와 Y의 샘플 수가 다릅니다. X:{X.shape}, Y:{Y.shape}")

        num_classes = Y.shape[1]
        class_names = load_class_names(args.class_names_json, num_classes)
    else:
        class_names = None

    # 저장 로그
    print(f"[INFO] Loaded NPZ: {npz_path}")
    print(f"[INFO] X key={x_key}, X shape={X.shape}, dtype={X.dtype}")
    if Y is not None:
        print(f"[INFO] Y key={y_key}, Y shape={Y.shape}, dtype={Y.dtype}")
        print(f"[INFO] num_classes={len(class_names)}")
    else:
        print("[INFO] No label array found (Y is None). Saving without label folders.")

    # 메타 저장
    meta = {
        "npz": str(npz_path),
        "x_key": x_key,
        "x_shape": list(X.shape),
        "y_key": y_key,
        "y_shape": list(Y.shape) if Y is not None else None,
        "class_names": class_names,
        "saved_format": args.fmt,
        "cmap": args.cmap,
        "dpi": args.dpi,
        "limit": args.limit,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 저장
    for i in range(N):
        wafer = X[i]

        if (wafer.ndim != 2):
            wafer = np.squeeze(wafer)
        if wafer.ndim != 2:
            raise ValueError(f"wafer[{i}]가 2D가 아닙니다. shape={wafer.shape}")

        if args.no_label_folders or Y is None:
            subdir = out_dir
            folder_name = None
        else:
            folder_name = label_to_folder(Y[i], class_names)
            subdir = out_dir / folder_name

        out_path = subdir / f"wafer_{i:06d}.{args.fmt}"
        save_one_image(wafer, out_path, cmap=args.cmap, dpi=args.dpi)

        if (i + 1) % 1000 == 0:
            print(f"[INFO] saved {i+1}/{N} ...")

    print(f"[DONE] Saved {N} images to: {out_dir}")
    print(f"[DONE] meta.json saved to: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
