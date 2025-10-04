"""Image anonymization helpers."""

import cv2
import numpy as np


def irreversible_params(level):
    level_clamped = int(np.clip(level, 1, 10))
    if level_clamped <= 3:
        return dict(target=16, noise_sigma=8, sp=0.01, q_levels=16, shuffle=0)
    if level_clamped <= 6:
        return dict(target=12, noise_sigma=12, sp=0.02, q_levels=12, shuffle=2)
    if level_clamped <= 8:
        return dict(target=10, noise_sigma=18, sp=0.03, q_levels=10, shuffle=3)
    return dict(target=8, noise_sigma=22, sp=0.04, q_levels=8, shuffle=4)


def quantize(img, levels):
    step = 256 // levels
    return (img // step) * step


def add_noise(img, sigma, sp):
    if sigma > 0:
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if sp > 0:
        mask = np.random.rand(*img.shape[:2])
        img[mask < sp / 2] = 0
        img[mask > 1 - sp / 2] = 255
    return img


def block_shuffle(img, blocks):
    if blocks <= 0:
        return img
    h, w = img.shape[:2]
    gx = max(1, min(blocks, w // 16))
    gy = max(1, min(blocks, h // 16))
    xs = np.array_split(np.arange(w), gx)
    ys = np.array_split(np.arange(h), gy)
    tiles = [[img[y[0]:y[-1] + 1, x[0]:x[-1] + 1] for x in xs] for y in ys]
    idxs = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    np.random.shuffle(idxs)
    out = np.zeros_like(img)
    k = 0
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            src_i, src_j = idxs[k]
            tile = tiles[src_i][src_j]
            oh, ow = y.size, x.size
            tile_resized = cv2.resize(tile, (ow, oh), interpolation=cv2.INTER_NEAREST)
            out[y[0]:y[-1] + 1, x[0]:x[-1] + 1] = tile_resized
            k += 1
    return out


def irreversible_anonymize(roi, level):
    params = irreversible_params(level)
    h, w = roi.shape[:2]
    tw = max(4, min(params['target'], w))
    th = max(4, min(params['target'], h))
    small = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
    small = quantize(small, params['q_levels'])
    small = add_noise(small, params['noise_sigma'], params['sp'])
    small = block_shuffle(small, params['shuffle'])
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = cv2.GaussianBlur(out, (0, 0), 0.7)
    return out


def feather_mask(mask, k):
    if k <= 0:
        return mask
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def draw_square_mask(mask, box, feather):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2
    sx1 = int(np.clip(cx - half, 0, w - 1))
    sy1 = int(np.clip(cy - half, 0, h - 1))
    sx2 = int(np.clip(sx1 + side, sx1 + 1, w))
    sy2 = int(np.clip(sy1 + side, sy1 + 1, h))

    m = np.zeros_like(mask)
    m[sy1:sy2, sx1:sx2] = 255
    if feather > 0:
        m = feather_mask(m, feather)
    mask[:] = np.maximum(mask, m)
    return mask


def expand_box(box, scale_x, scale_y, W, H):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale_x
    h = (y2 - y1) * scale_y
    nx1 = int(np.clip(cx - 0.5 * w, 0, W - 1))
    ny1 = int(np.clip(cy - 0.5 * h, 0, H - 1))
    nx2 = int(np.clip(cx + 0.5 * w, 0, W - 1))
    ny2 = int(np.clip(cy + 0.5 * h, 0, H - 1))
    return [nx1, ny1, nx2, ny2]


def select_boxes(boxes, only_largest, W, H, expand_ratio):
    scale = max(1.0, 1.0 + expand_ratio)
    out = [expand_box(b, scale, scale, W, H) for b in boxes]
    if only_largest and out:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in out]
        k = int(np.argmax(areas))
        out = [out[k]]
    return out


def blend_roi(dst, src, mask):
    m = mask.astype(np.float32) / 255.0
    if len(dst.shape) == 3:
        m = m[..., None]
    out = (dst.astype(np.float32) * (1 - m) + src.astype(np.float32) * m).astype(np.uint8)
    return out
