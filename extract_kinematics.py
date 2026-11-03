"""Shared helpers of the Lab App: the limbs' MediaPipe points, finding the 5 switches, measuring their lights,
touching, and following small camera movements (used by session_engine.py, session_analysis.py, lab_app.py).
"""
import cv2
import numpy as np

# -----------------------------
# Limb definitions
# -----------------------------
# MediaPipe Pose landmark indices per side. Left/Right are the SUBJECT's own sides.
LIMBS = {
    'upper': {
        'joints': {'left':  {'shoulder': 11, 'elbow': 13, 'wrist': 15},
                   'right': {'shoulder': 12, 'elbow': 14, 'wrist': 16}},
        'segments': [('shoulder', 'elbow'), ('elbow', 'wrist')],
        'hand': True,
        # Pose points that count for touching a switch (wrist, pinky, index, thumb) - plus the 21 hand points
        'touch_points': {'left': [15, 17, 19, 21], 'right': [16, 18, 20, 22]},
    },
    'lower': {
        'joints': {'left':  {'hip': 23, 'knee': 25, 'ankle': 27, 'heel': 29, 'toe': 31},
                   'right': {'hip': 24, 'knee': 26, 'ankle': 28, 'heel': 30, 'toe': 32}},
        'segments': [('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'heel'), ('heel', 'toe'), ('ankle', 'toe')],
        'hand': False,
        'touch_points': {'left': [27, 29, 31], 'right': [28, 30, 32]},  # ankle, heel, toe
    },
}


# -----------------------------
# Switch events
# -----------------------------
# The switch boxes are found automatically (or marked by hand) in the Lab App.
# Each box follows small camera movements, its red light is measured every frame, and touching =
# a point of the tracked hand (or foot) inside the box.
N_SWITCHES = 5
LIGHT_BOX_MARGIN = 0.25   # light is measured in the switch box enlarged by 25% (LED + its glow)
# Touching = a hand/foot point inside the switch box or within this fraction of the box size outside it.
# The button sticks out on the subject's side, so fingertips pressing it can sit just outside the box.
TOUCH_DIST = 0.25
MIN_EVENT_S = 0.07        # light changes shorter than this are ignored (noise, arm passing through the beam)
MERGE_GAP_S = 0.15        # a light hidden for less than this (hand/foot passing over the LED) is one event
# A switch whose light level between events is above this fraction of its level during events is lit all
# the time (blinking switches: <= 0.15, always-lit: >= 0.58); its "events" are only the hand passing by.
MAX_OFF_ON_RATIO = 0.35
LIGHT_ON_FRACTION = 0.2   # light counts as on above 20% of the way from its off level to its on level
# Slow changes (daylight, camera moving, glow of a neighbouring switch) are removed by subtracting the
# light level's running median over this window. Light events must be shorter than half of it (~0.5 s here).
BASELINE_WINDOW_S = 4.0
# On must stand out from off by this many times the off-level noise. Blinking switches score ~30-1500,
# steady (always-lit or never-lit) ones ~2-13. (A ratio test failed for switches on a reddish background.)
MIN_LIGHT_CONTRAST = 20.0


# -----------------------------
# Automatic switch finding
# -----------------------------
# A switch = black housing (white label on top) + red mushroom button, and its pink LED lights up during
# the video. One pass over the video (about as long as decoding it) builds a background image (median of
# frames aligned to the first one, so the subject mostly disappears) and a map of where pink light appeared.
SCAN_SCALE = 0.5            # the scan works on a half-size copy
SCAN_PER_S = 4              # frames per second checked for pink light
SCAN_BACKGROUND_FRAMES = 20  # frames in the median background
SWITCH_BOX_PAD = 0.1        # found boxes are enlarged by 10% of their size on each side


def _disk(d):
    """A round "brush" of diameter d pixels, used to grow, shrink or smooth shapes in a mask.

    Input: d - diameter in pixels. Returns: a small 0/1 image of a filled circle.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _fill_holes(mask):
    """Fill the holes inside every shape of a mask (e.g. a ring becomes a full disk).

    Input: mask - a 0/1 image. Returns: a new 0/1 image with each outer shape filled in.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 1, -1)
    return out


def _pink(img):
    """(mask of the pixels lit by a switch's pink LED, R - max(G, B) per pixel) of an 8-bit BGR image.

    Lit LED/glow pixels are bright pink: red near saturation, green and blue clearly lower but still high, blue
    not below green. On the 8-bit channels (no wider copy - this runs 5 times per 4K frame): R - max(G, B) and
    G - B saturate at 0, which changes nothing for the pink pixels (excess > 20, B >= G - 15 <=> G - B <= 15)."""
    B, G, R = cv2.split(img)
    excess = cv2.subtract(R, cv2.max(G, B))
    return (R > 200) & (G > 90) & (cv2.subtract(G, B) <= 15) & (excess > 20), excess


def pink_mask(frame):
    """Pixels lit by a switch's pink LED (same colour test as light_energy)."""
    return _pink(frame)[0]


def scan_video(cap, first_frame, total_frames, fps, show, start_frame=0, max_frames=None, should_stop=None):
    """(background image, fraction of checked frames each pixel was pink), both half size, first-frame coords.
    start_frame / max_frames: only scan that part of the video (a long session: a few minutes with lights)."""
    camera = CameraMotion(first_frame)
    h, w = first_frame.shape[:2]
    sw, sh = int(w * SCAN_SCALE), int(h * SCAN_SCALE)
    step = max(1, int(round(fps / SCAN_PER_S)))
    span = min(total_frames - start_frame, max_frames or total_frames)
    bg_every = max(1, span // (step * SCAN_BACKGROUND_FRAMES))
    pink_count = np.zeros((sh, sw), np.float32)
    background, checked = [], 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    i = 0
    while i < span and cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            camera.update(frame)
            M = cv2.invertAffineTransform(camera.M)  # current frame -> first frame
            M[:, 2] *= SCAN_SCALE
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            aligned = cv2.warpAffine(small, M, (sw, sh), borderMode=cv2.BORDER_REPLICATE)
            pink_count += pink_mask(aligned)
            if checked % bg_every == 0:
                background.append(aligned)
            checked += 1
            if checked % 20 == 0:
                show(i / max(1, span))
                if should_stop and should_stop():
                    break
        i += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(background), axis=0).astype(np.uint8), pink_count / max(1, checked)


def switch_candidates(img, size=1.0):
    """Boxes (pixels) of black housings with a red button and a white label. `size` scales the
    clean-up kernels: 1.0 for near switches, 0.5 for small far-away ones."""
    k = lambda d: max(3, int(d * size) | 1)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    red = (((H < 8) | (H > 170)) & (S > 130) & (V > 70)).astype(np.uint8)
    white = (V > 150) & (S < 70)
    dark = cv2.morphologyEx((V < 70).astype(np.uint8), cv2.MORPH_OPEN, _disk(k(11)))  # drops cables
    # Housing + button as one blob (the button splits the housing's dark front face), label holes filled
    body = _fill_holes(cv2.morphologyEx(dark | red, cv2.MORPH_CLOSE, _disk(k(9))))
    # Shrink, split, grow back: separates switches that touch in the image
    n, labels, _, _ = cv2.connectedComponentsWithStats(cv2.erode(body, _disk(k(21))))
    out = []
    for i in range(1, n):
        region = cv2.dilate((labels == i).astype(np.uint8), _disk(k(21))) & body
        area = int(region.sum())
        x, y, w, h = cv2.boundingRect(region)
        if area < 150 * size * size or area > 0.02 * img.shape[0] * img.shape[1]:
            continue
        if area / (w * h) < 0.45 or max(w, h) / min(w, h) > 3:  # compact, box-like
            continue
        inside = region > 0
        if not 0.015 <= red[inside].mean() <= 0.5 or dark[inside].mean() < 0.4:
            continue
        if white[max(0, y - h // 2):y + h, x:x + w].mean() < 0.01:  # label (on top, may be above the front)
            continue
        out.append((x, y, x + w, y + h))
    return out


def _overlap(a, b):
    """Intersection area / area of the smaller box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def order_switches(centers):
    """Indices of the switches from the subject's RIGHT to LEFT.

    The switches stand in an arc around the subject: follow the arc from one end to the other, then
    pick the direction from the way the arc bulges (the subject faces it; camera looking down)."""
    c = np.asarray(centers, float)
    n = len(c)
    if n < 2:
        return list(range(n))
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    a, b = np.unravel_index(np.argmax(d), d.shape)  # the two ends of the arc
    chain, rest = [int(a)], set(range(n)) - {int(a)}
    while rest:
        nxt = min(rest, key=lambda j: d[chain[-1], j])
        chain.append(nxt)
        rest.remove(nxt)
    if n >= 3:
        facing = c[chain[1:-1]].mean(axis=0) - (c[a] + c[b]) / 2
        right = np.array([-facing[1], facing[0]])  # image y points down
        if np.dot(c[chain[-1]] - c[chain[0]], right) > 0:
            chain.reverse()
    return chain


def find_switches(background, pink_frac):
    """Up to 5 switch boxes as [x0, y0, x1, y1] fractions, numbered from the subject's right to left."""
    h, w = background.shape[:2]
    boxes = switch_candidates(background, 1.0) + switch_candidates(background, 0.5)

    def light(box):  # share of the area around the switch where its pink LED/glow appeared
        """Score a candidate box by its light: the share of the area around it where pink LED light was seen.

        Input: box - [x0, y0, x1, y1] in pixels. Returns: a number from 0 (never lit) to 1.
        """
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        return float((pink_frac[max(0, y0 - bh):y1 + bh, max(0, x0 - bw):x1 + bw] > 0.02).mean())

    kept = []
    for box in sorted(boxes, key=light, reverse=True):
        if light(box) > 0.005 and all(_overlap(box, k) <= 0.3 for k in kept):
            kept.append(box)
    kept = kept[:N_SWITCHES]
    fracs = []
    for x0, y0, x1, y1 in kept:
        px, py = (x1 - x0) * SWITCH_BOX_PAD, (y1 - y0) * SWITCH_BOX_PAD
        fracs.append([round(max(0, x0 - px) / w, 4), round(max(0, y0 - py) / h, 4),
                      round(min(w, x1 + px) / w, 4), round(min(h, y1 + py) / h, 4)])
    return sort_switches(fracs)


def sort_switches(boxes, flip=False):
    """Put switch boxes in their standard order: switch 1, 2, 3 ... from the subject's right to left.

    Inputs: boxes - switch boxes as [x0, y0, x1, y1] fractions of the frame; flip - True to reverse the order
    (used when the automatic order came out mirrored).
    Returns: the same boxes, re-ordered.
    """
    order = order_switches([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes])
    boxes = [boxes[i] for i in order]
    return boxes[::-1] if flip else boxes


def box_corners(box, frame_w, frame_h, margin):
    """Corners of a fractional [x0, y0, x1, y1] box in pixels, enlarged by `margin` of its size."""
    x0, y0, x1, y1 = box[0] * frame_w, box[1] * frame_h, box[2] * frame_w, box[3] * frame_h
    mx, my = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
    return np.float32([[x0 - mx, y0 - my], [x1 + mx, y0 - my], [x1 + mx, y1 + my], [x0 - mx, y1 + my]])


def box_size(box, frame_w, frame_h):
    """Size of a fractional switch box in pixels (square root of its area): the unit of touch distances."""
    return float(np.sqrt((box[2] - box[0]) * frame_w * (box[3] - box[1]) * frame_h))


def switch_geometry(boxes, frame_w, frame_h):
    """Per switch box, in first-frame pixels: the corners where its light is measured (enlarged by
    LIGHT_BOX_MARGIN), the corners touching is measured against, and its size."""
    return ([box_corners(b, frame_w, frame_h, LIGHT_BOX_MARGIN) for b in boxes],
            [box_corners(b, frame_w, frame_h, 0.0) for b in boxes],
            np.array([box_size(b, frame_w, frame_h) for b in boxes]))


def touch_distance(poly, points, size):
    """Distance of the closest point (x, y) to the polygon (negative = inside), in box sizes; inf if no points."""
    return min((-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in points),
               default=np.inf) / size


def to_frame(pts, M):
    """First-frame points (..., 2) -> a frame's coordinates (M: the camera movement, first frame -> that frame)."""
    return np.asarray(pts) @ M[:, :2].T + M[:, 2]


def to_first_frame(pts, M):
    """A frame's points -> first-frame coordinates (the camera movement undone). M: (2, 3) with points (..., 2),
    or one matrix per frame (n, 2, 3) with points (n, 2) or (n, k, 2)."""
    M = np.asarray(M, float)
    A, t = M[..., :2], M[..., 2]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    inv = np.stack([np.stack([A[..., 1, 1], -A[..., 0, 1]], -1), np.stack([-A[..., 1, 0], A[..., 0, 0]], -1)], -2)
    inv = inv / det[..., None, None]
    pts = np.asarray(pts, float)
    if M.ndim == 2:
        return (pts - t) @ inv.T
    if pts.ndim == 2:
        return np.einsum('nij,nj->ni', inv, pts - t)
    return np.einsum('nij,nkj->nki', inv, pts - t[:, None])


class CameraMotion:
    """Follows small camera movements: maps first-frame coordinates to the current frame.

    Matches ORB features of each frame to the first frame (no drift) and fits a rotation +
    shift + scale; the moving subject is rejected as outliers by RANSAC."""
    SCALE = 0.25  # features are found on a quarter-size copy (fast, enough for whole-frame motion)

    def __init__(self, first_frame):
        """Remember the first frame's features; all later frames are compared to it.

        Input: first_frame - the video's first color frame.
        """
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self._small_gray(first_frame), None)
        self.M = np.float32([[1, 0, 0], [0, 1, 0]])

    def _small_gray(self, frame):
        """A small grayscale copy of a frame (features are found faster on it).

        Input: frame - a color frame. Returns: the shrunken gray image.
        """
        small = cv2.resize(frame, None, fx=self.SCALE, fy=self.SCALE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def update(self, frame):
        """Measure how much the camera moved in this frame compared with the first frame, and remember it.
        If the frame can't be matched well enough, the last known movement is kept.

        Input: frame - the current color frame. Returns: nothing (updates self.M).
        """
        kp, des = self.orb.detectAndCompute(self._small_gray(frame), None)
        if des is None or self.ref_des is None or len(kp) < 10:
            return  # keep the last known camera position
        matches = self.matcher.match(self.ref_des, des)
        if len(matches) < 15:
            return
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]) / self.SCALE
        dst = np.float32([kp[m.trainIdx].pt for m in matches]) / self.SCALE
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=12)
        if M is not None and inliers is not None and inliers.sum() >= 15:
            self.M = M.astype(np.float32)

    def apply(self, pts):
        """Move first-frame points to where they are in the current frame (using the last update()).

        Input: pts - points (x, y) in first-frame pixels. Returns: the same points in current-frame pixels (N x 2).
        """
        return cv2.transform(pts.reshape(-1, 1, 2), self.M).reshape(-1, 2)


def light_energy(frame, corners):
    """Amount of pink LED light (the LED and its glow) inside the polygon's bounding box, per pixel.

    The pink test (_pink) leaves out the deep-red mushroom button (low green/blue), the white table (R = G = B)
    and skin (orange: blue well below green), which all sit in or near the box."""
    h, w = frame.shape[:2]
    x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, [w, h])
    x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, [w, h])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    pink, excess = _pink(frame[y0:y1, x0:x1])
    return float(excess[pink].sum(dtype=np.int64)) * 100 / ((x1 - x0) * (y1 - y0))


def otsu_threshold(values):
    """Value that best splits the samples into two groups (Otsu's method on a 256-bin histogram)."""
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]


def find_light_events(signal, fps):
    """(first ON frame, first OFF frame or None) for each time the light was on, or ([], reason)."""
    raw = np.asarray(signal, dtype=float)
    if len(raw) < 3 or raw.max() - raw.min() < 1e-6:
        return [], 'light never changed'
    # Mirrored padding, so a light that is on at the very start or end doesn't lift the baseline there
    half = max(1, int(BASELINE_WINDOW_S * fps) // 2)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(raw, half, mode='reflect'), 2 * half + 1)
    s = raw - np.median(windows, axis=1)
    thr = otsu_threshold(s)
    low, high = s[s <= thr], s[s > thr]
    # On/off must be clearly different relative to the off-level noise (robust spread: scaled MAD), so an
    # unlit or always-lit switch with only small changes (drift, arm passing) gives no events
    if len(high) == 0:
        return [], 'no clear on/off changes of the light'
    noise = max(1.4826 * np.median(np.abs(low - np.median(low))), 1e-6)
    if (np.median(high) - np.median(low)) / noise < MIN_LIGHT_CONTRAST:
        return [], 'no clear on/off changes of the light'
    # Final on/off threshold close to the off level, so a light partly hidden by the hand/foot still counts as on
    on = s > np.median(low) + LIGHT_ON_FRACTION * (np.median(high) - np.median(low))
    min_len = max(2, int(round(MIN_EVENT_S * fps)))
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(on)])
    merged = []
    max_gap = max(min_len, int(round(MERGE_GAP_S * fps)))
    for r in runs:  # join an event the hand/foot briefly hid
        if merged and r[0] - merged[-1][1] < max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    # Real events reach the light's full level at some point; weaker bumps are the hand/foot passing by
    full = np.median(low) + 0.5 * (np.median(high) - np.median(low))
    merged = [r for r in merged if r[1] - r[0] >= min_len and s[r[0]:r[1]].max() >= full]
    if not merged:
        return [], 'no clear on/off changes of the light'
    lit = np.zeros(len(raw), bool)
    for a, b in merged:
        lit[a:b] = True
    if lit.all() or np.median(raw[~lit]) > MAX_OFF_ON_RATIO * np.median(raw[lit]):
        return [], 'light is on the whole time (only brief changes, e.g. the hand passing)'
    return [(a, b if b < len(raw) else None) for a, b in merged], ''
