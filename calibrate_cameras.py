"""Where two cameras are relative to each other (and their lens settings), for 3D in the Lab App.

The calibration itself is done in the Lab App (Session step: calibrate with the board, or by clicking matching
points). This file holds the maths, and prints the calibration board:

    venv/bin/python calibrate_cameras.py make-board
    (outputs/charuco_board.pdf and .png; print the PDF at 100% / actual size on A4, glue it to something flat,
    and measure one black square - it should be 38 mm).
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import video_source as vs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# ChArUco board: 5 x 7 squares of 38 mm on A4 (190 x 266 mm), 4x4 ArUco markers
BOARD_SQUARES = (5, 7)
BOARD_SQUARE_MM = 38.0
BOARD_MARKER_RATIO = 0.75
BOARD_DICT = cv2.aruco.DICT_4X4_50


def save_cameras(path, cam_a, cam_b, R, t, units, method, extra=None):
    """Camera A at the origin; a point X (camera A coordinates) is at R @ X + t in camera B coordinates."""
    data = dict(method=method, units=units,
                camera_a=dict(K=np.asarray(cam_a['K']).tolist(), dist=np.asarray(cam_a['dist']).ravel().tolist(),
                              size=list(map(int, cam_a['size']))),
                camera_b=dict(K=np.asarray(cam_b['K']).tolist(), dist=np.asarray(cam_b['dist']).ravel().tolist(),
                              size=list(map(int, cam_b['size'])),
                              R=np.asarray(R).tolist(), t=np.asarray(t).ravel().tolist()),
                **(extra or {}))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {path}')


def load_cameras(path):
    with open(path) as f:
        d = json.load(f)
    a, b = d['camera_a'], d['camera_b']
    cams = [dict(K=np.array(a['K']), dist=np.array(a['dist']), R=np.eye(3), t=np.zeros(3), size=a['size']),
            dict(K=np.array(b['K']), dist=np.array(b['dist']), R=np.array(b['R']), t=np.array(b['t']), size=b['size'])]
    return cams, d['units']


def K_matrix(f, size):
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], float)


# -----------------------------
# Clicks
# -----------------------------
def calibrate_from_points(pa, pb, size_a, size_b, f_a, f_b, refine_focal=None):
    """Relative pose (R, t with |t| = 1) and focal lengths from matched points, plus the 3D points and
    the reprojection errors (px)."""
    n = len(pa)
    if refine_focal is None:
        refine_focal = n >= 20
    na = (pa - np.array(size_a) / 2) / f_a
    nb = (pb - np.array(size_b) / 2) / f_b
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.LMEDS)
    best = None
    for Ei in E.reshape(-1, 3, 3):  # the 5-point method can return several solutions
        good, R, t, _ = cv2.recoverPose(Ei, na, nb, np.eye(3))
        if best is None or good > best[0]:
            best = (good, R, t.ravel())
    _, R, t = best

    def unpack(p):  # one focal scale shared by both cameras (similar lenses): steadier with few points
        fa, fb = (f_a * p[0], f_b * p[0]) if refine_focal else (f_a, f_b)
        q = p[1:] if refine_focal else p
        return fa, fb, cv2.Rodrigues(q[:3])[0], q[3:6] / np.linalg.norm(q[3:6]), q[6:].reshape(n, 3)

    def residuals(p):
        fa, fb, Rb, tb, X = unpack(p)
        ra = project(X, K_matrix(fa, size_a), np.zeros(5), np.eye(3), np.zeros(3)) - pa
        rb = project(X, K_matrix(fb, size_b), np.zeros(5), Rb, tb) - pb
        return np.concatenate([ra.ravel(), rb.ravel()])

    X0 = triangulate_px(pa, pb, K_matrix(f_a, size_a), K_matrix(f_b, size_b), R, t)
    p0 = np.concatenate([[1.0] if refine_focal else [], cv2.Rodrigues(R)[0].ravel(), t, X0.ravel()])
    res = least_squares(residuals, p0, x_scale='jac', loss='soft_l1', f_scale=5.0)
    fa, fb, R, t, X = unpack(res.x)
    err = np.hypot(*res.fun.reshape(2, n, 2).transpose(2, 0, 1)).max(axis=0)
    return fa, fb, R, t, X, err


def project(X, K, dist, R, t):
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), cv2.Rodrigues(R)[0], np.asarray(t, float),
                             K, np.asarray(dist, float))
    return p.reshape(-1, 2)


def triangulate_px(pa, pb, Ka, Kb, R, t, dist_a=np.zeros(5), dist_b=np.zeros(5)):
    """3D points (camera A coordinates) from pixel positions in both views (lens distortion removed first)."""
    na = cv2.undistortPoints(np.asarray(pa, np.float64).reshape(-1, 1, 2), Ka, dist_a).reshape(-1, 2)
    nb = cv2.undistortPoints(np.asarray(pb, np.float64).reshape(-1, 1, 2), Kb, dist_b).reshape(-1, 2)
    X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, np.reshape(t, (3, 1))]),
                              na.T, nb.T)
    return (X[:3] / X[3]).T


# -----------------------------
# Board
# -----------------------------
def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_MM, BOARD_SQUARE_MM * BOARD_MARKER_RATIO, dictionary)
    return board, cv2.aruco.CharucoDetector(board)


A4_MM = (210.0, 297.0)
BOARD_TEXT = f'Print at 100% / Actual size \u2014 each square must measure {BOARD_SQUARE_MM:.0f} mm'
CHECK_BAR_MM = 100.0


def _layout():
    """Board, text and check bar positions on the A4 page (mm from the top-left corner)."""
    bw, bh = BOARD_SQUARES[0] * BOARD_SQUARE_MM, BOARD_SQUARES[1] * BOARD_SQUARE_MM
    x0, y0 = (A4_MM[0] - bw) / 2, (A4_MM[1] - bh) / 2
    return dict(board=(x0, y0, bw, bh), text_y=y0 / 2 + 1.5, bar=(x0, y0 + bh + (A4_MM[1] - y0 - bh) / 2, CHECK_BAR_MM))


def board_image(px_per_square):
    board, _ = make_board()
    return board.generateImage((BOARD_SQUARES[0] * px_per_square, BOARD_SQUARES[1] * px_per_square), marginSize=0)


def write_board_png(path, dpi=300):
    px = dpi / 25.4
    lay = _layout()
    page = np.full((int(round(A4_MM[1] * px)), int(round(A4_MM[0] * px))), 255, np.uint8)
    x0, y0, bw, bh = lay['board']
    img = board_image(int(round(BOARD_SQUARE_MM * px)))
    X, Y = int(round(x0 * px)), int(round(y0 * px))
    page[Y:Y + img.shape[0], X:X + img.shape[1]] = img
    cv2.putText(page, BOARD_TEXT.replace('\u2014', '-'), (X, int(lay['text_y'] * px)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    bx, by, blen = lay['bar']
    x1, x2, yb = int(round(bx * px)), int(round((bx + blen) * px)), int(round(by * px))
    cv2.rectangle(page, (x1, yb - int(0.6 * px)), (x2, yb + int(0.6 * px)), 0, -1)
    for k in range(11):
        xt = int(round((bx + 10 * k) * px))
        cv2.line(page, (xt, yb - int((2.5 if k % 5 == 0 else 1.5) * px)), (xt, yb), 0, 2)
    cv2.putText(page, f'{CHECK_BAR_MM:.0f} mm check bar', (x2 + int(3 * px), yb + int(1 * px)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.imwrite(path, page, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def write_board_pdf(path, px_per_mm=20):
    """One A4 page (595.276 x 841.890 pt): the board as an exact-size image, the text and a 100 mm check bar
    as vector lines - printed at 100% / actual size, every square measures BOARD_SQUARE_MM."""
    import zlib
    pt = 72 / 25.4
    W, H = A4_MM[0] * pt, A4_MM[1] * pt
    lay = _layout()
    x0, y0, bw, bh = lay['board']
    img = board_image(int(round(BOARD_SQUARE_MM * px_per_mm)))   # 760 px per square: whole pixels per marker cell
    data = zlib.compress(img.tobytes(), 9)
    bx, by, blen = lay['bar']
    yb = H - by * pt
    ops = [f'1 g 0 0 {W:.4f} {H:.4f} re f',                     # white page (not transparent)
           f'q {bw * pt:.4f} 0 0 {bh * pt:.4f} {x0 * pt:.4f} {H - (y0 + bh) * pt:.4f} cm /Im1 Do Q',
           f'0 g {bx * pt:.4f} {yb - 0.6 * pt:.4f} {blen * pt:.4f} {1.2 * pt:.4f} re f', '0 G 0.6 w']
    for k in range(11):
        xt = (bx + 10 * k) * pt
        ops.append(f'{xt:.4f} {yb:.4f} m {xt:.4f} {yb + (2.5 if k % 5 == 0 else 1.5) * pt:.4f} l S')
    text = BOARD_TEXT.replace('\u2014', '\\227')            # em dash in WinAnsiEncoding
    ops.append(f'BT /F1 11 Tf {x0 * pt:.4f} {H - lay["text_y"] * pt:.4f} Td ({text}) Tj ET')
    ops.append(f'BT /F1 9 Tf {(bx + blen) * pt + 8:.4f} {yb - 3:.4f} Td ({CHECK_BAR_MM:.0f} mm check bar) Tj ET')
    content = '\n'.join(ops).encode('latin-1')
    objs = [b'<< /Type /Catalog /Pages 2 0 R >>',
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.4f} {H:.4f}] /Contents 4 0 R '
             f'/Resources << /XObject << /Im1 5 0 R >> /Font << /F1 6 0 R >> >> >>').encode(),
            f'<< /Length {len(content)} >>\nstream\n'.encode() + content + b'\nendstream',
            (f'<< /Type /XObject /Subtype /Image /Width {img.shape[1]} /Height {img.shape[0]} /ColorSpace /DeviceGray '
             f'/BitsPerComponent 8 /Interpolate false /Filter /FlateDecode /Length {len(data)} >>\nstream\n').encode()
            + data + b'\nendstream',
            b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>']
    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f'{i} 0 obj\n'.encode() + o + b'\nendobj\n'
    xref = len(out)
    out += f'xref\n0 {len(objs) + 1}\n0000000000 65535 f \n'.encode()
    out += b''.join(f'{o:010d} 00000 n \n'.encode() for o in offsets)
    out += f'trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    with open(path, 'wb') as f:
        f.write(out)


def run_make_board(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    png, pdf = os.path.join(OUT_DIR, 'charuco_board.png'), os.path.join(OUT_DIR, 'charuco_board.pdf')
    write_board_png(png)
    write_board_pdf(pdf)
    print(f'Saved: {pdf} (A4, exact scale - print at 100% / Actual size)\n       {png} (A4, 300 dpi)')


def detect_board(detector, frame, min_corners=6):
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < min_corners:
        return None
    return corners.reshape(-1, 2), ids.ravel()


class _Sequential:
    """Reads a video forward and hands out the frame closest to a time (ms on the video's own clock)."""

    def __init__(self, src, fps):
        self.src, self.half = src, 500.0 / fps
        self.t, self.done = None, False

    def at(self, target_ms):
        while not self.done and (self.t is None or self.t < target_ms - self.half):
            if not self.src.grab():
                self.done = True
                return None
            self.t = self.src.get(cv2.CAP_PROP_POS_MSEC)
        if self.done or self.t is None or self.t > target_ms + self.half:
            return None          # the video has no frame at that moment (ended, or not started yet)
        ok, frame = self.src.retrieve()
        return frame if ok else None


def sample_times(duration_a_s, duration_b_s, offset_s, start_s=0.0, seconds=None, per_second=4):
    """Moments (s, on video A's clock) to check for the board: where both videos recorded, from start_s, for
    `seconds`. Camera B's frame for moment T is the one at T - offset_s on B's clock (offset_s = seconds into
    video A at which video B started)."""
    lo = max(0.0, offset_s, start_s)
    hi = min(duration_a_s, offset_s + duration_b_s)
    if seconds:
        hi = min(hi, max(start_s, lo) + seconds)
    return np.arange(lo, hi, 1.0 / per_second)


def paired_frames(src_a, src_b, offset_s, times_s, fps_a, fps_b):
    """(T, frame of A at T, frame of B at T - offset_s) for each moment T (s on A's clock). Frames are taken
    by time, so videos that started at different moments (timecode-synced GoPros) are paired correctly."""
    a, b = _Sequential(src_a, fps_a), _Sequential(src_b, fps_b)
    for T in times_s:
        yield T, a.at(T * 1000), b.at((T - offset_s) * 1000)


def _grid_neighbors(ids):
    """Pairs of detected corner ids that are next to each other on the board (one square apart)."""
    nx = BOARD_SQUARES[0] - 1
    have = set(int(i) for i in ids)
    return [(i, j) for i in have for j in (i + 1, i + nx) if j in have and (j != i + 1 or (i + 1) % nx)]


def calibrate_board(video_a, video_b, offset_s=0.0, square_mm=BOARD_SQUARE_MM, start_s=0.0, seconds=60.0,
                    per_second=4.0, progress=None, should_stop=None):
    """Board calibration of two cameras. Returns a dict with the cameras (K, dist, size, R, t in cm), the
    numbers of frames with the board, the fit errors, the check (reconstructed square size, reprojection
    error) and warnings."""
    board, detector = make_board()
    scale = square_mm / BOARD_SQUARE_MM          # the board's object points at the measured size (mm)
    board_xyz = board.getChessboardCorners().reshape(-1, 3) * scale
    srcs = [vs.VideoSource(video_a, hardware=True), vs.VideoSource(video_b, hardware=True)]
    fps = [s.get(cv2.CAP_PROP_FPS) or 30 for s in srcs]
    sizes = [(int(s.get(cv2.CAP_PROP_FRAME_WIDTH)), int(s.get(cv2.CAP_PROP_FRAME_HEIGHT))) for s in srcs]
    warnings = []
    if sizes[0] != sizes[1]:
        warnings.append(f'The videos have different resolutions ({sizes[0][0]}x{sizes[0][1]} / '
                        f'{sizes[1][0]}x{sizes[1][1]}) - use the same settings on both cameras.')
    if abs(fps[0] - fps[1]) > 0.01:
        warnings.append(f'The videos have different frame rates ({fps[0]:.3f} / {fps[1]:.3f} fps) - use the same '
                        'settings on both cameras.')
    times = sample_times(srcs[0].total_ms / 1000, srcs[1].total_ms / 1000, offset_s, start_s, seconds, per_second)
    if not len(times):
        raise ValueError('The two videos have no common time in that window - check the offset and the time window.')
    seen = [{}, {}]                             # per camera: sample index -> (corners, ids)
    for k, (T, fa, fb) in enumerate(paired_frames(srcs[0], srcs[1], offset_s, times, fps[0], fps[1])):
        for v, frame in enumerate((fa, fb)):
            if frame is None:
                continue
            small = cv2.resize(frame, None, fx=0.5, fy=0.5) if max(sizes[v]) > 2000 else frame
            if detect_board(detector, small) is not None:     # quick check at half size first
                d = detect_board(detector, frame)
                if d is not None:
                    seen[v][k] = d
        if progress and k % 4 == 0:
            progress((k + 1) / len(times), f'board seen in {len(seen[0])} (A) / {len(seen[1])} (B) frames')
        if should_stop and should_stop():
            break
    for s in srcs:
        s.release()
    for v in range(2):
        if len(seen[v]) < 10:
            raise ValueError(f'The board was found in only {len(seen[v])} frames of video {"AB"[v]} - need 10+ '
                             '(is it in view in that time window? is the offset right?).')

    # Each camera's lens (focal length, distortion) from its own board views (up to 60, spread over time)
    cams, lens_rms = [], []
    for v in range(2):
        keys = sorted(seen[v])
        views = [seen[v][k] for k in keys[::max(1, len(keys) // 60)]]
        obj = [board_xyz[ids].astype(np.float32) for _, ids in views]
        img = [c.astype(np.float32) for c, _ in views]
        rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, sizes[v], None, None, flags=cv2.CALIB_FIX_K3)
        cams.append(dict(K=K, dist=dist, size=sizes[v]))
        lens_rms.append(float(rms))

    # Where camera B is: the moments both cameras saw the board (the same corners in both)
    obj_s, img_a, img_b, common_ids = [], [], [], []
    for k in sorted(set(seen[0]) & set(seen[1])):
        in_a, in_b = dict(zip(seen[0][k][1], seen[0][k][0])), dict(zip(seen[1][k][1], seen[1][k][0]))
        common = sorted(set(in_a) & set(in_b))
        if len(common) < 6:
            continue
        obj_s.append(board_xyz[common].astype(np.float32))
        img_a.append(np.float32([in_a[i] for i in common]))
        img_b.append(np.float32([in_b[i] for i in common]))
        common_ids.append(common)
    if len(obj_s) < 5:
        raise ValueError(f'Both cameras saw the board at the same moment in only {len(obj_s)} frames - need 5+ '
                         '(hold it where both cameras see it; check the offset).')
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(obj_s, img_a, img_b, cams[0]['K'], cams[0]['dist'],
                                                     cams[1]['K'], cams[1]['dist'], sizes[0],
                                                     flags=cv2.CALIB_FIX_INTRINSIC)
    t = t.ravel()

    # Check: triangulate the corners and measure the squares
    sq, reproj = [], []
    for pa, pb, ids in zip(img_a, img_b, common_ids):
        X = triangulate_px(pa, pb, cams[0]['K'], cams[1]['K'], R, t, cams[0]['dist'], cams[1]['dist'])
        pos = dict(zip(ids, X))
        sq += [np.linalg.norm(pos[i] - pos[j]) for i, j in _grid_neighbors(ids)]
        reproj += list(np.linalg.norm(project(X, cams[0]['K'], cams[0]['dist'], np.eye(3), np.zeros(3)) - pa, axis=1))
        reproj += list(np.linalg.norm(project(X, cams[1]['K'], cams[1]['dist'], R, t) - pb, axis=1))
    sq = np.array(sq)
    res = dict(cameras=cams, R=R, t_cm=t / 10, frames_a=len(seen[0]), frames_b=len(seen[1]), frames_both=len(obj_s),
               lens_rms_px=lens_rms, stereo_rms_px=float(rms), distance_cm=float(np.linalg.norm(t) / 10),
               square_mm=float(square_mm), square_mean_mm=float(sq.mean()), square_sd_mm=float(sq.std()),
               reprojection_px=float(np.mean(reproj)), offset_s=float(offset_s), sample_window_s=[float(times[0]),
                                                                                                  float(times[-1])])
    if res['frames_both'] < 20:
        warnings.append(f"Only {res['frames_both']} frames with the board seen by both cameras (20+ is better) - "
                        'film the board longer where both cameras see it.')
    if res['stereo_rms_px'] > 1.0:
        warnings.append(f"Stereo fit error {res['stereo_rms_px']:.2f} px (> 1 px) - check the offset (wrong pairing "
                        'of frames) and that the board is flat.')
    for v in range(2):
        if lens_rms[v] > 1.0:
            warnings.append(f'Camera {"AB"[v]} lens fit error {lens_rms[v]:.2f} px (> 1 px).')
    if abs(res['square_mean_mm'] - square_mm) > 0.02 * square_mm:
        warnings.append(f"The squares come out {res['square_mean_mm']:.1f} mm instead of {square_mm:.1f} mm.")
    res['warnings'] = warnings
    res['ok'] = not warnings
    return res


def save_board_calibration(path, res, video_a, video_b):
    ca, cb = res['cameras']
    save_cameras(path, ca, cb, res['R'], res['t_cm'], 'cm', 'board',
                 dict(video_a=os.path.abspath(video_a), video_b=os.path.abspath(video_b),
                      **{k: res[k] for k in ('frames_a', 'frames_b', 'frames_both', 'lens_rms_px', 'stereo_rms_px',
                                             'distance_cm', 'square_mm', 'square_mean_mm', 'square_sd_mm',
                                             'reprojection_px', 'offset_s', 'sample_window_s', 'warnings')}))


def board_report(res):
    lines = [f"Board found in {res['frames_a']} frames of A, {res['frames_b']} of B, {res['frames_both']} of both.",
             f"Lens fit error: A {res['lens_rms_px'][0]:.2f} px, B {res['lens_rms_px'][1]:.2f} px; "
             f"stereo fit error {res['stereo_rms_px']:.2f} px.",
             f"Cameras are {res['distance_cm']:.1f} cm apart.",
             f"Check: squares measure {res['square_mean_mm']:.2f} ± {res['square_sd_mm']:.2f} mm "
             f"(entered {res['square_mm']:.1f} mm); mean reprojection error {res['reprojection_px']:.2f} px."]
    lines.append('PASS' if res['ok'] else 'WARNING:\n  ' + '\n  '.join(res['warnings']))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Print the camera calibration board (the calibration itself is '
                                                 'done in the Lab App).')
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('make-board', help='Save the printable calibration board')
    run_make_board(parser.parse_args())

if __name__ == '__main__':
    main()
