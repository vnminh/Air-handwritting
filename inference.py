import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks
import cv2
import time
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from model_conformer_fix_128_rff import load_model, predict_from_coordinates, normalize_coordinates
# from model_conformer import load_model, predict_from_coordinates, normalize_coordinates
from model_conformer_no_mask import load_model, predict_from_coordinates, normalize_coordinates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HAND_MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/mediapipe_para/hand_landmarker.task"
LANGUAGE_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/language.txt"
# MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/model_para/best_model_cf_128-4-6-512.pth"
# MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/model_para/best_model_128_rff(2).pth"
# MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/model_para/final_best_model.pth"
MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/model_para/best_model_128_rff_relPos-26.pth"
# MODEL_PATH = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/code/model_para/best_model_128_rff_APOS-26dim.pth"

# ---------------------------------------------------------------------------
# Load handwriting model
# ---------------------------------------------------------------------------
DMODEL = 128
NHEAD = 4
NECLAYER = 6
DFF = DMODEL*4

model, tokens, blank_idx, device = load_model(MODEL_PATH, LANGUAGE_PATH, d_model=DMODEL, nhead=NHEAD, num_encoder_layers=NECLAYER, dim_feedforward=DFF)
print(f"Loaded model on {device}")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
LOCK = threading.Lock()
LASTEST = None          # latest annotated frame for display
RESULT_TEXT = ""        # latest prediction result

STATUS = "PEN_UP"      # PEN_UP | PEN_DOWN
COOR = []              # collected (x, y) during PEN_DOWN
LEFT_SIGNAL_STATE = "NO_LEFT"  # OPEN | CLOSED | UNKNOWN | NO_LEFT
RIGHT_SIGNAL_STATE = "NO_RIGHT"  # RIGHT | NO_RIGHT
MIN_POINTS = 15         # minimum points to run inference
SAVE_DIR = "/home/minh-le-vo-nhat/Documents/Minh-DUT/Ky-8-2025-2026/XLA/BTNhom/infer_visualize" 
LEFT_OPEN_MIN_FINGERS = 4
LEFT_CLOSE_MAX_FINGERS = 1
SWAP_HAND_LABELS = True
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _load_vietnamese_font(font_size=28):
    """Load a Unicode-capable font for rendering Vietnamese text."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except OSError:
                continue
    return ImageFont.load_default()


VI_FONT = _load_vietnamese_font(font_size=28)


def _normalize_handed_label(label):
    """Optionally swap labels if camera mirroring flips perceived handedness."""
    if not SWAP_HAND_LABELS:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def _get_handed_label(result, hand_idx):
    """Get handedness label for a detected hand if available."""
    if hand_idx >= len(result.handedness):
        return None
    handed = result.handedness[hand_idx]
    if not handed:
        return None
    label = handed[0].category_name if len(handed) > 0 else None
    return _normalize_handed_label(label)


def _count_extended_fingers(hand_landmarks, handed_label):
    """Estimate number of extended fingers from a hand landmark list."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return 0

    # Thumb direction depends on handedness.
    if handed_label == "Left":
        thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x
    elif handed_label == "Right":
        thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x
    else:
        thumb_extended = abs(hand_landmarks[4].x - hand_landmarks[3].x) > 0.03

    fingers = 1 if thumb_extended else 0
    fingertip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip_id, pip_id in zip(fingertip_ids, pip_ids):
        if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y:
            fingers += 1
    return fingers


def _select_left_right_hands(result):
    """Select left/right hands with label-based mapping and robust fallbacks."""
    hands = []
    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        hands.append({
            "landmarks": hand_landmarks,
            "label": _get_handed_label(result, hand_idx),
        })

    left_hand = next((h["landmarks"] for h in hands if h["label"] == "Left"), None)
    right_hand = next((h["landmarks"] for h in hands if h["label"] == "Right"), None)

    if len(hands) >= 2:
        if left_hand is not None and right_hand is None:
            right_hand = next((h["landmarks"] for h in hands if h["landmarks"] is not left_hand), None)
        elif right_hand is not None and left_hand is None:
            left_hand = next((h["landmarks"] for h in hands if h["landmarks"] is not right_hand), None)
        elif left_hand is None and right_hand is None:
            # Fallback when labels are missing: split by wrist x-position.
            sorted_hands = sorted(hands, key=lambda h: h["landmarks"][0].x)
            left_hand = sorted_hands[0]["landmarks"]
            right_hand = sorted_hands[-1]["landmarks"]

    return left_hand, right_hand


def draw_unicode_text_bgr(image_bgr, text, org, color=(255, 255, 0)):
    """Draw Unicode text on a BGR image using PIL, then convert back to BGR."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # PIL uses RGB; convert from OpenCV's BGR color order.
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(org, text, font=VI_FONT, fill=rgb_color)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# Inference (runs on daemon thread — never blocks main loop)
# ---------------------------------------------------------------------------



def save_trajectory_plot(raw_coords, normalized_coords, prediction, save_path):
    """Save a 2-panel plot of raw and normalized trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw trajectory
    ax = axes[0]
    ax.plot(raw_coords[:, 0], raw_coords[:, 1], "b-", linewidth=1)
    ax.plot(raw_coords[0, 0], raw_coords[0, 1], "go", markersize=8, label="start")
    ax.plot(raw_coords[-1, 0], raw_coords[-1, 1], "ro", markersize=8, label="end")
    ax.set_title("Raw coordinates")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normalized trajectory
    ax = axes[1]
    ax.plot(normalized_coords[:, 0], normalized_coords[:, 1], "b-", linewidth=1)
    ax.plot(normalized_coords[0, 0], normalized_coords[0, 1], "go", markersize=8, label="start")
    ax.plot(normalized_coords[-1, 0], normalized_coords[-1, 1], "ro", markersize=8, label="end")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Normalized  |  Pred: {prediction}")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def run_inference(coords):
    """Run model prediction in a background thread."""
    global RESULT_TEXT
    coord_array = np.array(coords, dtype=np.float32)  # (N, 2)
    if len(coord_array) < MIN_POINTS:
        return

    # Normalize (for plotting)
    normalized = normalize_coordinates(coord_array)

    decoded_tokens, raw_string = predict_from_coordinates(
        model, coord_array, tokens, blank_idx, device,
    )
    with LOCK:
        RESULT_TEXT = raw_string

    # Save plot
    ts = int(time.time() * 1000)
    save_path = os.path.join(SAVE_DIR, f"{ts}.png")
    save_trajectory_plot(coord_array, normalized, raw_string, save_path)

    print(f"[predict] {raw_string}  ->  saved {save_path}")


def _trigger_inference():
    """Copy collected coords, reset state, launch inference thread."""
    global COOR, STATUS
    coords_copy = list(COOR)
    COOR = []
    STATUS = "PEN_UP"
    if len(coords_copy) >= MIN_POINTS:
        threading.Thread(target=run_inference, args=(coords_copy,), daemon=True).start()

# ---------------------------------------------------------------------------
# MediaPipe callback (called on its own thread by MediaPipe)
# ---------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode


def print_result(result, output_image, timestamp_ms: int):
    global LASTEST, STATUS, COOR, LEFT_SIGNAL_STATE, RIGHT_SIGNAL_STATE

    np_img = np.copy(output_image.numpy_view())
    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        draw_landmarks(np_img, hand_landmarks)

    left_hand, right_hand = _select_left_right_hands(result)

    if right_hand is not None:
        pen = right_hand[8]
        pen_xy = np.array([pen.x * output_image.width, (1 - pen.y) * output_image.height])
    else:
        pen_xy = None

    with LOCK:
        LASTEST = np_img
        RIGHT_SIGNAL_STATE = "RIGHT" if right_hand is not None else "NO_RIGHT"

        # --- Left-hand trigger / right-hand writing ---
        left_fingers = _count_extended_fingers(left_hand, "Left") if left_hand is not None else -1
        left_is_open = left_fingers >= LEFT_OPEN_MIN_FINGERS
        left_is_closed = 0 <= left_fingers <= LEFT_CLOSE_MAX_FINGERS

        if left_hand is None:
            LEFT_SIGNAL_STATE = "NO_LEFT"
        elif left_is_open:
            LEFT_SIGNAL_STATE = "OPEN"
        elif left_is_closed:
            LEFT_SIGNAL_STATE = "CLOSED"
        else:
            LEFT_SIGNAL_STATE = "UNKNOWN"

        if STATUS == "PEN_UP" and left_is_open:
            STATUS = "PEN_DOWN"
            COOR = []

        if STATUS == "PEN_DOWN":
            # Record only right-hand index tip while left hand stays open.
            if pen_xy is not None:
                COOR.append(pen_xy.tolist())
            if left_is_closed:
                _trigger_inference()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit. Open LEFT hand to record RIGHT-hand writing, close LEFT hand to stop and predict.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Build display frame
        with LOCK:
            display = np.copy(LASTEST) if LASTEST is not None else None
            result_text = RESULT_TEXT
            current_status = STATUS
            left_signal = LEFT_SIGNAL_STATE
            right_signal = RIGHT_SIGNAL_STATE
            n_points = len(COOR)
            trace_points = np.array(COOR, dtype=np.int32) if len(COOR) > 0 else None
        if display is not None:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            # Draw the currently collected trajectory for live visual feedback.
            if trace_points is not None and len(trace_points) >= 2:
                draw_points = trace_points.copy()
                draw_points[:, 1] = display.shape[0] - draw_points[:, 1]
                cv2.polylines(display, [draw_points], False, (0, 255, 255), 2)
                cv2.circle(display, tuple(draw_points[0]), 4, (0, 255, 0), -1)
                cv2.circle(display, tuple(draw_points[-1]), 4, (0, 0, 255), -1)

            # Draw status
            color = (0, 255, 0) if current_status == "PEN_DOWN" else (0, 0, 255)
            cv2.putText(display, f"[{current_status}] left={left_signal} right={right_signal} pts={n_points}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw prediction result
            if result_text:
                try:
                    display = draw_unicode_text_bgr(
                        display,
                        f"Pred: {result_text}",
                        (10, 50),
                        color=(255, 255, 0),
                    )
                except Exception:
                    # Fallback if PIL/font is unavailable.
                    cv2.putText(display, f"Pred: {result_text}",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("FRAME", display)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
