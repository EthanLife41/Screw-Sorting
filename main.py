import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from dotenv import load_dotenv
load_dotenv()

import cv2
import time
import base64
import numpy as np
from openai import OpenAI

#config
CAMERA_INDEX_CANDIDATES = [1, 0]
PREFERRED_CAMERA_ORDER = [1, 0]

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_NAME = "Screw Detection Prototype"
MASK_WINDOW_NAME = "ROI Debug Mask"

# ROI
ROI_WIDTH_RATIO = 0.34
ROI_HEIGHT_RATIO = 0.30
ROI_OFFSET_X = -90
ROI_OFFSET_Y = 20

BLUR_KERNEL_SIZE = (7, 7)

SETTLE_FRAMES_REQUIRED = 20
BACKGROUND_WARMUP_FRAMES = 35

BACKGROUND_DIFF_THRESHOLD = 28

# Empty ROI detection
EMPTY_BASELINE_PIXEL_MARGIN = 2200
EMPTY_BASELINE_BLOB_MARGIN = 2600
EMPTY_MAX_AREA_FRACTION = 0.10
EMPTY_FRAMES_REQUIRED_FOR_ARMING = 12

# Object detection
OBJECT_MIN_CHANGED_PIXELS = 300
OBJECT_MIN_LARGEST_BLOB = 120
OBJECT_MIN_AREA_FRACTION = 0.003
OBJECT_MAX_AREA_FRACTION = 0.32
OBJECT_FRAMES_REQUIRED = 4

FRAME_DIFF_THRESHOLD = 20
MAX_MOTION_AREA_FOR_EXTREME_MOVEMENT = 20000

COOLDOWN_SECONDS = 2.0
RESULT_SCREEN_SECONDS = 10.0

CAPTURE_DIR = "captures"
CAPTURE_PATH = os.path.join(CAPTURE_DIR, "latest_capture.jpg")

OPENAI_MODEL = "gpt-4.1-mini"

SHOW_DEBUG_MASK = True

CAMERA_OPEN_RETRIES = 3
CAMERA_WARMUP_READS = 12
CAMERA_RECONNECT_DELAY = 0.6

resume_button_rect = None

def create_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def parse_yes_no(text):
    if not text:
        return None

    cleaned = text.strip().upper()
    if cleaned == "YES":
        return "YES"
    if cleaned == "NO":
        return "NO"
    if "YES" in cleaned and "NO" not in cleaned:
        return "YES"
    if "NO" in cleaned and "YES" not in cleaned:
        return "NO"
    return None


def encode_image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")


def analyze_image_for_screw(client, image_path):
    try:
        image_data_url = encode_image_to_data_url(image_path)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a strict visual inspection assistant. "
                                "The image is a cropped inspection region from a work surface. "
                                "Answer only YES or NO. "
                                "Answer YES only if a screw is clearly visible as the main object. "
                                "If a hand, arm, large tool, or large unrelated object dominates the image, answer NO. "
                                "Do not include any other words."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Is there a screw clearly visible in this cropped inspection image? Answer only YES or NO."
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url
                        }
                    ]
                }
            ]
        )

        return parse_yes_no(response.output_text.strip())

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def action_from_result(result):
    if result == "YES":
        return "ACTION: OPEN TRAPDOOR A"
    return "ACTION: DO NOTHING"


#Camera
def configure_camera(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


def open_camera_once(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None

    configure_camera(cap)
    time.sleep(0.4)

    ok = False
    for _ in range(CAMERA_WARMUP_READS):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.05)

    if not ok:
        cap.release()
        return None

    for _ in range(CAMERA_WARMUP_READS):
        cap.read()

    return cap


def open_camera(index):
    for attempt in range(CAMERA_OPEN_RETRIES):
        cap = open_camera_once(index)
        if cap is not None:
            return cap
        print(f"Camera {index} open retry {attempt + 1}/{CAMERA_OPEN_RETRIES} failed...")
        time.sleep(CAMERA_RECONNECT_DELAY)
    return None


def safe_release(cap):
    if cap is not None:
        cap.release()


def reconnect_camera(index):
    print(f"Reconnecting camera index {index}...")
    return open_camera(index)


def get_available_camera_indices():
    working = []
    print("Scanning cameras...")
    for idx in CAMERA_INDEX_CANDIDATES:
        cap = open_camera(idx)
        if cap is not None:
            print(f"Found working camera: {idx}")
            working.append(idx)
            cap.release()
    return working


def choose_starting_camera_position(camera_indices):
    for preferred in PREFERRED_CAMERA_ORDER:
        if preferred in camera_indices:
            return camera_indices.index(preferred)
    return 0


def open_camera_from_list(camera_indices, position):
    position = position % len(camera_indices)
    camera_index = camera_indices[position]
    cap = open_camera(camera_index)
    return cap, camera_index, position


def ensure_capture_dir():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def preprocess_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
    return gray


def compute_central_roi(frame_shape):
    h, w = frame_shape[:2]

    roi_w = int(w * ROI_WIDTH_RATIO)
    roi_h = int(h * ROI_HEIGHT_RATIO)

    x1 = (w - roi_w) // 2 + ROI_OFFSET_X
    y1 = (h - roi_h) // 2 + ROI_OFFSET_Y

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + roi_w)
    y2 = min(h, y1 + roi_h)

    return x1, y1, x2, y2


def draw_status_text(frame, lines, colour=(0, 255, 0)):
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA
        )
        y += 28


def save_roi_image(roi_bgr):
    ensure_capture_dir()
    ok = cv2.imwrite(CAPTURE_PATH, roi_bgr)
    if not ok:
        raise RuntimeError("Failed to save capture image.")
    print(f"Captured image: {CAPTURE_PATH}")
    return CAPTURE_PATH


def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, (int(w * scale), int(h * scale)))

# state
class DetectionState:
    def __init__(self):
        self.prev_roi_gray = None
        self.background_roi_gray = None

        self.settle_frames = 0
        self.background_frames_collected = 0

        self.empty_frame_count = 0
        self.object_frame_count = 0

        self.armed = False
        self.last_trigger_time = 0.0

        self.status_message = "Starting..."
        self.last_result_message = "No capture yet"

        self.last_changed_pixels = 0
        self.last_largest_blob = 0.0
        self.last_motion_area = 0.0
        self.last_area_fraction = 0.0
        self.last_presence_mask = None

        self.empty_baseline_pixels = None
        self.empty_baseline_blob = None

    def full_reset(self):
        self.prev_roi_gray = None
        self.background_roi_gray = None

        self.settle_frames = 0
        self.background_frames_collected = 0

        self.empty_frame_count = 0
        self.object_frame_count = 0

        self.armed = False
        self.status_message = "Settling camera..."
        self.last_changed_pixels = 0
        self.last_largest_blob = 0.0
        self.last_motion_area = 0.0
        self.last_area_fraction = 0.0
        self.last_presence_mask = None

        self.empty_baseline_pixels = None
        self.empty_baseline_blob = None

    def reset_after_capture(self):
        self.empty_frame_count = 0
        self.object_frame_count = 0
        self.armed = False
        self.status_message = "Remove object from ROI to re-arm"
        self.last_changed_pixels = 0
        self.last_largest_blob = 0.0
        self.last_motion_area = 0.0
        self.last_area_fraction = 0.0
        self.last_presence_mask = None

    def set_empty_baseline(self, changed_pixels, largest_blob):
        self.empty_baseline_pixels = changed_pixels
        self.empty_baseline_blob = largest_blob


class AppState:
    def __init__(self):
        self.mode = "LIVE"
        self.result_start_time = None
        self.result_background_frame = None
        self.result_roi_frame = None
        self.result_label = None
        self.result_action = None
        self.result_image_path = None
        self.resume_requested = False


def update_background_model(state, roi_gray, alpha):
    if state.background_roi_gray is None:
        state.background_roi_gray = roi_gray.astype(np.float32)
    else:
        cv2.accumulateWeighted(roi_gray, state.background_roi_gray, alpha)


def analyse_roi(state, roi_bgr):
    roi_gray = preprocess_gray(roi_bgr)
    kernel = np.ones((3, 3), np.uint8)

    motion_area = 0.0
    if state.prev_roi_gray is not None:
        frame_diff = cv2.absdiff(state.prev_roi_gray, roi_gray)
        _, motion_mask = cv2.threshold(frame_diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        contours_motion, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours_motion)

    state.prev_roi_gray = roi_gray
    state.last_motion_area = motion_area

    if state.settle_frames < SETTLE_FRAMES_REQUIRED:
        state.settle_frames += 1
        blank = np.zeros_like(roi_gray)
        state.last_presence_mask = blank
        return "SETTLING", 0, 0.0, 0.0, motion_area, cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

    if state.background_frames_collected < BACKGROUND_WARMUP_FRAMES:
        update_background_model(state, roi_gray, alpha=0.15)
        state.background_frames_collected += 1
        blank = np.zeros_like(roi_gray)
        state.last_presence_mask = blank
        return "LEARNING", 0, 0.0, 0.0, motion_area, cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

    background_uint8 = cv2.convertScaleAbs(state.background_roi_gray)
    bg_diff = cv2.absdiff(background_uint8, roi_gray)

    _, presence_mask = cv2.threshold(bg_diff, BACKGROUND_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    presence_mask = cv2.morphologyEx(presence_mask, cv2.MORPH_OPEN, kernel)
    presence_mask = cv2.dilate(presence_mask, kernel, iterations=2)

    changed_pixels = int(cv2.countNonZero(presence_mask))
    roi_area = roi_gray.shape[0] * roi_gray.shape[1]
    area_fraction = changed_pixels / float(roi_area)

    contours, _ = cv2.findContours(presence_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_blob_area = max((cv2.contourArea(c) for c in contours), default=0.0)

    looks_empty_enough = False
    if state.empty_baseline_pixels is not None and state.empty_baseline_blob is not None:
        looks_empty_enough = (
            changed_pixels <= state.empty_baseline_pixels + EMPTY_BASELINE_PIXEL_MARGIN
            and largest_blob_area <= state.empty_baseline_blob + EMPTY_BASELINE_BLOB_MARGIN
            and area_fraction <= EMPTY_MAX_AREA_FRACTION
        )
    else:
        looks_empty_enough = area_fraction <= EMPTY_MAX_AREA_FRACTION

    if not state.armed and looks_empty_enough:
        update_background_model(state, roi_gray, alpha=0.03)

    state.last_changed_pixels = changed_pixels
    state.last_largest_blob = largest_blob_area
    state.last_area_fraction = area_fraction
    state.last_presence_mask = presence_mask

    return "READY", changed_pixels, largest_blob_area, area_fraction, motion_area, cv2.cvtColor(presence_mask, cv2.COLOR_GRAY2BGR)

#results
def draw_button(frame, x1, y1, x2, y2, text):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def build_result_screen(app_state):
    global resume_button_rect

    frame = app_state.result_background_frame.copy()
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

    preview = resize_to_fit(app_state.result_roi_frame, int(w * 0.42), int(h * 0.55))
    ph, pw = preview.shape[:2]
    px = 40
    py = (h - ph) // 2

    frame[py:py + ph, px:px + pw] = preview
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), 2)

    tx = int(w * 0.58)
    y = 90

    cv2.putText(frame, "CAPTURE COMPLETE", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    y += 70

    colour = (0, 220, 0) if app_state.result_label == "YES" else (0, 180, 255)
    cv2.putText(frame, f"Screw detected: {app_state.result_label}", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 3, cv2.LINE_AA)
    y += 60

    cv2.putText(frame, app_state.result_action, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    y += 70

    cv2.putText(frame, "Testing note:", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 35
    cv2.putText(frame, "Sawdust may affect vision reliability.", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
    y += 55

    remaining = max(0.0, RESULT_SCREEN_SECONDS - (time.time() - app_state.result_start_time))
    cv2.putText(frame, f"Reopening camera in: {remaining:.1f}s", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y += 70

    btn_x1, btn_y1, btn_x2, btn_y2 = tx, y, tx + 240, y + 60
    draw_button(frame, btn_x1, btn_y1, btn_x2, btn_y2, "Resume Now")
    resume_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)

    y += 100
    cv2.putText(frame, "Saved image:", (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(frame, CAPTURE_PATH, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def mouse_callback(event, x, y, flags, param):
    global resume_button_rect
    app_state = param

    if event == cv2.EVENT_LBUTTONDOWN and app_state.mode == "RESULT" and resume_button_rect is not None:
        x1, y1, x2, y2 = resume_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            app_state.resume_requested = True


def capture_and_process(client, roi_bgr):
    image_path = save_roi_image(roi_bgr)
    result = analyze_image_for_screw(client, image_path)
    result_label = result if result is not None else "NO"
    result_action = action_from_result(result_label)
    return image_path, result_label, result_action

def main():
    print("Starting screw detection prototype...")
    print("Controls: q quit, c capture, n next camera, p previous camera, r reset")

    try:
        client = create_openai_client()
    except Exception as e:
        print(f"Startup error: {e}")
        return

    camera_indices = get_available_camera_indices()
    if not camera_indices:
        print("Error: Could not find any working cameras.")
        return

    active_position = choose_starting_camera_position(camera_indices)
    cap, active_camera_index, active_position = open_camera_from_list(camera_indices, active_position)

    if cap is None:
        print("Error: Could not open selected camera.")
        return

    print(f"Starting on camera index {active_camera_index}")

    detection_state = DetectionState()
    detection_state.full_reset()

    app_state = AppState()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, app_state)

    if SHOW_DEBUG_MASK:
        cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(MASK_WINDOW_NAME, 420, 260)

    try:
        while True:
            if app_state.mode == "RESULT":
                cv2.imshow(WINDOW_NAME, build_result_screen(app_state))

                if SHOW_DEBUG_MASK:
                    cv2.imshow(MASK_WINDOW_NAME, np.zeros((300, 300, 3), dtype=np.uint8))

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break

                timed_out = (time.time() - app_state.result_start_time) >= RESULT_SCREEN_SECONDS
                if timed_out or app_state.resume_requested:
                    app_state.mode = "LIVE"
                    app_state.resume_requested = False
                    detection_state.reset_after_capture()
                    detection_state.last_trigger_time = time.time()
                continue

            ok, frame = cap.read()
            if not ok:
                print("Live camera read failed. Attempting reconnect...")
                safe_release(cap)
                cap = reconnect_camera(active_camera_index)
                if cap is None:
                    print("Reconnect failed. Try pressing n/p to switch cameras or rerun.")
                    break
                detection_state.full_reset()
                continue

            display_frame = frame.copy()
            x1, y1, x2, y2 = compute_central_roi(frame.shape)
            roi = frame[y1:y2, x1:x2]

            phase, changed_pixels, largest_blob, area_fraction, motion_area, mask_bgr = analyse_roi(
                detection_state, roi
            )

            current_time = time.time()
            in_cooldown = (current_time - detection_state.last_trigger_time) < COOLDOWN_SECONDS

            roi_colour = (0, 255, 0) if detection_state.armed else (0, 255, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), roi_colour, 2)

            if phase == "SETTLING":
                detection_state.status_message = f"Settling camera... {detection_state.settle_frames}/{SETTLE_FRAMES_REQUIRED}"
                detection_state.empty_frame_count = 0
                detection_state.object_frame_count = 0
                detection_state.armed = False

            elif phase == "LEARNING":
                detection_state.status_message = (
                    f"Learning empty desk... {detection_state.background_frames_collected}/{BACKGROUND_WARMUP_FRAMES}"
                )
                detection_state.empty_frame_count = 0
                detection_state.object_frame_count = 0
                detection_state.armed = False

            elif in_cooldown:
                remaining = COOLDOWN_SECONDS - (current_time - detection_state.last_trigger_time)
                detection_state.status_message = f"Cooldown: {remaining:.1f}s"
                detection_state.empty_frame_count = 0
                detection_state.object_frame_count = 0
                detection_state.armed = False

            else:
                if detection_state.empty_baseline_pixels is None:
                    detection_state.set_empty_baseline(changed_pixels, largest_blob)

                looks_empty = (
                    changed_pixels <= detection_state.empty_baseline_pixels + EMPTY_BASELINE_PIXEL_MARGIN
                    and largest_blob <= detection_state.empty_baseline_blob + EMPTY_BASELINE_BLOB_MARGIN
                    and area_fraction <= EMPTY_MAX_AREA_FRACTION
                )

                looks_like_object = (
                    changed_pixels >= OBJECT_MIN_CHANGED_PIXELS
                    and largest_blob >= OBJECT_MIN_LARGEST_BLOB
                    and area_fraction >= OBJECT_MIN_AREA_FRACTION
                    and area_fraction <= OBJECT_MAX_AREA_FRACTION
                )

                if not detection_state.armed:
                    if looks_empty:
                        detection_state.empty_frame_count += 1
                        detection_state.object_frame_count = 0

                        if detection_state.last_result_message in ("Last trigger: automatic", "Last trigger: manual"):
                            detection_state.status_message = (
                                f"Clearing ROI... {detection_state.empty_frame_count}/{EMPTY_FRAMES_REQUIRED_FOR_ARMING}"
                            )
                        else:
                            detection_state.status_message = (
                                f"Arming on empty desk... {detection_state.empty_frame_count}/{EMPTY_FRAMES_REQUIRED_FOR_ARMING}"
                            )

                        if detection_state.empty_frame_count >= EMPTY_FRAMES_REQUIRED_FOR_ARMING:
                            detection_state.armed = True
                            detection_state.status_message = "Armed: waiting for next object..."
                            detection_state.last_result_message = "Ready for next capture"
                    else:
                        detection_state.empty_frame_count = 0
                        detection_state.object_frame_count = 0

                        if detection_state.last_result_message in ("Last trigger: automatic", "Last trigger: manual"):
                            detection_state.status_message = "Remove object from ROI to re-arm"
                        else:
                            detection_state.status_message = "Waiting for a clear empty ROI..."

                else:
                    if looks_empty:
                        detection_state.object_frame_count = 0
                        detection_state.status_message = "Armed: waiting for next object..."
                    else:
                        if looks_like_object and motion_area <= MAX_MOTION_AREA_FOR_EXTREME_MOVEMENT:
                            detection_state.object_frame_count += 1
                            detection_state.status_message = (
                                f"New object detected: {detection_state.object_frame_count}/{OBJECT_FRAMES_REQUIRED}"
                            )

                            if detection_state.object_frame_count >= OBJECT_FRAMES_REQUIRED:
                                image_path, result_label, result_action = capture_and_process(client, roi.copy())

                                app_state.mode = "RESULT"
                                app_state.result_start_time = time.time()
                                app_state.result_background_frame = frame.copy()
                                app_state.result_roi_frame = roi.copy()
                                app_state.result_label = result_label
                                app_state.result_action = result_action
                                app_state.result_image_path = image_path
                                app_state.resume_requested = False

                                detection_state.last_trigger_time = time.time()
                                detection_state.object_frame_count = 0
                                detection_state.empty_frame_count = 0
                                detection_state.armed = False
                                detection_state.last_result_message = "Last trigger: automatic"
                        else:
                            if area_fraction > OBJECT_MAX_AREA_FRACTION:
                                detection_state.status_message = "Large object rejected..."
                            else:
                                detection_state.status_message = "Object uncertain or moving too much..."
                            detection_state.object_frame_count = max(0, detection_state.object_frame_count - 1)

            lines = [
                detection_state.status_message,
                f"Camera index: {active_camera_index}",
                f"Changed pixels: {changed_pixels}",
                f"Largest blob: {largest_blob:.0f}",
                f"Area fraction: {area_fraction:.3f}",
                f"Motion area: {motion_area:.0f}",
                f"Armed: {detection_state.armed}",
                f"Object frames: {detection_state.object_frame_count}/{OBJECT_FRAMES_REQUIRED}",
                detection_state.last_result_message,
                "Keys: c capture | n next | p previous | r reset | q quit"
            ]
            draw_status_text(display_frame, lines)

            cv2.putText(
                display_frame,
                "Keep hands and large objects outside ROI. Place only the screw in the box.",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(WINDOW_NAME, display_frame)

            if SHOW_DEBUG_MASK:
                debug_mask = cv2.resize(mask_bgr, (420, 260))
                cv2.putText(
                    debug_mask,
                    "White = changed from background",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(MASK_WINDOW_NAME, debug_mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                image_path, result_label, result_action = capture_and_process(client, roi.copy())

                app_state.mode = "RESULT"
                app_state.result_start_time = time.time()
                app_state.result_background_frame = frame.copy()
                app_state.result_roi_frame = roi.copy()
                app_state.result_label = result_label
                app_state.result_action = result_action
                app_state.result_image_path = image_path
                app_state.resume_requested = False

                detection_state.last_trigger_time = time.time()
                detection_state.object_frame_count = 0
                detection_state.empty_frame_count = 0
                detection_state.armed = False
                detection_state.last_result_message = "Last trigger: manual"

            elif key == ord("r"):
                print("Resetting settle + background learning...")
                detection_state.full_reset()

            elif key == ord("n"):
                safe_release(cap)
                active_position = (active_position + 1) % len(camera_indices)
                cap, active_camera_index, active_position = open_camera_from_list(camera_indices, active_position)

                if cap is None:
                    print("Failed to open next camera.")
                    break

                print(f"Switched to camera index {active_camera_index}")
                detection_state.full_reset()

            elif key == ord("p"):
                safe_release(cap)
                active_position = (active_position - 1) % len(camera_indices)
                cap, active_camera_index, active_position = open_camera_from_list(camera_indices, active_position)

                if cap is None:
                    print("Failed to open previous camera.")
                    break

                print(f"Switched to camera index {active_camera_index}")
                detection_state.full_reset()

    finally:
        safe_release(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()