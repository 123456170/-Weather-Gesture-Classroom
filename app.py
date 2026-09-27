import streamlit as st
import cv2
import numpy as np
import time
import math
import random
import io
import wave
import os
import tempfile
import collections

try:
    from streamlit import fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as fragment
    except ImportError:
        fragment = None

st.set_page_config(
    page_title="Weather Gesture Classroom",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRAME_W = 960
FRAME_H = 540

RECORD_W = 480
RECORD_H = 270
RECORD_FPS = 8
MAX_RECORD_SECONDS = 60
MAX_FRAMES = RECORD_FPS * MAX_RECORD_SECONDS

HOLD_THRESHOLD = 0.8
RESET_HOLD = 3.0

WEATHERS = ["sun", "wind", "rain"]

GESTURE_LABELS = {
    "sun": "Sunshine",
    "wind": "Wind",
    "rain": "Rain",
    "reset": "Teacher reset",
}

WEATHER_MESSAGES = {
    "sun": "Both arms raised = Sunshine!",
    "wind": "Arms waving side-to-side = Wind!",
    "rain": "Jumping-jack arm motion = Rain!",
    "reset": "Teacher reset: bingo cleared.",
}

FACTS = {
    "sun": [
        "Sunlight takes about 8 minutes to reach Earth.",
        "The Sun is a giant ball of hot glowing gas.",
        "Sunlight helps plants make food.",
        "Hats and sunglasses can protect your eyes from bright sun.",
    ],
    "wind": [
        "Wind is moving air.",
        "Wind can help kites fly and boats sail.",
        "Trees and grass can show us which way the wind blows.",
        "Strong wind can make waves at the beach.",
    ],
    "rain": [
        "Did you know raindrops aren't actually teardrop-shaped? They're round!",
        "Rain is part of the water cycle.",
        "Plants, rivers, and animals need rain.",
        "Rain falls when clouds become heavy with water.",
    ],
}

DEMO_SCRIPT = [
    (3.0, "sun"),
    (11.0, "wind"),
    (20.0, "rain"),
    (30.0, "reset"),
    (38.0, "sun"),
    (46.0, "wind"),
    (54.0, "rain"),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


@st.cache_data(show_spinner=False)
def make_wav_bytes(name):
    """Create tiny synthetic sound cues without requiring bundled audio files."""
    sr = 22050
    dur = 1.2
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)

    if name == "sun":
        notes = [523.25, 659.25, 783.99, 1046.50]
        for i, f in enumerate(notes):
            start = i * 0.12
            mask = (t >= start) & (t < start + 0.25)
            if np.any(mask):
                env = np.exp(-8.0 * (t[mask] - start))
                y[mask] += 0.28 * np.sin(2.0 * np.pi * f * t[mask]) * env

    elif name == "wind":
        noise = np.random.randn(len(t)).astype(np.float32)
        b = 0.045
        for i in range(1, len(noise)):
            noise[i] = noise[i - 1] + b * (noise[i] - noise[i - 1])
        env = np.sin(np.pi * t / dur) ** 2
        y = 0.65 * noise * env

    elif name == "rain":
        noise = np.random.randn(len(t)).astype(np.float32) * 0.18
        env = np.exp(-1.8 * t)
        y = noise * env
        for start in [0.08, 0.32, 0.58, 0.82]:
            mask = t >= start
            if np.any(mask):
                f = random.choice([1150.0, 1500.0, 1900.0])
                y[mask] += 0.20 * np.sin(
                    2.0 * np.pi * f * (t[mask] - start)
                ) * np.exp(-24.0 * (t[mask] - start))

    elif name == "reset":
        y = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) * np.exp(-2.2 * t)
        y += 0.18 * np.sin(2.0 * np.pi * 165.0 * t) * np.exp(-2.6 * t)

    else:
        y = 0.22 * np.sin(2.0 * np.pi * 440.0 * t) * np.exp(-3.0 * t)

    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def get_sound_paths():
    """Write temporary WAV files so optional pygame host audio can play them."""
    tmpdir = tempfile.gettempdir()
    paths = {}
    for name in ["sun", "wind", "rain", "reset"]:
        path = os.path.join(tmpdir, f"classroom_weather_{name}.wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(make_wav_bytes(name))
        paths[name] = path
    return paths


def play_local_sound(name):
    """Optional local host sound through pygame. Browser sound is separate."""
    if not st.session_state.get("local_sound", False):
        return

    try:
        import pygame

        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

        sound = pygame.mixer.Sound(get_sound_paths()[name])
        sound.set_volume(0.75)
        sound.play()
    except Exception as e:
        st.session_state.local_sound_error = str(e)


def init_state():
    defaults = {
        "weather": "idle",
        "expression": "neutral",
        "message": "Welcome! Make a big gesture.",
        "bingo": {k: False for k in WEATHERS},
        "fact_idx": {k: 0 for k in WEATHERS},
        "auto_demo": True,
        "browser_sound": True,
        "local_sound": False,
        "use_camera": False,
        "hold_gesture": None,
        "hold_start": 0.0,
        "hold_progress": 0.0,
        "hold_source": None,
        "demo_start": time.time(),
        "demo_script_index": 0,
        "sound_token": 0,
        "last_sound": None,
        "last_sound_time": 0.0,
        "last_event_time": 0.0,
        "last_record_time": 0.0,
        "pose_candidate": None,
        "pose_candidate_start": None,
        "pose_reset_start": None,
        "pose_cooldown": 0.0,
        "camera_frame": None,
        "camera_error": "",
        "local_sound_error": "",
        "scripted_video_bytes": None,
        "scripted_video_mime": "video/mp4",
        "live_video_bytes": None,
        "live_video_mime": "video/mp4",
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if "frame_buffer" not in st.session_state:
        st.session_state.frame_buffer = collections.deque(maxlen=MAX_FRAMES)

    if "pose_history" not in st.session_state:
        st.session_state.pose_history = collections.deque(maxlen=60)


init_state()


def required_hold(kind):
    return RESET_HOLD if kind == "reset" else HOLD_THRESHOLD


def trigger_event(kind, now=None):
    now = now or time.time()

    if kind != "reset" and now - st.session_state.last_event_time < 1.2:
        return

    st.session_state.last_event_time = now

    if kind == "reset":
        st.session_state.bingo = {k: False for k in WEATHERS}
        st.session_state.weather = "idle"
        st.session_state.expression = "surprised"
        st.session_state.message = WEATHER_MESSAGES["reset"]
        st.session_state.last_sound = "reset"
        st.session_state.last_sound_time = now
        st.session_state.sound_token += 1
        play_local_sound("reset")
        return

    if kind not in WEATHERS:
        return

    st.session_state.weather = kind
    st.session_state.bingo[kind] = True
    st.session_state.fact_idx[kind] = (
        st.session_state.fact_idx[kind] + 1
    ) % len(FACTS[kind])
    st.session_state.message = WEATHER_MESSAGES[kind]
    st.session_state.last_sound = kind
    st.session_state.last_sound_time = now
    st.session_state.sound_token += 1

    if kind == "sun":
        st.session_state.expression = "joy"
    elif kind == "wind":
        st.session_state.expression = "excited"
    else:
        st.session_state.expression = "playful"

    play_local_sound(kind)


def press_gesture(kind):
    """
    Button callback.

    This runs before the Streamlit script reruns, so it can safely change
    session state values used by sidebar controls.
    """
    st.session_state.auto_demo = False

    allow_new_hold = (
        kind == "reset"
        or st.session_state.hold_gesture is None
        or st.session_state.hold_source == "demo"
    )

    if allow_new_hold:
        st.session_state.hold_gesture = kind
        st.session_state.hold_start = time.time()
        st.session_state.hold_source = "manual"
        st.session_state.hold_progress = 0.0


def restart_demo():
    """Sidebar callback to restart the automatic demo safely."""
    st.session_state.auto_demo = True
    st.session_state.demo_start = time.time()
    st.session_state.demo_script_index = 0
    st.session_state.bingo = {k: False for k in WEATHERS}
    st.session_state.weather = "idle"
    st.session_state.expression = "neutral"
    st.session_state.message = "Demo restarted!"
    st.session_state.hold_gesture = None
    st.session_state.hold_progress = 0.0
    st.session_state.hold_source = None


def clear_bingo():
    """Sidebar callback to clear bingo safely."""
    st.session_state.bingo = {k: False for k in WEATHERS}
    st.session_state.weather = "idle"
    st.session_state.expression = "neutral"
    st.session_state.message = "Bingo cleared."


def update_simulation(now):
    # If a manual or demo hold is in progress, let it complete.
    if st.session_state.hold_gesture is not None:
        gesture = st.session_state.hold_gesture
        hold_need = required_hold(gesture)
        elapsed = now - st.session_state.hold_start
        st.session_state.hold_progress = min(1.0, elapsed / hold_need)

        if elapsed >= hold_need:
            source = st.session_state.hold_source
            st.session_state.hold_gesture = None
            st.session_state.hold_progress = 0.0
            st.session_state.hold_source = None

            if source == "demo":
                st.session_state.demo_script_index += 1

            trigger_event(gesture, now)
        return

    st.session_state.hold_progress = 0.0

    if not st.session_state.auto_demo:
        return

    # Loop the automatic classroom demo after the final event finishes.
    if st.session_state.demo_script_index >= len(DEMO_SCRIPT):
        if now - st.session_state.demo_start > DEMO_SCRIPT[-1][0] + 7.0:
            st.session_state.demo_start = now
            st.session_state.demo_script_index = 0
            st.session_state.bingo = {k: False for k in WEATHERS}
            st.session_state.weather = "idle"
            st.session_state.expression = "neutral"
            st.session_state.message = "New demo round starting!"
        return

    event_time, event_kind = DEMO_SCRIPT[st.session_state.demo_script_index]
    hold_need = required_hold(event_kind)
    start_time = event_time - hold_need
    elapsed_demo = now - st.session_state.demo_start

    if elapsed_demo >= start_time:
        st.session_state.hold_gesture = event_kind
        st.session_state.hold_start = now
        st.session_state.hold_source = "demo"
        st.session_state.hold_progress = 0.0


@st.cache_resource(show_spinner=False)
def get_pose():
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )


def get_camera():
    if st.session_state.get("camera") is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        st.session_state.camera = cap
    return st.session_state.camera


def auto_adjust_brightness(frame, target=125):
    """Simple calibration-free brightness compensation for changing classroom lights."""
    if frame is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())

    if mean < 1.0:
        return frame

    alpha = target / mean
    alpha = max(0.55, min(alpha, 2.05))

    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


def classify_pose(lm, now):
    """
    Gross-body gesture classification using MediaPipe Pose.

    Approximations:
    - both arms raised -> sun
    - side-to-side arm movement -> wind
    - repeated up/down arm movement -> rain
    - both wrists held close near torso for 3 seconds -> teacher reset
    """
    if len(lm) < 25:
        return None

    important = [11, 12, 15, 16, 23, 24]
    if any(lm[i].visibility < 0.35 for i in important):
        return None

    lsh, rsh, lw, rw, lhip, rhip = [lm[i] for i in important]

    hist = st.session_state.pose_history
    hist.append(
        (
            now,
            lw.x,
            lw.y,
            rw.x,
            rw.y,
            lsh.x,
            lsh.y,
            rsh.x,
            rsh.y,
        )
    )

    while hist and now - hist[0][0] > 2.0:
        hist.popleft()

    # Teacher-reset approximation: both wrists close together near torso.
    wrist_x = (lw.x + rw.x) / 2.0
    wrist_y = (lw.y + rw.y) / 2.0
    shoulder_y = (lsh.y + rsh.y) / 2.0
    hip_y = (lhip.y + rhip.y) / 2.0
    torso_top = shoulder_y
    torso_bottom = max(shoulder_y + 0.15, hip_y)

    close_hands = (
        abs(lw.x - rw.x) < 0.13
        and torso_top < wrist_y < torso_bottom + 0.10
    )

    if close_hands:
        if st.session_state.pose_reset_start is None:
            st.session_state.pose_reset_start = now
        elif now - st.session_state.pose_reset_start >= RESET_HOLD:
            st.session_state.pose_reset_start = None
            return "reset"
    else:
        st.session_state.pose_reset_start = None

    # Both arms raised.
    if lw.y < lsh.y - 0.07 and rw.y < rsh.y - 0.07:
        return "sun"

    if len(hist) < 10:
        return None

    ts = [h[0] for h in hist]
    lxs = [h[1] for h in hist]
    lys = [h[2] for h in hist]
    rxs = [h[3] for h in hist]
    rys = [h[4] for h in hist]
    lsys = [h[6] for h in hist]
    rsys = [h[8] for h in hist]

    duration = max(0.2, ts[-1] - ts[0])

    dx = 0.0
    for i in range(1, len(hist)):
        dx += abs(lxs[i] - lxs[i - 1]) + abs(rxs[i] - rxs[i - 1])

    x_speed = dx / duration

    rel_y = []
    for i in range(len(hist)):
        sh_y = (lsys[i] + rsys[i]) / 2.0
        wrist_y_avg = (lys[i] + rys[i]) / 2.0
        rel_y.append(wrist_y_avg - sh_y)

    v_range = max(rel_y) - min(rel_y)

    changes = 0
    for i in range(2, len(rel_y)):
        d1 = rel_y[i - 1] - rel_y[i - 2]
        d2 = rel_y[i] - rel_y[i - 1]
        if d1 * d2 < 0:
            changes += 1

    # Waving side-to-side.
    if x_speed > 0.55 and v_range < 0.28:
        return "wind"

    # Jumping-jack-like arm movement.
    if v_range > 0.18 and (changes >= 2 or x_speed > 0.45):
        return "rain"

    return None


def handle_pose_label(label, now):
    if now < st.session_state.pose_cooldown:
        return

    if label is None:
        st.session_state.pose_candidate = None
        st.session_state.pose_candidate_start = None
        return

    if label == "reset":
        trigger_event("reset", now)
        st.session_state.pose_cooldown = now + 3.0
        st.session_state.pose_candidate = None
        st.session_state.pose_candidate_start = None
        return

    if st.session_state.hold_gesture is not None:
        return

    if st.session_state.pose_candidate == label:
        if now - st.session_state.pose_candidate_start >= HOLD_THRESHOLD:
            trigger_event(label, now)
            st.session_state.pose_cooldown = now + 2.5
            st.session_state.pose_candidate = None
            st.session_state.pose_candidate_start = None
    else:
        st.session_state.pose_candidate = label
        st.session_state.pose_candidate_start = now


def process_camera(now):
    if not st.session_state.use_camera:
        return

    try:
        cap = get_camera()
        if cap is None or not cap.isOpened():
            raise RuntimeError("Camera could not be opened.")

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera frame could not be read.")

        frame = auto_adjust_brightness(frame)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose = get_pose()
        results = pose.process(rgb)

        st.session_state.camera_frame = frame

        label = None
        if results.pose_landmarks:
            label = classify_pose(results.pose_landmarks.landmark, now)

        handle_pose_label(label, now)

    except Exception as e:
        st.session_state.camera_error = str(e)
        st.session_state.use_camera = False
        st.session_state.camera_frame = None


def draw_text(
    img,
    text,
    org,
    scale=0.7,
    color=(0, 0, 0),
    thickness=2,
    outline_color=None,
    outline_thickness=0,
):
    if outline_color is not None and outline_thickness > 0:
        cv2.putText(
            img,
            text,
            org,
            FONT,
            scale,
            outline_color,
            thickness + outline_thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        text,
        org,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def put_centered_text(
    img,
    text,
    cx,
    cy,
    scale=0.7,
    color=(0, 0, 0),
    thickness=2,
    outline_color=None,
    outline_thickness=0,
):
    text_size, baseline = cv2.getTextSize(text, FONT, scale, thickness)
    tw, th = text_size
    org = (int(cx - tw / 2), int(cy + th / 2))

    draw_text(
        img,
        text,
        org,
        scale=scale,
        color=color,
        thickness=thickness,
        outline_color=outline_color,
        outline_thickness=outline_thickness,
    )


def wrap_text(text, max_chars=52):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_background(img, weather, t):
    h, w = img.shape[:2]

    if weather == "sun":
        img[:] = (70, 205, 255)
    elif weather == "wind":
        img[:] = (210, 235, 200)
    elif weather == "rain":
        img[:] = (150, 120, 80)
    else:
        img[:] = (235, 225, 180)

    cv2.rectangle(img, (0, int(h * 0.85)), (w, h), (70, 170, 70), -1)


def draw_weather_effects(img, weather, t):
    h, w = img.shape[:2]
    s = h / 540.0

    if weather == "sun":
        cx = int(w * 0.82)
        cy = int(h * 0.20)
        r = int(70 * s)

        cv2.circle(img, (cx, cy), r, (0, 225, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (0, 180, 255), max(2, int(4 * s)), cv2.LINE_AA)

        for k in range(12):
            ang = 2.0 * math.pi * k / 12.0 + t * 0.8
            x1 = int(cx + math.cos(ang) * (r + 10 * s))
            y1 = int(cy + math.sin(ang) * (r + 10 * s))
            x2 = int(cx + math.cos(ang) * (r + 45 * s))
            y2 = int(cy + math.sin(ang) * (r + 45 * s))
            cv2.line(img, (x1, y1), (x2, y2), (0, 200, 255), max(3, int(6 * s)), cv2.LINE_AA)

    elif weather == "wind":
        for i in range(6):
            y_base = int(h * (0.18 + 0.12 * i))
            phase = t * 2.2 + i * 1.4
            pts = []
            step = max(8, int(18 * s))

            for x in range(-100, w + 100, step):
                y = y_base + int(18 * s * math.sin(x / max(20.0, 70 * s) + phase))
                pts.append((x, y))

            if len(pts) >= 2:
                cv2.polylines(
                    img,
                    np.array([pts], dtype=np.int32),
                    False,
                    (250, 250, 250),
                    max(2, int(4 * s)),
                    cv2.LINE_AA,
                )

    elif weather == "rain":
        limit_y = int(h * 0.82)

        for i in range(110):
            x = (i * 89 + int(t * 280)) % w
            y = (i * 53 + int(t * 640)) % limit_y
            length = int(22 * s)
            cv2.line(
                img,
                (x, y),
                (x - int(4 * s), y + length),
                (230, 200, 120),
                max(1, int(3 * s)),
                cv2.LINE_AA,
            )


def draw_cloud_character(img, cx, cy, r, expression, weather, t, s):
    cy += int(6 * s * math.sin(t * 2.0))

    outline = (35, 35, 35)
    fill = (250, 250, 250)

    if weather == "rain":
        fill = (235, 235, 235)
    elif weather == "wind":
        fill = (245, 255, 255)

    circles = [
        (cx - int(r * 0.78), cy + int(r * 0.28), int(r * 0.55)),
        (cx, cy - int(r * 0.32), int(r * 0.68)),
        (cx + int(r * 0.78), cy + int(r * 0.28), int(r * 0.55)),
        (cx, cy + int(r * 0.18), int(r * 0.82)),
    ]

    for x, y, rr in circles:
        cv2.circle(img, (x, y), rr + max(2, int(4 * s)), outline, -1, cv2.LINE_AA)

    for x, y, rr in circles:
        cv2.circle(img, (x, y), rr, fill, -1, cv2.LINE_AA)

    eye_y = cy - int(r * 0.08)
    left_eye = (cx - int(r * 0.30), eye_y)
    right_eye = (cx + int(r * 0.30), eye_y)
    eye_color = (30, 30, 30)

    if expression == "joy":
        for eye in (left_eye, right_eye):
            cv2.ellipse(
                img,
                eye,
                (int(r * 0.13), int(r * 0.08)),
                0,
                200,
                340,
                eye_color,
                max(2, int(4 * s)),
                cv2.LINE_AA,
            )

    elif expression == "playful":
        cv2.ellipse(
            img,
            left_eye,
            (int(r * 0.13), int(r * 0.08)),
            0,
            200,
            340,
            eye_color,
            max(2, int(4 * s)),
            cv2.LINE_AA,
        )
        cv2.circle(img, right_eye, max(2, int(r * 0.07)), eye_color, -1, cv2.LINE_AA)

    else:
        eye_r = max(2, int(r * 0.07))
        if expression == "excited":
            eye_r = max(3, int(r * 0.09))
        if expression == "surprised":
            eye_r = max(3, int(r * 0.11))

        cv2.circle(img, left_eye, eye_r, eye_color, -1, cv2.LINE_AA)
        cv2.circle(img, right_eye, eye_r, eye_color, -1, cv2.LINE_AA)

    mouth_center = (cx, cy + int(r * 0.28))

    if expression == "joy":
        cv2.ellipse(
            img,
            mouth_center,
            (int(r * 0.34), int(r * 0.22)),
            0,
            15,
            165,
            eye_color,
            max(2, int(5 * s)),
            cv2.LINE_AA,
        )

    elif expression == "excited":
        cv2.ellipse(
            img,
            mouth_center,
            (int(r * 0.26), int(r * 0.20)),
            0,
            0,
            360,
            eye_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            img,
            (mouth_center[0], mouth_center[1] + int(r * 0.08)),
            (int(r * 0.12), int(r * 0.07)),
            0,
            0,
            360,
            (90, 90, 235),
            -1,
            cv2.LINE_AA,
        )

    elif expression == "playful":
        cv2.ellipse(
            img,
            mouth_center,
            (int(r * 0.30), int(r * 0.18)),
            0,
            15,
            165,
            eye_color,
            max(2, int(5 * s)),
            cv2.LINE_AA,
        )
        cv2.ellipse(
            img,
            (mouth_center[0] + int(r * 0.08), mouth_center[1] + int(r * 0.09)),
            (int(r * 0.09), int(r * 0.06)),
            0,
            0,
            360,
            (90, 90, 235),
            -1,
            cv2.LINE_AA,
        )

    elif expression == "surprised":
        cv2.circle(img, mouth_center, max(3, int(r * 0.10)), eye_color, -1, cv2.LINE_AA)

    else:
        cv2.line(
            img,
            (cx - int(r * 0.18), mouth_center[1]),
            (cx + int(r * 0.18), mouth_center[1]),
            eye_color,
            max(2, int(4 * s)),
            cv2.LINE_AA,
        )

    if weather == "sun":
        cheek_color = (120, 170, 255)
        cv2.circle(
            img,
            (cx - int(r * 0.52), cy + int(r * 0.12)),
            max(3, int(r * 0.10)),
            cheek_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            img,
            (cx + int(r * 0.52), cy + int(r * 0.12)),
            max(3, int(r * 0.10)),
            cheek_color,
            -1,
            cv2.LINE_AA,
        )

    if weather == "rain":
        for i in range(3):
            dx = int(r * (-0.35 + 0.35 * i))
            drop_y = cy + int(r * 0.95) + int((t * 170 + i * 37) % max(1, int(65 * s)))
            cv2.ellipse(
                img,
                (cx + dx, drop_y),
                (max(2, int(5 * s)), max(3, int(8 * s))),
                0,
                0,
                360,
                (235, 205, 120),
                -1,
                cv2.LINE_AA,
            )


def draw_status(img, message, weather, s):
    h, w = img.shape[:2]

    weather_label = {
        "idle": "READY!",
        "sun": "SUNSHINE",
        "wind": "WIND",
        "rain": "RAIN",
    }.get(weather, "READY")

    message = str(message)
    if len(message) > 62:
        message = message[:59] + "..."

    panel_w = int(min(w * 0.58, 620 * s))
    panel_h = int(72 * s)
    x = int((w - panel_w) / 2)
    y = int(14 * s)

    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), max(2, int(3 * s)))

    put_centered_text(
        img,
        weather_label,
        x + panel_w / 2,
        y + panel_h * 0.32,
        0.95 * s,
        (0, 120, 220),
        max(2, int(3 * s)),
    )

    put_centered_text(
        img,
        message,
        x + panel_w / 2,
        y + panel_h * 0.78,
        0.52 * s,
        (40, 40, 40),
        max(1, int(2 * s)),
    )


def draw_bingo(img, bingo, s):
    h, w = img.shape[:2]

    pw = int(235 * s)
    ph = int(175 * s)
    x = w - pw - int(14 * s)
    y = int(14 * s)

    cv2.rectangle(img, (x, y), (x + pw, y + ph), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (30, 30, 30), max(2, int(3 * s)))

    draw_text(
        img,
        "WEATHER BINGO",
        (x + int(12 * s), y + int(30 * s)),
        0.62 * s,
        (30, 30, 30),
        max(1, int(2 * s)),
    )

    labels = {
        "sun": "SUN",
        "wind": "WIND",
        "rain": "RAIN",
    }

    for i, k in enumerate(WEATHERS):
        done = bool(bingo.get(k, False))
        mark = "[X]" if done else "[ ]"
        color = (0, 170, 0) if done else (90, 90, 90)
        text = f"{mark} {labels[k]}"

        draw_text(
            img,
            text,
            (x + int(16 * s), y + int(62 * s + i * 32 * s)),
            0.60 * s,
            color,
            max(1, int(2 * s)),
        )

    if all(bingo.get(k, False) for k in WEATHERS):
        draw_text(
            img,
            "BINGO!",
            (x + int(16 * s), y + ph - int(16 * s)),
            0.72 * s,
            (0, 180, 255),
            max(2, int(3 * s)),
            outline_color=(30, 30, 30),
            outline_thickness=2,
        )


def draw_fact_panel(img, weather, fact_idx, s):
    h, w = img.shape[:2]

    x = int(14 * s)
    y = h - int(132 * s)
    pw = w - int(28 * s)
    ph = int(118 * s)

    cv2.rectangle(img, (x, y), (x + pw, y + ph), (90, 45, 15), -1)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (220, 220, 220), max(2, int(2 * s)))

    if weather in FACTS:
        title = f"{weather.upper()} FACT"
        idx = int(fact_idx.get(weather, 0)) % len(FACTS[weather])
        fact = FACTS[weather][idx]
    else:
        title = "WEATHER FACT"
        fact = "Raise arms for sun, wave for wind, and jumping-jack arms for rain."

    draw_text(
        img,
        title,
        (x + int(18 * s), y + int(32 * s)),
        0.72 * s,
        (0, 230, 255),
        max(2, int(2 * s)),
    )

    lines = wrap_text(fact, 54)

    for i, line in enumerate(lines[:3]):
        draw_text(
            img,
            line,
            (x + int(18 * s), y + int(64 * s + i * 30 * s)),
            0.66 * s,
            (255, 255, 255),
            max(1, int(2 * s)),
        )


def draw_hold_progress(img, gesture, progress, s):
    h, w = img.shape[:2]

    labels = {
        "sun": "Hold both arms up high",
        "wind": "Wave arms side to side",
        "rain": "Move arms up and down",
        "reset": "Teacher reset: hold both hands still",
    }

    label = labels.get(gesture, "Hold gesture")
    hold_need = required_hold(gesture)

    pw = int(min(w * 0.72, 680 * s))
    ph = int(120 * s)
    x = int((w - pw) / 2)
    y = int(h * 0.30)

    cv2.rectangle(img, (x, y), (x + pw, y + ph), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (255, 255, 255), max(2, int(3 * s)))

    put_centered_text(
        img,
        label,
        x + pw / 2,
        y + ph * 0.28,
        0.75 * s,
        (255, 255, 255),
        max(2, int(2 * s)),
    )

    bar_x = x + int(24 * s)
    bar_y = y + int(68 * s)
    bar_w = pw - int(48 * s)
    bar_h = int(26 * s)

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (240, 240, 240), -1)

    fill_w = int(bar_w * max(0.0, min(1.0, progress)))
    fill_color = (0, 80, 255) if gesture == "reset" else (0, 180, 0)

    if fill_w > 0:
        cv2.rectangle(
            img,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            fill_color,
            -1,
        )

    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 20), 2)

    put_centered_text(
        img,
        f"{progress * hold_need:.1f}s / {hold_need:.1f}s",
        x + pw / 2,
        y + ph * 0.86,
        0.50 * s,
        (220, 220, 220),
        max(1, int(1 * s)),
    )


def draw_badges(img, badges, s):
    h, w = img.shape[:2]
    y = int(h * 0.52)

    for badge in badges:
        draw_text(
            img,
            badge,
            (int(16 * s), y),
            0.55 * s,
            (255, 255, 255),
            max(1, int(2 * s)),
            outline_color=(0, 0, 0),
            outline_thickness=3,
        )
        y += int(28 * s)


def draw_bingo_complete(img, t, s):
    h, w = img.shape[:2]
    scale = (1.1 + 0.1 * math.sin(t * 4.0)) * s

    put_centered_text(
        img,
        "BINGO! Great job!",
        w / 2,
        h * 0.22,
        scale,
        (0, 200, 255),
        max(2, int(4 * s)),
        outline_color=(0, 0, 0),
        outline_thickness=4,
    )


def draw_scene(img, t, view):
    h, w = img.shape[:2]
    s = h / 540.0

    weather = view.get("weather", "idle")
    expression = view.get("expression", "neutral")
    bingo = view.get("bingo") or {k: False for k in WEATHERS}

    camera_frame = view.get("camera_frame")

    if camera_frame is not None:
        cam = cv2.resize(camera_frame, (w, h))
        img[:] = cv2.addWeighted(cam, 0.72, np.zeros_like(cam), 0.28, 0)
    else:
        draw_background(img, weather, t)

    draw_weather_effects(img, weather, t)

    draw_cloud_character(
        img,
        int(w * 0.14),
        int(h * 0.27),
        int(82 * s),
        expression,
        weather,
        t,
        s,
    )

    draw_status(img, view.get("message", ""), weather, s)
    draw_bingo(img, bingo, s)
    draw_fact_panel(img, weather, view.get("fact_idx", {}), s)

    if view.get("hold_gesture"):
        draw_hold_progress(img, view["hold_gesture"], view.get("hold_progress", 0.0), s)

    draw_badges(img, view.get("badges", []), s)

    if all(bingo.get(k, False) for k in WEATHERS) and weather != "idle":
        draw_bingo_complete(img, t, s)


def create_writer(filename, fps, size):
    codecs = []

    if filename.lower().endswith(".mp4"):
        codecs = ["mp4v", "avc1"]
    else:
        codecs = ["MJPG"]

    for codec in codecs:
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer

    return None


def render_scripted_video(seconds=60, fps=8):
    """Render a self-contained demo video using the same drawing engine."""
    size = (RECORD_W, RECORD_H)
    stamp = int(time.time() * 1000)

    filename = os.path.join(tempfile.gettempdir(), f"weather_demo_{stamp}.mp4")
    writer = create_writer(filename, fps, size)
    mime = "video/mp4"

    if writer is None:
        filename = os.path.join(tempfile.gettempdir(), f"weather_demo_{stamp}.avi")
        writer = create_writer(filename, fps, size)
        mime = "video/avi"

    if writer is None:
        return None, None

    local_bingo = {k: False for k in WEATHERS}
    local_fact_idx = {k: 0 for k in WEATHERS}
    weather = "idle"
    message = "Welcome! Make a big gesture."
    expression = "neutral"

    triggered = [False] * len(DEMO_SCRIPT)

    try:
        total_frames = int(seconds * fps)

        for frame_i in range(total_frames):
            t = frame_i / float(fps)

            hold = None
            progress = 0.0

            for i, (event_time, event_kind) in enumerate(DEMO_SCRIPT):
                if triggered[i]:
                    continue

                hold_need = required_hold(event_kind)
                if event_time - hold_need <= t < event_time:
                    hold = event_kind
                    progress = (t - (event_time - hold_need)) / hold_need
                    break

            for i, (event_time, event_kind) in enumerate(DEMO_SCRIPT):
                if triggered[i] or t < event_time:
                    continue

                triggered[i] = True

                if event_kind == "reset":
                    local_bingo = {k: False for k in WEATHERS}
                    weather = "idle"
                    message = WEATHER_MESSAGES["reset"]
                    expression = "surprised"
                else:
                    weather = event_kind
                    local_bingo[event_kind] = True
                    local_fact_idx[event_kind] = (
                        local_fact_idx[event_kind] + 1
                    ) % len(FACTS[event_kind])
                    message = WEATHER_MESSAGES[event_kind]

                    if event_kind == "sun":
                        expression = "joy"
                    elif event_kind == "wind":
                        expression = "excited"
                    else:
                        expression = "playful"

            view = {
                "weather": weather,
                "expression": expression,
                "message": message,
                "bingo": local_bingo,
                "fact_idx": local_fact_idx,
                "hold_gesture": hold,
                "hold_progress": progress,
                "camera_frame": None,
                "badges": ["AUTO DEMO VIDEO", "OFFLINE READY"],
            }

            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            draw_scene(frame, t, view)

            small = cv2.resize(frame, size)
            writer.write(small)

        writer.release()

        with open(filename, "rb") as f:
            data = f.read()

        return data, mime

    finally:
        try:
            if os.path.exists(filename):
                os.unlink(filename)
        except Exception:
            pass


def frames_to_video_bytes(frames):
    if not frames:
        return None, None

    stamp = int(time.time() * 1000)
    filename = os.path.join(tempfile.gettempdir(), f"weather_live_{stamp}.mp4")
    writer = create_writer(filename, RECORD_FPS, (RECORD_W, RECORD_H))
    mime = "video/mp4"

    if writer is None:
        filename = os.path.join(tempfile.gettempdir(), f"weather_live_{stamp}.avi")
        writer = create_writer(filename, RECORD_FPS, (RECORD_W, RECORD_H))
        mime = "video/avi"

    if writer is None:
        return None, None

    try:
        for jpg in frames:
            arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if arr is None:
                continue

            if arr.shape[1] != RECORD_W or arr.shape[0] != RECORD_H:
                arr = cv2.resize(arr, (RECORD_W, RECORD_H))

            writer.write(arr)

        writer.release()

        with open(filename, "rb") as f:
            data = f.read()

        return data, mime

    finally:
        try:
            if os.path.exists(filename):
                os.unlink(filename)
        except Exception:
            pass


def tick():
    now = time.time()

    try:
        if st.session_state.use_camera:
            process_camera(now)
        else:
            st.session_state.camera_frame = None

        update_simulation(now)

        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        badges = []
        if st.session_state.auto_demo:
            badges.append("AUTO DEMO")
        if st.session_state.use_camera:
            badges.append("CAMERA + POSE")
        if st.session_state.local_sound:
            badges.append("PYGAME SOUND")

        view = {
            "weather": st.session_state.weather,
            "expression": st.session_state.expression,
            "message": st.session_state.message,
            "bingo": st.session_state.bingo,
            "fact_idx": st.session_state.fact_idx,
            "hold_gesture": st.session_state.hold_gesture,
            "hold_progress": st.session_state.hold_progress,
            "camera_frame": st.session_state.camera_frame,
            "badges": badges,
        }

        draw_scene(frame, now, view)

        st.image(frame, channels="BGR", use_container_width=True)

        if now - st.session_state.last_record_time >= 1.0 / RECORD_FPS:
            small = cv2.resize(frame, (RECORD_W, RECORD_H))
            ok, jpg = cv2.imencode(
                ".jpg",
                small,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            if ok:
                st.session_state.frame_buffer.append(jpg.tobytes())
            st.session_state.last_record_time = now

        if (
            st.session_state.browser_sound
            and st.session_state.last_sound
            and now - st.session_state.last_sound_time < 1.2
        ):
            st.audio(
                make_wav_bytes(st.session_state.last_sound),
                format="audio/wav",
                autoplay=True,
                key=f"audio_{st.session_state.sound_token}",
            )

    except Exception as e:
        st.error(f"Live demo error: {e}")


if fragment is not None:

    @fragment(run_every="200ms")
    def live_canvas():
        tick()

else:

    def live_canvas():
        tick()
        if st.button("Refresh live view"):
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()


st.title("🌦️ Weather Gesture Classroom")
st.caption(
    "Big, deliberate classroom gestures trigger playful weather. "
    "The demo starts automatically. Use the buttons below for manual interaction."
)

st.sidebar.title("Teacher controls")

# These checkboxes intentionally do not use `key=...` because some values
# are changed programmatically by button callbacks.
st.session_state.auto_demo = st.sidebar.checkbox(
    "Auto demo",
    value=st.session_state.auto_demo,
)
st.session_state.browser_sound = st.sidebar.checkbox(
    "Browser sound",
    value=st.session_state.browser_sound,
)
st.session_state.local_sound = st.sidebar.checkbox(
    "Local pygame sound",
    value=st.session_state.local_sound,
)
st.session_state.use_camera = st.sidebar.checkbox(
    "Use camera + MediaPipe Pose",
    value=st.session_state.use_camera,
)

if st.session_state.camera_error:
    st.sidebar.error(f"Camera issue: {st.session_state.camera_error}")

if st.session_state.local_sound_error:
    st.sidebar.warning(f"Pygame sound issue: {st.session_state.local_sound_error}")

st.sidebar.button(
    "Restart demo",
    on_click=restart_demo,
    use_container_width=True,
)

st.sidebar.button(
    "Clear bingo now",
    on_click=clear_bingo,
    use_container_width=True,
)

st.sidebar.caption(
    "Teacher reset is approximated from pose when camera mode is enabled: "
    "hold both wrists close together near the torso for 3 seconds."
)

bcols = st.columns(4)

bcols[0].button(
    "☀️ Sunshine",
    on_click=press_gesture,
    args=("sun",),
    use_container_width=True,
)

bcols[1].button(
    "💨 Wind",
    on_click=press_gesture,
    args=("wind",),
    use_container_width=True,
)

bcols[2].button(
    "🌧️ Rain",
    on_click=press_gesture,
    args=("rain",),
    use_container_width=True,
)

bcols[3].button(
    "🔁 Teacher Reset",
    on_click=press_gesture,
    args=("reset",),
    use_container_width=True,
)

left_col, right_col = st.columns([3, 2])

with left_col:
    live_canvas()

with right_col:
    current_weather = st.session_state.weather

    weather_title = {
        "idle": "Ready for gestures",
        "sun": "Sunshine",
        "wind": "Wind",
        "rain": "Rain",
    }.get(current_weather, "Ready")

    st.markdown(
        f"<div style='font-size:40px; font-weight:800;'>{weather_title}</div>",
        unsafe_allow_html=True,
    )

    if current_weather in FACTS:
        idx = st.session_state.fact_idx[current_weather] % len(FACTS[current_weather])
        fact = FACTS[current_weather][idx]
    else:
        fact = (
            "Raise both arms for sunshine, wave arms side-to-side for wind, "
            "and make jumping-jack arm motions for rain."
        )

    st.markdown(
        f"<div style='font-size:26px; line-height:1.35;'>{fact}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    bingo_html = ["<div style='font-size:30px; font-weight:700;'>Weather Bingo</div>"]

    labels = {
        "sun": "Sunshine",
        "wind": "Wind",
        "rain": "Rain",
    }

    for k in WEATHERS:
        icon = "✅" if st.session_state.bingo.get(k, False) else "⬜"
        bingo_html.append(
            f"<div style='font-size:26px; margin-top:8px;'>{icon} {labels[k]}</div>"
        )

    if all(st.session_state.bingo.get(k, False) for k in WEATHERS):
        bingo_html.append(
            "<div style='font-size:32px; font-weight:900; color:#b45309; margin-top:12px;'>"
            "🎉 BINGO!"
            "</div>"
        )

    st.markdown("".join(bingo_html), unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:20px; color:#333;'>"
        "Tip: turn off Auto demo in the sidebar and let students use the big buttons "
        "or camera gestures."
        "</div>",
        unsafe_allow_html=True,
    )

st.subheader("Demo video export")

vcol1, vcol2 = st.columns(2)

if vcol1.button("Generate instant 60-second demo video", use_container_width=True):
    with st.spinner("Rendering 60-second demo video..."):
        data, mime = render_scripted_video(seconds=60, fps=RECORD_FPS)
        st.session_state.scripted_video_bytes = data
        st.session_state.scripted_video_mime = mime or "video/mp4"

        if data is None:
            st.warning("OpenCV could not open a video writer on this machine.")

if vcol2.button("Export current live recording (up to 60s)", use_container_width=True):
    with st.spinner("Encoding current live recording..."):
        frames = list(st.session_state.frame_buffer)
        data, mime = frames_to_video_bytes(frames)
        st.session_state.live_video_bytes = data
        st.session_state.live_video_mime = mime or "video/mp4"

        if data is None:
            st.warning("No recorded frames are available yet.")

if st.session_state.scripted_video_bytes:
    st.video(
        st.session_state.scripted_video_bytes,
        format=st.session_state.scripted_video_mime,
    )

    ext = "mp4" if st.session_state.scripted_video_mime == "video/mp4" else "avi"

    st.download_button(
        "Download 60-second demo video",
        data=st.session_state.scripted_video_bytes,
        file_name=f"weather_classroom_demo_60s.{ext}",
        mime=st.session_state.scripted_video_mime,
        use_container_width=True,
    )

if st.session_state.live_video_bytes:
    st.video(
        st.session_state.live_video_bytes,
        format=st.session_state.live_video_mime,
    )

    ext = "mp4" if st.session_state.live_video_mime == "video/mp4" else "avi"

    st.download_button(
        "Download live recording",
        data=st.session_state.live_video_bytes,
        file_name=f"weather_live_recording.{ext}",
        mime=st.session_state.live_video_mime,
        use_container_width=True,
    )

st.caption(
    "Offline-first classroom build: local sprite-style rendering, local synthetic sounds, "
    "no API key, and no external internet calls."
)