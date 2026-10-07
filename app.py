
import os
import cv2
import json
import time
import math
import threading
import tempfile
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Dict

from pydantic import BaseModel
from ollama import chat

import numpy as np
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

VISION_MODEL = os.environ.get(
    "COFFEE_VISION_MODEL",
    "qwen3-vl:4b-instruct"
)
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

CAMERA_INDEX = int(os.environ.get("COFFEE_CAMERA", "0"))

VISION_INTERVAL = 1.0
FRAME_HISTORY_SECONDS = 5.0
VISION_FRAME_COUNT = 2

STEEP_SECONDS = 30
ENABLE_TTS = os.environ.get("COFFEE_TTS", "1") != "0"

MODEL_PATH = "models/hand_landmarker.task"
WINDOW_NAME = "Embodied Coffee Coach v1 Vision"


# ============================================================
# CLOSED WORLD VOCABULARY
# ============================================================

OBJECTS = [
    "coffee_bag",
    "hand_grinder",
    "aeropress",
    "stir_paddle",
    "kettle",
    "glass_cup",
]

ACTIONS = [
    "idle",
    "pour_beans_to_grinder",
    "grinding",
    "pour_grounds_to_aeropress",
    "pour_water_to_aeropress",
    "stirring",
    "waiting",
    "flip_aeropress",
    "pressing",
    "pour_coffee_to_cup",
    "unknown",
]

STAGES = [
    {
        "id": "LOAD_BEANS",
        "title": "1 / 7  Load beans",
        "instruction": "Pour coffee beans from the bag into the hand grinder.",
        "expected": ["pour_beans_to_grinder"],
    },
    {
        "id": "GRIND",
        "title": "2 / 7  Grind",
        "instruction": "Grind the beans with a steady circular motion.",
        "expected": ["grinding"],
    },
    {
        "id": "TRANSFER",
        "title": "3 / 7  Transfer grounds",
        "instruction": "Pour the ground coffee into the AeroPress.",
        "expected": ["pour_grounds_to_aeropress"],
    },
    {
        "id": "POUR_WATER",
        "title": "4 / 7  Add water",
        "instruction": "Pour water into the AeroPress.",
        "expected": ["pour_water_to_aeropress"],
    },
    {
        "id": "STIR",
        "title": "5 / 7  Stir",
        "instruction": "Stir the coffee with the AeroPress paddle.",
        "expected": ["stirring"],
    },
    {
        "id": "STEEP",
        "title": "6 / 7  Steep",
        "instruction": "Leave the coffee untouched for 30 seconds.",
        "expected": ["waiting", "idle"],
    },
    {
        "id": "PRESS",
        "title": "7 / 7  Flip and press",
        "instruction": "Place the cap, carefully invert the AeroPress onto the cup, and press all the way down.",
        "expected": ["flip_aeropress", "pressing"],
    },
    {
        "id": "DONE",
        "title": "Coffee ready",
        "instruction": "Brewing complete. Remove the AeroPress and enjoy your coffee.",
        "expected": [],
    },
]


# ============================================================
# BASIC HELPERS
# ============================================================

def speak(text):
    if not ENABLE_TTS:
        return
    try:
        subprocess.Popen(
            ["say", str(text)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))



def wrap_text(text, max_chars=42):
    words = str(text).split()
    lines, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def put_lines(img, lines, x, y, scale=0.50, gap=22, max_lines=4):
    for line in lines[:max_lines]:
        cv2.putText(
            img, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, (235, 235, 235), 1, cv2.LINE_AA
        )
        y += gap
    return y


# ============================================================
# MEDIAPIPE HAND MOTION
# ============================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


@dataclass
class HandSample:
    t: float
    centers: Dict[str, tuple]


class MotionTracker:
    def __init__(self, seconds=3.0):
        self.seconds = seconds
        self.samples = deque()

    def clear(self):
        self.samples.clear()

    def add(self, sample):
        self.samples.append(sample)
        cutoff = sample.t - self.seconds
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def summarize(self):
        samples = list(self.samples)
        if len(samples) < 2:
            return {
                "hands_visible": False,
                "both_hands_ratio": 0.0,
                "path_length": 0.0,
                "repetitive_motion": False,
                "vertical_direction": "none",
                "motion_level": "none",
            }

        valid = [s for s in samples if s.centers]
        if not valid:
            return {
                "hands_visible": False,
                "both_hands_ratio": 0.0,
                "path_length": 0.0,
                "repetitive_motion": False,
                "vertical_direction": "none",
                "motion_level": "none",
            }

        def centroid(s):
            pts = list(s.centers.values())
            return (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )

        pts = [centroid(s) for s in valid]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        path = sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))

        def reversals(vals, deadband=0.003):
            d = np.diff(vals)
            signs = [1 if x > 0 else -1 for x in d if abs(x) > deadband]
            return sum(a != b for a, b in zip(signs[:-1], signs[1:]))

        rev = reversals(xs) + reversals(ys)
        repetitive = bool(rev >= 4 and path > 0.15)

        dy = ys[-1] - ys[0]
        if dy > 0.05:
            vertical = "down"
        elif dy < -0.05:
            vertical = "up"
        else:
            vertical = "stable"

        duration = max(0.1, valid[-1].t - valid[0].t)
        speed = path / duration
        if speed < 0.04:
            level = "still"
        elif speed < 0.14:
            level = "low"
        elif speed < 0.30:
            level = "medium"
        else:
            level = "high"

        both = sum(len(s.centers) >= 2 for s in samples) / len(samples)

        return {
            "hands_visible": True,
            "both_hands_ratio": round(float(both), 2),
            "path_length": round(float(path), 3),
            "repetitive_motion": repetitive,
            "vertical_direction": vertical,
            "motion_level": level,
        }


def create_landmarker():
    if not os.path.exists(MODEL_PATH):
        return None

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def centers_from_result(result):
    out = {}
    if result is None or not result.hand_landmarks:
        return out

    for i, landmarks in enumerate(result.hand_landmarks):
        label = f"hand_{i}"
        if i < len(result.handedness) and result.handedness[i]:
            label = result.handedness[i][0].category_name.lower()

        ids = [0, 5, 9, 13, 17]
        x = sum(landmarks[j].x for j in ids) / len(ids)
        y = sum(landmarks[j].y for j in ids) / len(ids)
        out[label] = (float(x), float(y))

    return out


def draw_hands(frame, result):
    if result is None or not result.hand_landmarks:
        return

    h, w = frame.shape[:2]
    for landmarks in result.hand_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (230,230,230), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (255,255,255), -1, cv2.LINE_AA)


# ============================================================
# MULTI FRAME VISION LANGUAGE MODEL
# ============================================================

class VisionOutput(BaseModel):
    visible_objects: list[str]
    action: str
    confidence: float
    evidence: str
    guidance: str
    stage_complete: bool


class VisionReasoner:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.error = None
        self.request_serial = 0
        self.result_serial = 0
        self.result = {
            "visible_objects": [],
            "action": "idle",
            "confidence": 0.0,
            "evidence": "Waiting for visual analysis.",
            "guidance": STAGES[0]["instruction"],
            "stage_complete": False,
            "stage_id": "LOAD_BEANS",
            "serial": 0,
        }

    def request(self, frames, stage, hand_motion, history):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.request_serial += 1
            serial = self.request_serial

        copies = [f.copy() for f in frames]
        threading.Thread(
            target=self._worker,
            args=(copies, stage, hand_motion, history, serial),
            daemon=True
        ).start()
        return True

    def _worker(self, frames, stage, hand_motion, history, serial):
        descriptions = """
The physical objects in this prototype are:

coffee_bag:
A colorful resealable coffee bean bag.

hand_grinder:
A tall black cylindrical manual coffee grinder with a long metal crank and black knob.
It is thick, cylindrical, and much larger than the stirring paddle.

stir_paddle:
A long, thin, flat black plastic AeroPress paddle. It has no crank, no cylindrical body,
and is usually held by one end and moved inside the transparent AeroPress chamber.

aeropress:
A transparent plastic AeroPress chamber and plunger. During the inverted method it may
stand vertically with the open chamber at the top. It can contain visible coffee grounds.

kettle:
A hot water container used for coffee brewing. It may be a glass kettle, insulated bottle, thermos, travel tumbler, or similar container. Identify it by its role as the source of hot water rather than by a specific shape, color, or material.

glass_cup:
A short clear cylindrical drinking glass.

The user performs the inverted AeroPress workflow:
coffee bag -> hand grinder -> grind -> grounds into AeroPress -> water -> stir ->
wait 30 seconds -> place cap -> flip AeroPress onto glass -> press until complete.
"""

        prompt = f"""
You are the visual perception and reasoning component of an embodied coffee coach.

You are given TWO webcam frames ordered from EARLIEST to LATEST. Treat them as a short
temporal sequence and compare the visible change between them.

{descriptions}

SUPPLEMENTARY HAND MOTION:
{json.dumps(hand_motion)}

RECENT COMPLETED STAGES:
{json.dumps(history[-4:])}

Only use these object labels:
{json.dumps(OBJECTS)}

Only use these action labels:
{json.dumps(ACTIONS)}

Important reasoning rules:
1. Identify objects from appearance and context.
2. Infer an action from changes across the three frames.
3. The user's hands often partially occlude objects.
4. If a black cylinder with a long crank is being rotated repeatedly, prefer "grinding".
5. If the colorful coffee bag is tipped above the grinder, prefer "pour_beans_to_grinder".
6. If the grinder is tipped over the transparent AeroPress, prefer "pour_grounds_to_aeropress".
7. If a hot water container is tipped toward the AeroPress, prefer "pour_water_to_aeropress".
8. If the black paddle moves inside the AeroPress, prefer "stirring".
9. If the AeroPress is being inverted toward the glass, prefer "flip_aeropress".
10. If both hands push vertically down on the AeroPress over the glass, prefer "pressing".
11. Pressing is the final brewing action. Once a sustained downward press has been observed and then stops, the brew is complete.
12. Be conservative if visibility is poor.
13. stage_complete should mean there is enough evidence that the user has successfully
    performed the current expected action, not merely that an object is visible.
14. Object presence is NOT evidence that an action happened. A coffee bag next to a grinder
    is not "pour_beans_to_grinder". A kettle next to an AeroPress is not "pour_water_to_aeropress".
15. If the three frames do not show a meaningful temporal change, return action="idle".
16. You are NOT told what the user is supposed to do next. Classify only what is visibly happening.
17. For pouring, require the source object and destination object to both be visible, with the
    source above the destination and a changed tilt across frames.
18. For grinding, require changed crank orientation or repeated hand movement around the grinder.
19. For stirring, require the paddle to visibly move inside the AeroPress.
20. For pressing, require visible downward displacement of the plunger across frames.

Return valid JSON only:
{{
  "visible_objects": ["..."],
  "action": "one action label",
  "confidence": 0.0,
  "evidence": "one concise sentence explaining the visible evidence",
  "guidance": "one concise coaching sentence",
  "stage_complete": false
}}
"""

        temp_dir = None

        try:
            # Use the official Ollama Python client and real JPEG file paths.
            # This follows Ollama's documented vision API path and avoids
            # hand-building the base64 request.
            temp_dir = tempfile.mkdtemp(prefix="coffee_vision_")
            image_paths = []

            for i, frame in enumerate(frames):
                h, w = frame.shape[:2]
                if w > 512:
                    scale = 512 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                path = os.path.join(temp_dir, f"frame_{i}.jpg")
                ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if not ok:
                    raise RuntimeError(f"Could not save temporary vision frame {i}")
                image_paths.append(path)

            response = chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": image_paths,
                }],
                format=VisionOutput.model_json_schema(),
                think=False,
                stream=False,
                options={
                    "temperature": 0,
                    "num_predict": 120,
                },
                keep_alive="15m",
            )

            parsed_model = VisionOutput.model_validate_json(response.message.content)
            parsed = parsed_model.model_dump()

            objects = [
                x for x in parsed.get("visible_objects", [])
                if x in OBJECTS
            ]

            action = parsed.get("action", "unknown")
            if action not in ACTIONS:
                action = "unknown"

            result = {
                "visible_objects": objects,
                "action": action,
                "confidence": clamp(float(parsed.get("confidence", 0.0)), 0.0, 1.0),
                "evidence": str(parsed.get("evidence", "")),
                "guidance": str(parsed.get("guidance", "")),
                "stage_complete": bool(parsed.get("stage_complete", False)),
                "stage_id": stage["id"],
                "serial": serial,
            }

            print("VISION OK:", json.dumps(result, ensure_ascii=False))

            with self.lock:
                self.result = result
                self.result_serial = serial
                self.error = None

        except Exception as e:
            print("VISION ERROR:", repr(e))
            with self.lock:
                self.error = repr(e)

        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            with self.lock:
                self.busy = False


# ============================================================
# STATE MACHINE
# ============================================================

class CoffeeCoach:
    def __init__(self):
        self.stage_index = 0
        self.stage_started = time.time()

        self.completed_history = []
        self.vision = VisionReasoner()
        self.motion = MotionTracker()

        self.frame_history = deque()
        self.last_vision_request = 0.0

        # Only process each completed VLM inference once.
        self.last_processed_serial = 0

        # Fresh, independent visual confirmations for the current stage.
        self.confirmations = 0
        self.action_was_seen = False
        self.last_confirmed_action = None

        self.last_spoken_stage = None
        self.last_spoken_guidance = ""

        # Steep-stage disturbance feedback
        self.steep_warning_text = ""
        self.last_steep_warning_time = 0.0
        self.steep_warning_cooldown = 4.0

    @property
    def stage(self):
        return STAGES[self.stage_index]

    def reset(self):
        self.__init__()

    def add_frame(self, frame):
        now = time.time()
        self.frame_history.append((now, frame.copy()))
        cutoff = now - FRAME_HISTORY_SECONDS
        while self.frame_history and self.frame_history[0][0] < cutoff:
            self.frame_history.popleft()

    def sample_frames(self):
        items = list(self.frame_history)
        if len(items) < 2:
            return []

        # Two frames are enough for this constrained action vocabulary and
        # substantially reduce local VLM latency.
        first = items[max(0, len(items)//3)][1]
        last = items[-1][1]
        return [first, last]

    def advance(self, reason):
        if self.stage["id"] == "DONE":
            return

        old = self.stage["id"]
        self.completed_history.append({
            "stage": old,
            "seconds": round(time.time() - self.stage_started, 1),
            "reason": reason,
        })

        self.stage_index = min(self.stage_index + 1, len(STAGES)-1)
        self.stage_started = time.time()
        self.motion.clear()

        self.confirmations = 0
        self.action_was_seen = False
        self.last_confirmed_action = None
        self.steep_warning_text = ""

        # Do not allow a result produced for the previous stage to drive the new stage.
        with self.vision.lock:
            self.last_processed_serial = self.vision.result.get("serial", self.last_processed_serial)

        speak(self.stage["instruction"])

    def process_vision_result(self):
        with self.vision.lock:
            r = dict(self.vision.result)

        serial = int(r.get("serial", 0))
        result_stage = r.get("stage_id")

        # Critical stability rule:
        # one completed VLM inference can affect the state machine only once.
        if serial <= self.last_processed_serial:
            return

        self.last_processed_serial = serial

        # Ignore a late inference that was requested for the previous stage.
        if result_stage != self.stage["id"]:
            print(
                "IGNORING STALE VISION RESULT:",
                result_stage,
                "current stage:",
                self.stage["id"],
            )
            return

        action = r["action"]
        confidence = r["confidence"]
        stage = self.stage["id"]
        elapsed = time.time() - self.stage_started
        visible = set(r.get("visible_objects", []))

        expected = set(self.stage.get("expected", []))
        is_expected = action in expected

        # Stage-specific object requirements. This prevents the model from advancing
        # merely because a hand movement resembles the expected action.
        object_gate = True

        if stage == "LOAD_BEANS":
            object_gate = {"coffee_bag", "hand_grinder"}.issubset(visible)

        elif stage == "GRIND":
            object_gate = "hand_grinder" in visible

        elif stage == "TRANSFER":
            object_gate = {"hand_grinder", "aeropress"}.issubset(visible)

        elif stage == "POUR_WATER":
            object_gate = {"kettle", "aeropress"}.issubset(visible)

        elif stage == "STIR":
            # The black AeroPress paddle can occasionally be mistaken for the
            # black hand grinder. For stirring, the AeroPress itself is the
            # important object anchor. Do not require a perfect paddle label.
            object_gate = "aeropress" in visible

        elif stage == "PRESS":
            object_gate = {"aeropress", "glass_cup"}.issubset(visible)

        # A recognition must be reasonably confident, match the expected action,
        # AND show the required physical objects.
        stir_motion_fallback = False
        if stage == "STIR":
            motion = self.motion.summarize()
            stir_motion_fallback = (
                "aeropress" in visible
                and motion.get("repetitive_motion", False)
                and motion.get("motion_level") in {"low", "medium", "high"}
                and action in {"stirring", "grinding", "unknown"}
                and confidence >= 0.45
            )

        if (is_expected and object_gate and confidence >= 0.68) or stir_motion_fallback:
            self.confirmations += 1
            self.action_was_seen = True
            self.last_confirmed_action = "stirring" if stir_motion_fallback else action
            print(
                f"STAGE EVIDENCE {stage}: {self.confirmations} confirmation(s), "
                f"action={action}, confidence={confidence:.2f}, "
                f"objects={sorted(visible)}, stir_fallback={stir_motion_fallback}"
            )
        elif action in {"idle", "waiting", "unknown"}:
            # Preserve previous evidence. Stopping the action is meaningful.
            pass
        else:
            # Conflicting evidence does not instantly erase a real action,
            # but prevents accidental accumulation.
            self.confirmations = max(0, self.confirmations - 1)

        # Never advance immediately after entering a stage.
        if elapsed < 2.0:
            return

        # The action should be observed in at least two independent VLM analyses.
        # For finite actions, advancement happens after the user STOPS the action.
        stopped = action in {"idle", "waiting", "unknown"}

        if stage == "LOAD_BEANS":
            bag_removed = "coffee_bag" not in visible
            if (
                elapsed >= 4.0
                and self.confirmations >= 2
                and self.action_was_seen
                and (stopped or bag_removed)
            ):
                self.advance("coffee bag + grinder visible during pour, then pour ended")
                return

        elif stage == "GRIND":
            if self.confirmations >= 3 and stopped:
                self.advance("grinding observed repeatedly, then stopped")
                return

        elif stage == "TRANSFER":
            if self.confirmations >= 2 and stopped:
                self.advance("grounds transfer observed twice, then stopped")
                return

        elif stage == "POUR_WATER":
            if self.confirmations >= 2 and stopped:
                self.advance("water pouring observed twice, then stopped")
                return

        elif stage == "STIR":
            if self.confirmations >= 2 and stopped:
                self.advance("stirring observed twice, then stopped")
                return

        elif stage == "PRESS":
            # Pressing is longer, so require repeated independent evidence.
            if self.confirmations >= 2 and stopped:
                self.advance("pressing observed twice, then stopped")
                return


    def update(self):
        now = time.time()

        if self.stage["id"] == "STEEP":
            elapsed = now - self.stage_started
            remaining = max(0, int(STEEP_SECONDS - elapsed))

            if elapsed >= STEEP_SECONDS:
                self.steep_warning_text = ""
                self.advance("30 second steep complete")
                return

            # During steeping, any clear hand movement is treated as a disturbance.
            # We do not advance the workflow; instead we warn the user and report
            # the remaining steep time.
            motion = self.motion.summarize()
            disturbed = (
                motion.get("hands_visible", False)
                and motion.get("motion_level") in {"medium", "high"}
                and motion.get("path_length", 0.0) >= 0.12
            )

            if disturbed and (now - self.last_steep_warning_time) >= self.steep_warning_cooldown:
                self.steep_warning_text = (
                    f"Steeping is not finished yet. {remaining} seconds remaining."
                )
                print("STEEP WARNING:", self.steep_warning_text)
                speak(self.steep_warning_text)
                self.last_steep_warning_time = now

            # Clear stale warning after a few seconds if the user stops moving.
            if (
                not disturbed
                and self.steep_warning_text
                and (now - self.last_steep_warning_time) > 3.0
            ):
                self.steep_warning_text = ""

            return

        if self.stage["id"] == "DONE":
            return

        # Read completed inference and use it for transition logic.
        self.process_vision_result()

        # Request a new multi-frame analysis periodically.
        if now - self.last_vision_request >= VISION_INTERVAL:
            frames = self.sample_frames()
            if len(frames) == VISION_FRAME_COUNT:
                sent = self.vision.request(
                    frames,
                    self.stage,
                    self.motion.summarize(),
                    self.completed_history,
                )
                if sent:
                    self.last_vision_request = now


# ============================================================
# UI
# ============================================================

def draw_ui(camera_frame, coach):
    camera = cv2.resize(camera_frame, (900, 700))
    canvas = np.zeros((700, 1280, 3), dtype=np.uint8)
    canvas[:, :900] = camera

    cv2.rectangle(canvas, (900,0), (1279,699), (24,24,24), -1)

    x = 925
    y = 38

    cv2.putText(
        canvas, "EMBODIED COFFEE COACH",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255,255,255), 2, cv2.LINE_AA
    )
    y += 42

    cv2.putText(
        canvas, coach.stage["title"],
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (255,255,255), 2, cv2.LINE_AA
    )
    y += 34

    elapsed = time.time() - coach.stage_started

    if coach.stage["id"] == "STEEP":
        remaining = max(0, STEEP_SECONDS - int(elapsed))
        timer = f"Steep timer: {remaining}s"
    else:
        timer = f"Stage time: {elapsed:.1f}s"

    cv2.putText(
        canvas, timer, (x,y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46, (180,180,180), 1, cv2.LINE_AA
    )
    y += 34

    cv2.putText(
        canvas, "EXPECTED",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (170,170,170), 1, cv2.LINE_AA
    )
    y += 24

    y = put_lines(
        canvas,
        wrap_text(coach.stage["instruction"], 39),
        x, y, 0.50, 22, 3
    )
    y += 14

    with coach.vision.lock:
        r = dict(coach.vision.result)
        busy = coach.vision.busy
        err = coach.vision.error

    cv2.putText(
        canvas, "VISION OBJECTS",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (170,170,170), 1, cv2.LINE_AA
    )
    y += 24

    objects_text = ", ".join(r["visible_objects"]) if r["visible_objects"] else "none / uncertain"
    y = put_lines(canvas, wrap_text(objects_text, 39), x, y, 0.48, 21, 3)
    y += 12

    cv2.putText(
        canvas, "INFERRED ACTION",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (170,170,170), 1, cv2.LINE_AA
    )
    y += 25

    cv2.putText(
        canvas, r["action"],
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.56, (245,245,245), 1, cv2.LINE_AA
    )
    y += 23

    cv2.putText(
        canvas, f"confidence: {r['confidence']:.2f}",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (190,190,190), 1, cv2.LINE_AA
    )
    y += 32

    cv2.putText(
        canvas, "VISUAL EVIDENCE",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (170,170,170), 1, cv2.LINE_AA
    )
    y += 24

    y = put_lines(canvas, wrap_text(r["evidence"], 39), x, y, 0.48, 21, 4)
    y += 12

    cv2.putText(
        canvas, "AI GUIDANCE",
        (x,y), cv2.FONT_HERSHEY_SIMPLEX,
        0.44, (170,170,170), 1, cv2.LINE_AA
    )
    y += 24

    if coach.stage["id"] == "STEEP":
        elapsed = time.time() - coach.stage_started
        remaining = max(0, int(STEEP_SECONDS - elapsed))
        if coach.steep_warning_text:
            guidance_text = coach.steep_warning_text
        else:
            guidance_text = f"Do not disturb the brew yet. {remaining} seconds remaining."
    else:
        guidance_text = r["guidance"]

    y = put_lines(canvas, wrap_text(guidance_text, 39), x, y, 0.50, 22, 4)

    motion = coach.motion.summarize()
    motion_text = (
        f"hands: {motion['hands_visible']} | "
        f"motion: {motion['motion_level']} | "
        f"repeat: {motion['repetitive_motion']}"
    )
    cv2.putText(
        canvas, motion_text,
        (925, 626),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39, (160,160,160), 1, cv2.LINE_AA
    )

    required = 3 if coach.stage["id"] == "GRIND" else 2
    if coach.stage["id"] in {"STEEP", "DONE"}:
        evidence_text = "evidence: timer / complete"
    else:
        evidence_text = f"evidence confirmations: {coach.confirmations}/{required}"

    cv2.putText(
        canvas, evidence_text,
        (925, 648),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39, (175,175,175), 1, cv2.LINE_AA
    )

    if err:
        status = "VISION ERROR"
        short_error = str(err).replace("\n", " ")[:48]
        cv2.putText(
            canvas, short_error,
            (925, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31, (190,190,190), 1, cv2.LINE_AA
        )
    elif busy:
        status = "QWEN3-VL: ANALYZING..."
    else:
        status = "QWEN3-VL: READY"

    cv2.putText(
        canvas, status,
        (925, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (190,190,190), 1, cv2.LINE_AA
    )

    return canvas


# ============================================================
# MAIN
# ============================================================

def main():
    coach = CoffeeCoach()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = create_landmarker()
    start = time.time()

    speak("Embodied Coffee Coach ready. " + coach.stage["instruction"])

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            raw_for_vision = frame.copy()

            result = None
            if landmarker is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                timestamp_ms = int((time.time() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                centers = centers_from_result(result)
                coach.motion.add(HandSample(time.time(), centers))
                draw_hands(frame, result)

            coach.add_frame(raw_for_vision)
            coach.update()

            ui = draw_ui(frame, coach)
            cv2.imshow(WINDOW_NAME, ui)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                coach.reset()
                speak(coach.stage["instruction"])
            elif key == ord("n"):
                coach.advance("developer debug override")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
