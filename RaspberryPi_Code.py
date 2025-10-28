#!/usr/bin/env python3
import json, queue, sys, time, threading, subprocess
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Resampling 44.1k -> 16k (OK on Py 3.11/3.12)
import audioop  # NOTE: deprecated in Python 3.13+

# ------------------ CONFIG ------------------
MODEL_PATH       = "/home/hyper/models/vosk-model-small-en-us-0.15"
DEVICE_INDEX     = 2                # your USB mic index
TARGET_RATE      = 16000            # Vosk expects 16 kHz mono
BLOCKSIZE_IN     = 4096
QUEUE_MAX_CHUNKS = 6
KEYWORDS         = ["help", "save me", "police"]
COOLDOWN_S       = 3

# Firestore
SERVICE_ACCOUNT  = "/home/hyper/serviceAccountKey.json"
FIRE_COLLECTION  = "Raspberry PI 4"
FIRE_DOC_ID      = "Firebase_ID"
SUBCOL_ALERTS    = "alerts"
FIELD_KEYWORD    = "keyWord"
FIELD_SLNO       = "Slot_No"
# ---------------------------------------------------------

# ------------------ FIREBASE INIT ------------------------
import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate(SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred)
db = firestore.client()
# ---------------------------------------------------------

# ------------------ PICAMERA2 LIVE PREVIEW ---------------
# Uses your working pattern: Preview.QTGL with fallback to Preview.QT
from picamera2 import Picamera2, Preview

_preview_thread = None
_picam2 = None

def start_picamera_preview():
    """Launch Picamera2 live preview in a background thread. No-op if already running."""
    global _preview_thread, _picam2
    if _preview_thread and _preview_thread.is_alive():
        print("[CAM] Preview already running.")
        return

    def _worker():
        global _picam2
        try:
            _picam2 = Picamera2()
            config = _picam2.create_preview_configuration(main={"size": (1280, 720)})
            _picam2.configure(config)
            try:
                _picam2.start_preview(Preview.QTGL)
            except Exception:
                print("[CAM] QTGL preview failed; falling back to QT.")
                _picam2.start_preview(Preview.QT)
            _picam2.start()
            print("[CAM] Live preview started (Picamera2). Close the window or Ctrl+C main app to stop.")
            while True:
                time.sleep(1)
        except Exception as e:
            print("[CAM] ERROR starting preview:", e)
        finally:
            try:
                if _picam2:
                    _picam2.stop()
                    print("[CAM] Preview stopped.")
            except Exception:
                pass

    _preview_thread = threading.Thread(target=_worker, daemon=True)
    _preview_thread.start()
# ---------------------------------------------------------

# ------------------ AUDIO / VOSK -------------------------
def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def get_device_and_rate(dev_index: int):
    dev = sd.query_devices()[dev_index]
    if dev["max_input_channels"] < 1:
        raise RuntimeError(f"Device #{dev_index} has no input channels: {dev}")
    return int(dev["default_samplerate"]), dev["name"], dev["max_input_channels"]
# ---------------------------------------------------------

# ------------------ FIRESTORE OPS ------------------------
def tx_increment_and_log(keyword: str):
    """
    Update main doc with latest (keyWord, Slot_No) and
    append a history record in sub-collection 'alerts' with server timestamp.
    """
    doc_ref = db.collection(FIRE_COLLECTION).document(FIRE_DOC_ID)

    @firestore.transactional
    def txn_fn(txn):
        snap = doc_ref.get(transaction=txn)
        current = 0
        if snap.exists:
            data = snap.to_dict() or {}
            try:
                current = int(data.get(FIELD_SLNO, 0))
            except Exception:
                current = 0
        new_val = current + 1

        txn.set(doc_ref, {FIELD_KEYWORD: keyword, FIELD_SLNO: new_val}, merge=True)
        txn.set(
            doc_ref.collection(SUBCOL_ALERTS).document(),
            {FIELD_KEYWORD: keyword, FIELD_SLNO: new_val, "ts": firestore.SERVER_TIMESTAMP},
        )
        return new_val

    return txn_fn(db.transaction())

def print_full_history():
    doc_ref = db.collection(FIRE_COLLECTION).document(FIRE_DOC_ID)
    alerts_ref = doc_ref.collection(SUBCOL_ALERTS)
    docs = alerts_ref.order_by(FIELD_SLNO).stream()
    print("\n--- Alerts history ---")
    count = 0
    for d in docs:
        print(d.to_dict())
        count += 1
    if count == 0:
        print("(empty)")
    print("----------------------\n")
# ---------------------------------------------------------

def on_keyword_detected(keyword: str, full_text: str, result_json: dict):
    print(f"\n[ALERT] Keyword: '{keyword}' | Transcript: {full_text}")

    # Start Picamera2 live preview (non-blocking); if already running, it's a no-op.
    start_picamera_preview()

    # Firestore updates
    new_sl = tx_increment_and_log(keyword)
    print(f"Uploaded to Firebase: {FIELD_KEYWORD}={keyword}, {FIELD_SLNO}={new_sl}")
    print_full_history()

def main():
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    input_rate, dev_name, max_in = get_device_and_rate(DEVICE_INDEX)
    channels = 1 if max_in >= 1 else max_in
    try:
        sd.check_input_settings(device=DEVICE_INDEX, samplerate=input_rate,
                                channels=channels, dtype="int16")
    except Exception:
        if max_in >= 2:
            channels = 2
            sd.check_input_settings(device=DEVICE_INDEX, samplerate=input_rate,
                                    channels=channels, dtype="int16")
        else:
            raise

    print(f"Using device #{DEVICE_INDEX} (in={max_in}): {dev_name}")
    print(f"Capturing @ {input_rate} Hz, channels={channels} → resample to {TARGET_RATE} Hz mono")
    print("Listening for:", KEYWORDS, " | Ctrl+C to stop")

    grammar = json.dumps(KEYWORDS + ["[unk]"])
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, TARGET_RATE, grammar)
    rec.SetWords(True)

    q = queue.Queue()
    warned_overflow = False

    def audio_callback(indata, frames, t, status):
        nonlocal warned_overflow
        # keep queue bounded (real-time)
        if q.qsize() >= QUEUE_MAX_CHUNKS:
            try:
                q.get_nowait()
                if not warned_overflow:
                    print("Audio status: input overflow (dropped old frames)")
                    warned_overflow = True
            except queue.Empty:
                pass
        q.put(bytes(indata))

    ratecv_state = None
    last_fire = 0.0

    with sd.RawInputStream(
        samplerate=input_rate,
        blocksize=BLOCKSIZE_IN,
        dtype="int16",
        channels=channels,
        device=DEVICE_INDEX,
        callback=audio_callback,
        latency="low",
    ):
        try:
            while True:
                data = q.get()
                if channels > 1:
                    data = audioop.tomono(data, 2, 1, 1)
                data, ratecv_state = audioop.ratecv(
                    data, 2, 1, input_rate, TARGET_RATE, ratecv_state
                )
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = normalize(res.get("text", ""))
                    if not text:
                        continue
                    for kw in KEYWORDS:
                        if normalize(kw) in text:
                            now = time.time()
                            if now - last_fire >= COOLDOWN_S:
                                on_keyword_detected(kw, text, res)
                                last_fire = now
                            break
        except KeyboardInterrupt:
            print("\nStopping...")
            print(json.loads(rec.FinalResult()))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
