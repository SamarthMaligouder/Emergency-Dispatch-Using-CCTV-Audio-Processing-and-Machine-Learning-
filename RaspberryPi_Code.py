
import os, json, time, warnings, numpy as np, requests
import sounddevice as sd
import librosa

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal",
    category=UserWarning,
    module="numpy.core.getlimits",
)

MODEL_PATH = "kws_3cls_int8.tflite"
META_PATH  = "kws_3cls_metadata.json"
MIC_DEVICE = "hw:3,0"
MIC_SR     = 44100
FRAME_S    = None

THRESH = {"help": 0.80, "save_me": 0.80, "police": 0.80}
MARGIN = 0.15
ENTROPY_MAX = 0.80
CONSEC_REQ = 3
COOLDOWN_S = 4.0

ENERGY_MIN = 0.02
FLUX_MIN   = 7e-4
SILENCE_END_S = 0.6
DEBUG_PRINT_PROBS = True
ALERT_URL  = os.environ.get("ALERT_URL", "")

def list_input_devices():
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"[{idx}] {dev['name']}  (default SR: {dev.get('default_samplerate')})")

def sanitize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -1.0, 1.0)

def safe_logmel(y: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    EPS = 1e-10
    y = sanitize_audio(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    mel = np.maximum(mel, EPS)
    db = librosa.power_to_db(mel, ref=np.max)
    db = np.clip(np.nan_to_num(db, nan=-80.0, posinf=0.0, neginf=-80.0), -80.0, 0.0)
    return db.astype(np.float32)

def ensure_bhwc1(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2: x = x[..., np.newaxis]
    if x.ndim == 3: x = x[np.newaxis, ...]
    return x

def quantize_input(x_float: np.ndarray, input_details) -> np.ndarray:
    dtype = input_details[0]['dtype']
    scale, zero = input_details[0]['quantization']
    if dtype == np.float32: return x_float.astype(np.float32)
    if not scale: return x_float.astype(dtype)
    return np.round(x_float / scale + zero).astype(dtype)

def dequantize_output(y_quant: np.ndarray, output_details) -> np.ndarray:
    dtype = output_details[0]['dtype']
    scale, zero = output_details[0]['quantization']
    if dtype == np.float32: return y_quant.astype(np.float32)
    if not scale: return y_quant.astype(np.float32)
    return (y_quant.astype(np.float32) - zero) * scale

def softmax_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def spectral_flux(mel_db: np.ndarray) -> float:
    mel_lin = 10.0 ** (mel_db / 10.0)
    diff = np.diff(mel_lin, axis=1)
    if diff.size == 0: return 0.0
    return float(np.maximum(diff, 0.0).sum(axis=0).mean())

def decide(probs: np.ndarray, classes, streak: dict):
    idx_top = int(np.argmax(probs))
    cls = classes[idx_top]
    p_top = float(probs[idx_top])
    p2 = float(np.partition(probs, -2)[-2]) if probs.size > 1 else 0.0
    ent = softmax_entropy(probs)
    if p_top >= THRESH[cls] and (p_top - p2) >= MARGIN and ent <= ENTROPY_MAX:
        streak[cls] += 1
        for c in classes:
            if c != cls: streak[c] = 0
        if streak[cls] >= CONSEC_REQ:
            streak[cls] = 0
            return cls, p_top
    else:
        for c in classes: streak[c] = 0
    return None, None

def send_alert(keyword: str, prob: float):
    if not ALERT_URL:
        print(f"(no ALERT_URL) Would send alert: {keyword} ({prob:.2f})")
        return
    try:
        r = requests.post(ALERT_URL, json={"keyword": keyword, "prob": prob, "ts": time.time()}, timeout=5)
        print("Alert:", r.status_code, r.text[:200])
    except Exception as e:
        print("Alert error:", e)

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

meta = json.load(open(META_PATH, "r"))
CLASSES = meta["classes"]
SR      = int(meta["sample_rate"])
WIN_S   = float(meta["window_s"])
N_MELS  = int(meta["n_mels"])
if FRAME_S is None:
    FRAME_S = WIN_S

interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interp.allocate_tensors()
in_det  = interp.get_input_details()
out_det = interp.get_output_details()

def run_infer_from_spec(spec_db):
    x = ensure_bhwc1(spec_db).astype(np.float32)
    x_in = quantize_input(x, in_det)
    interp.set_tensor(in_det[0]['index'], x_in)
    interp.invoke()
    out = interp.get_tensor(out_det[0]['index'])
    probs = np.squeeze(dequantize_output(out, out_det)).astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0)
    return probs / np.sum(probs)

def mk_spec(y): return safe_logmel(y, SR, N_MELS)

try:
    di = sd.query_devices(MIC_DEVICE, 'input') if MIC_DEVICE is not None else sd.query_devices(kind='input')
    MIC_SR = int(di.get('default_samplerate') or MIC_SR)
except Exception:
    MIC_SR = int(MIC_SR)

list_input_devices()
print(f"Using input device: {MIC_DEVICE if MIC_DEVICE is not None else 'default'} @ {MIC_SR} Hz")
print("=== Keyword Detector (help / save me / police) ===")
print("Press Ctrl+C to exit.")

HOP_S     = FRAME_S / 2.0
frame_len = int(MIC_SR * FRAME_S)
hop_len   = int(MIC_SR * HOP_S)

from collections import deque
rms_hist = deque(maxlen=max(1, int(SILENCE_END_S / HOP_S)))
can_trigger = True
last_fire = 0.0
streak = {c: 0 for c in CLASSES}
_last_rms_print = 0.0

try:
    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        samplerate=MIC_SR,
        dtype="float32",
        blocksize=hop_len,
    ) as stream:
        ring = np.zeros(frame_len, dtype=np.float32)
        while True:
            audio, _ = stream.read(hop_len)
            y_hop = audio[:, 0]
            if len(y_hop) < hop_len:
                y_hop = np.pad(y_hop, (0, hop_len - len(y_hop)))
            ring = np.concatenate([ring[hop_len:], y_hop])
            now = time.time()
            if now - _last_rms_print >= 2.0:
                print(f"Mic RMS ~ {float(np.sqrt(np.mean(ring**2))):.4f}")
                _last_rms_print = now
            y = ring
            if MIC_SR != SR:
                y = librosa.resample(y, orig_sr=MIC_SR, target_sr=SR)
            y = sanitize_audio(y)
            frame_rms = float(np.sqrt(np.mean(y**2)))
            if frame_rms < ENERGY_MIN:
                rms_hist.append(frame_rms)
                if not can_trigger and len(rms_hist) == rms_hist.maxlen and all(v < ENERGY_MIN for v in rms_hist):
                    can_trigger = True
                continue
            rms_hist.append(frame_rms)
            spec = mk_spec(y)
            if spectral_flux(spec) < FLUX_MIN:
                continue
            probs = run_infer_from_spec(spec)
            if DEBUG_PRINT_PROBS:
                print("probs:", {CLASSES[i]: round(float(probs[i]), 3) for i in range(len(CLASSES))})
            if can_trigger and (now - last_fire) >= COOLDOWN_S:
                cls, p = decide(probs, CLASSES, streak)
                if cls is not None:
                    print(f"DETECTED: {cls} ({p:.2f})")
                    send_alert(cls, p)
                    last_fire = now
                    can_trigger = False
except KeyboardInterrupt:
    print("\nBye! (Ctrl+C)")
