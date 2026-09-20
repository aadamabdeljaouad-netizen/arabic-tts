import json
import logging
import os
import tempfile
import textwrap
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

# Limit internal library threads for low memory environment
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import onnxruntime as ort
from flask import Flask, jsonify, render_template, request, send_file
from piper import PiperVoice, PiperConfig
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"

VOICE_NAME = "ar_JO-kareem-low"
MODEL_PATH = VOICES_DIR / f"{VOICE_NAME}.onnx"
CONFIG_PATH = VOICES_DIR / f"{VOICE_NAME}.onnx.json"

VOICE_BASE_URL = (
    "[https://huggingface.co/rhasspy/piper-voices/resolve/main/](https://huggingface.co/rhasspy/piper-voices/resolve/main/)"
    "ar/ar_JO/kareem/low"
)

MAX_TEXT_LENGTH = 400
CHUNK_LENGTH = 100
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_MAX_SECONDS = 180
DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arabic-tts")

_voice = None
_model_lock = threading.Lock()
_generation_lock = threading.Lock()

def download_file(url, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        return

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")

    request_object = Request(
        url,
        headers={"User-Agent": "arabic-tts/1.0"},
    )

    started_at = time.monotonic()
    downloaded_bytes = 0

    try:
        with urlopen(request_object, timeout=DOWNLOAD_TIMEOUT) as response:
            expected_length = response.headers.get("Content-Length")
            expected_length = (
                int(expected_length) if expected_length else None
            )

            with temporary_path.open("wb") as output:
                while True:
                    if time.monotonic() - started_at > DOWNLOAD_MAX_SECONDS:
                        raise TimeoutError("انتهت مهلة تنزيل النموذج.")

                    block = response.read(256 * 1024)
                    if not block:
                        break

                    downloaded_bytes += len(block)

                    if downloaded_bytes > DOWNLOAD_MAX_BYTES:
                        raise ValueError("حجم ملف النموذج أكبر من الحد المتوقع.")

                    output.write(block)

        if downloaded_bytes == 0:
            raise ValueError("ملف النموذج الذي تم تنزيله فارغ.")

        if (
            expected_length is not None
            and downloaded_bytes != expected_length
        ):
            raise ValueError("لم يكتمل تنزيل ملف النموذج.")

        if destination.suffix == ".json":
            with temporary_path.open("r", encoding="utf-8") as config_file:
                json.load(config_file)

        temporary_path.replace(destination)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def get_voice():
    global _voice

    with _model_lock:
        if _voice is not None:
            return _voice

        download_file(
            f"{VOICE_BASE_URL}/{MODEL_PATH.name}?download=true",
            MODEL_PATH,
        )
        download_file(
            f"{VOICE_BASE_URL}/{CONFIG_PATH.name}?download=true",
            CONFIG_PATH,
        )

        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            voice_config = PiperConfig.from_dict(json.load(config_file))

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False

        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        _voice = PiperVoice(
            config=voice_config,
            session=session,
        )

        logger.info("اكتملت تهيئة النموذج العربي.")
        return _voice

def prepare_voice_at_startup():
    try:
        get_voice()
    except Exception:
        logger.exception("تعذرت تهيئة النموذج عند بدء التشغيل.")

def error_response(message, status_code):
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify(error=message), status_code

    return (
        render_template(
            "index.html",
            error=message,
            max_length=MAX_TEXT_LENGTH,
        ),
        status_code,
    )

@app.get("/")
def index():
    return render_template(
        "index.html",
        error=None,
        max_length=MAX_TEXT_LENGTH,
    )

@app.post("/generate")
def generate():
    text = request.form.get("text", "").strip()

    if not text:
        return error_response("اكتب النص الذي تريد تحويله أولًا.", 400)

    if len(text) > MAX_TEXT_LENGTH:
        return error_response(
            f"الحد الأقصى للنص هو {MAX_TEXT_LENGTH} حرفًا.",
            400,
        )

    if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
        return error_response("النص يحتوي على محارف تحكم غير مسموحة.", 400)

    text = " ".join(text.split())

    if not _generation_lock.acquire(blocking=False):
        response, status = error_response(
            "الخادم يحوّل نصًا آخر الآن. انتظر قليلًا ثم أعد المحاولة.",
            503,
        )
        return response, status, {"Retry-After": "10"}

    audio_file = None

    try:
        try:
            voice = get_voice()
        except Exception:
            logger.exception("تعذر تنزيل النموذج أو تحميله.")
            return error_response(
                "تعذر تجهيز نموذج الصوت. انتظر قليلًا ثم حاول مجددًا.",
                503,
            )

        audio_file = tempfile.TemporaryFile(mode="w+b")

        chunks = textwrap.wrap(
            text,
            width=CHUNK_LENGTH,
            break_long_words=True,
            break_on_hyphens=False,
        )

        written_bytes = 0

        with wave.open(audio_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)

            for text_chunk in chunks:
                for audio_chunk in voice.synthesize(text_chunk):
                    audio_bytes = audio_chunk.audio_int16_bytes
                    wav_file.writeframesraw(audio_bytes)
                    written_bytes += len(audio_bytes)

        if written_bytes == 0:
            raise ValueError("لم ينتج النموذج بيانات صوتية.")

        audio_file.seek(0)

        response = send_file(
            audio_file,
            mimetype="audio/wav",
            download_name="arabic-tts.wav",
            as_attachment=False,
            conditional=False,
            etag=False,
            max_age=0,
        )

        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"

        response.call_on_close(audio_file.close)
        return response

    except Exception:
        if audio_file is not None:
            audio_file.close()

        logger.exception("حدث خطأ أثناء توليد الصوت.")
        return error_response(
            "تعذر توليد الصوت. جرّب نصًا أقصر ثم أعد المحاولة.",
            500,
        )

    finally:
        _generation_lock.release()

@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(_error):
    return error_response(
        "حجم الطلب كبير جدًا. أرسل نصًا أقصر.",
        413,
    )

threading.Thread(
    target=prepare_voice_at_startup,
    name="voice-startup",
    daemon=True,
).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )