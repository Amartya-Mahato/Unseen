import streamlit as st
import cv2
import base64
import requests
import tempfile
import re

# ── Config ─────────────────────────────────────────────────────────────────────
LM_BASE        = "http://localhost:1234/v1"
COMPLETIONS_URL = f"{LM_BASE}/chat/completions"
MODELS_URL      = f"{LM_BASE}/models"
MAX_FRAMES      = 20          # keep low — local models struggle with 60+ images
BATCH_SIZE      = 5           # frames per API call (batched to avoid 400s)
JPEG_QUALITY    = 60


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_model() -> str:
    """Auto-fetch the loaded model name from LM Studio."""
    try:
        r = requests.get(MODELS_URL, timeout=5)
        models = r.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "gemma-4"   # fallback


def extract_frames(path: str) -> list[tuple[float, any]]:
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // MAX_FRAMES)

    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # Resize to 512px wide to shrink payload
            h, w = frame.shape[:2]
            if w > 512:
                frame = cv2.resize(frame, (512, int(h * 512 / w)))
            frames.append((idx / fps, frame))
        idx += 1
    cap.release()
    return frames[:MAX_FRAMES]


def to_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode()


SUMMARY_KEYWORDS = [
    "how many", "count", "total", "number of", "describe", "what is happening",
    "what's happening", "overview", "summarize", "summary", "what do you see",
    "what's in", "what is in", "who is", "who are", "what are people",
]

def question_type(q: str) -> str:
    return "summary" if any(kw in q.lower() for kw in SUMMARY_KEYWORDS) else "search"


def call_llm(model: str, system: str, content: list) -> str:
    resp = requests.post(
        COMPLETIONS_URL,
        json={
            "model":       model,
            "messages":    [{"role": "system", "content": system},
                            {"role": "user",   "content": content}],
            "max_tokens":  768,
            "temperature": 0.1,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def build_content(frames: list, question: str) -> list:
    content = [{"type": "text", "text": f"Question: {question}"}]
    for ts, frame in frames:
        content.append({"type": "text",      "text": f"[Frame @ {ts:.1f}s]"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{to_b64(frame)}"
        }})
    return content


def analyze_summary(model: str, frames: list, question: str) -> tuple[str, list[dict], str]:
    system = (
        "You are a CCTV security analyst.\n"
        "First, give a direct plain-English answer to the question.\n"
        "Then, for each frame where you can count or observe something notable, output:\n"
        "FRAME | TIMESTAMP: <seconds> | NOTE: <what you see in this frame>\n"
        "Be factual. Only describe what is clearly visible."
    )
    raw = call_llm(model, system, build_content(frames, question))

    evidence = []
    for line in raw.splitlines():
        if line.startswith("FRAME"):
            ts_m = re.search(r"TIMESTAMP:\s*([\d.]+)", line)
            no_m = re.search(r"NOTE:\s*(.+)",          line)
            if ts_m:
                evidence.append({
                    "timestamp": float(ts_m.group(1)),
                    "evidence":  no_m.group(1).strip() if no_m else "Notable frame",
                })

    summary = " ".join(l for l in raw.splitlines() if not l.startswith("FRAME")).strip()
    return summary, evidence, raw


def analyze_search(model: str, frames: list, question: str) -> tuple[list[dict], str]:
    system = (
        "You are a CCTV security analyst. Analyze the labeled frames.\n"
        "For each frame that shows activity relevant to the question, output EXACTLY:\n"
        "MATCH | TIMESTAMP: <seconds> | EVIDENCE: <short description>\n"
        "If nothing relevant in these frames, output: NO_MATCH\n"
        "Only report what is clearly visible. Do not guess."
    )
    all_raw, results = [], []
    batches  = [frames[i:i+BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]
    progress = st.progress(0, text="Scanning frames...")

    for i, batch in enumerate(batches):
        raw = call_llm(model, system, build_content(batch, question))
        all_raw.append(raw)
        for line in raw.splitlines():
            if line.startswith("MATCH"):
                ts_m = re.search(r"TIMESTAMP:\s*([\d.]+)", line)
                ev_m = re.search(r"EVIDENCE:\s*(.+)",      line)
                if ts_m:
                    results.append({
                        "timestamp": float(ts_m.group(1)),
                        "evidence":  ev_m.group(1).strip() if ev_m else "Relevant activity",
                    })
        progress.progress((i + 1) / len(batches), text=f"Batch {i+1}/{len(batches)}...")

    progress.empty()
    return results, "\n---\n".join(all_raw)





def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Surveillance", page_icon="🎥", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d0d0d; }
    [data-testid="stSidebar"]          { background: #111; }
    .evidence-card {
        background: #1a1a1a;
        border-left: 3px solid #ff4444;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .ts-badge { font-size: 20px; font-weight: 700; color: #ff4444; font-family: monospace; }
    .ev-text  { color: #ccc; font-size: 14px; margin-top: 4px; }
    .model-tag { color: #888; font-size: 12px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🎥 AI Surveillance System")

# ── Model status bar ───────────────────────────────────────────────────────────
model = get_model()
st.markdown(f'<span class="model-tag">⬤ LM Studio · {model} · {MAX_FRAMES} frames · batch {BATCH_SIZE}</span>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

# ── Left: upload + video ───────────────────────────────────────────────────────
with left:
    upload = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"])
    if upload:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(upload.read())
        tmp.flush()
        st.session_state["video_path"] = tmp.name

    if "video_path" in st.session_state:
        st.video(st.session_state["video_path"])

# ── Right: Q&A + results ───────────────────────────────────────────────────────
with right:
    st.subheader("Ask a security question")
    question = st.text_input(
        label="query", label_visibility="collapsed",
        placeholder="e.g. Is anyone wearing a blue t-shirt?",
    )

    with st.expander("💡 Example questions"):
        for ex in [
            "How many people are in the video?",
            "What is happening in this footage?",
            "Is there a person in a blue t-shirt?",
            "Did anyone try to barge in?",
            "Is anyone acting suspiciously near the counter?",
        ]:
            if st.button(ex, key=ex):
                question = ex

    if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
        if "video_path" not in st.session_state:
            st.warning("Please upload a video first.")
        elif not question:
            st.warning("Please enter a question.")
        else:
            mode = question_type(question)
            with st.status("Working...", expanded=True) as status:
                st.write("📽️ Extracting frames with OpenCV...")
                frames = extract_frames(st.session_state["video_path"])
                dur    = frames[-1][0] if frames else 0
                st.write(f"   → {len(frames)} frames · {fmt_time(dur)} duration · mode: **{mode}**")

                st.write("🤖 Querying Gemma 4 via LM Studio...")
                try:
                    if mode == "summary":
                        summary, evidence, raw = analyze_summary(model, frames, question)
                        st.session_state.update(mode=mode, summary=summary, results=evidence, raw=raw)
                    else:
                        results, raw = analyze_search(model, frames, question)
                        st.session_state.update(mode=mode, summary=None, results=results, raw=raw)
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.stop()

                status.update(label="Done!", state="complete")

    # ── Results ────────────────────────────────────────────────────────────────
    if "results" in st.session_state:
        mode    = st.session_state.get("mode", "search")
        results = st.session_state["results"]
        summary = st.session_state.get("summary")

        # Summary answer box (for count/describe questions)
        if summary:
            st.markdown(f"""
            <div class="evidence-card" style="border-color:#4a9eff">
                <div style="color:#4a9eff;font-weight:700;font-size:13px;margin-bottom:6px">
                    💬 AI ANSWER
                </div>
                <div class="ev-text" style="font-size:15px">{summary}</div>
            </div>""", unsafe_allow_html=True)

        # Evidence / frame clips
        if results:
            label = "📍 Notable frames" if mode == "summary" else f"🚨 Found **{len(results)}** evidence moment(s)"
            st.success(label) if mode == "search" else st.info(label)

            for i, r in enumerate(results, 1):
                ts, ev = r["timestamp"], r["evidence"]
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="ts-badge">#{i} &nbsp; ⏱ {fmt_time(ts)}</div>
                    <div class="ev-text">{ev}</div>
                </div>""", unsafe_allow_html=True)
                with st.expander(f"▶ Play clip at {fmt_time(ts)}"):
                    st.video(st.session_state["video_path"], start_time=int(ts))
        elif not summary:
            st.info("✅ No relevant activity found in the footage.")

        with st.expander("📄 Raw AI response"):
            st.code(st.session_state.get("raw", ""), language="text")