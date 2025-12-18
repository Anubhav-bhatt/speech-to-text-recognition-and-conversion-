import gradio as gr
import whisper
import os
import subprocess
import uuid
import language_tool_python

# ----------------------------
# Load models
# ----------------------------
model = whisper.load_model("tiny")
tool = language_tool_python.LanguageTool('en-US')

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------
# Audio cleaner
# ----------------------------
def clean_audio(input_audio):
    output_audio = os.path.join(
        TEMP_DIR, f"clean_{uuid.uuid4()}.wav"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_audio,
            "-ac", "1",
            "-ar", "16000",
            output_audio
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return output_audio

# ----------------------------
# Make English meaningful
# ----------------------------
def improve_english(text):
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected

# ----------------------------
# Telugu → Meaningful English
# ----------------------------
def telugu_to_meaningful_english(audio_file):
    if audio_file is None:
        return "❌ No audio received"

    try:
        clean_wav = clean_audio(audio_file)

        # Telugu → English
        result = model.transcribe(
            clean_wav,
            task="translate",
            fp16=False
        )

        raw_english = result["text"].strip()

        if not raw_english:
            return "⚠️ No speech detected."

        # Improve sentence quality
        final_english = improve_english(raw_english)

        return final_english

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------------------
# UI
# ----------------------------
ui = gr.Interface(
    fn=telugu_to_meaningful_english,
    inputs=gr.Audio(
        type="filepath",
        label="🎤 Speak Telugu"
    ),
    outputs=gr.Textbox(
        label="📝 Meaningful English Sentence"
    ),
    title="Telugu Speech → Meaningful English",
    description="Speech translation with grammar & sentence correction"
)

if __name__ == "__main__":
    ui.launch()
