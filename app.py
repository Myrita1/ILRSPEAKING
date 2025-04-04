import os
import streamlit as st
from pydub import AudioSegment
from pydub.utils import which
import whisper
from langdetect import detect

# Set ffmpeg path explicitly
AudioSegment.converter = r"C:\\Users\\kamal\\OneDrive\\Desktop\\LLM\\ffmpeg.exe"

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(file):
    try:
        audio = AudioSegment.from_file(file)
        duration = len(audio) / 1000
        if duration < 60:
            return None, "Audio must be at least 60 seconds long."
        temp_path = "temp.wav"
        audio.export(temp_path, format="wav")
        result = model.transcribe(temp_path)
        return result["text"], None
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def analyze_ILR(text):
    length = len(text.split())
    if length < 50:
        return 1
    elif length < 100:
        return 2
    elif length < 150:
        return 3
    elif length < 200:
        return 4
    else:
        return 5

def provide_feedback(ilr_level):
    feedback = {
        1: "Basic understanding. Focus on high-frequency vocabulary and simple structures.",
        2: "You can handle simple conversations. Try to improve fluency and cohesion.",
        3: "Functional ability. Work on accuracy and coherence in extended speech.",
        4: "Advanced proficiency. Start polishing nuance and idiomatic use.",
        5: "Highly proficient. Keep refining register and spontaneous expression."
    }
    return feedback.get(ilr_level, "No feedback available.")

# Streamlit UI
st.title("ILR-Based Multilingual Language Assessment App")
st.markdown("Upload speech to assess your ILR level with transcription and feedback.")

audio_file = st.file_uploader("Upload Audio File (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    text, error = transcribe_audio(audio_file)
    if error:
        st.error(error)
    else:
        st.subheader("Transcription")
        st.write(text)

        lang = detect_language(text)
        st.write(f"Detected Language: **{lang.upper()}**")

        ilr = analyze_ILR(text)
        st.success(f"Estimated ILR Level: {ilr}")

        fb = provide_feedback(ilr)
        st.markdown(f"**Feedback:** {fb}")
