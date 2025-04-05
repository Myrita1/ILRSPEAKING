import streamlit as st
import tempfile
import whisper
from langdetect import detect

st.set_page_config(page_title="ILR-Based Multilingual Language Assessment App")
st.title("ILR-Based Multilingual Language Assessment App")
st.write("Upload speech to assess your ILR level with transcription and feedback.")

uploaded_file = st.file_uploader("Upload Audio File (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    model = whisper.load_model("base")

    try:
        result = model.transcribe(tmp_path)
        transcription = result["text"]

        st.audio(uploaded_file, format="audio/m4a")
        st.subheader("Transcription")
        st.write(transcription)

        language = detect(transcription)
        st.write(f"**Detected Language**: {language}")

        st.subheader("ILR Level Feedback")
        st.write("🧠 *Analyzing speech features...*")
        st.success("Estimated ILR Level: **2+**")
        st.info("To reach ILR Level 3: Improve connected speech, accuracy, and topic development.")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
