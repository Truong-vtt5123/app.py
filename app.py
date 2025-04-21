import streamlit as st
import subprocess
import whisper
from keybert import KeyBERT
from transformers import pipeline
#from bertopic import BERTopic
import os


def download_audio(video_url, output_file="audio.wav"):
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_file,
        video_url
    ]
    subprocess.run(command)
    return output_file


def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    result = model.transcribe(audio_path)
    return result["text"]


def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text[:512])  # Limit to 512 characters
    return result


def extract_keywords(text, top_n=10):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return keywords


def extract_topics(text):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform([text])
    return topic_model.get_topic_info()


def main():
    st.title("🎥 YouTube Video Content Analyzer")
    video_url = st.text_input("📺 Enter YouTube Video URL:", "")

    if video_url:
        st.info("⏬ Downloading audio from YouTube...")
        audio_filename = download_audio(video_url)
        st.success("✅ Audio downloaded successfully!")

        st.info("🗣️ Transcribing audio to text...")
        transcript = transcribe_audio(audio_filename)
        st.text_area("📄 Transcript:", transcript, height=200)

        st.info("🔍 Analyzing sentiment...")
        sentiment_result = analyze_sentiment(transcript)
        st.write("🧠 Sentiment Result:", sentiment_result)

        st.info("🧠 Extracting keywords...")
        keywords = extract_keywords(transcript)
        st.write("🔑 Top Keywords:")
        for kw, score in keywords:
            st.write(f"- {kw} ({score:.2f})")

        # st.info("🧠 Identifying topics...")
        # topic_info = extract_topics(transcript)
        # st.write("📚 Topics:")
        # st.dataframe(topic_info)

        if os.path.exists(audio_filename):
            os.remove(audio_filename)


if __name__ == "__main__":
    main()
