import pickle
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# File paths
MODEL_PATH = r"D:\Herald\AI and ML\Coursework\Task3\sentiment_model.keras"
TOKENIZER_PATH = r"D:\Herald\AI and ML\Coursework\Task3\tokenizer.pkl"
MAX_LEN_PATH = r"D:\Herald\AI and ML\Coursework\Task3\max_len.pkl"

# Load model and saved objects
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(MAX_LEN_PATH, "rb") as f:
    max_len = pickle.load(f)

# App UI
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review and the trained model will predict its sentiment.")

review = st.text_area("Enter your movie review here:", height=180)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a movie review first.")
    else:
        sequence = tokenizer.texts_to_sequences([review])

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_len,
            padding="post"
        )

        prediction = model.predict(padded_sequence)
        score = float(prediction[0][0])

        st.subheader("Prediction Result")

        if score > 0.5:
            confidence = score * 100
            st.success(f"Positive Review")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            confidence = (1 - score) * 100
            st.error(f"Negative Review")
            st.write(f"Confidence: {confidence:.2f}%")

        st.write(f"Raw Prediction Score: {score:.4f}")