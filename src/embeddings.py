from openai import OpenAI
import streamlit as st

from src.config import OPENAI_EMBEDDING_MODEL
import os


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)