import streamlit as st
from langchain.llms import OpenAI
from Chatbot import get_openai_api_key

st.title("🔗 CDER Langchain Chatbot")

with st.sidebar:
    openai_api_key = get_openai_api_key()

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "When was Liptior approved?")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        generate_response(text)
