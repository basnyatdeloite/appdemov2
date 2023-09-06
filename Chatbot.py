import openai
import streamlit as st
import os
import openai
from dotenv import load_dotenv
from streamlit_feedback import streamlit_feedback
import trubrics
from trubrics.integrations.streamlit import FeedbackCollector
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


def get_openai_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


with st.sidebar:
    openai_api_key = get_openai_api_key()

st.title("📝 Chat with CDER-Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
    ]
if "response" not in st.session_state:
    st.session_state["response"] = None

messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask me anything!"):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    st.session_state["response"] = response.choices[0].message.content
    with st.chat_message("assistant"):
        messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])

if st.session_state["response"]:
    # feedback = streamlit_feedback(
    #     feedback_type="thumbs",
    #     optional_text_label="[Optional] Please provide an explanation",
    #     key=f"feedback_{len(messages)}",
    # )
    # This app is logging feedback to Trubrics backend, but you can send it anywhere.
    # The return value of streamlit_feedback() is just a dict.
    # Configure your own account at https://trubrics.streamlit.app/
    # if feedback and "TRUBRICS_EMAIL" in st.secrets:
    # config = trubrics.init(
    #     email=st.secrets.TRUBRICS_EMAIL,
    #     password=st.secrets.TRUBRICS_PASSWORD,
    # )
    #
    # collection = trubrics.collect(
    #     component_name="default",
    #     model="gpt-3.5-turbo",
    #     response=feedback,
    #     metadata={"chat": messages},
    # )
    # trubrics.save(config, collection)
    # st.toast("Feedback recorded!", icon="📝")

    collector = FeedbackCollector(
        project="default",
        email=st.secrets.TRUBRICS_EMAIL,
        password=st.secrets.TRUBRICS_PASSWORD,
    )

    if "feedback_collected" not in st.session_state:
        st.session_state.feedback_collected = False

    feedback = collector.st_feedback(
        component="default",
        feedback_type="thumbs",
        model="gpt-3.5-turbo",
        prompt_id=None,  # see prompts to log prompts and model generations
        open_feedback_label='Optional Provide additional feedback'
    )

    if feedback is not None:  # or another condition that indicates feedback was given
        st.session_state.feedback_collected = True

    # send_email(collector, messages)
