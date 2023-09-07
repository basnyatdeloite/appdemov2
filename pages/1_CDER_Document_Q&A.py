import streamlit as st
# import anthropic
import openai
from Generic_ChatBot import get_openai_api_key
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def pdf_to_text(pdf_docs):
    """
    Function to extract text from pdf documents

    Inputs:
        pdf_docs - list of uploaded pdf documents
    Outputs:
        text - extracted text
    """
    text = ""
    for pdf in pdf_docs:
        # initialize pdfreader
        pdf_reader = PdfReader(pdf)
        # loop through the pages of the pdf and extract the text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def embedder(model_type):
    """
    Function that returns the embedding model that will be used to embed the text

    Inputs:
        model_type - Hugging Face or OpenAI
    Outputs:
        embeddings - embedding model
    """
    if model_type == "Open AI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    return embeddings


def create_text_chunks(text):
    """
    This function takes the extracted text from the documents and splits it into smaller text chunks
    The resulting text chunks are returned

    Inputs:
        text - the text extracted from documents
    Outputs:
        chunks - a list containing the chunks of texts
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(text, embedding_model):
    """
    This function embeds the text from the document and stores in a vectorstore

    Inputs:
        text - extracted text from document
        embedding_model - embedding model
    Outputs:
        vectorstore - vector store containing the embedded text
    """
    # create vectorstore and store embedded text
    vectorstore = FAISS.from_texts(texts=text, embedding=embedding_model)
    return vectorstore


def create_conversation_chain(model_type, vectorstore):
    """
    This function creates a conversation chain for conversation retrieval

    Inputs:
        model_type - huggingface or openai
        vectorstore - The vector database containing the embeddings of the text
    Outputs:
        conversation_chain
    """

    if model_type == "Open AI":
        # Load OpenAI conversational model
        model = ChatOpenAI()
    else:
        # Load hugging face model
        model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

        # memory buffer is used to store and retrieve previous conversation history. Used to maintain context and continuity
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def process_pdf_docs(pdf_docs, model_type="Open AI"):
    try:
        # Extract text from pdf documents
        text = pdf_to_text(pdf_docs)

        if not text:
            # If no text is extracted, return None
            st.error("Can't extract text from this document. Try another one.")
            return None

        # Create text chunks
        text_chunks = create_text_chunks(text)
        # Load embedding
        embedding = embedder(model_type)
        # Create vector store
        vectorstore = create_vectorstore(text_chunks, embedding)
        # Create conversation chain
        return create_conversation_chain(model_type, vectorstore)
    except Exception as e:
        # Catch and handle any exception that might occur during processing
        st.error(e)
        return None

st.sidebar.title("📝 CDER Document Q&A")
st.sidebar.subheader("LLM: GTP 3.5 Turbo (Open AI)")

with st.sidebar:
    # anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    openai_api_key = get_openai_api_key()




if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

uploaded_file = st.sidebar.file_uploader("Upload a file", accept_multiple_files=True)
if st.sidebar.button("Process"):
    with st.spinner("Processing"):
        st.session_state.conversation = process_pdf_docs(uploaded_file)

if uploaded_file:
    prompt = st.chat_input("How can I help you?")
    if prompt:
        response = st.session_state.conversation({"question": prompt})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message("user" if i % 2 == 0 else "assistant"):
                st.markdown(message.content)

##add sidebar warning
st.sidebar.subheader("""Disclaimer: """)
st.sidebar.markdown("""Do not upload sensitive or proprietary data to this application. It is still in development and utilizes API calls to Open AI as opposed to an internal open source model running on an FDA controlled server""")
    




# question = st.text_input(
#     "Ask something about the article",
#     placeholder="Can you give me a short summary?",
#     disabled=not uploaded_file,
# )

# if uploaded_file and question and not anthropic_api_key:
#     st.info("Please add your Anthropic API key to continue.")

# if uploaded_file and question and not openai_api_key:
#     st.info("Please add your Openai API key to continue.")

# if uploaded_file and question and anthropic_api_key:
#     article = uploaded_file.read().decode()
#     prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
#     {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""
#
#     client = anthropic.Client(api_key=anthropic_api_key)
#     response = client.completions.create(
#         prompt=prompt,
#         stop_sequences=[anthropic.HUMAN_PROMPT],
#         model="claude-v1", #"claude-2" for Claude 2 model
#         max_tokens_to_sample=100,
#     )

# if uploaded_file and question and openai_api_key:
#     article = uploaded_file.read().decode()
#     prompt = f"Here's an article:\n\n<article>\n{article}\n</article>\n\nQuestion: {question}"
#
#     openai.api_key = openai_api_key
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=200,
#         stop=None
#     )
#
#     st.write("### Answer")
#     # st.write(response.completion)
#     st.write(response['choices'][0]['text'])
