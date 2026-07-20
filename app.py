import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎬")
st.title("🎬 YouTube Video Chatbot")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
video_url = st.sidebar.text_input("YouTube Video URL")
process_btn = st.sidebar.button("Load Video")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Invalid YouTube URL")

def fetch_transcript(url):
    video_id = get_video_id(url)
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en"])
        return fetched.to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def build_chain(video_url, groq_api_key):
    transcript_list = fetch_transcript(video_url)
    if not transcript_list:
        return None, "No transcript available for this video."

    full_text = " ".join([seg["text"] for seg in transcript_list])
    doc = Document(page_content=full_text, metadata={"source": video_url})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents([doc])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        Chroma(collection_name="youtube-transcript", embedding_function=embeddings).delete_collection()
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="youtube-transcript"
    )

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about a YouTube video based on its transcript.
Use only the following context to answer the question. If the answer isn't in the context, say you don't have enough information from this video to answer.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt | llm | StrOutputParser()
    )
    return chain, None

if process_btn:
    if not groq_api_key or not video_url:
        st.sidebar.error("Please provide both Groq API key and video URL.")
    else:
        with st.spinner("Processing video transcript..."):
            chain, error = build_chain(video_url, groq_api_key)
            if error:
                st.sidebar.error(error)
            else:
                st.session_state.chain = chain
                st.session_state.chat_history = []
                st.sidebar.success("Video loaded! Ask away.")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

if st.session_state.chain:
    user_question = st.chat_input("Ask something about the video...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.info("👈 Enter your Groq API key and a YouTube URL in the sidebar, then click 'Load Video'.")
