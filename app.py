import streamlit as st
from dotenv import load_dotenv





HF_TOKEN = st.secrets["HF_TOKEN"]

from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    max_new_tokens=512,
    temperature=0.7
)

chat_model = ChatHuggingFace(llm=llm)


#Streamlit UI
st.title("Youtube  AI ChatBot")

st.write("Ask  questions from any Youtube video ")

youtube_url=st.text_input(
    "Enter Youtube video url"
)

question=st.text_input(
    "Ask your question"
)

def get_video_id(url):

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    elif "youtu.be" in url:
        return url.split("/")[-1]
    
def create_vector_db(video_id):

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)


    text = " ".join(
        [x.text for x in transcript]
    )


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    docs = splitter.create_documents(
        [text]
    )


    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )


    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )


    return vectorstore


def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash",
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        max_new_tokens=512,
        temperature=0.7
    )


    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model

if st.button("Create Chatbot"):

    video_id = get_video_id(
        youtube_url
    )


    with st.spinner("Processing video..."):

        st.session_state.vectorstore = create_vector_db(
            video_id
        )

        st.session_state.llm = load_llm()


    st.success(
        "Video processed successfully!"
    )



if "vectorstore" in st.session_state:

    if st.button("Ask"):

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_kwargs={"k":4}
            )
        )


        def format_docs(docs):
            return "\n\n".join(
                doc.page_content for doc in docs
            )


        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )


        prompt = PromptTemplate(
            template="""
You are a helpful assistant.

Answer the question using only the given context.

Context:
{context}

Question:
{question}
""",
            input_variables=[
                "context",
                "question"
            ]
        )


        parser = StrOutputParser()


        main_chain = (
            parallel_chain
            | prompt
            | st.session_state.llm
            | parser
        )


        answer = main_chain.invoke(
            question
        )


        st.write(answer)