import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from google import genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -------- Extract PDF Text --------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# -------- Split into Chunks --------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# -------- Create Vector Store --------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"   # ✅ FIXED
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -------- Create Gemini Chain --------
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say:
    "Answer is not available in the context"

    Context:
    {context}

    Question:
    {input}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # ✅ FIXED
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    return create_stuff_documents_chain(model, prompt)


# -------- Handle User Question --------
def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"   # ✅ FIXED HERE TOO
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    combine_docs_chain = get_conversational_chain()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = retrieval_chain.invoke({"input": user_question})

    st.write("Reply:", response["answer"])


# -------- Streamlit UI --------
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Done")


if __name__ == "__main__":
    main()