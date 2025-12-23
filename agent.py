from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import time

@st.cache_resource
def load_vector_db():
    return getPdfFile()


def getPdfFile():
    local_path = "Podstawy_inwestowania1.pdf"
    if local_path:
        loader = PyPDFLoader(file_path=local_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
            collection_name="local-rag"
        )
        return vector_db
    else:
        print("Upload a PDF file")

def safe_api_call(func):
    """Calls func() and catches all exceptions.
    If an error occurs, returns (None, str(error)).
    If OK, returns the function result"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(str(e))
            return None, str(e)
    return wrapper


@safe_api_call
def call_bielik(prompt, question, vector_db):
    bielik_model = ChatOllama(
        model="hf.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0",
        temperature=0.1
    )

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=prompt,
    )
    
    try:
        if vector_db:
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                bielik_model,
                prompt=QUERY_PROMPT
            )
        else:
            retriever = None
    except Exception as e:
        print("Błąd MultiQueryRetriever:")
        print(e)
        retriever = None

    final_template = """odpowiedz na pytanie bazujac na ponizszym kontekstu:
    {context}
    Podaj fragment pochodzący z kontekstu
    Question: {question}
    """


    chat_prompt = ChatPromptTemplate.from_template(final_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | chat_prompt
        | bielik_model
        | StrOutputParser()
    )

    response = chain.invoke({"question": question})
    return response, None

# PROMPT = """
#         Jestes Polskim asystenem inwestowania, 
#         odpowiadaj na pytanie tylko i wyłącznie dotyczące inwestowania
#         na pytania po za tematem inewstownia odpowiadaj, że nie masz o nich pojęcia
#         Original question: {question}
#     """

PROMPT = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""

def runStreamlit():
    st.title("LLM Model Comparison")

    vector_db = load_vector_db()


    tests = [
        ("Bielik", call_bielik),
    ]

    prompts = {
        "text": PROMPT
    }

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Zadaj pytanie do modelu:")

    if st.button("▶️ Wyślij") and question:
        responses = []
        for name, func in tests:
            for nameP, prompt in prompts.items():
                with st.spinner(f"Wywołanie: {name}..."):
                    start = time.time()
                    text, error = func(prompt, question, vector_db)
                    elapsed = time.time() - start
                    responses.append((name, text, elapsed, error))

        st.session_state.history.append((question, responses))

    if st.session_state.history:
        st.header("Historia pytań i odpowiedzi")
        for i, (q, res_list) in enumerate(st.session_state.history[::-1]):
            st.subheader(f"Q{i+1}: {q}")
            for name, text, elapsed, error in res_list:
                st.markdown(f"**Model {name}** ({elapsed:.2f}s):")
                if error:
                    st.error(error)
                else:
                    st.info(text)
            st.markdown("---")

if __name__ == "__main__":
    runStreamlit()
