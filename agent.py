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
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import time
import yfinance as yf
from dotenv import load_dotenv
from langchain.tools import tool 

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

    bielik_model_with_tools = bielik_model.bind_tools([stock_price], 
    tool_choice="stock_price"
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
    Jeśli pytanie dotyczy aktualnej ceny akcji, kursu giełdowego lub wartości rynkowej, MUSISZ użyć narzędzia 'stock_price'. Nie zgaduj.
    Question: {question}
    """


    chat_prompt = ChatPromptTemplate.from_template(final_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | chat_prompt
        | bielik_model_with_tools
    )

    response = chain.invoke({"question": question})
    if response.tool_calls:
        return response, "TOOL_CALL"
    else:
        return response.content, None


@safe_api_call
def call_gemini(prompt, question, vector_db):
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0.1
    )

    gemini_model_with_tools = gemini_model.bind_tools([stock_price], 
    tool_choice="stock_price"
    )

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=prompt,
    )
    
    try:
        if vector_db:
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                gemini_model,
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
    Jeśli pytanie dotyczy aktualnej ceny akcji, kursu giełdowego lub wartości rynkowej, MUSISZ użyć narzędzia 'stock_price'. Nie zgaduj.
    Question: {question}
    """


    chat_prompt = ChatPromptTemplate.from_template(final_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | chat_prompt
        | gemini_model_with_tools
    )

    response = chain.invoke({"question": question})
    if response.tool_calls:
        return response, "TOOL_CALL"
    else:
        return response.content, None

@tool
def stock_price(ticker: str) -> str:
    """
    Pobiera aktualną cenę akcji dla danego tickera (np. AAPL, TSLA).
    """

    stock = yf.Ticker(ticker)
    price = stock.info['regularMarketPrice']
    print("wykorzystalem toola")

    if price is None:
        return "Nie udało się pobrać ceny."

    return f"Aktualna cena {ticker.upper()} wynosi {price} USD"
    

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

def run_streamlit():
    load_dotenv()

    st.set_page_config(page_title="LLM Compare", layout="wide")
    st.title("🤖 Porównanie modeli LLM")

    vector_db = load_vector_db()

    models = [
        ("Bielik", call_bielik),
        ("Gemini", call_gemini),
    ]

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Zadaj pytanie:")

    if st.button("▶️ Wyślij") and question:
        responses = []

        for name, func in models:
            with st.spinner(f"{name} odpowiada..."):
                start = time.time()
                res, status = func(PROMPT, question, vector_db)
                
                if status == "TOOL_CALL":
                    tool_call = res.tool_calls[0]
                    args = tool_call["args"]
                    
                    try:
                        result = stock_price.invoke(args)
                        display_text = result 
                    except Exception as e:
                        display_text = f"Błąd podczas pobierania ceny: {e}"
                else:
                    display_text = res

                elapsed = time.time() - start
                
                responses.append((name, display_text, elapsed, None))

        st.session_state.history.append((question, responses))

    if st.session_state.history:
        st.header("📜 Historia")

        for q, results in reversed(st.session_state.history):
            st.subheader(f"❓ {q}")

            cols = st.columns(len(results))

            for col, (name, text, elapsed, error) in zip(cols, results):
                with col:
                    st.markdown(f"### {name}")
                    st.caption(f"⏱ {elapsed:.2f}s")

                    if error:
                        st.error(error)
                    else:
                        st.chat_message("assistant").markdown(text)

            st.markdown("---")


if __name__ == "__main__":
    run_streamlit()

