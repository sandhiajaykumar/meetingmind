import os

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

VECTOR_DB_PATH = "data/vectorstore"


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# ============================================================
# CREATE / UPDATE VECTOR DATABASE
# ============================================================

def create_vectorstore(
    transcript: str,
    meeting_id: str
):

    """
    Add a meeting transcript to the FAISS vector database.

    If the database does not exist, create it.

    If it already exists, add the new meeting to it.
    """

    # Make sure data directory exists
    os.makedirs(
        "data",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create document
    # --------------------------------------------------------

    document = Document(
        page_content=transcript,
        metadata={
            "meeting_id": meeting_id
        }
    )

    # --------------------------------------------------------
    # Split transcript into chunks
    # --------------------------------------------------------

    documents = text_splitter.split_documents(
        [document]
    )

    # --------------------------------------------------------
    # Load existing vector database
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Create new vector database
    # --------------------------------------------------------

    if vectorstore is None:

        vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )

    # --------------------------------------------------------
    # Add to existing database
    # --------------------------------------------------------

    else:

        vectorstore.add_documents(
            documents
        )

    # --------------------------------------------------------
    # Save vector database
    # --------------------------------------------------------

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    return vectorstore


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vectorstore():

    """
    Load the existing FAISS vector database.
    """

    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(index_file):

        return None

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# SEARCH MEETINGS
# ============================================================

def search_meetings(
    question: str,
    k: int = 3
):

    """
    Perform semantic similarity search
    against stored meeting transcripts.
    """

    vectorstore = load_vectorstore()

    if vectorstore is None:

        return []

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results