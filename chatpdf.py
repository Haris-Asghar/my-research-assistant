import os
from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.knowledge_base import list_canopy_indexes
from canopy.models.data_models import Document
from canopy.chat_engine import ChatEngine
from canopy.context_engine import ContextEngine
from canopy.models.data_models import UserMessage, AssistantMessage
import arxiv
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from pinecone import ServerlessSpec

PINECONE_API = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# initialize tokenizer
Tokenizer.initialize()

# set up index
INDEX_NAME = "knowledge-base"

kb = KnowledgeBase(index_name=INDEX_NAME)

spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

if not any(name.endswith(INDEX_NAME) for name in list_canopy_indexes()):
    kb.create_canopy_index(spec)

kb.connect()

context_engine = ContextEngine(kb)
chat_engine = ChatEngine(context_engine)


def chat(new_message, history):
    messages = history + [UserMessage(content=new_message)]
    response = chat_engine.chat(messages)
    assistant_response = response.choices[0].message.content
    return assistant_response, messages + [AssistantMessage(content=assistant_response)]


def upload(document):
    # upload a single document to the knowledge base
    return kb.upsert([document])


def fetch_arxiv_paper(arxiv_id):
    # Query the arXiv API for the paper using the provided arXiv ID
    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[arxiv_id])), None)
    if paper is None:
        raise ValueError("No paper found with the provided arXiv ID.")

    # Download the PDF
    response = requests.get(paper.pdf_url)
    if response.status_code != 200:
        raise IOError("Failed to download the PDF.")

    # Save the PDF temporarily
    with open("temp_paper.pdf", "wb") as f:
        f.write(response.content)

    # Read the PDF content using PyMuPDFLoader
    loader = PyMuPDFLoader("temp_paper.pdf")
    data = loader.load()
    text_content = ""
    for page in data:
        page_text = page.page_content
        if "References" in page_text:
            text_content += page_text.split("References")[0]
            break
        text_content += page_text

    # Clean up the temporary file
    os.remove("temp_paper.pdf")

    # Extracting details
    return paper.title, [author.name for author in paper.authors], text_content, paper.entry_id


def add_paper_to_kb(arxiv_id):
    '''
    Use this function to upload a paper to the pinecone index.
    '''
    title, authors, content, link = fetch_arxiv_paper(arxiv_id)

    # create a canopy document
    document = Document(
        id=arxiv_id,
        source=link,
        text=content,
        metadata={
            "title": title,
            "authors": ",".join(authors)
        }
    )

    upload(document)


def clear_kb():
    '''
    Use this function to clear the pinecone index.
    '''
    kb.delete_index()
    kb.create_canopy_index(spec)


def cli_chat():
    print("Welcome to the CLI Chat. Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat...")
            break
        try:
            response, history = chat(user_input, history)
            print("Assistant:", response)
        except Exception as e:
            print("Error:", str(e))
            break


if __name__ == "__main__":
    cli_chat()
