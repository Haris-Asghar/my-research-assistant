from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import pandas as pd
from chatui import llm_chat
from chatpdf import add_paper_to_kb, clear_kb
from pdf2pdf import extract_text, generate_embeddings, query_pinecone


def search_papers():
    with st.form(key='search_form'):
        search_query = st.text_input(
            "Enter search terms or a prompt to find research papers:")
        search_button = st.form_submit_button(label='Search')
        if search_button:
            with st.spinner('Searching for relevant research papers...'):
                embeddings = generate_embeddings(search_query)
                query_results = query_pinecone(embeddings)
                query_matches = query_results[0]["matches"]

                similar_papers = {"DOI": [], "Title": [], "Date": []}
                for match in query_matches:
                    similar_papers["DOI"].append(match["metadata"]["doi"])
                    similar_papers["Title"].append(match["metadata"]["title"])
                    similar_papers["Date"].append(
                        match["metadata"]["latest_creation_date"])
                similar_papers = pd.DataFrame(similar_papers)
                similar_papers_sorted = similar_papers.sort_values(
                    by="Date", ascending=False)
                st.write(similar_papers_sorted)


def upload_pdf():
    with st.form(key='upload_form'):
        uploaded_file = st.file_uploader(
            "Upload a PDF file of a Research Paper, to find a Similar Research Paper", type=['pdf'])
        upload_button = st.form_submit_button(label='Upload')
        if upload_button and uploaded_file:
            with st.spinner('Processing your PDF...'):
                st.write(uploaded_file.name)
                file_path = os.path.join("PDFs", uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                data = extract_text("PDFs/" + uploaded_file.name)
                if data:
                    st.write(data)
                else:
                    st.write("ggwp")
                if len(data) > 5:
                    embeddings = generate_embeddings(data)
                    query_results = query_pinecone(embeddings)
                    query_matches = query_results[0]["matches"]

                    similar_papers = {"DOI": [], "Title": [], "Date": []}
                    for match in query_matches:
                        similar_papers["DOI"].append(match["metadata"]["doi"])
                        similar_papers["Title"].append(
                            match["metadata"]["title"])
                        similar_papers["Date"].append(
                            match["metadata"]["latest_creation_date"])
                    similar_papers = pd.DataFrame(similar_papers)
                    similar_papers_sorted = similar_papers.sort_values(
                        by="Date", ascending=False)
                    st.write(similar_papers_sorted)


def update_knowledge_base():
    st.title("Update Knowledge Base")
    arxiv_id = st.text_input("Enter the arXiv ID of the paper to add:")
    submit_button = st.button("Add Paper")
    clear_kb_button = st.button("Clear Knowledge Base")

    if submit_button:
        with st.spinner("Adding paper to the knowledge base..."):
            try:
                add_paper_to_kb(arxiv_id)
                st.success("Paper successfully added to the knowledge base!")
            except Exception as e:
                st.error(f"Failed to add paper: {str(e)}")

    if clear_kb_button:
        with st.spinner("Clearing the knowledge base..."):
            try:
                clear_kb()
                st.success("Knowledge base cleared successfully!")
            except Exception as e:
                st.error(f"Failed to clear the knowledge base: {str(e)}")


def main():
    st.set_page_config(page_title="Research Assistant", layout="wide")

    # Sidebar for selecting functionality
    st.sidebar.title("Features")
    app_mode = st.sidebar.selectbox("Choose a feature",
                                    ["Chat with Research", "Update Knowledge Base", "Prompt to Paper", "PDF to Paper", ])

    if app_mode == "Prompt to Paper":
        search_papers()
    elif app_mode == "PDF to Paper":
        upload_pdf()
    elif app_mode == "Chat with Research":
        llm_chat()
    elif app_mode == "Update Knowledge Base":
        update_knowledge_base()


if __name__ == "__main__":
    main()
