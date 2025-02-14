from dotenv import load_dotenv
import os
import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage



def main():
    # Load environment variables
    #load_dotenv()
    cohere_api = 'z1gu5tAWkkl2nBeTS8EAZLKo8q0TWNLWMK8DsfTF'

    if not cohere_api:
        st.error("Cohere API Key not found. Please add it to your .env file.")
        return

    # Display contents of UI
    ui_content()

    # Contains extracted PDF
    pdf = pdf_reader()

    # Extract the text from the uploaded PDF if it's not None
    if pdf is not None:
        extracted_text = pdf_extract(pdf)

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(extracted_text)

        # Initialize Cohere embeddings
        user ='Studdybuddy'
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api, user_agent=user)
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)
        client_question = st.text_input("Ask a question about your PDF")

        if client_question:
            docs = knowledge_base.similarity_search(query=client_question)

            # Initialize the Cohere LLM
            llm = ChatCohere(
            cohere_api_key=cohere_api,
            model="command-r-plus-08-2024"
        )


            # Pass the correctly initialized LLM to the QA chain
            chain = load_qa_chain(llm, chain_type="stuff")  # The proper use of chain type
            response = chain.run(input_documents=docs, question=client_question)

            st.write(response)

def ui_content():
    st.header("Welcome :blue[_Carleton Students_] to StudyBuddy")
    st.header("Come ask your PDF! 📑", divider="blue")


def pdf_reader():
    pdf = st.file_uploader("Upload your PDF here", type="pdf")
    return pdf


def pdf_extract(pdf):
    # Convert the uploaded file to text directly without saving it
    extracted_text = extract_text(pdf)
    return extracted_text


if __name__ == '__main__':
    main()
