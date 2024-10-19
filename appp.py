import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector  # Use PGVector for PostgreSQL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import psycopg2
import uuid

# Load environment variables from the .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DATABASE_URL=os.getenv("PGVECTOR_CONNECTION_STRING")

# Connect to PostgreSQL for executing raw SQL queries
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Generate a unique UUID for each PDF upload session
session_uuid = str(uuid.uuid4())

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = text_splitter.split_text(text)
    return chunks



def delete_existing_embeddings():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Delete existing entries for this session's UUID
    delete_query = f"DELETE FROM langchain_pg_embedding WHERE custom_id = '{session_uuid}'"
    cur.execute(delete_query)
    
    conn.commit()
    cur.close()
    conn.close()


def get_vector_store(text_chunks):
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Ensure this model returns 1536 dimensions
    
    # Create a vector store from text chunks
    vector_store = PGVector.from_texts(
        texts=text_chunks,                
        embedding=embeddings,   
        connection_string=DATABASE_URL,                                    # Specify the table name where data will be stored
    )
    
    # Store vector data in the cloud database
      # Save to the cloud PostgreSQL database
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Use the following pieces of information and context to answer the user's question in brief with the help of llm model, and don't generate answer with a single word; try to curate an answer.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.

       Context: {context}
       Question: {question}

    Only return the helpful answer, curate answer using to reply in brief and try to answer in bullet points, but if the previous line is contextually not associated with it, then don't make bullet unnecessarily. Answer must be detailed and well explained.
    Helpful answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    new_db = PGVector(DATABASE_URL, embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the response in Streamlit
    print(response)
    st.write("🤖Reply-   ", response["output_text"])



# Main Streamlit app
def main():

    st.set_page_config(
        page_title="ToyStack AI - RAG Chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        # You can replace the URL below with your own logo URL or local image path
        st.image("logo.png", use_column_width=True)
        st.markdown("---")
        
        # Navigation Menu
        menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
        choice = st.selectbox("Navigate", menu)
    
    
    # Home Page
    if choice == "🏠 Home":
        st.title("📄 ToyStack AI - RAG Chatbot")
        st.markdown("""
        Welcome to **ToyStack AI - RAG Chatbot**!
    
        
        - **Upload Documents**: Easily upload your PDF documents.
        - **Summarize**: Get concise summaries of your documents.
        - **Chat**: Interact with your documents through our intelligent chatbot.
    
        
        """)
    
    # Chatbot Page
    elif choice == "🤖 Chatbot":
    
        st.header("🤖 Chatbot ")
        st.markdown(" ")
        
        # Input for the user to ask a question
        user_question = st.text_input("Ask a Question from the PDF Files")
        
        # If the user asks a question, process it
        if user_question:
            user_input(user_question)
        
        #Sidebar for uploading PDFs
        with st.sidebar:
            st.title(" ")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
    
    # Contact Page
    elif choice == "📧 Contact":
        st.title("📬 Contact Us")
        st.markdown("""
        We'd love to hear from you! Whether you have a question, feedback, or want to contribute, feel free to reach out.
                    
        - **Website:** [**ToyStack.ai**](https://toystack.ai)
        - [**Join us on Discord**](https://discord.com/invite/s5yVxSRmBN)   
        
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2024 ToyStack AI - RAG Chatbot. All rights reserved. ")
    
if __name__ == "__main__":
    main()  