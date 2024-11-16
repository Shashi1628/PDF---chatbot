import json
import os
import sys
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

# LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock)
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="us.meta.llama3-2-1b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Function to add custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #F9F9F9; /* Very light pastel background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #333333; /* Dark text */
        }

        .stApp {
            width: 100%;
            height: 100%;
            padding: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .stHeader {
            background-color: #FFFFFF; /* White background for header */
            color: #333333;
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        h1 {
            font-size: 3em;
            font-weight: 700;
            color: #333333; /* Dark Text */
        }

        .stTextInput>div>div>input {
          
            padding: 15px;
            font-size: 1.1em;
            border: 2px solid #333333; /* Dark Border */
         
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 100%;
            transition: 0.3s ease;
        }

        .stTextInput>div>div>input:focus {
            outline: none;
        }

        .stButton>button {
            background-color: #0078d4; /* Blue */
            color: white;
            font-size: 1.2em;
            border-radius: 25px;
            padding: 14px 30px;
            border: none;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #005b9f; /* Darker blue */
        }

        .stSidebar {
            background-color: #FFFFFF; /* White Sidebar */
            color: #333333;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
        }

        .stSidebar h2, .stSidebar p {
            color: #333333;
        }

        .stSpinner {
            color: #0078d4; /* Blue */
        }

        .stAlert {
            background-color: rgba(172, 177, 195, 0.15);
            color: #333333;
            font-size: 1.1em;
            border-radius: 15px;
            margin-bottom: 10px;
        }

        /* Full-screen input */
        .stTextInput input {
            width: 100%;
            max-width: 800px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Apply custom CSS
    add_custom_css()

    st.header("Interactive PDF Chatbot 📚")
    st.markdown("### Powered by AWS Bedrock")

    user_question = st.text_input("Ask a Question About the PDF 📄", placeholder="Type your question here...")

    with st.sidebar:
        st.header("Vector Store Operations")
        st.subheader("Data Ingestion and Update")
        if st.button("Ingest PDF Data and Update Vector Store"):
            with st.spinner("Ingesting Data..."):
                try:
                    docs = data_ingestion()
                    get_vector_store(docs)
                    st.success("Vector Store Updated Successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.subheader("Choose Language Model:")
        llm_option = st.radio("Select Language Model", ("Claude", "Llama2"))

    if llm_option:
        st.info(f"You have selected the **{llm_option}** model.")

    # Output based on selected LLM model
    if user_question:
        if llm_option == "Claude":
            if st.button("Generate Claude Response"):
                with st.spinner("Processing..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = get_claude_llm()
                        response = get_response_llm(llm, faiss_index, user_question)
                        st.write(response)
                        st.success("Done")
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif llm_option == "Llama2":
            if st.button("Generate Llama2 Response"):
                with st.spinner("Processing..."):
                    try:
                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = get_llama2_llm()
                        response = get_response_llm(llm, faiss_index, user_question)
                        st.write(response)
                        st.success("Done")
                    except Exception as e:
                        st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question to get started!")

if __name__ == "__main__":
    main()
