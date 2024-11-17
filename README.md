# PDF Chatbot Using AWS Bedrock

This application allows users to interact with PDF documents by asking questions. The app utilizes AWS Bedrock services, specifically the Claude and Llama2 models, to provide detailed responses based on the PDF content. It incorporates vector embeddings to retrieve relevant document sections, ensuring high-quality answers.

## Features

- **PDF Document Ingestion**: Upload and process PDFs to generate embeddings and create a vector store.
- **AI Model Integration**: Use the Claude and Llama2 models to generate responses based on the context from PDF files.
- **Question Answering**: Users can ask questions about the PDF content, and the app will return concise, accurate answers.
- **Interactive Interface**: A clean and professional user interface built with Streamlit, featuring easy navigation and a pleasant user experience.

## Technologies Used

- **Streamlit**: Used to create a simple web-based UI for the application.
- **AWS Bedrock**: Provides access to the Claude and Llama2 models for natural language processing tasks.
- **Langchain**: Utilized for document loading, splitting, embedding, and interacting with vector stores.
- **FAISS**: Used for fast retrieval of document chunks based on similarity search.
- **Python**: The core programming language used for backend logic.

## Setup

### Prerequisites

- Python 3.7 or later
- AWS Account with access to AWS Bedrock and the necessary permissions
- Streamlit installed for running the app

### Dependencies

- `boto3` (for AWS SDK)
- `langchain` (for working with language models and embeddings)
- `faiss-cpu` (for vector store handling)
- `streamlit` (for frontend)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Shashi1628/PDF-chatbot.git
   cd PDF-chatbot
