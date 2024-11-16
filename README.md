PDF Chatbot Using AWS Bedrock
This application allows users to interact with PDF documents by asking questions. The app utilizes AWS Bedrock services, specifically the Claude and Llama2 models, to provide detailed responses based on the PDF content. It incorporates vector embeddings to retrieve relevant document sections, ensuring high-quality answers.

Features
PDF Document Ingestion: Upload and process PDFs to generate embeddings and create a vector store.
AI Model Integration: Use the Claude and Llama2 models to generate responses based on the context from PDF files.
Question Answering: Users can ask questions about the PDF content, and the app will return concise, accurate answers.
Interactive Interface: A clean and professional user interface built with Streamlit, featuring easy navigation and a pleasant user experience.
Technologies Used
Streamlit: Used to create a simple web-based UI for the application.
AWS Bedrock: Provides access to the Claude and Llama2 models for natural language processing tasks.
Langchain: Utilized for document loading, splitting, embedding, and interacting with vector stores.
FAISS: Used for fast retrieval of document chunks based on similarity search.
Python: The core programming language used for backend logic.
Setup
Prerequisites
Python 3.7 or later
AWS Account with access to AWS Bedrock and the necessary permissions
Streamlit installed for running the app
Dependencies:
boto3 (for AWS SDK)
langchain (for working with language models and embeddings)
faiss-cpu (for vector store handling)
streamlit (for frontend)
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-repository-url.git
cd your-repository-folder
Install the necessary Python packages:

bash
Copy code
pip install -r requirements.txt
Set up your AWS credentials (make sure they have access to AWS Bedrock):

Either set up your credentials using AWS CLI:
bash
Copy code
aws configure
Or manually configure them via environment variables in your ~/.bashrc or ~/.zshrc:
bash
Copy code
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
Running the Application
Once all the dependencies are installed and your AWS credentials are configured, you can run the application:

bash
Copy code
streamlit run app.py
This will launch the app in your default browser.

Uploading PDFs
Upload a directory of PDFs (ensure the PDF files are inside the "data" folder in your project directory).
The app will process these files and generate embeddings for them.
You can then query the app by typing a question related to the contents of the PDFs.
Using the Application
Input Field: The main input field allows users to enter questions related to the uploaded PDFs.
Vector Store Update: Use the "Update Vectors" button to regenerate embeddings and update the vector store.
Model Selection: Choose between the Claude or Llama2 models to generate responses.
Response Area: The generated answer will be displayed here, along with the relevant context from the PDF.
Customization
You can adjust the following settings to customize the application:

PDF Ingestion: Modify the data_ingestion() function to include different file directories or formats.
Embedding Model: The get_vector_store() function uses the Titan Embedding model, but you can switch it to another model if needed.
AI Model: The get_claude_llm() and get_llama2_llm() functions can be customized to connect with other models if required.
UI Customization: Modify the CSS inside the add_custom_css() function to change the appearance of the app (e.g., colors, layout).
Screenshots
Example of the Main Interface:

Vector Store Update Section:

Model Selection and Responses:

Troubleshooting
Model Not Responding: If the models fail to load, ensure your AWS credentials are configured properly and that your account has access to AWS Bedrock services.
PDF Upload Failures: Ensure that the PDF files are in the correct format and placed in the data folder.
