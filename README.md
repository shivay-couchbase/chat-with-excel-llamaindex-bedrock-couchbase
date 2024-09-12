# Chat with Excel using LlamaIndex, Couchbase & Bedrock

This project demonstrates a Streamlit application that allows users to chat with their Excel files using LlamaIndex, Couchbase Vector Store, and Amazon Bedrock.

## Features

- Upload Excel files and chat with their contents
- Utilizes Retrieval-Augmented Generation (RAG) for accurate responses
- Secure authentication system
- Integrates with Couchbase Vector Store for efficient document storage and retrieval
- Leverages Amazon Bedrock for LLM and embeddings

## Prerequisites

- Python 3.8+
- Couchbase Server
- Amazon Bedrock access
- LlamaCloud API key

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
   - Fill in the required credentials and settings

## Usage

1. Run the Streamlit app:
   ```
   streamlit run chat_with_excel.py
   ```
2. Open the provided URL in your browser
3. If authentication is enabled, enter the password
4. Upload an Excel file and start chatting!

## How it works

1. The app uses LlamaParse to extract structured data from Excel files
2. Extracted data is stored in Couchbase Vector Store
3. User queries are processed using RAG:
   - Relevant context is retrieved from Couchbase
   - Amazon Bedrock generates responses based on the context and query

## Configuration

Adjust the following settings in `chat_with_excel.py` as needed: in `chat_with_excel.py`

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
