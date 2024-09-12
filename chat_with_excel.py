import os
import tempfile

import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)

from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.couchbase import CouchbaseVectorStore

from llama_parse import LlamaParse
import nest_asyncio




def parse_bool(value: str):
    """Parse boolean values from environment variables"""
    return value.lower() in ("yes", "true", "t", "1")


def check_environment_variable(variable_name):
    """Check if environment variable is set"""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"
        )
        st.stop()


nest_asyncio.apply()

def store_document(uploaded_file, storage_context):
    """Parse the Excel file & store it in Couchbase Vector Store."""
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        parser = LlamaParse(
            api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
            parsing_instruction="""You are parsing the open positions from a stock trading book. The column Symbol contains the company name.
            Please extract Symbol, Buy Price, Qty and Buy Date information from the columns.""",
            result_type="markdown"
        )

        file_extractor = {".xlsx": parser}
        documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        st.info(f"Excel file parsed and loaded into vector store in {len(documents)} documents")
        return index
    return None


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string, db_username, db_password):
    """Connect to couchbase"""
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    from couchbase.options import ClusterOptions
    from datetime import timedelta

    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    connect_string = connection_string
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@st.cache_resource()
def get_vector_store(
    _cluster,
    db_bucket,
    db_scope,
    db_collection,
    index_name,
):
    """Return the Couchbase vector store."""
    return CouchbaseVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        index_name=index_name,
    )


if __name__ == "__main__":
    # Authorization
    if "auth" not in st.session_state:
        st.session_state.auth = False

    st.set_page_config(
        page_title="Chat with your Excel file using LlamaIndex, Couchbase & Bedrock",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
    else:
        # Authorization
        if "auth" not in st.session_state:
            st.session_state.auth = False

        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        # Authentication
        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")

    if st.session_state.auth:
        # Load environment variables
        DB_CONN_STR = os.getenv("DB_CONN_STR")
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_BUCKET = os.getenv("DB_BUCKET")
        DB_SCOPE = os.getenv("DB_SCOPE")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        INDEX_NAME = os.getenv("INDEX_NAME")

        # Ensure that all environment variables are set
        check_environment_variable("BEDROCK_API_KEY")
        check_environment_variable("DB_CONN_STR")
        check_environment_variable("DB_USERNAME")
        check_environment_variable("DB_PASSWORD")
        check_environment_variable("DB_BUCKET")
        check_environment_variable("DB_SCOPE")
        check_environment_variable("DB_COLLECTION")
        check_environment_variable("INDEX_NAME")
        check_environment_variable("LLAMA_CLOUD_API_KEY")


        # Connect to Couchbase Vector Store
        cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)

        vector_store = get_vector_store(
            cluster,
            DB_BUCKET,
            DB_SCOPE,
            DB_COLLECTION,
            INDEX_NAME,
        )

        # Build the prompt for the RAG
        template_rag = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        # Frontend
        couchbase_logo = (
            "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"
        )

        st.title("Chat with Excel file")
        st.markdown(
            "Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* "
        )

        # Use Bedrock as the llm & for embeddings
        llm = Bedrock(model="mistral.mistral-large-2402-v1:0", region_name="us-east-1")

        embeddings = BedrockEmbedding(model="amazon.titan-embed-text-v1")

        # Set the global settings for loading documents
        Settings.embed_model = embeddings
        Settings.chunk_size = 1500
        Settings.chunk_overlap = 150
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    

        with st.sidebar:
            st.header("Upload your Excel file")
            with st.form("upload excel file"):
                uploaded_file = st.file_uploader(
                    "Choose a Excel file.",
                    help="The document will be deleted after one hour of inactivity (TTL).",
                    type="xlsx",
                )
                submitted = st.form_submit_button("Upload")
                if submitted:
                    index = store_document(uploaded_file, storage_context)
                    if not index:
                        st.warning("Please upload a valid Excel file")

                    # Create the chat engine with context from the uploaded data
                    st.session_state.chat_engine_rag = index.as_chat_engine(
                        chat_mode="context",
                        llm=llm,
                        system_prompt=template_rag,
                    )

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you will get two answers:
                * one using RAG ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
                """
            )

            st.markdown(
                "For RAG, we are using [LlamaIndex](https://www.llamaindex.ai/), [Couchbase Vector Search](https://couchbase.com/) & [Bedrock](https://Bedrock.com/). We fetch parts of the Excel file relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store."
            )

            # View Code
            if st.checkbox("View Code"):
                st.write(
                    "View the code here: [Github](https://github.com/couchbase-examples/rag-demo-llama-index/blob/main/chat_with_excel.py)"
                )

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the excel file. How can I help you?",
                    "avatar": "🤖",
                }
            )
            st.session_state.chat_llm = None
            st.session_state.chat_engine_rag = None

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        # React to user input
        if question := st.chat_input("Ask a question based on the Excel file"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": "👤"}
            )

            # Add placeholder for streaming the response
            with st.chat_message("assistant", avatar=couchbase_logo):
                message_placeholder = st.empty()

            # stream the response from the RAG
            rag_response = ""
            rag_stream_response = st.session_state.chat_engine_rag.stream_chat(question)
            for chunk in rag_stream_response.response_gen:
                rag_response += chunk
                message_placeholder.markdown(rag_response + "▌")

            message_placeholder.markdown(rag_response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    "avatar": couchbase_logo,
                }
            )
