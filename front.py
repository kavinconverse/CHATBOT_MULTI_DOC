from backend import *
from dotenv import load_dotenv
def main():
    load_dotenv()

    # Initialize session state
    initialize_session_state()

    st.set_page_config(page_title='chatbot interaction for pdf,txt.doc',layout='centered',page_icon=':books:')
    st.title("Multi-Documents ChatBot using LLaMA2 :books:")
    st.markdown('<style>h1{color: green; text-align: center;}</style>', unsafe_allow_html=True)

    # Initialize Streamlit
    st.sidebar.title('Multiple Document Uploader')
    uploaded_files = st.sidebar.file_uploader("Upload your files", accept_multiple_files=True)

    if uploaded_files:
        st.spinner("Document Processing")
        vector_store = build_vector_store(uploaded_files)
        st.sidebar.success('Vector Store Created')
        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)


if __name__ == "__main__":
    main()