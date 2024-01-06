from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username='gilbut76', floating=True, width=221)

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

# 제목
st.title('ChatPDF')
st.write('---')

# 파일업로드
uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')
st.write('---')

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

#Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header('PDF에게 질문 해보세요.')
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({'query':question})
            st.write(result['result'])