from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader



def loadingTheFiles():
    loader = DirectoryLoader('files/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(documents)
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    chunks = loadingTheFiles()
    embeddings = OpenAIEmbeddings()
    st.write(embeddings)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    docs = knowledge_base.similarity_search(user_question)
    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    
    
#     llm = OpenAI()
#     chain = load_qa_chain(llm, chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=user_question)
#         print(cb)
        
#     st.write(response)
    

if __name__ == '__main__':
    main()