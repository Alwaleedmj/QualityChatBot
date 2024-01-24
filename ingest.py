from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma 
import os 
from constants import CHROMA_SETTINGS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR

persist_directory = "db"

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(text)
    return texts



def main():
    load_dotenv()
    loader = DirectoryLoader('files/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_chunks = get_text_chunks(documents)
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
    embedding = instructor_embeddings
    db = Chroma.from_documents(text_chunks, embedding, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None




if __name__ == "__main__":
    main()