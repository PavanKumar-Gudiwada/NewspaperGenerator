# Assuming this is in the same file as your Gradio app or imported:
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
)
from langchain_core.documents import Document

def load_user_files_to_documents(files_to_load: list) -> list[Document]:
    """
    Takes a list of file objects/paths from the frontend and loads them
    into LangChain Document objects using the appropriate loaders.
    """
    docs = []
    
    # Process each file object provided by Gradio
    for file_obj in files_to_load:
        file_path = file_obj.name # Gradio file objects have a .name attribute for the path
        
        print(f"Loading file: {file_path}")
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            loader = UnstructuredImageLoader(file_path, mode="elements")
        else:
            print(f"Skipping unsupported file type: {os.path.basename(file_path)}")
            continue

        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return docs