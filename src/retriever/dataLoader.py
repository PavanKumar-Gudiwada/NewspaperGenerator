import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
)
from langchain_core.documents import Document


def load_all_data(data_folder="data"):
    """
    Load and combine documents from structured (CSV, Excel)
    and unstructured (PDF, images) files using LangChain loaders.

    Returns:
        List[Document]: List of LangChain Document objects with metadata.
    """
    structured_folder = os.path.join(data_folder, "structured")
    unstructured_folder = os.path.join(data_folder, "un_structured")

    docs = []

    # --- Load structured data ---
    for file in os.listdir(structured_folder):
        file_path = os.path.join(structured_folder, file)
        if file.endswith(".csv"):
            loader = CSVLoader(file_path)
            docs.extend(loader.load())
        elif file.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            docs.extend(loader.load())

    # --- Load unstructured data ---
    for file in os.listdir(unstructured_folder):
        file_path = os.path.join(unstructured_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif file.endswith((".png", ".jpeg", ".jpg")):
            loader = UnstructuredImageLoader(file_path)
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents.")
    return docs
