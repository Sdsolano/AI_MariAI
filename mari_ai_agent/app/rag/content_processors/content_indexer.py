import os
import pandas as pd
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, TextLoader,
    UnstructuredFileLoader, CSVLoader
)
from langchain.schema import Document


def excel_loader(filepath):
    documents = []
    xls = pd.ExcelFile(filepath)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        text = df.to_string(index=False)
        metadata = {"source": os.path.basename(filepath), "sheet_name": sheet}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def get_loader(filepath):
    ext = filepath.lower()
    if ext.endswith(".pdf"):
        return PyMuPDFLoader(filepath)
    elif ext.endswith(".docx"):
        return UnstructuredWordDocumentLoader(filepath)
    elif ext.endswith(".pptx"):
        return UnstructuredPowerPointLoader(filepath)
    elif ext.endswith(".txt"):
        return TextLoader(filepath, encoding="utf-8")
    elif ext.endswith(".csv"):
        return CSVLoader(filepath)
    elif ext.endswith((".xlsx", ".xlsm")):
        return "custom_excel_loader"
    else:
        return UnstructuredFileLoader(filepath)




