from pdf_loader import load_data_from_pickle, load_pdfs_from_directory, save_data_to_pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
import pymupdf4llm
import os
import pickle


# 특정 디렉토리와 하위 디렉토리의 모든 PDF 파일을 로드하여 마크다운으로 변환
def test_load_pdfs_from_directory(directory_path):
    documents = []

    # os.walk()를 사용해 디렉토리 내 모든 파일 탐색
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):  # PDF 파일만 처리
                file_path = os.path.join(root, file)  # 파일 경로 결합
                print(file_path)
                try:
                    # PDF 파일을 마크다운 형식으로 변환
                    markdown_text = pymupdf4llm.to_markdown(file_path)
                    documents.append(markdown_text)  # 변환된 텍스트 추가
                    print(f"Loaded markdown from {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return documents

test_pickle_file_path = "venv/data/processed/modern_documents.pkl"
directory_path = "venv/data/raw/현대사"
# documents = load_data_from_pickle(test_pickle_file_path) 
documents = []
print(documents)

    # 문서 로드
print(directory_path)
documents = test_load_pdfs_from_directory(directory_path)
save_data_to_pickle(documents, test_pickle_file_path)