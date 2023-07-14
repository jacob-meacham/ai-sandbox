from functools import cache
import os
import csv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

EMBEDDING_MODEL = "text-embedding-ada-002"


@cache
def get_openai_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        key_file = os.path.join(os.path.expanduser("~"),
                                ".openai")
        api_key = open(key_file).read().strip()

    return api_key


def build_embeddings(embedding_fn, docs_dir, cache_dir):
    input_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            input_files.append(os.path.join(root, file))

    # Consider using a Markdown-aware Splitter instead
    texts = []
    metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in input_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                texts.append(row['content'])
                metadatas.append({k: v for k, v in row.items() if k != 'content'})

    docs = text_splitter.create_documents(texts, metadatas=metadatas)
    embedding_db = Chroma.from_documents(docs, embedding_fn, persist_directory=cache_dir)
    return embedding_db


# TODO: Probably separate out building the vector DB with loading it.
def get_embeddings(docs_dir, cache_dir, force_cache_rebuild=False):
    embedding_fn = OpenAIEmbeddings(openai_api_key=get_openai_api_key(), model=EMBEDDING_MODEL)
    embedding_db = Chroma(embedding_function=embedding_fn, persist_directory=cache_dir)
    if force_cache_rebuild or not embedding_db._collection.count():
        embedding_db = build_embeddings(embedding_fn, docs_dir, cache_dir)

    return embedding_db
