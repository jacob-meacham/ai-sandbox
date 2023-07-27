import argparse
import os
import urllib

import chromadb
import streamlit as st
import langchain.llms
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from markdownify import markdownify

from embeddings.llm_with_embeddings import ModelWithEmbeddings
from embeddings.openai_utils import get_openai_api_key

EMBEDDING_MODEL = "text-embedding-ada-002"

# TODO: Move into a template file
PROMPT_TEMPLATE = "You are a helpful chat bot for engineers and you have knowledge of an internal document repository. You should prioritize accuracy. If you don't know the answer, " \
                  "you should respond with a link to the most relevant article. You are chatting with an engineer. Use only information from the following internal knowledge articles " \
                  "to answer the question. Articles:\n\n" \
                  "{embedded_docs}" \
                  "\nThis engineer's question is:" \
                  "\n{input_text}"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_embeddings(embedding_db, docs_dir):
    input_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if '.md' not in file:
                continue
            input_files.append(os.path.join(root, file))

    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in input_files:
        with open(file, 'r') as f:
            texts.append(markdownify(f.read()))

    docs = text_splitter.create_documents(texts)

    print(f'Created {len(docs)} sections to embed')

    # TODO: Having some stability issues with the OpenAI API
    for doc_chunk in chunks(docs, 500):
        print(f'Adding {len(doc_chunk)} docs!')
        retries = 0
        while retries < 3:
            try:
                embedding_db.add_documents(doc_chunk)
                retries = 10
            except:
                retries += 1

    embedding_db.persist()
    return embedding_db


def get_embeddings_db(docs_dir, chroma_db_path, force_cache_rebuild=False):
    embedding_fn = OpenAIEmbeddings(openai_api_key=get_openai_api_key(), model=EMBEDDING_MODEL)
    if 'http' in chroma_db_path:
        parsed_url = urllib.parse.urlparse(chroma_db_path)
        client = chromadb.HttpClient(host=parsed_url.hostname, port=parsed_url.port)

        embedding_db = Chroma(client=client, embedding_function=embedding_fn)
    else:
        embedding_db = Chroma(embedding_function=embedding_fn, persist_directory=chroma_db_path)

    if force_cache_rebuild or not embedding_db._collection.count():
        embedding_db = build_embeddings(embedding_db, docs_dir)

    return embedding_db


def chat_message(role, text):
    st.chat_message(role).markdown(text)
    st.session_state.messages.append({'role': role, 'content': text})


def st_main(docs_dir, chroma_db_path, force_cache_rebuild=False):
    st.set_page_config(
        page_title="Mindbody Engineering Oracle",
        page_icon="🤖",
    )

    st.title("Mindbody Engineering Oracle")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'role': 'assistant',
                                          'content': "I am the Mindbody Engineering Oracle. What can I help you with?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar.expander('Advanced'):
        debug = st.checkbox(
            'Debug'
        )

        model_scale = st.selectbox(
            'Model Scale',
            ('small', 'large')
        )

        match model_scale:
            case 'large':
                model = 'gpt-4'
            case 'small':
                model = 'gpt-3.5-turbo'

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    llm = langchain.llms.OpenAI(temperature=temperature, model_name=model, openai_api_key=get_openai_api_key())
    embeddings_db = get_embeddings_db(docs_dir, chroma_db_path, force_cache_rebuild)

    model = ModelWithEmbeddings(llm, embeddings_db, PROMPT_TEMPLATE)

    if prompt := st.chat_input("What can I answer for you?"):
        chat_message('user', prompt)

        with st.spinner():
            response, final_prompt = model.submit_query(prompt, {})
        if debug:
            chat_message('system', final_prompt)
        chat_message('assistant', response)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-dir')
    parser.add_argument('--chroma-db', default='./.mb_chroma', help='Either local directory or server address')
    parser.add_argument('--force-cache-rebuild', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()

    st_main(options.docs_dir, chroma_db_path=options.chroma_db, force_cache_rebuild=options.force_cache_rebuild)
