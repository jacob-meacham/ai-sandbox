import argparse
import os

import streamlit as st
import langchain.llms
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from markdownify import markdownify

from embeddings.llm_with_embeddings import ModelWithEmbeddings
from embeddings.openai_utils import get_openai_api_key

EMBEDDING_MODEL = "text-embedding-ada-002"

# TODO: Move into a template file
PROMPT_TEMPLATE = "You are a helpful chat bot for engineers and you have knowledge of an internal document repository. You should prioritize accuracy. If you don't know the answer, " \
                  "you should respond with \"I'm not sure and don't want to speculate\". You are chatting with an engineer. Use only information from the internal knowledge articles" \
                  "to answer the question. {embedded_docs}" \
                  "\nThis engineer's question is:" \
                  "\n{input_text}"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def build_embeddings(embedding_fn, docs_dir, cache_dir):
    input_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            input_files.append(os.path.join(root, file))

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ('###', "Header 3")
    ]


    docs = []
    # TODO: This had problems
    # text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    # for file in input_files:
    #     with open(file, 'r') as f:
    #         text = markdownify(f.read())
    #         docs.extend(text_splitter.split_text(text))

    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in input_files:
        with open(file, 'r') as f:
            texts.append(markdownify(f.read()))

    docs = text_splitter.create_documents(texts)

    print(f'Created {len(docs)} sections to embed')

    embedding_db = Chroma(embedding_function=embedding_fn, persist_directory=cache_dir)

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


def get_embeddings(docs_dir, cache_dir, force_cache_rebuild=False):
    embedding_fn = OpenAIEmbeddings(openai_api_key=get_openai_api_key(), model=EMBEDDING_MODEL)
    embedding_db = Chroma(embedding_function=embedding_fn, persist_directory=cache_dir)
    if force_cache_rebuild or not embedding_db._collection.count():
        embedding_db = build_embeddings(embedding_fn, docs_dir, cache_dir)

    return embedding_db


def chat_message(role, text):
    st.chat_message(role).markdown(text)
    st.session_state.messages.append({'role': role, 'content': text})


def st_main(docs_dir, cache_dir):
    st.title("ClassPass Engineering Knowledge Bot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'role': 'assistant',
                                      'content': "Welcome to ClassPass Engineering chat. What can I help you with?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar.expander('Advanced'):
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
    embeddings_db = get_embeddings(docs_dir, cache_dir)

    model = ModelWithEmbeddings(llm, embeddings_db, PROMPT_TEMPLATE)

    if prompt := st.chat_input("What can I answer for you?"):
        chat_message('user', prompt)

        with st.spinner():
            response = model.submit_query(prompt, {})
        chat_message('assistant', response)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_dir')
    parser.add_argument('--cache-dir', default='./.classpass_cache')
    parser.add_argument('--force-cache-rebuild', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()

    st_main(options.docs_dir, options.cache_dir)
