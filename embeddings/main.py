import argparse

import streamlit as st
import os
import langchain.llms
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import functools
import tiktoken
from functools import cache
from itertools import accumulate, zip_longest, takewhile

EMBEDDING_MODEL = "text-embedding-ada-002"
TOKEN_LIMIT = 4096

@cache
def get_openai_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        key_file = os.path.join(os.path.expanduser("~"),
                                ".openai")
        api_key = open(key_file).read().strip()

    return api_key


def num_tokens(text, model):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_embeddings(docs_dir, cache_dir='./.cache'):
    input_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            input_files.append(os.path.join(root, file))

    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in input_files:
        with open(file) as f:
            texts.append(f.read())

    docs = text_splitter.create_documents(texts)
    embedding_fn = OpenAIEmbeddings(openai_api_key=get_openai_api_key(), model=EMBEDDING_MODEL)
    embeddings = Chroma.from_documents(docs, embedding_fn, persist_directory=cache_dir)

    return embeddings


def submit_query(llm, embeddings, customer_platform, customer_tier, features, phone, input_text):
    docs = embeddings.similarity_search(input_text)

    # lol tired functional programming
    docs_to_add = [d[0].page_content for d in takewhile(lambda x: x[1] < TOKEN_LIMIT,
                                           zip_longest(docs, accumulate(docs, lambda total, doc: total + num_tokens(doc.page_content, llm.model_name), initial=0))) if d[0]]
    docs_string = '\n\n'.join(docs_to_add)

    # TODO: Use templates and system messages within this.
    prompt = f"You are a helpful chat agent. You should prioritize accuracy. If you don't know the answer, " \
             f"you should respond with \"Please call us at {phone}\". You are chatting with a business owner who uses {customer_platform}, and should only respond with information related to {customer_platform}." \
             f"This customer is on the {customer_tier} tier, and has the following features enabled: {', '.join(features)}. Use the below articles" \
             f"to answer the following question: {docs_string}" \
             f"\nThis customer's question is:" \
             f"\n{input_text}"
    return llm(prompt)


def chat_message(role, text):
    st.chat_message(role).markdown(text)
    st.session_state.messages.append({'role': role, 'content': text})

def main(docs_dir, cache_dir):
    st.title("Mindbody CX Bot")
    embeddings = get_embeddings(docs_dir, cache_dir)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'role': 'assistant', 'content': "Welcome to Mindbody chat. I'm Happy Robot, here to answer your questions!"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    features = []
    customer_platform = st.sidebar.selectbox(
        'Platform',
        ('Mindbody', 'Booker'))

    if customer_platform == 'Mindbody':
        phone = '1-877-755-4279'
        customer_tier = st.sidebar.selectbox(
            'Customer Tier',
            ('Starter', 'Accelerate', 'Ultimate', 'Ultimate Plus')
        )

        nmb_status = st.sidebar.selectbox(
            'Type',
            ('Classic', 'New Mindbody Experience')
        )

        features = st.sidebar.multiselect(
            'What features are enabled?',
            (['New Check-in', 'Staff Identity', 'Consumer Identity'])
        )

        features.append(nmb_status)
    else:
        phone = '1-866-966-9798'
        customer_tier = 'v1'

    with st.sidebar.expander('Advanced'):
        model_scale = st.selectbox(
            'Model Scale',
            ('large', 'small')
        )

        match model_scale:
            case 'large':
                model = 'gpt-4'
            case 'small':
                model = 'gpt-3.5-turbo'

    llm = langchain.llms.OpenAI(temperature=0.7, model_name=model, openai_api_key=get_openai_api_key())

    if prompt := st.chat_input("What do you need help with?"):
        chat_message('user', prompt)

        with st.spinner():
            response = submit_query(llm, embeddings, customer_platform, customer_tier, features, phone, prompt)
        chat_message('assistant', response)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_dir')
    parser.add_argument('--cache-dir', default='./.cache')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()
    main(options.docs_dir, options.cache_dir)
