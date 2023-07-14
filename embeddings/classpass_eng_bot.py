import argparse

import streamlit as st
import langchain.llms

from embeddings.llm_with_embeddings import ModelWithEmbeddings
from embeddings.openai import get_embeddings, get_openai_api_key

# TODO: Move into a template file
PROMPT_TEMPLATE = "You are a helpful chat bot for engineers and you have knowledge of an internal document repository. You should prioritize accuracy. If you don't know the answer, " \
                  "you should respond with \"I'm not sure and don't want to speculate\". You are chatting with an engineer. Use the below articles" \
                  "to answer the question. {embedded_docs}" \
                  "\nThis engineer's question is:" \
                  "\n{input_text}"


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

        temperature = st.slider(min_value=0.0, max_value=1.0)

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
