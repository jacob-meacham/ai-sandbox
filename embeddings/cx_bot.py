import argparse

import streamlit as st
import langchain.llms

from embeddings.llm_with_embeddings import ModelWithEmbeddings
from embeddings.openai import get_embeddings, get_openai_api_key

# TODO: Move into a template file
PROMPT_TEMPLATE = "You are a helpful chat agent. You should prioritize accuracy. If you don't know the answer, " \
                  "you should respond with \"Please call us at {phone}\". You are chatting with a business owner who uses {customer_platform}, and should only respond with information related to {customer_platform}." \
                  "This customer is on the {customer_tier} tier, and has the following features enabled: {features}. Use the below articles" \
                  "to answer the question. {embedded_docs}" \
                  "\nThis customer's question is:" \
                  "\n{input_text}"


def chat_message(role, text):
    st.chat_message(role).markdown(text)
    st.session_state.messages.append({'role': role, 'content': text})


def st_main(docs_dir, cache_dir):
    st.title("Mindbody CX Bot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'role': 'assistant',
                                          'content': "Welcome to Mindbody chat. I'm Happy Robot, here to answer your questions!"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    features = []
    customer_platform = st.sidebar.selectbox(
        'Platform',
        ('MINDBODY', 'Booker'))

    if customer_platform == 'MINDBODY':
        phone = '1-877-755-4279'
        customer_tier = st.sidebar.selectbox(
            'Customer Tier',
            ('Starter', 'Accelerate', 'Ultimate', 'Ultimate Plus')
        )

        nmb_status = st.sidebar.selectbox(
            'Type',
            ('this customer uses Classic', 'this customer uses the New Mindbody Experience')
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
            ('small', 'large')
        )

        match model_scale:
            case 'large':
                model = 'gpt-4'
            case 'small':
                model = 'gpt-3.5-turbo'

    llm = langchain.llms.OpenAI(temperature=0.7, model_name=model, openai_api_key=get_openai_api_key())
    embeddings_db = get_embeddings(docs_dir, cache_dir)

    model = ModelWithEmbeddings(llm, embeddings_db, PROMPT_TEMPLATE)

    if prompt := st.chat_input("What do you need help with?"):
        chat_message('user', prompt)

        with st.spinner():
            response = model.submit_query(prompt, {
                'customer_platform': customer_platform,
                'phone': phone,
                'customer_tier': customer_tier,
                'features': ', '.join(features)
            }, {'product': customer_platform})
        chat_message('assistant', response)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_dir')
    parser.add_argument('--cache-dir', default='./.cx_cache')
    parser.add_argument('--force-cache-rebuild', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()

    st_main(options.docs_dir, options.cache_dir)
