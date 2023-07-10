import streamlit as st
import pandas as pd
import os

def get_openai_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        key_file = os.path.join(os.path.expanduser("~"),
                                ".openai")
        api_key = open(key_file).read().strip()

    return api_key


df = pd.DataFrame({
      'first column': [1, 2, 3, 4],
      'second column': [10, 20, 30, 40]
})
df
