from functools import cache
import os

@cache
def get_openai_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key

    api_key_location = os.environ.get('OPENAI_API_KEY_LOCATION')
    if api_key_location:
        api_key = open(api_key_location).read().strip()
        if api_key:
            return api_key

    key_file = os.path.join(os.path.expanduser("~"),
                            ".openai")
    api_key = open(key_file).read().strip()

    return api_key

