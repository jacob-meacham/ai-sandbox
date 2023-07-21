import tiktoken
from itertools import accumulate, zip_longest, takewhile
TOKEN_LIMIT = 1000

def num_tokens(text, model):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# TODO: Probably gets changed again when we set this up as a ChatModel with history
class ModelWithEmbeddings:
    def __init__(self, llm, embedding_db, prompt_template):
        self.llm = llm
        self.embedding_db = embedding_db
        self.prompt_template = prompt_template

    def submit_query(self, input_text, context, metadata_filter={}):
        docs = self.embedding_db.similarity_search(input_text, filter=metadata_filter)

        # lol tired functional programming
        docs_to_add = [d[0].page_content for d in takewhile(lambda x: x[1] < TOKEN_LIMIT,
                                                            zip_longest(docs, accumulate(docs, lambda total,
                                                                                                      doc: total + num_tokens(
                                                                doc.page_content, self.llm.model_name), initial=0))) if d[0]]
        embedded_docs = '\n\nHelp Article:\n'.join(docs_to_add)

        # TODO: Use templates and system messages within this.
        t = {
            'input_text': input_text,
            'embedded_docs': embedded_docs,
            **context
        }
        prompt = self.prompt_template.format(**{
            'input_text': input_text,
            'embedded_docs': embedded_docs,
            **context
        })

        return self.llm(prompt), prompt
