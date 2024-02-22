from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


def get_prompts():

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    return DEFAULT_DOCUMENT_PROMPT, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT