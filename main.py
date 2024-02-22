
import dotenv
import argparse

from db import create_vectordb
from prompts import get_prompts

from operator import itemgetter

from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.schema import format_document


dotenv.load_dotenv()

db_retriever = create_vectordb()

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

DEFAULT_DOCUMENT_PROMPT, CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT = get_prompts()

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | db_retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

# Now we can run the chain
def run_chain(question):
    inputs = {"question": question}
    result = final_chain.invoke(inputs)
    return result

# use argparse to get question from terminal
parser = argparse.ArgumentParser(description='Get question')
parser.add_argument('--question', type=str, default="What is the legal age for working?", help='The question to ask')

if __name__ == "__main__":
    args = parser.parse_args()
    print(run_chain(args.question))



