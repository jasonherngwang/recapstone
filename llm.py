import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

load_dotenv()

OPENAI_LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
TOKENIZER_ENCODING = "cl100k_base"
PINECONE_INDEX_NAME = "recapstone"

embeddings = OpenAIEmbeddings(client="", openai_api_key=os.environ["OPENAI_API_KEY"])

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
)

vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME, embedding=embeddings
)

llm = ChatOpenAI(
    client="",
    temperature=LLM_TEMPERATURE,
    model=OPENAI_LLM_MODEL,
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    verbose=False,
)


def process_chat_history(chat_history, last_k=3):
    # Ignore introductory AI message(s)
    first_human_index = next(
        (
            index
            for index, message in enumerate(chat_history)
            if message.sender == "human"
        ),
        None,
    )
    if first_human_index is None:
        return []

    chat_history = chat_history[first_human_index:]

    history = []
    current_question = ""

    for message in chat_history:
        # If there are successive human messages, only use the last
        if message.sender == "human":
            current_question = message.message
        elif message.sender == "ai":
            # AI messages must be paired with a human message
            if current_question:
                history.append((current_question, message.message))
            current_question = ""

    # Consider only the last K chats, to reduce token amount
    return history[-last_k:]


def answer_question(query, history):
    chat_history = process_chat_history(history)
    response = chain({"question": query, "chat_history": chat_history})
    return {
        "answer": response["answer"],
    }
