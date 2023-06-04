from langchain.prompts import PromptTemplate

condense_question_template = """You will be provided a Chat History and a Follow-Up Question. Review both of these.
If the Chat History is irrelevant to the Follow-Up Question, ignore the Chat History, and just return the Follow-Up Question as the Standalone Question.
If the Chat History is relevant, combine both the Chat History and Follow-Up Question into a single condensed Standalone Question.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

qa_template = """You are a helpful assistant named Seamie, named after the Seamless CI/CD product. Your job is to answer questions about Seamless CI/CD.
You will be provided a Question and associated Context. Review both of these.
If the question is irrelevant to software engineering, Seamless CI/CD, or you, say that you are unable to answer the question.
If the question is relevant, but you don't know the answer, say that you don't know the answer.
If you are able to answer to the question, do so using precise language and a helpful tone.

Question: {question}

Context:
```
{context}
```

Answer:"""

QA_PROMPT = PromptTemplate(
    template=qa_template, input_variables=["question", "context"]
)
