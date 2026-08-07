import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Helps identify your application when WebBaseLoader requests the website.
os.environ.setdefault(
    "USER_AGENT",
    "LangChainCourseProject/1.0"
)


# ---------------------------------------------------------
# 1. Load the webpage
# ---------------------------------------------------------

url = "https://365datascience.com/upcoming-courses"

loader = WebBaseLoader(url)
raw_documents = loader.load()


# ---------------------------------------------------------
# 2. Divide the webpage into smaller chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1_000,
    chunk_overlap=200,
)

documents = text_splitter.split_documents(raw_documents)


# ---------------------------------------------------------
# 3. Create embeddings and store them in FAISS
# ---------------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ---------------------------------------------------------
# 4. Initialize the language model
# ---------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


# ---------------------------------------------------------
# 5. Create a prompt that converts follow-up questions
#    into standalone search queries
# ---------------------------------------------------------

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the conversation history and the user's latest question,
rewrite the latest question as a standalone search query.

Do not answer the question. Only return the rewritten question.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------
# 6. Create the final question-answering prompt
# ---------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the user's question using only the supplied context.

If the answer is not present in the context, say that you could
not find the answer on the loaded webpage.

Context:
{context}
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------
# 7. Store the conversation history
# ---------------------------------------------------------

chat_history = []


def ask_question(question: str) -> str:
    """Answer a question using the webpage and conversation history."""

    # Turn a possible follow-up question into a standalone query.
    contextualize_messages = contextualize_prompt.format_messages(
        chat_history=chat_history,
        question=question,
    )

    standalone_response = llm.invoke(contextualize_messages)
    standalone_question = standalone_response.content

    # Retrieve webpage chunks related to the standalone question.
    relevant_documents = retriever.invoke(standalone_question)

    context = "\n\n".join(
        document.page_content
        for document in relevant_documents
    )

    # Generate the final answer.
    answer_messages = answer_prompt.format_messages(
        chat_history=chat_history,
        question=question,
        context=context,
    )

    response = llm.invoke(answer_messages)
    answer = response.content

    # Save this exchange for the next question.
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )

    return answer


# ---------------------------------------------------------
# 8. Ask questions about the webpage
# ---------------------------------------------------------

query = "What is the next course to be uploaded?"

answer = ask_question(query)
print(answer)

follow_up = ask_question("Who is teaching it?")
print(follow_up)