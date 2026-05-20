import os
from operator import itemgetter
from textwrap import dedent

from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableParallel

load_dotenv()
OPENAI_KEY = os.environ.get('OPENAI_KEY')
YOUTUBE_USER_HANDLE = os.environ.get('YOUTUBE_USER_HANDLE')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'youtube-videos')


class Question(BaseModel):
    question: str
    chat_history: str = ""


def make_multiquery(model, retriever, input):
  MULTI_QUERY_PT_RAW = dedent("""
  You are a search engine expert.

  TASK: Generate 2 different versions of the given USER QUESTION and (optionally) CHAT HISTORY
  in order to retrive the most relevant documents from a (vector) database based on semantic similarity.

  RULES:
      - Generate the question from multiple different perspectives, focusing on unique angles
      about the important aspects of the question.
      - Provide these alternative  questions separated by newlines.

  QUESTION:
  {question}

  CHAT HISTORY (blank if none):
  {chat_history}
  """)

  pt_formatted = MULTI_QUERY_PT_RAW.format(
      question="{question}",
      chat_history=input.get('chat_history', 'N/A')
  )
  return MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    prompt=PromptTemplate.from_template(pt_formatted)
  )

def get_retrieval_chain(youtube_user_handle, openai_api_key):
    missing = [
        name for name, value in [
            ('OPENAI_KEY', openai_api_key),
            ('YOUTUBE_USER_HANDLE', youtube_user_handle),
            ('PINECONE_API_KEY', PINECONE_API_KEY),
            ('PINECONE_ENVIRONMENT', PINECONE_ENVIRONMENT),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Make sure these are set in your .env or environment before starting the app."
        )

    try:
        db = Pinecone.from_existing_index(
            PINECONE_INDEX_NAME,
            namespace=youtube_user_handle,
            embedding=OpenAIEmbeddings(
                model='text-embedding-3-small',
                openai_api_key=openai_api_key
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "Could not load the Pinecone index. Ensure your Pinecone project has an active "
            f"index named '{PINECONE_INDEX_NAME}', and that PINECONE_API_KEY/PINECONE_ENVIRONMENT "
            "are correct. Then run the indexing pipeline or create the index in the Pinecone dashboard."
        ) from exc

    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_KEY)

    multi_retriever = make_multiquery(model, retriever, {})

    # RAG prompt
    SYSTEM_PROMPT = dedent(
        """
        Answer the question concisely based only on the following transcript snippets from a youtube video.
        Do not mention the snippets directly. If you don't know the answer, simply say 'I don't know'.
        A chat history may be provided for additional context.

        CHAT HISTORY:
        {chat_history}


        SNIPPETS:
        {context}
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question:\n{question}")
        ]
    )

    # RAG
    chain = (
        {
            "context": itemgetter("question") | multi_retriever,
            "question": lambda input: input['question'],
            "chat_history": lambda input: input.get('chat_history', 'N/A')
        }
        | prompt
        | model
        | StrOutputParser()
    )

    chain = chain.with_types(input_type=Question)

    return chain


# Expose chain
chain = get_retrieval_chain(youtube_user_handle=YOUTUBE_USER_HANDLE, openai_api_key=OPENAI_KEY)
