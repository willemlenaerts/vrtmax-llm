import sagemaker
import boto3
import json
import time
import subprocess
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from os import environ
import streamlit as st
import subprocess
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

# Reference: https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history
st.title("📖 VRT Max Chat")

# Fix for issue in huggingface embedding, see https://github.com/langchain-ai/langchain/discussions/17773
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()
    
# Load AWS Profile/credentials from .env
if "sagemaker_session" not in st.session_state:
    # Login AWS
    aws_profile = "vrt-analytics-engineer-nonsensitive"
    envvars = subprocess.check_output(['aws-vault', 'exec', aws_profile, '--', 'env'])
    for envline in envvars.split(b'\n'):
        line = envline.decode('utf8')
        eqpos = line.find('=')
        if eqpos < 4:
            continue
        k = line[0:eqpos]
        v = line[eqpos+1:]
        if k == 'AWS_ACCESS_KEY_ID':
            aws_access_key_id = v
        if k == 'AWS_SECRET_ACCESS_KEY':
            aws_secret_access_key = v
        if k == 'AWS_SESSION_TOKEN':
            aws_session_token = v

    environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    environ["AWS_SESSION_TOKEN"] = aws_session_token

    session = boto3.Session(
    aws_access_key_id, aws_secret_access_key, aws_session_token, region_name="eu-west-1"
    )   
    credentials = session.get_credentials() 
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
    auth = AWSV4SignerAuth(credentials, 'eu-west-2', 'aoss')

    # Retriever
    model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    # encode_kwargs = {'normalize_embeddings': False}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     encode_kwargs=encode_kwargs
    # )

    embeddings = CustomEmbeddings(model_name=model_name)

    # Init OpenSearch client connection
    docsearch = OpenSearchVectorSearch(
        index_name="aoss-index",  # TODO: use the same index-name used in the ingestion script
        embedding_function=embeddings,
        opensearch_url="https://epcavlvwitam2ivpwv4k.eu-west-2.aoss.amazonaws.com:443",  # TODO: e.g. use the AWS OpenSearch domain instantiated previously
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
    )

    retriever = docsearch.as_retriever(search_kwargs={'k': 5,"vector_field":"vrtmax_catalog_vector"})

    # LLM endpoint
    llm = ChatBedrockConverse(
        region_name="eu-west-2",
        model="meta.llama3-70b-instruct-v1:0",
        temperature=0.6,
        top_p=0.6,
        max_tokens=512
    )

msgs = StreamlitChatMessageHistory(key='langchain_messages')
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Generate prompt
# SYSTEM
system_prompt = """
Je bent een hulpvaardige assistent die Nederlands spreekt.
Je krijgt vragen over de catalogus van programma's van het videoplatform VRT MAX. 
Je probeert mensen te helpen om programma's aan te bevelen die aansluiten bij hun vraag.
Daarvoor krijg je een beschrijving van een aantal programma's.
Aan jou om wat relevant is aan te bevelen.
Begin niet met het geven van jouw mening over de programma's. Begin direct met het aanbevelen van relevante content aan de gebruiker.
"""

# HUMAN
message = """Beantwoordt de vraag op basis van de volgende context:

{context}

Vraag: {question}
"""
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", message),
#     ]
# )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", message),
    ]
)

def format_docs(docs):
    context = []
    for d in docs:
        program_description = ""
        if d.metadata["mediacontent_page_description_program"]  != "":
            program_description = docs[0].metadata["mediacontent_page_description_program"]
        elif d.metadata["mediacontent_page_editorialtitle_program"]  != "":
            program_description = docs[0].metadata["mediacontent_page_editorialtitle_program"]

        context.append("Het programma met de naam " + d.metadata["mediacontent_pagetitle_program"] +\
                " heeft volgende beschrijving: " + program_description  +\
                " De episode van dit programma heeft als beschrijving: " + d.metadata["mediacontent_page_description"])
        
    return "\n\n".join(context)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     lambda session_id: msgs,
#     input_messages_key="question",
#     history_messages_key="history",
# )

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "special_key"}}
    response = chain.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response)


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)