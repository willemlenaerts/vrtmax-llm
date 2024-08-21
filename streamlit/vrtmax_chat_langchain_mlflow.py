import sagemaker
import boto3
import subprocess
from os import environ
import streamlit as st
from typing import List
import pandas as pd

# Langchain imports
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


st.set_page_config(layout="wide")

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

model_options = [
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0"
]


def update_llm():
    llm = ChatBedrockConverse(
        region_name="eu-west-2",
        model=st.session_state.model,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        max_tokens=st.session_state.max_tokens
    )
    return llm   

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
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    embeddings = CustomEmbeddings(model_name=model_name)

    # Init OpenSearch client connection
    st.session_state.temperature = 0.6
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
    st.session_state.model = "meta.llama3-70b-instruct-v1:0"
    st.session_state.temperature = 0.6
    st.session_state.top_p = 0.4
    st.session_state.max_tokens = 512
    st.session_state.llm = update_llm()

col1, col2 = st.columns(2)

with col1:
    # Model version
    st.session_state.model = st.selectbox(label="Selecteer model",index=model_options.index(st.session_state.model),options=model_options,on_change=update_llm)

    # Model settings
    st.session_state.temperature = st.slider(label="Temperature", min_value=0.0,max_value=1.0,value=st.session_state.temperature,on_change=update_llm)
    st.session_state.top_p = st.slider(label="Top p", min_value=0.0,max_value=1.0,value=st.session_state.top_p,on_change=update_llm)
    st.session_state.max_tokens = st.slider(label="Max tokens", min_value=0,max_value=4096,value=st.session_state.max_tokens,on_change=update_llm)

    # Generate prompt
    # SYSTEM
    system_prompt_start = """Je bent een hulpvaardige assistent die Nederlands spreekt.
Je krijgt vragen over de catalogus van episodes van het videoplatform VRT MAX. 
Je probeert mensen te helpen om episodes aan te bevelen die aansluiten bij hun vraag.
Daarvoor krijg je een beschrijving van een aantal episodes.
Aan jou om wat relevant is aan te bevelen.
Begin niet met het geven van jouw mening over de episodes. Begin direct met het aanbevelen van relevante content aan de gebruiker.
Leg ook telkens uit waarom je een episode aanbeveelt.
Geef voor alle episodes die je aanbeveelt ook de URL mee als referentie."""

    system_prompt = st.text_area("Systeem prompt",system_prompt_start,height=200)

    # HUMAN
    message_start = """Beantwoordt de vraag op basis van de volgende context:

{context}

Vraag: {question}"""

    message = st.text_area("Vraag met context",message_start, height=200)

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
                " De episode van dit programma heeft als beschrijving: " + d.metadata["mediacontent_page_description"] +\
                " De episode zal online staan tot " + d.metadata["offering_publication_planneduntil"] +\
                " De episode heeft als URL " + d.metadata["mediacontent_pageurl"] +\
                " De episode heeft als foto " + "https:" +d.metadata["mediacontent_imageurl"] 
        )
        
    return "\n\n".join(context)

# chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | st.session_state.llm
#     | StrOutputParser()
# )

from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | st.session_state.llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     lambda session_id: msgs,
#     input_messages_key="question",
#     history_messages_key="history",
# )


with col2:
    prompt = st.chat_input()
    if prompt:
        st.chat_message("human").write(prompt)
        config = {"configurable": {"session_id": "special_key"}}
        response = rag_chain_with_source.invoke({"question": prompt}, config)
        print(len(response["context"]))
        st.chat_message("ai").write(response["answer"])

        context = response["context"]
        metadata = []
        for item in response["context"]:
            metadata.append(item.metadata)
        df = pd.DataFrame.from_records(metadata)
        st.dataframe(df)
