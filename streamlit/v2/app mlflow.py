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
from fuzzywuzzy import fuzz
import time 
import datetime
from dateutil import tz

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
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0"
]

def fuzzy_match(boolean_filter,user_input, type):
    if len(user_input) < 5: # filter out short strings that are part of a program name by accident
        return boolean_filter
    if type == "progr am":
        df = st.session_state["programmas"]
        metadata_filter = "metadata.mediacontent_pagetitle_program"
    elif type == "person":
        df = st.session_state["personen"]
        metadata_filter = "metadata.mediacontent_episode_castlist"

    output = []
    for naam in list(df.naam):
        try:
            score = fuzz.partial_ratio(naam.lower(),user_input.lower())
        except:
            score = 0
        output.append((naam,score))


    output = sorted(
        output, 
        key=lambda x: x[1],
        reverse=True
    )[0]

    if output[1] > 95:
        match = output[0]
        boolean_filter["bool"]["must"].append({"match_phrase" : {metadata_filter:match}})
    
    return boolean_filter

def update_llm():
    print(st.session_state.temperature_slider)
    llm = ChatBedrockConverse(
        region_name="eu-west-2",
        model=st.session_state.model,
        temperature=st.session_state.temperature_slider,
        top_p=st.session_state.top_p_slider,
        max_tokens=st.session_state.max_tokens_slider
    )
    return llm   

# Initial load OR reload if aws credentials expired
if "model" not in st.session_state or datetime.datetime.now(datetime.timezone.utc) > datetime.datetime.strptime(environ["AWS_CREDENTIAL_EXPIRATION"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=tz.gettz("UTC")):
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
        if k == 'AWS_CREDENTIAL_EXPIRATION':
            aws_credential_expiration = v

    environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    environ["AWS_SESSION_TOKEN"] = aws_session_token
    environ["AWS_CREDENTIAL_EXPIRATION"] = aws_credential_expiration

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

    st.session_state["opensearch"] = OpenSearchVectorSearch(
    index_name="vrtmax-catalog-index",  # TODO: use the same index-name used in the ingestion script
    embedding_function=embeddings,
    opensearch_url="https://epcavlvwitam2ivpwv4k.eu-west-2.aoss.amazonaws.com:443",  # TODO: e.g. use the AWS OpenSearch domain instantiated previously
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    is_appx_search=False
)
    
    # programs en personen
    st.session_state["programmas"] = pd.read_csv("programmas.csv")
    st.session_state["personen"] = pd.read_csv("personen.csv")

    # LLM endpoint
    print("allo")
    st.session_state.model = "mistral.mistral-large-2402-v1:0"
    st.session_state.temperature_start_value = 0.6
    st.session_state.top_p_start_value = 0.4
    st.session_state.max_tokens_start_value = 512
    st.session_state.llm = ChatBedrockConverse(
        region_name="eu-west-2",
        model=st.session_state.model,
        temperature=st.session_state.temperature_start_value,
        top_p=st.session_state.top_p_start_value,
        max_tokens=st.session_state.max_tokens_start_value
    )

if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = """Je bent een hulpvaardige assistent die Nederlands spreekt.
    Je krijgt vragen over de catalogus van episodes van het videoplatform VRT MAX. 
    Je probeert mensen te helpen om episodes aan te bevelen die aansluiten bij hun vraag.
    Daarvoor krijg je een beschrijving van een aantal episodes én eventueel ook een stukje van de ondertitels van de episode.
    Aan jou om wat relevant is aan te bevelen. Als je niets vindt dat voldoet aan de vraag van de gebruiker dan mag je dat ook zeggen.
    Begin niet met het geven van jouw mening over de episodes. Begin direct met het aanbevelen van relevante content aan de gebruiker.
    Leg ook telkens uit waarom je een episode aanbeveelt.
    Geef voor alle episodes die je aanbeveelt ook de URL mee als referentie."""

col1, col2 = st.columns(2)

with col1:
    # Model version
    st.session_state.model = st.selectbox(label="Selecteer model",index=model_options.index(st.session_state.model),options=model_options,on_change=update_llm)

    # Model settings
    st.slider(label="Temperature",key="temperature_slider", min_value=0.0,max_value=1.0,value=st.session_state.temperature_start_value,on_change=update_llm)
    st.slider(label="Top p",key="top_p_slider", min_value=0.0,max_value=1.0,value=st.session_state.top_p_start_value,on_change=update_llm)
    st.slider(label="Max tokens",key="max_tokens_slider", min_value=0,max_value=4096,value=st.session_state.max_tokens_start_value,on_change=update_llm)

    # Generate prompt
    # SYSTEM
    with st.form("system_prompt_form"): 
        st.write("System prompt")
        system_prompt = st.text_area("System prompt",st.session_state["system_prompt"],height=200, label_visibility="collapsed")

        # Every form must have a submit button.
        submitted_system_prompt = st.form_submit_button("Submit")
        if submitted_system_prompt:
            st.session_state["system_prompt"] = system_prompt


def format_docs(docs):
    context = []
    for d in docs:
        program_description = ""
        if d.metadata["mediacontent_page_description_program"]  != "":
            program_description = docs[0].metadata["mediacontent_page_description_program"]
        elif d.metadata["mediacontent_page_editorialtitle_program"]  != "":
            program_description = docs[0].metadata["mediacontent_page_editorialtitle_program"]

        context.append("Het programma met de naam " + d.metadata["mediacontent_pagetitle_program"] +\
                " heeft volgende beschrijving: " + program_description + "." +\
                " De episode van dit programma heeft als naam: " + d.metadata["mediacontent_pagetitle"] + "."  +\
                " De episode van dit programma heeft als beschrijving: " + d.metadata["mediacontent_page_description"] + "."  +\
                " De episode zal online staan tot " + d.metadata["offering_publication_planneduntil"] + "."  +\
                " De episode heeft als URL " + d.metadata["mediacontent_pageurl"] + "."  +\
                " De episode heeft als foto " + "https:" +d.metadata["mediacontent_imageurl"] + "."  
        )
        
    return "\n\n".join(context)

with col2:
    prompt = st.chat_input()
    if prompt:
        start = time.time()
        st.chat_message("human").write(prompt)
        stop = time.time()
        print("time: ", str(stop-start))

        # Fuzzy matching
        # Initialize filter dict
        start = time.time()
        boolean_filter = {"bool":{"must":[]}}

        boolean_filter = fuzzy_match(boolean_filter, prompt, "person")
        boolean_filter = fuzzy_match(boolean_filter, prompt, "program")
        stop = time.time()
        print("fuzzy matching: ", str(stop-start)) 

        # Docsearch (ideally in langchain, can't get it to work)
        start = time.time()
        docs = st.session_state["opensearch"].similarity_search(
            prompt,
            k=5,
            vector_field="vrtmax_catalog_vector",
            search_type="script_scoring",
            pre_filter=boolean_filter
        )
        stop = time.time()
        print("opensearch query: ", str(stop-start)) 

        context = format_docs(docs)

        message = """Beantwoordt de vraag op basis van de volgende context:

        """ + context + """

        Vraag: {question}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state["system_prompt"]),
                ("human", message),
            ]
        )

        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
        )

        start = time.time()
        response = chain.stream(prompt)
        stop = time.time()
        print("llm invoke: ", str(stop-start)) 
        print(response)
        st.chat_message("ai").write_stream(response)

        metadata = []
        for item in docs:
            metadata.append(item.metadata)
        df = pd.DataFrame.from_records(metadata)

        st.dataframe(df)

        st.write(boolean_filter)
