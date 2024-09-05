import sagemaker
import boto3
import json
import subprocess
from langchain_aws import ChatBedrockConverse
from os import environ
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


# LLM endpoint
llm = ChatBedrockConverse(
    region_name="eu-west-2",
    model="mistral.mistral-large-2402-v1:0" , # "meta.llama3-70b-instruct-v1:0"
    temperature=0.6,
    top_p=0.6,
    max_tokens=512
)

model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
encode_kwargs = {'normalize_embeddings': False}

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()
    
model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

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


system_prompt = """
Je bent een hulpvaardige assistent die Nederlands spreekt.
Je krijgt vragen over de catalogus van programma's van het videoplatform VRT MAX. 
Je probeert mensen te helpen om programma's aan te bevelen die aansluiten bij hun vraag.
Daarvoor krijg je een beschrijving van een aantal programma's.
Aan jou om wat relevant is aan te bevelen.
Begin niet met het geven van jouw mening over de programma's. Begin direct met het aanbevelen van relevante content aan de gebruiker.
"""

message = """Beantwoordt de vraag op basis van de volgende context:

{context}

Vraag: {question}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", message),
    ]
)


from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
mlflow.models.set_model(rag_chain_with_source)
