import torch
import pandas as pd
from huggingface_hub import login
from langchain import hub
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# login to hugginface (~/.cache/huggingface/token)
login()

# Track to local MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create a new experiment that the model and the traces will be logged to
mlflow.set_experiment("StableLM Tracing")

# Enable LangChain autologging
mlflow.langchain.autolog(log_models=True, log_input_examples=True)

###############################################################################
# Set up embeddings for later retrieval
###############################################################################
df = pd.read_csv("test.csv")

# Postprocessing
# Concatenate description & subtitles
df["info_to_embed"] = df["mediacontent_page_description"] + " " + df["subtitle"]
df = df[df.info_to_embed.notnull()]
df= df[["mediacontent_page_description","mediacontent_pagetitle_program","info_to_embed","mediacontent_pageid"]]

loader = DataFrameLoader(df, page_content_column="info_to_embed")
catalog = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
catalog_chunks = text_splitter.split_documents(catalog)

model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
encode_kwargs = {'normalize_embeddings': False}
hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)
vector_store = FAISS.from_documents(documents=catalog_chunks,
                                    embedding=hf_embedding)


###############################################################################
# Set up embeddings for later retrieval
###############################################################################
retriever = vector_store.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
###############################################################################
# Make Chain
###############################################################################

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")







template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

query = "programma over geld en beleggen"

results_with_scores = vector_store.similarity_search_with_score(query)
for doc, score in results_with_scores:
    print(f"Metadata: {doc.metadata}, Score: {score}")

# Build prompt
beschrijving = ""
for item in results_with_scores:
    item[0].metadata["mediacontent_page_description"]
    item[0].metadata["mediacontent_pageid"]
    item[0].metadata["mediacontent_pagetitle_program"]
    beschrijving += "Episode van programm " + item[0].metadata["mediacontent_pagetitle_program"] + "en met episodenummer " + str(item[0].metadata["mediacontent_pageid"]) + " heeft volgende beschrijving: " + item[0].metadata["mediacontent_page_description"]

# prompt = f"""Hieronder een lijst van beschrijving van TV episodes. Kan je op basis van deze informatie zeggen in welke episode Isolde zat? De nummer is voldoende.
prompt = f"""{query}

{beschrijving}

"""



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
