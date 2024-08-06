#Import Dependencies
import streamlit as sl
import sagemaker
import boto3
from sagemaker.huggingface.model import HuggingFacePredictor
import json
import random
import time

# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# AWS login to use sagemaker endpoints
if "sagemaker_session" not in sl.session_state:
    session = boto3.Session(profile_name='vrt-analytics-engineer-nonsensitive')
    sagemaker_session = sagemaker.Session(boto_session=session)
    sl.session_state.sagemaker_session = sagemaker_session
    role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)

    # Vector store endpoint
    endpoint_name = "faiss-endpoint-1722716142"
    faiss_vector_store = sagemaker.predictor.Predictor(endpoint_name,sagemaker_session=sl.session_state.sagemaker_session)
    sl.session_state.faiss = faiss_vector_store


    # LLM endpoint
    llm = HuggingFacePredictor(endpoint_name="huggingface-pytorch-tgi-inference-2024-07-30-20-23-30-977",sagemaker_session=sl.session_state.sagemaker_session)
    sl.session_state.llm = llm

# Set up chatbot
sl.title("VRT MAX chat")

# Initialize chat history
if "messages" not in sl.session_state:
    sl.session_state.messages = []

    # System prompt
    system_prompt = """
    Je bent een hulpvaardige assistent die Nederlands spreekt.
    Je krijgt vragen over de catalogus van programma's van het videoplatform VRT MAX. 
    Je probeert mensen te helpen om programma's aan te bevelen die aansluiten bij hun vraag.
    Daarvoor krijg je een beschrijving van een aantal programma's.
    Aan jou om wat relevant is aan te bevelen.
    Begin niet met het geven van jouw mening over de programma's. Begin direct met het aanbevelen van relevante content aan de gebruiker.
    """

    sl.session_state.messages.append({ "role": "system", "content": system_prompt })

# Display chat messages from history on app rerun
if "messages_to_display" not in sl.session_state:
    sl.session_state.messages_to_display = []

for message in sl.session_state.messages_to_display:
    if message["role"] != "system":
        with sl.chat_message(message["role"]):
            sl.markdown(message["content"])

if __name__== '__main__':

    query=sl.chat_input('Wat is je vraag?')
    if(query):

        # Display user message in chat message container
        sl.chat_message("user").markdown(query)
        sl.session_state.messages_to_display.append({ "role": "user", "content": query })

        # Fetch relevant content
        payload = json.dumps({
            "text": query,
        })

        out = sl.session_state.faiss.predict(
            payload,
            initial_args={"ContentType": "application/json", "Accept": "application/json"}
        )
        out = json.loads(out)
        
        # Generate prompt
        user_prompt = ""
        for item in out.keys():
            
            user_prompt += "Het programma met de naam " + out[item]["metadata"]["mediacontent_pagetitle_program"] +\
                " heeft volgende beschrijving: " + out[item]["metadata"]["mediacontent_page_description"] 
            
            # + " en in het programma wordt er onder meer het volgende gezegd dat misschien relevant is voor de vraag: " + \out[item]["page_content"]

        user_prompt += " Tot zover alle informatie over de programmas."

        user_prompt += " Gelieve met die informatie een antwoord te geven op volgende vraag: " + query

        # Ask question
        # Prompt to generate
        sl.session_state.messages.append({ "role": "user", "content": user_prompt })

        # Generation arguments
        parameters = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct", # placeholder, needed
            "top_p": 0.6,
            "temperature": 0.9,
            "max_tokens": 512,
            "stop": ["<|eot_id|>"],
        }

        chat = sl.session_state.llm.predict({"messages" :sl.session_state.messages, **parameters})

        response = chat["choices"][0]["message"]["content"].strip()
        sl.session_state.messages.append({ "role": "assistant", "content": response })
        sl.session_state.messages_to_display.append({ "role": "assistant", "content": response })

        # Display assistant response in chat message container
        with sl.chat_message("assistant"):
            response = sl.write_stream(response_generator(response))