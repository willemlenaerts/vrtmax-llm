#Import Dependencies
import streamlit as sl
import sagemaker
import boto3
from sagemaker.huggingface.model import HuggingFacePredictor
import json


# AWS login to use sagemaker endpoints
print(sl.session_state)
if "sagemaker_session" not in sl.session_state:
    session = boto3.Session(profile_name='vrt-analytics-engineer-nonsensitive')
    sagemaker_session = sagemaker.Session(boto_session=session)
    sl.session_state.sagemaker_session = sagemaker_session
    role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
print(sl.session_state)
# Vector store endpoint
endpoint_name = "faiss-endpoint-1722716142"
faiss_vector_store = sagemaker.predictor.Predictor(endpoint_name,sagemaker_session=sl.session_state.sagemaker_session)
assert faiss_vector_store.endpoint_context().properties['Status'] == 'InService'

# LLM endpoint
llm = HuggingFacePredictor(endpoint_name="huggingface-pytorch-tgi-inference-2024-07-30-20-23-30-977",sagemaker_session=sl.session_state.sagemaker_session)

if __name__=='__main__':
    sl.header("VRT MAX chat")

    
    query=sl.text_input('Wat is je vraag?')
    if(query):
            # Fetch relevant content
            payload = json.dumps({
                "text": query,
            })

            out = faiss_vector_store.predict(
                payload,
                initial_args={"ContentType": "application/json", "Accept": "application/json"}
            )
            out = json.loads(out)
            # Generate prompt
            system_prompt = """
            Je bent een hulpvaardige assistent die Nederlands spreekt.
            Je krijgt vragen over de catalogus van programma's van het videoplatform VRT MAX. 
            Je probeert mensen te helpen om programma's aan te bevelen die aansluiten bij hun vraag.
            Daarvoor krijg je een beschrijving van een aantal programma's.
            Aan jou om wat relevant is aan te bevelen.
            Begin niet met het geven van jouw mening over de programma's. Begin direct met het aanbevelen van relevante content aan de gebruiker.
            """

            user_prompt = ""
            for item in out.keys():
                
                user_prompt += "Het programma met de naam " + out[item]["metadata"]["mediacontent_pagetitle_program"] +\
                    " heeft volgende beschrijving: " + out[item]["metadata"]["mediacontent_page_description"] 
                
                # + " en in het programma wordt er onder meer het volgende gezegd dat misschien relevant is voor de vraag: " + \out[item]["page_content"]

            user_prompt += " Tot zover alle informatie over de programmas."

            user_prompt += " Gelieve met die informatie een antwoord te geven op volgende vraag: " + query

            # Ask question
            # Prompt to generate
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ]

            # Generation arguments
            parameters = {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct", # placeholder, needed
                "top_p": 0.6,
                "temperature": 0.9,
                "max_tokens": 512,
                "stop": ["<|eot_id|>"],
            }

            chat = llm.predict({"messages" :messages, **parameters})
            sl.write(chat["choices"][0]["message"]["content"].strip())