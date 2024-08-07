#Import Dependencies
import streamlit as sl
import sagemaker
import boto3
import json
import time
import subprocess
from langchain_aws import ChatBedrockConverse

# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if "sagemaker_session" not in sl.session_state:
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

    session = boto3.Session(
    aws_access_key_id, aws_secret_access_key, aws_session_token, region_name="eu-west-1"
    )    
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)

    endpoint_name = "faiss-endpoint-1723058950"
    faiss_vector_store = sagemaker.predictor.Predictor(endpoint_name,sagemaker_session=sagemaker_session)

    # LLM endpoint
    llm = ChatBedrockConverse(
        credentials_profile_name=aws_profile, 
        region_name="eu-west-2",
        model="meta.llama3-70b-instruct-v1:0",
        temperature=0.6,
        top_p=0.6,
        max_tokens=512
    )


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

    sl.session_state.messages.append(("system",system_prompt ))

# Display chat messages from history on app rerun
if "messages_to_display" not in sl.session_state:
    sl.session_state.messages_to_display = []

for message in sl.session_state.messages_to_display:
    if message[0] != "system":
        with sl.chat_message(message[0]):
            sl.markdown(message[1])

if __name__== '__main__':

    query=sl.chat_input('Wat is je vraag?')

    if(query):

        # Display user message in chat message container
        sl.chat_message("user").markdown(query)
        sl.session_state.messages_to_display.append(( "human",  query ))

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
        user_prompt = ""
        for item in out.keys():
            
            user_prompt += "Het programma met de naam " + out[item]["metadata"]["mediacontent_pagetitle_program"] +\
                " heeft volgende beschrijving: " + out[item]["metadata"]["mediacontent_page_description"] 
            
            # + " en in het programma wordt er onder meer het volgende gezegd dat misschien relevant is voor de vraag: " + \out[item]["page_content"]

        user_prompt += " Tot zover alle informatie over de programmas."

        user_prompt += " Gelieve met die informatie een antwoord te geven op volgende vraag: " + query

        # Ask question
        # Prompt to generate
        sl.session_state.messages.append(( "human",  user_prompt ))

        chat = llm.invoke(sl.session_state.messages)

        stream = llm.stream(sl.session_state.messages)

        sl.session_state.messages.append(("assistant", chat ))
        sl.session_state.messages_to_display.append(("assistant", chat ))

        # Display assistant response in chat message container
        sl.write_stream(stream)
        # sl.chat_message("assistant").markdown(chat)