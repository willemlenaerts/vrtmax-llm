import os

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import mlflow

assert (
    "OPENAI_API_KEY" in os.environ
), "Please set your OPENAI_API_KEY environment variable."

# Using a local MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create a new experiment that the model and the traces will be logged to
mlflow.set_experiment("LangChain Tracing")

# Enable LangChain autologging
# Note that models and examples are not required to be logged in order to log traces.
# Simply enabling autolog for LangChain via mlflow.langchain.autolog() will enable trace logging.
mlflow.langchain.autolog(log_models=True, log_input_examples=True)

llm = OpenAI(temperature=0.7, max_tokens=1000)

prompt_template = (
    "Imagine that you are {person}, and you are embodying their manner of answering questions posed to them. "
    "While answering, attempt to mirror their conversational style, their wit, and the habits of their speech "
    "and prose. You will emulate them as best that you can, attempting to distill their quirks, personality, "
    "and habits of engagement to the best of your ability. Feel free to fully embrace their personality, whether "
    "aspects of it are not guaranteed to be productive or entirely constructive or inoffensive."
    "The question you are asked, to which you will reply as that person, is: {question}"
)

chain = prompt_template | llm

# Test the chain
chain.invoke(
    {
        "person": "Richard Feynman",
        "question": "Why should we colonize Mars instead of Venus?",
    }
)

# Let's test another call
chain.invoke(
    {
        "person": "Linus Torvalds",
        "question": "Can I just set everyone's access to sudo to make things easier?",
    }
)