import os
import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def model_fn(model_dir):
    """
    Load the FAISS vector store and huggingface model into memory.
    """

    # load huggingface embedding model
    model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    # load vector store
    vs_path = os.path.join(model_dir, 'faiss')
    vs = FAISS.load_local(vs_path, hf)
    return vs


def input_fn(request_body, request_content_type):
    """
    Takes in request and transforms it to necessary input type - in this case we use json inputs
    """
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
    else:
        raise ValueError("Content type must be application/json")
    return request_body


def predict_fn(input_data, model):
    """
    SageMaker model server invokes `predict_fn` on the return value of `input_fn`.

    This function returns the similarity search results
    """
    vs = model
    results = vs.similarity_search_with_score(input_data['text'])
    return results


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    We wrap the output text into a json format here.
    """
    out_dict = dict()
    for ind in range(len(predictions)):
        out_dict[ind] = dict()
        out_dict[ind]["page_content"] = predictions[ind][0].page_content
        out_dict[ind]["metadata"]= predictions[ind][0].metadata
    return json.dumps(out_dict)