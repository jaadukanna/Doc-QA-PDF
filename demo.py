import os 
import sys 
from dotenv import load_dotenv
import gradio as gr 
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex, VectorStoreIndex, LLMPredictor, PromptHelper
from llama_index import StorageContext,load_index_from_storage

load_dotenv()

import openai
openai.api_key = "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"

os.environ['OPENAI_API_KEY'] = "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"

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    chunk_overlap_ratio = 0.1
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio=chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

    #llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = VectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist()
    

    return index

def qabot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response


iface = gr.Interface(fn=qabot, inputs=gr.Textbox(label="Enter your query"), outputs="text")
index = construct_index("docs")
iface.launch(share=True)
