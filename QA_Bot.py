from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from uuid import uuid4
import json
from pathlib import Path
import os
from dataclasses import dataclass

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("title")

    if "source" in metadata:
        source = metadata["source"].split("\\")
        source = source[source.index("langchain"):]
        metadata["source"] = "\\".join(source)

    return metadata
def load_docs(vector_store):
    #Load data from json file
    loader = JSONLoader(
        file_path='./langchain/dataTK.json',
        jq_schema='.[]',
        content_key="content",
        metadata_func=metadata_func)

    docs = loader.load()

    #Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    uuids = [str(uuid4()) for _ in range(len(splits))]
    #Add docs to vector db
    vector_store.add_documents(documents=splits, ids=uuids)

#Convert relevant docs to json format
def format_docs(docs):
    list_doc = []
    for doc in docs:
        curDoc = {
            'id':doc.metadata['title'],
            'title':doc.metadata['title'],
            'content':doc.page_content
        }
        list_doc.append(curDoc)
    return json.dumps(list_doc, ensure_ascii=False)

@dataclass
class BotConfig:
    biencoder_model_name: str = "sergeyzh/rubert-tiny-turbo"
    vector_db_directory: str = "./chroma_langchain_db"
    k: int = 15 
    cross_encoder_model_name: str = "DiTy/cross-encoder-russian-msmarco"
    llm_model_name: str = "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24"

class QABot():
    def __init__(self, config : BotConfig):
        #Embedding model for bi encoder
        self.biencoder = HuggingFaceEmbeddings(model_name=config.biencoder_model_name)
        #Load vector db
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.biencoder,
            persist_directory=config.vector_db_directory,
        )
        if len(self.vector_store.get()['metadatas']) == 0:
            load_docs(self.vector_store)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.k}
        )

        # Add cross-encoder for rerank docs from bi encoder
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=config.cross_encoder_model_name)
        self.compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=5)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

        #Add LLM model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        llm_model_name = config.llm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", quantization_config=quantization_config)
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=512, top_k=50, temperature=0.1)
        self.llm = HuggingFacePipeline(pipeline=pipe)

        #prompt
        template = """system: Your task is to answer the user's questions using only the information from the provided documents. Give two answers to each question: one with a list of relevant document identifiers and the second with the answer to the question itself, using documents with these identifiers.
        document: {context}
        user: {question}
        assistant: """
        self.custom_rag_prompt = PromptTemplate.from_template(template)

        #create chain for app
        self.rag_chain = (
            {"context": self.compression_retriever | format_docs, "question": RunnablePassthrough()}
            | self.custom_rag_prompt
            | self.llm
            | StrOutputParser())
    def forward(self, input):
        asi_msg = self.rag_chain.invoke(input)
        return asi_msg.split("assistant: ")[-1]

