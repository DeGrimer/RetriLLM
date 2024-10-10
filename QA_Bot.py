from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_postgres import PGVector
from store import PostgresByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from database import COLLECTION_NAME, CONNECTION_STRING
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from uuid import uuid4
import json
from pathlib import Path
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

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
def format_context_docs(context):
    list_doc = []
    for doc in context:
        list_doc.append(doc.metadata['title'])
    return list_doc
@dataclass
class BotConfig:
    biencoder_model_name: str = "sergeyzh/rubert-tiny-turbo"
    vector_db_directory: str = "./chroma_langchain_db"
    k: int = 5 
    cross_encoder_model_name: str = "DiTy/cross-encoder-russian-msmarco"
    llm_model_name: str = "t-bank-ai/T-lite-instruct-0.1"

class QABot():
    def __init__(self, config : BotConfig):
        #Embedding model for bi encoder
        self.biencoder = HuggingFaceEmbeddings(model_name=config.biencoder_model_name)
        #Load vector db
        vectorstore = PGVector(
            embeddings=self.biencoder,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )

        store = PostgresByteStore(CONNECTION_STRING, COLLECTION_NAME)
        id_key = "doc_id"

        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore, 
            docstore=store, 
            id_key=id_key,
            search_kwargs={"k": config.k}
        )

        # Add cross-encoder for rerank docs from bi encoder
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=config.cross_encoder_model_name)
        self.compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=3)
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
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.llm_model_name,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                return_full_text=False,
                top_k=50,
                do_sample = True,
                temperature=0.1
            ),
            model_kwargs={"quantization_config": quantization_config},
        )
        self.chat_model = ChatHuggingFace(llm=llm)
    def get_ans(self, input):
        #prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate(
            [
                ('assistant', system_prompt),
                ('user', '{question}'),
            ]
        )

        #create chain for app
        rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | self.chat_model
        | StrOutputParser())
        rag_chain_with_source = RunnableParallel(
        {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain, source=lambda x: format_context_docs(x["context"]))

        asi_msg = rag_chain_with_source.invoke(input)
        answer = f"""
         Для ответа были использованы следующие документы: {asi_msg['source']}
        \n
        Ответ: {asi_msg['answer']}
        """
        return answer.split('://://')[0]
    def forward(self, input):
        asi_msg = self.get_ans(input)
        return asi_msg

