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

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("title")

    if "source" in metadata:
        source = metadata["source"].split("\\")
        source = source[source.index("langchain"):]
        metadata["source"] = "\\".join(source)

    return metadata

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

#Embedding model for bi encoder
embeddings = HuggingFaceEmbeddings(model_name="sergeyzh/rubert-tiny-turbo")

#Create vector db
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_dbV3",  # Where to save data locally, remove if not necessary
)

uuids = [str(uuid4()) for _ in range(len(splits))]
#Add docs to vector db
vector_store.add_documents(documents=splits, ids=uuids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 15}
)

# Add cross-encoder for rerank docs from bi encoder
model = HuggingFaceCrossEncoder(model_name='DiTy/cross-encoder-russian-msmarco')
compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

#Add LLM model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)
model_name = "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)

#Create pipeline for LLM model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, top_k=50, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

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

#prompt
template = """system: Your task is to answer the user's questions using only the information from the provided documents. Give two answers to each question: one with a list of relevant document identifiers and the second with the answer to the question itself, using documents with these identifiers.
 document: {context}
 user: {question}
 assistant: """
custom_rag_prompt = PromptTemplate.from_template(template)

#create chain for app
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser())

asi_msg = rag_chain.invoke("Могу ли я взять отпуск на время сессии при обучении в университете?")
print(asi_msg.split("assistant: ")[-1])

