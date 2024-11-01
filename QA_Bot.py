from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import  ChatPromptTemplate
from langchain_postgres import PGVector
from store import PostgresByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from database import COLLECTION_NAME, CONNECTION_STRING
from transformers import AutoTokenizer, BitsAndBytesConfig
from langchain.schema.document import Document
from uuid import uuid4
import json
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
def load_docs(retriever):
    #Load data from json file
    loader = JSONLoader(
        file_path='./langchain/dataTK.json',
        jq_schema='.[]',
        content_key="content",
        metadata_func=metadata_func)

    documents = loader.load()
    doc_ids = [str(uuid4()) for _ in documents]
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    id_key = "doc_id"
    all_sub_docs = []
    for i, doc in enumerate(documents):
        doc_id = doc_ids[i]
        sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[id_key] = doc_id
        all_sub_docs.extend(sub_docs)
    #Add chunked sub docs to vectorstore
    retriever.vectorstore.add_documents(all_sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents)))
    text_summaries = []
    with open('summarization.json', 'r', encoding='utf-8') as f:
        text_summaries = json.load(f)

    #Add summary to vectorstore
    summary_docs = []
    for i, (summary, doc_id) in enumerate(zip(text_summaries, doc_ids)):
        # Define your new metadata here
        new_metadata = {"page": i, "doc_id": doc_id}

        # Create a new Document instance for each summary
        doc = Document(page_content=str(summary))

        # Replace the metadata
        doc.metadata = new_metadata

        # Add the Document to the list
        summary_docs.append(doc)
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents)))

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
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
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

        #Add LLM model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.llm_model_name,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                return_full_text=False,
                top_k=50,
                do_sample = True,
                temperature=0.1,
                eos_token_id = terminators
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

