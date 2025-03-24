import os
import sys
from numpy import array
from pyspark.sql import SparkSession
from datasets import load_dataset, Dataset

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName('AIG_code_test')\
                            .master("local[7]") \
                            .config("spark.driver.memory", "50g") \
                            .config("spark.default.parallelism", "14") \
                            .config("spark.sql.shuffle.partitions", "14") \
                            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                            .getOrCreate()


dataset_name = "eloukas/edgar-corpus"
dataset_save_dir = "./dataset"

def download_data(year):
    _dataset = load_dataset(dataset_name, f"year_{year}", split="train")
    save_path = f"{dataset_save_dir}/dataset_{year}.parquet"
    _dataset.to_parquet(save_path)

def load_data(year):
    save_path = f"{dataset_save_dir}/dataset_{year}.parquet"
    return spark.read.parquet(save_path)

SENTENCE_EMBED_MODEL = "all-MiniLM-L12-v2"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import col, udf, explode
from sentence_transformers import SentenceTransformer
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from langchain_openai import OpenAIEmbeddings

def chunk_text(text, chunk_size=200): 
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0
    )
    return text_splitter.split_text(text)

def embed(chunk, dv=True):
    model = SentenceTransformer(SENTENCE_EMBED_MODEL)
    embedding = model.encode(chunk).tolist()
    return DenseVector(embedding) if dv else array(embedding)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

rerank_model_name = 'BAAI/bge-reranker-large'
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
rerank_model.eval()