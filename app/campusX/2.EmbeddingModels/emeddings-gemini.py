from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import  cosine_similarity
import numpy as np

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


query = "tell me about virat kohli"

vectors_emb = embeddings.embed_documents(documents)
query_emb = embeddings.embed_query(query)



score  = cosine_similarity([query_emb],vectors_emb)[0]

score = list(score)
print(list(score))
print(documents[score.index(max(score))])






 
