from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=10   # HARD CAP
)


response = llm.invoke("Write a  poem about cricket.")
print(response.content)
