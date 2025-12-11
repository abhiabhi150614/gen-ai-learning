from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()



#model = ChatGoogleGenerativeAI(model = "gemma-3-12b-it")
models = ChatOllama(model="gemma3:1b")

chat_history = [
    SystemMessage(content = "your name is abhishek b shetty and you are helpfull ai agent")


]


while True:
    a = input("You : ")
    chat_history.append(HumanMessage(content = a))
    if(a == "exit"):
        break
    result = models.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("Bot : ",result.content)