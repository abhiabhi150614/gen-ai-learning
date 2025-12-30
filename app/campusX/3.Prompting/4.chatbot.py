from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from dotenv import load_dotenv



load_dotenv()


chat_history = []



model = ChatOllama(model="gemma3:1b")


while(True):

    a = input("Human : ")
    chat_history.append(a)
    if a == "exit":
        break

    response = model.invoke(chat_history)

    chat_history.append(response.content)
  
    print("AI :",response.content)


print(chat_history)