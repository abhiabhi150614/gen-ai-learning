from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv



load_dotenv()


chat_history = [SystemMessage("you are an ai mentor today keep messages conversation very short")]



model = ChatOllama(model="gemma3:1b")


while(True):

    a = input("Human : ")
    chat_history.append(HumanMessage(a))
    if a == "exit":
        break

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(response.content))
  
    print("AI :",response.content)


print(chat_history)