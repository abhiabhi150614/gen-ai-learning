from langchain_ollama import ChatOllama 



model = ChatOllama(model="gemma3:1b")




result = model.invoke("design a 1 day travel itinerary for paris give output in stricly json not more not less")


print(result.content)






