from langchain_ollama import ChatOllama


model = ChatOllama(model = "gemma3:1b")



p = model.invoke("hii")
print(p.content)


