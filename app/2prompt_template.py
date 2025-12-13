from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate([
    ("system","You are an {domain} expert"),
    ("human","what is {topic}")
])




prompt = chat_template.invoke({
    'domain' : 'Deep learning',
    'topic' : 'Regularization'
})

model = ChatOllama(model = "gemma3:1b")


answer = model.invoke(prompt)

print(answer.content)