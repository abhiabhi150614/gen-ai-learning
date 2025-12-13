from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="gemma3:1b")


prompt = PromptTemplate(template = "write a 4 point summary on {topic}",
                        input_variables =['topic'])



a1 = prompt.invoke({'topic' : "Operation sindoor"})



res1 = model.invoke(a1)


prompt2 = PromptTemplate(template = "identify key points in {text}",
                        input_variables = ['text'])


a2 = prompt.invoke({'text':str(res1.content)})


res2 = model.invoke(a2)

print(res2.content)

