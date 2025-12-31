
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()






llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
)

p1 = PromptTemplate(
    template = "Wite a details report on {topic}",
    input_variables = ['topic']
)

p2 = PromptTemplate(
    template = "write a 1 line short summary on {text}",
    input_variables = ['text']
)


chat = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = p1 | chat | parser | p2 | chat | parser
'''
prompt1 = p1.invoke({'topic':'cricket'})

resp = chat.invoke(prompt1)

print(resp.content)
print()
print()
print()
print()
print()
prompt2 = p2.invoke({'text' : resp.content})



resp2 = chat.invoke(prompt2)
'''


chained =  chain.invoke({'topic':'cricket'})
print(chained)
