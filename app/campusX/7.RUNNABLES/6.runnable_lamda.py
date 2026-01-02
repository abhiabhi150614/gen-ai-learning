from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables   import RunnableSequence ,RunnableParallel, RunnablePassthrough , RunnableLambda 


load_dotenv()

def word_counter(text):
    return len(text.split())



run_counter = RunnableLambda(word_counter)


model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

pt = PromptTemplate(
    template = "genrate a joke on the {topic}"

)


chain1 = RunnableSequence(pt,model,parser)

chain2 = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'count' : RunnableLambda(word_counter)
})

chain = RunnableSequence(chain1,chain2)


answer = chain.invoke({'topic':'cricket'})

print(answer)








