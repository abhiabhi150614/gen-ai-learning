from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables   import RunnableSequence ,RunnableParallel, RunnablePassthrough , RunnableLambda, RunnableBranch

load_dotenv()

pt = PromptTemplate(
    template = "write a detailed report on the {topic}",
    input_variables = ['topic']
)

pt1 = PromptTemplate(
    template = "summarize the '{text}' in short",
    input_variables = ['text']
)

model = ChatOllama(model = "gemma3:1b")
parser = StrOutputParser()
chain1 = RunnableSequence(pt,model,parser)


chain2 = RunnableBranch(
    (lambda x : len(x.split()) > 100, RunnableSequence(pt1,model,parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(chain1,chain2)

res = final_chain.invoke({'topic':'AI for machinary'})

print(res)


















































