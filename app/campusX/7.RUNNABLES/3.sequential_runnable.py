from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables   import RunnableSequence 

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


p = StrOutputParser()

pt = PromptTemplate(
    template = "genrate a joke on the {topic}",
    input_variables = 'topic'
)


pt2 = PromptTemplate(
    template = 'explain the following joke {text}',
    input_variable = 'text'
)
chain = RunnableSequence(pt,model,p,pt2,model,p)


k = chain.invoke({'topic' : 'cricket'})


print(k)
