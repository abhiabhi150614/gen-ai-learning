from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables   import RunnableSequence ,RunnableParallel, RunnablePassthrough 

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")



#example usage 
passthrough = RunnablePassthrough()
print(passthrough.invoke({'topic':'cricket'}))



p = StrOutputParser()

pt1 = PromptTemplate(
    template = "genrate a joke on the {topic}",
    input_variables = 'topic'
)


pt2 = PromptTemplate(
    template = 'explain the following joke in 2 line{text}',
    input_variable = 'text'
)


joke_gen_chain = RunnableSequence(pt1,model,p)


par = RunnableParallel(
    {
      'explain' : RunnableSequence(pt2,model,p),
      'joke': RunnablePassthrough()
    }
)


final_chain = RunnableSequence(joke_gen_chain,par)


res = final_chain.invoke("cricket")


print(res)