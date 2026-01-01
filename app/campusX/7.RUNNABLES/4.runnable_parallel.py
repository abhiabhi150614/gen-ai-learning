from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables   import RunnableSequence, RunnableParallel 


load_dotenv()


pt1 = PromptTemplate(
    template = "write a short linkedin post on {topic} stictly give only 1 direct post",
    input_variables = ['topic']
)


pt2 = PromptTemplate(
    template = "write a short twitter post on {topic} strictly give 1 direct post",
    input_variables = ['topic']
)
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")



parser = StrOutputParser()
ans = RunnableParallel(
    {
        'tweet_genrate' : RunnableSequence(pt1,model,parser),
        'linked_genrate' : RunnableSequence(pt2,model,parser)
    }
)

res = ans.invoke({'topic':'ai'})
print(res['tweet_genrate'])
print(res['linked_genrate'])

