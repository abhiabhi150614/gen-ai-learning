from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import StrOutputParser
from langchain_ollama import ChatOllama
load_dotenv()








pt1 = PromptTemplate(
    template = "generate 5 interesting facts about {topic}",
    input_variables = ['topic']
)

#model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model = ChatOllama(model = "gemma3:1b")
parser =  StrOutputParser()


chain = pt1 | model | parser 



response = chain.invoke({'topic':'black hole'})


print(response)

chain.get_graph().print_ascii()

chain.get_graph()