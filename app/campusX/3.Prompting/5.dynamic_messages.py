from langchain_core.messages import SystemMessage,HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate 



from  dotenv import load_dotenv


load_dotenv()


chat_template = ChatPromptTemplate([
    ("system" , "You are a {domain} expert"),

    ("human","explain shortly about {topic}") 
])



prompt = chat_template.invoke({
    "domain" : "cricket",
    "topic" : "batting"
})

print(prompt)