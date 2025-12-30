from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

from langchain_core.messages import HumanMessage,AIMessage, SystemMessage


#chat template 


message = ChatPromptTemplate(
    [
        ("system" , "you are helpfull customer support agent"),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human" , '{query}')
    ]
)



#load chat history 
chat_history = []

with open('7.chat_history.txt') as f:
    chat_history.extend(f.readlines())

k = message.invoke({'chat_history':chat_history,'query' : 'where is my refund'})

print(k)
