from langchain_ollama import ChatOllama

model = ChatOllama(model="gemma3:1b")  #use other ollama models 
chat_history = [
    {"role": "system", "content": "your name is abhishek b shetty and you are a helpful ai agent"}
]

while True:
    user_input = input("You : ")

    if user_input == "exit":
        break

    chat_history.append({"role": "user", "content": user_input})

    response = model.invoke(chat_history)

    print("Bot :", response.content)

    chat_history.append({"role": "assistant", "content": response.content})
