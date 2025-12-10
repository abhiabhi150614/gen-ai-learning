from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

k = input("enter the prompt : ")
load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemma-3-12b-it")  #you can also use gemini-2.5-flash
prompt = "image your name is simal and give response for boy abhishek like you are mentor of abhishek so answer his questions : "
prompt = prompt + "\n" + k
result = model.invoke(prompt)


print(result.content)
