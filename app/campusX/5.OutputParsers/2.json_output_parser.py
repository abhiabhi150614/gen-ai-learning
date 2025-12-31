
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()






llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=200,
    temperature=0,
)


chat = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()

p1 = PromptTemplate(
    template = "Give me details about {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}


)


chain = p1 | chat | parser
'''
prompt1 = p1.invoke({})



res = chat.invoke(prompt1)

par = parser.parse(res.content)


'''

res = chain.invoke({'topic':'black hole'})
print(res)

