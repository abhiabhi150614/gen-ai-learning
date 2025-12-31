
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate

from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema






load_dotenv()






llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=200,
    temperature=0,
)




chat = ChatHuggingFace(llm = llm)


schema = [
    ResponseSchema(name = 'fact_1', description = "fact 1 about the topic "),
    ResponseSchema(name = 'fact_2', description = "fact 2 about the topic "),
    ResponseSchema(name = 'fact_3', description = "fact 3 about the topic "),
    ResponseSchema(name = 'fact_4', description = "fact 4 about the topic ")
]


parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template = "give me 4 facts about {topic}  \n {format_ins}",
    input_variables = {'topic'},
    partial_variables = {'format_ins': parser.get_format_instructions()}
)


prompt = template.invoke({'topic' : 'black hole'})

print(prompt)