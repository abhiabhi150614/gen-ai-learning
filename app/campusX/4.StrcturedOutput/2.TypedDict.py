from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()


class Review(TypedDict):
    key_themes : Annotated[list[str],"write down all the key themes dicussed in the review in a list"]
    
    summary : Annotated[str, "a short 1 line summary of the review"] 
    sentiment :  Annotated[Literal["pos","neutral","neg"],"return in terms of true if positive and false if negative"]
    pros : Annotated[Optional[list[str]],"write down all the pros of the review in short 1 words"]
    cons : Annotated[Optional[list[str]],"write down all cons of the review in short 1 words"]
    name : Annotated[Optional[str],"Write  the name of the person who gave  reviewer"]







#model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

model = ChatOllama(model = "gemma3:1b")


strct = model.with_structured_output(Review)



response = strct.invoke("""
 recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

                                 
Reviewer name : 'Abhishek B shetty'
                        
""")






print(response)
