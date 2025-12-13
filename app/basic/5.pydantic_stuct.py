from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama 
from typing import TypedDict,Optional, Annotated,Literal
from pydantic import BaseModel, Field


model = ChatOllama(model = "gemma3:1b")



class Review(BaseModel):
    keythemes:list[str] = Field(description = "writen down key things")
    summary:str = Field(description = "write in depth summary")
    sentiment: Literal["true","false"]
    name: Optional[str] = Field(description = "write name of the reviewer")


prompt = """
Google’s Pixel phones have never been the most powerful handsets, with their Tensor chipsets falling behind rivals in benchmarks. But surprisingly, the Google Pixel 10 series might be even worse.

In a post on X, @lafaiel (via Phone Arena) shared a screenshot of a Geekbench listing for the Google Pixel 10 Pro, in which the Pixel 10 Pro achieved a GPU score of just 3,707.

So based on this result the Pixel 10 Pro is way behind, though the fact that it scored even less than its predecessor is especially worrying.

Now, GPU performance is only one part of the power picture, and the Pixel 10 Pro should outperform the Pixel 9 Pro on Geekbench overall, with the same source recording a single-core score.

But such a low GPU score means the Google Pixel 10 Pro might still struggle with demanding games.

Google claims the Pixel 10 series will feel faster in everyday use, and perform better for AI tasks, and that’s probably true, so if you don’t care about games then this shouldn’t matter much.

That said, while this looks like a legitimate Geekbench screenshot, it’s still just one result, so it’s possible this will turn out to be an outlier. It’s also feasible that Google could improve performance before launch.
"""



answer = model.invoke(prompt)

print(answer)
