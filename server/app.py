import chatbot_utils

from litestar import Litestar , post
from litestar.config.cors import CORSConfig

import pandas as pd

from pydantic import BaseModel
from typing import List

cors_config = CORSConfig(allow_origins=["*"])


def parse_numbers(s):
    return [float(x) for x in s.strip("[]").split(",")]


def get_confluence_embeddings():
    # Current file where the embeddings of our internal Confluence document is saved
    Confluence_embeddings_file = "DOC_title_content_embeddings.csv"

    DOC_title_content_embeddings = pd.read_csv(
        Confluence_embeddings_file, dtype={"embeddings": object}
    )
    DOC_title_content_embeddings["embeddings"] = DOC_title_content_embeddings[
        "embeddings"
    ].apply(lambda x: parse_numbers(x))


    return DOC_title_content_embeddings

# Class definitions (Pydantic is like Typescript :P)
class UserQuery(BaseModel):
    query: str

class ConfluencePage (BaseModel):
    title: str
    link: str

class Reply(BaseModel):
    message: str
    pages: List[ConfluencePage]


@post(path="/message")
async def chat_message(data: UserQuery) -> Reply:
    DOC_title_content_embeddings = get_confluence_embeddings()

    message, links = chatbot_utils.generate_answer(data.query, DOC_title_content_embeddings)

    return { "message": message, "links": links }

# app = Litestar(
#     route_handlers=[chat_message],
#     cors_config=cors_config
#     )
chatbot_utils.update_internal_doc_embeddings()