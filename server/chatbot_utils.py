import os
from atlassian import Confluence
from transformers import GPT2TokenizerFast
import openai
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set the API key
openai.api_key =  os.getenv('API_KEY')

DOC_MODEL = 'text-embedding-ada-002'
COMPLETIONS_MODEL = "text-davinci-003"
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL')
CONFLUENCE_EMAIL = os.getenv('CONFLUENCE_EMAIL')
CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_API_TOKEN')

def connect_to_Confluence():
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_EMAIL,
        password=CONFLUENCE_API_TOKEN,
        cloud=True)

    return confluence

def get_all_pages(confluence, space='DTDD'):
    start = 0
    limit = 100
    _all_pages = []
    while True:
        pages = confluence.get_all_pages_from_space(space, start, limit, status=None, expand="body.storage", content_type='page')
        _all_pages = _all_pages + pages
        if len(pages) < limit:
            break
        start = start + limit
    return _all_pages
def get_all_pagess(confluence, parent_id=105676819):
    pages = []
    # get all pages in DTDD space
    DTDD_pages = confluence.get_all_pages_from_space('DTDD', expand='body.storage', limit=1000)
    # while more pages exist, get the next page
    while DTDD_pages['next']:
        pages.append(DTDD_pages['results'])
        DTDD_pages = confluence.get_all_pages_from_space('DTDD', expand='body.storage', limit=1000, start=DTDD_pages['next']['start'])

    for page in tqdm(DTDD_pages):
        pages.append(page)
    x = pages['more']
    return pages['more']
    # All child pages in the how to guide in DTPP Space.
    how_to_pages = confluence.get_child_pages(parent_id)

    # Confluence API is odd. Need to pull each page individually to get contents.
    for page in tqdm(how_to_pages):
        page = confluence.get_page_by_id(page['id'], expand='body.storage')
        pages.append(page)
    return pages

def get_embeddings(text: str, model: str) -> list[float]:
    '''
    Calculate embeddings.

    Parameters
    ----------
    text : str
        Text to calculate the embeddings for.
    model : str
        String of the model used to calculate the embeddings.

    Returns
    -------
    list[float]
        List of the embeddings
    '''
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_max_num_tokens():
    return 2046

def has_text(element):
    return element.string is not None and len(element.string.strip()) > 0

def collect_title_body_embeddings(pages, save_csv=True):
    collect = []
    for page in tqdm(pages):
        title = page['title']
        link = CONFLUENCE_URL + '/spaces/DTDD/pages/' + page['id']
        htmlbody = page['body']['storage']['value']
        htmlParse = BeautifulSoup(htmlbody, 'html.parser')
        body = []

        for para in htmlParse.find_all(has_text):
            sentence = para.get_text()
            body.append(sentence)

        body = '. '.join(body)

        # Calculate number of tokens
        tokens = tokenizer.encode(body)
        collect += [(title, link, body, len(tokens))]

    DOC_title_content_embeddings = pd.DataFrame(collect, columns=['title', 'link', 'body', 'num_tokens'])
    # Caculate the embeddings
    # Limit first to pages with less than 2046 tokens
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
    DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, DOC_MODEL))

    if save_csv:
        DOC_title_content_embeddings.to_csv('DOC_title_content_embeddings.csv', index=False)

    return DOC_title_content_embeddings

def update_internal_doc_embeddings():
    # Connect to Confluence
    confluence = connect_to_Confluence()
    # Get page contents
    pages = get_all_pages(confluence)
    # Extract title, body and number of tokens
    DOC_title_content_embeddings= collect_title_body_embeddings(pages, save_csv=True)
    return DOC_title_content_embeddings

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embeddings(query, model=DOC_MODEL)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)

    return doc_embeddings

def construct_prompt(query, doc_embeddings):

    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_links = []

    for section_index in range(len(doc_embeddings)):
        # Add contexts until we run out of space.
        document_section = doc_embeddings.loc[section_index]

        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        # if body is a valid string
        if isinstance(document_section.body, str):
            chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
            chosen_sections_links.append(document_section.link)

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

    return (prompt,  chosen_sections_links)

def generate_answer(query, DOC_title_content_embeddings):


    # Order docs by similarity of the embeddings with the query
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)

    # Construct the prompt
    prompt, links = construct_prompt(query, DOC_title_content_embeddings)
    # Ask the question with the context to ChatGPT


    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )
    output = response["choices"][0]["text"].strip(" \n")

    # Enable this code to use GPT4
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     temperature=0,
    #     max_tokens=300,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     messages=[
    #         {"role": "system", "content": prompt},
    #     ]
    # )


    # output = response['choices'][0]['message']['content']
    return output, links
