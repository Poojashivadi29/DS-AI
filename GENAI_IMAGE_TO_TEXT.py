import os
os.environ['GEMINI_API_KEY']='AIzaSyBfsjsydiDv8ILCFAntkYoO9E53-csGmz4'
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text=text.replace('.',' *')
    return Markdown(textwrap.indent(text,'> ',predicate=lambda _: True))

for m in genai.list_models():
  if 'genrateContent' in m.supported_generation_methods:
    print(m.name)