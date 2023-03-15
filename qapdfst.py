
from langchain import OpenAI, VectorDBQA, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import sys
from io import StringIO
import pandas as pd


llm = OpenAI(temperature=0.1, max_tokens= 1000, model_name="text-davinci-003")
prompt_template = """You are a biology textbook and helping students prepare for the exams. Use the following pieces of context to answer the question in details  at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in details and in bullet points \n\n :"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

prompt_chapters = """give me a list of all  chapters from the table of contents in this book as a CSV. Print only the chapter names"""
promptchaps = "give me the names of all chapters from the table of contents separated by commas."
CHAPPROMPT = PromptTemplate(
    template=promptchaps, input_variables=[])

prompttopics = "give me all key topics for the chapter on {chapter} in a CSV format"
TOPICPROMPT = PromptTemplate(
    template=prompttopics, input_variables=["chapter"])

strtry = "1 Energy, 2 Transformation, 3 Form, 4 Function, 5 Movement, 6 Interaction, 7 Balance, 8 Environment, 9 Patterns, 10 Consequences, 11 Evidence, 12 Models."

#simplechain = LLMChain(llm,prompt_template= CHAPPROMPT)
@st.cache_resource
def getChapterNames():
    chain_type_kwargs = {"prompt": CHAPPROMPT}
    #, chain_type_kwargs=chain_type_kwargs
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)
    resp = qa.run(promptchaps)
    strresp = StringIO(resp)
    cdf = pd.read_csv(strresp, sep=",")
    return cdf
    
def getTopicsByChapter(chapter):
    #give me all key topics for the chapter on energy in a CSV format"
    st.write(f'chapter is now {chapter}')
    pp={"prompt":TOPICPROMPT}
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)
    #st.write(qa)
    resp = qa.run(query=prompttopics,chain_type_kwargs=pp)
    strresp = StringIO(resp)
    tdf = pd.read_csv(strresp, sep=",")
    return tdf

@st.cache_resource
def ReadBioBookandCreateVectorStore():
    loader = PyMuPDFLoader("data/biotxtbookIB.pdf")
    doc = loader.load()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch

docsearch = ReadBioBookandCreateVectorStore()

#cnames = getChapterNames()
#st.write(cnames)
#with st.container():
    #for cname in cnames.columns:
        #st.button(cname)
    #selchapter = st.selectbox('Select a chapter',cnames.columns)
    #topics = getTopicsByChapter(selchapter)
    #st.write(topics)
    #seltopic = st.selectbox('Select a topic',topics.columns)
    #st.write(f'selected chapter: {selchapter} and selected topic {seltopic}')
chain_type_kwargs = {"prompt": PROMPT}
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, chain_type_kwargs=chain_type_kwargs)
st.header("MYP4-5 Biology Texbook")
query = st.text_area("Please enter a query for MYP 4-5 Biology Textbook ")

if st.button("Answer", "primary"):
    #qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, chain_type_kwargs=chain_type_kwargs)
    result = qa.run(query)
    st.write(result)
    st.download_button('Download result', result)

#resp = qa.run(query)
#print(resp)




