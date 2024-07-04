import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pickle

import warnings
warnings.filterwarnings("ignore")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

embedding_model = HuggingFaceEmbeddings()
vectordb_path = 'vector_db.pkl'

def create_vector_db():
    loader = CSVLoader(file_path='platform_faqs.csv', source_column='prompt', encoding='latin-1')
    data = loader.load()

    vector_db = FAISS.from_documents(documents=data, embedding=embedding_model)
    with open(vectordb_path, 'wb') as f:
        pickle.dump({
            'faiss_object': vector_db,  # Save the entire FAISS object
            'embedding_model': embedding_model,
            'data': data,
        }, f)

# print("done")

def get_chain():
    with open(vectordb_path, 'rb') as f:
        saved_data = pickle.load(f)

    # Recreate the vector_db object
    loaded_vector_db = saved_data['faiss_object']
    retriever = loaded_vector_db.as_retriever()
    # --------------------------------------------------------------
    # query = "can i add this to my resume and is emi available "
    # results = loaded_vector_db.search(query, k=5, search_type="similarity")  # Retrieve top 5 similar documents

    # Print the results
    # for i, result in enumerate(results):
    #     print(f"Result {i + 1}: {result}")
    # ----------------------------------------------------------------
        
    # prompt_template = """Given the following context and a question, generate an answer based on this context only.
    # Answer as if you are in between a conversation build up the answer with respect to complete given question. Use punctuations like  comma, full stop. 
    # If the question asked is out of provided context then say 'I apologize for inconvenience I don't have that information right now would you like to speak to our Senior for this query?' 

    # CONTEXT: {context}

    # QUESTION: {question}"""

    prompt_template = """Hey, I need your help with something!

    CONTEXT: {context}

    QUESTION: {question}

    Can you provide a detailed answer that sounds friendly and engaging? Also, feel free to add some follow-up questions to keep the conversation going. Thanks!
    """

    # prompt_template = """Imagine you're having a conversation with a friend who is interested in [topic]. They've been following [previous conversation/context] closely. Ask yourself, 'What would I tell them to make them understand this well?' Then, answer the following question for your friend:

    # Question: {question}

    # CONTEXT: {context}

    # I apologize if the answer goes beyond the provided context. If the question is outside this topic, answer with, "I apologize, but that information isn't included in our current conversation. Perhaps we can discuss something else related to [topic]?"
    # """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    
    return chain


if __name__ == "__main__":
    # create_vector_db() 
    while True:
        stop_words = ["stop", "exit", "quit"]
        querys = input("Enter your query: ").lower()
        if any(word in querys for word in stop_words):
            print("Thank you for reaching!")
            break
        chain=get_chain()
        response = chain(querys)
        print(response['result'])
