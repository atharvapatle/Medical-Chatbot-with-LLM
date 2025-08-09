from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
import uuid   
import secrets


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="openai/gpt-oss-20b:free")

# Store chat histories for each session
chat_histories = {}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']

    # Step 2: Create chat history for this session if not present
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()

    history = chat_histories[session_id]

    # Step 3: Get user message
    msg = request.form["msg"]
    print("User:", msg)

    # Step 4: Add user message to history
    history.add_user_message(msg)

    # Step 5: Run the RAG chain with current history
    response = rag_chain.invoke({
        "input": msg,
        "history": history.messages  # pass conversation so far
    })

    # Step 6: Add AI response to history
    ai_answer = response["answer"]
    history.add_ai_message(ai_answer)

    print("AI:", ai_answer)
    return str(ai_answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
