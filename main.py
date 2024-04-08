import os
import json
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import uvicorn
from pymongo import MongoClient
from fastapi import HTTPException, FastAPI, Query
from langchain.memory import ChatMessageHistory
from operator import itemgetter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


app = FastAPI()


client = MongoClient('client')
db = client['db']
collection = db['collection_name']

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


@app.post("/chat")
async def chat_response(user_id: str = Query(...), text: str = Query(...)):
    try:
        if not user_id or not text:
            return {"error": "User ID and Text are required."}, 400
        
        # Retrieve the conversation document for the user
        conversation_doc = collection.find_one({"user_id": user_id})
        if not conversation_doc:
            # If no conversation document exists for the user, create a new one
            conversation_doc = {"user_id": user_id, "history": []}

        history_values = conversation_doc.get('history', [])
        chat_history = ChatMessageHistory()
        for m in history_values:
            if m["role"] == "human":
                chat_history.add_user_message(m["content"])
            elif m["role"] == "ai":
                chat_history.add_ai_message(m["content"])
        
        model = ChatOpenAI(model="gpt-4-1106-preview")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful chatbot who answers based on the conversation history"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        memory = ConversationBufferMemory(return_messages=True, chat_memory=chat_history)
        chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | model
        )

        inputs = {"input": text}
        response = chain.invoke(inputs)
        
        # Append the new message to the conversation history
        conversation_doc["history"].append({"role": "human", "content": text})
        conversation_doc["history"].append({"role": "ai", "content": response.content})
        
        # Update the conversation document in the database
        collection.update_one({"user_id": user_id}, {"$set": conversation_doc}, upsert=True)

        return {"response": response.content, "history": conversation_doc["history"]}

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9091)
