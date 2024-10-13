# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import weaviate
import uuid
import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from weaviate.classes.init import Auth


# SQLAlchemy setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    description = Column(String)
    manuals = relationship('Manual', back_populates='item')

class Manual(Base):
    __tablename__ = 'manuals'
    id = Column(Integer, primary_key=True)
    content = Column(String)
    item_id = Column(Integer, ForeignKey('items.id'))
    item = relationship('Item', back_populates='manuals')

class Purchase(Base):
    __tablename__ = 'purchases'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    item_id = Column(Integer, ForeignKey('items.id'))
    user = relationship('User')
    item = relationship('Item')

engine = create_engine('sqlite:///example.db')
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# Initialize LangChain Weaviate Vector Store



# Best practice: store your credentials in environment variables
wcd_url = os.environ["WEAVAITE_URL"]
wcd_api_key = os.environ["WEAVAITE_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key}  # Replace with your OpenAI API key
)


# Initialize the embedding model
embedding_model = OpenAIEmbeddings(openai_api_key = openai_api_key, model = "text-embedding-3-large")

# Initialize LangChain Weaviate vector store
vectorstore = Weaviate(
    client=weaviate_client,
    embedding=embedding_model,
    by_text=False  # Since we're providing embeddings
)

# Define the classes and properties
def create_weaviate_schema():
    class_definitions = []

    # User class
    user_class = {
        'class': 'User',
        'description': 'A user in the system',
        'vectorizer': 'none',
        'properties': [
            {'name': 'name', 'dataType': ['text']},
            {'name': 'id', 'dataType': ['int']},
        ]
    }
    class_definitions.append(user_class)

    # Item class
    item_class = {
        'class': 'Item',
        'description': 'An item in the system',
        'vectorizer': 'none',
        'properties': [
            {'name': 'description', 'dataType': ['text']},
            {'name': 'id', 'dataType': ['int']},
        ]
    }
    class_definitions.append(item_class)

    # Manual class
    manual_class = {
        'class': 'Manual',
        'description': 'A manual for an item',
        'vectorizer': 'none',
        'properties': [
            {'name': 'content', 'dataType': ['text']},
            {'name': 'item_id', 'dataType': ['int']},
            {'name': 'id', 'dataType': ['int']},
        ]
    }
    class_definitions.append(manual_class)

    # Purchase class
    purchase_class = {
        'class': 'Purchase',
        'description': 'A purchase in the system',
        'vectorizer': 'none',
        'properties': [
            {'name': 'user_id', 'dataType': ['int']},
            {'name': 'item_id', 'dataType': ['int']},
            {'name': 'id', 'dataType': ['int']},
        ]
    }
    class_definitions.append(purchase_class)

    # Create classes if they don't exist
    existing_classes = weaviate_client.schema.get()['classes']
    existing_class_names = [cls['class'] for cls in existing_classes]

    for class_def in class_definitions:
        if class_def['class'] not in existing_class_names:
            weaviate_client.schema.create_class(class_def)

create_weaviate_schema()

# Initialize the LLM
llm = OpenAI(openai_api_key='YOUR_OPENAI_API_KEY')

# FastAPI setup
app = FastAPI()

# Pydantic models
class UserCreate(BaseModel):
    name: str

class ItemCreate(BaseModel):
    description: str

class ItemUpdate(BaseModel):
    description: str

class ManualCreate(BaseModel):
    content: str
    item_id: int

class ManualUpdate(BaseModel):
    content: str

class PurchaseCreate(BaseModel):
    user_id: int
    item_id: int

# Create a user
@app.post('/users/')
def create_user(user: UserCreate):
    db = SessionLocal()
    db_user = User(name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Generate embedding for the user name
    embedding = embedding_model.embed_text(db_user.name)

    # Add user to Weaviate using LangChain vector store
    metadata = {'name': db_user.name, 'id': db_user.id}
    vectorstore.add_texts(
        texts=[db_user.name],
        metadatas=[metadata],
        ids=[f'User_{db_user.id}'],
        namespace='User'
    )
    return db_user

# Delete a user
@app.delete('/users/{user_id}')
def delete_user(user_id: int):
    db = SessionLocal()
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail='User not found')
    db.delete(db_user)
    db.commit()

    # Delete user from Weaviate
    vectorstore.delete(ids=[f'User_{user_id}'], namespace='User')
    return {'detail': 'User deleted'}

# Create an item
@app.post('/items/')
def create_item(item: ItemCreate):
    db = SessionLocal()
    db_item = Item(description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    # Generate embedding for the item description
    embedding = embedding_model.embed_text(db_item.description)

    # Add item to Weaviate using LangChain vector store
    metadata = {'description': db_item.description, 'id': db_item.id}
    vectorstore.add_texts(
        texts=[db_item.description],
        metadatas=[metadata],
        ids=[f'Item_{db_item.id}'],
        namespace='Item'
    )
    return db_item

# Update an item
@app.put('/items/{item_id}')
def update_item(item_id: int, item: ItemUpdate):
    db = SessionLocal()
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail='Item not found')
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)

    # Update item in Weaviate
    # First delete the old entry
    vectorstore.delete(ids=[f'Item_{item_id}'], namespace='Item')

    # Add updated item
    metadata = {'description': db_item.description, 'id': db_item.id}
    vectorstore.add_texts(
        texts=[db_item.description],
        metadatas=[metadata],
        ids=[f'Item_{db_item.id}'],
        namespace='Item'
    )
    return db_item

# Delete an item
@app.delete('/items/{item_id}')
def delete_item(item_id: int):
    db = SessionLocal()
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail='Item not found')
    db.delete(db_item)
    db.commit()

    # Delete item from Weaviate
    vectorstore.delete(ids=[f'Item_{item_id}'], namespace='Item')
    return {'detail': 'Item deleted'}

# Create a manual
@app.post('/manuals/')
def create_manual(manual: ManualCreate):
    db = SessionLocal()
    db_manual = Manual(content=manual.content, item_id=manual.item_id)
    db.add(db_manual)
    db.commit()
    db.refresh(db_manual)

    # Generate embedding for the manual content
    embedding = embedding_model.embed_text(db_manual.content)

    # Add manual to Weaviate using LangChain vector store
    metadata = {'content': db_manual.content, 'item_id': db_manual.item_id, 'id': db_manual.id}
    vectorstore.add_texts(
        texts=[db_manual.content],
        metadatas=[metadata],
        ids=[f'Manual_{db_manual.id}'],
        namespace='Manual'
    )
    return db_manual

# Update a manual
@app.put('/manuals/{manual_id}')
def update_manual(manual_id: int, manual: ManualUpdate):
    db = SessionLocal()
    db_manual = db.query(Manual).filter(Manual.id == manual_id).first()
    if not db_manual:
        raise HTTPException(status_code=404, detail='Manual not found')
    db_manual.content = manual.content
    db.commit()
    db.refresh(db_manual)

    # Update manual in Weaviate
    # First delete the old entry
    vectorstore.delete(ids=[f'Manual_{manual_id}'], namespace='Manual')

    # Add updated manual
    metadata = {'content': db_manual.content, 'item_id': db_manual.item_id, 'id': db_manual.id}
    vectorstore.add_texts(
        texts=[db_manual.content],
        metadatas=[metadata],
        ids=[f'Manual_{db_manual.id}'],
        namespace='Manual'
    )
    return db_manual

# Delete a manual
@app.delete('/manuals/{manual_id}')
def delete_manual(manual_id: int):
    db = SessionLocal()
    db_manual = db.query(Manual).filter(Manual.id == manual_id).first()
    if not db_manual:
        raise HTTPException(status_code=404, detail='Manual not found')
    db.delete(db_manual)
    db.commit()

    # Delete manual from Weaviate
    vectorstore.delete(ids=[f'Manual_{manual_id}'], namespace='Manual')
    return {'detail': 'Manual deleted'}

# Create a purchase
@app.post('/purchases/')
def create_purchase(purchase: PurchaseCreate):
    db = SessionLocal()
    db_purchase = Purchase(user_id=purchase.user_id, item_id=purchase.item_id)
    db.add(db_purchase)
    db.commit()
    db.refresh(db_purchase)

    # Generate embedding for the purchase
    purchase_text = f"User ID: {db_purchase.user_id}, Item ID: {db_purchase.item_id}"
    embedding = embedding_model.embed_text(purchase_text)

    # Add purchase to Weaviate using LangChain vector store
    metadata = {'user_id': db_purchase.user_id, 'item_id': db_purchase.item_id, 'id': db_purchase.id}
    vectorstore.add_texts(
        texts=[purchase_text],
        metadatas=[metadata],
        ids=[f'Purchase_{db_purchase.id}'],
        namespace='Purchase'
    )
    return db_purchase

# Search endpoint
@app.get('/search/')
def search(query: str):
    embedding = embedding_model.embed_query(query)

    # Search across 'Item' and 'Manual' namespaces
    results = []

    for namespace in ['Item', 'Manual']:
        matches = vectorstore.similarity_search_with_score_by_vector(
            embedding=embedding,
            namespace=namespace,
            k=5
        )
        for doc, score in matches:
            metadata = doc.metadata
            metadata['class'] = namespace
            metadata['score'] = score
            results.append(metadata)

    if not results:
        return {'answer': 'No relevant results found.'}

    # Prepare the context for the LLM
    context = ""
    for item in results:
        if item['class'] == 'Item':
            context += f"Item: {item.get('description', '')}\n"
        elif item['class'] == 'Manual':
            context += f"Manual Content: {item.get('content', '')}\n"

    # Construct the prompt
    prompt = f"""
You are an assistant that provides helpful answers based on the provided information.

Information:
{context}

Question: '{query}'

Answer the question using the information above. If the information is insufficient, say so.
"""

    # Generate the answer using the LLM
    answer = llm.predict(prompt).strip()

    return {'answer': answer}
