# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import weaviate
import weaviate.classes as wvc
import os
from weaviate.classes.config import Configure
from langchain.llms import OpenAI
from weaviate.classes.init import Auth
from langchain.embeddings.openai import OpenAIEmbeddings

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

# Initialize Weaviate client
WCD_URL = os.environ["WEAVAITE_URL"]
WCD_API_KEY = os.environ["WEAVAITE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCD_URL,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(WCD_API_KEY),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': OPENAI_API_KEY}  # Replace with your OpenAI API key
)

# Initialize the embedding model
embedding_model = Configure.Vectorizer.text2vec_openai()
# embeddings_openai =  OpenAIEmbeddings( model ="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)

# Create Weaviate schema
def create_weaviate_schema():
    client.collections.delete("User")
    client.collections.delete("Item")
    client.collections.delete("Manual")
    client.collections.delete("Purchase")

    client.collections.create(
        name="User",
        properties=[
            wvc.config.Property(name="name", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="id", data_type=wvc.config.DataType.INT)
        ],
        vectorizer_config=embedding_model
    )

    client.collections.create(
        name="Item",
        properties=[
            wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="id", data_type=wvc.config.DataType.INT)
        ],
        vectorizer_config=embedding_model
    )

    client.collections.create(
        name="Manual",
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="item_id", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="id", data_type=wvc.config.DataType.INT)
        ],
        vectorizer_config=embedding_model
    )

    client.collections.create(
        name="Purchase",
        properties=[
            wvc.config.Property(name="user_id", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="item_id", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="id", data_type=wvc.config.DataType.INT)
        ],
        vectorizer_config=embedding_model
    )

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

    # Add user to Weaviate
    client.collections.get("User").data.insert({
        "name": db_user.name,
        "id": db_user.id
    })
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
    client.collections.get("User").data.delete(
        where={"path": ["id"], "operator": "Equal", "valueInt": user_id}
    )
    return {'detail': 'User deleted'}

# Create an item
@app.post('/items/')
def create_item(item: ItemCreate):
    db = SessionLocal()
    db_item = Item(description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    # Add item to Weaviate
    client.collections.get("Item").data.insert({
        "description": db_item.description,
        "id": db_item.id
    })
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
    client.collections.get("Item").data.update(
        where={"path": ["id"], "operator": "Equal", "valueInt": item_id},
        properties={"description": db_item.description}
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
    client.collections.get("Item").data.delete(
        where={"path": ["id"], "operator": "Equal", "valueInt": item_id}
    )
    return {'detail': 'Item deleted'}

# Create a manual
@app.post('/manuals/')
def create_manual(manual: ManualCreate):
    db = SessionLocal()
    db_manual = Manual(content=manual.content, item_id=manual.item_id)
    db.add(db_manual)
    db.commit()
    db.refresh(db_manual)

    # Add manual to Weaviate
    client.collections.get("Manual").data.insert({
        "content": db_manual.content,
        "item_id": db_manual.item_id,
        "id": db_manual.id
    })
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
    client.collections.get("Manual").data.update(
        where={"path": ["id"], "operator": "Equal", "valueInt": manual_id},
        properties={"content": db_manual.content}
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
    client.collections.get("Manual").data.delete(
        where={"path": ["id"], "operator": "Equal", "valueInt": manual_id}
    )
    return {'detail': 'Manual deleted'}

# Create a purchase
@app.post('/purchases/')
def create_purchase(purchase: PurchaseCreate):
    db = SessionLocal()
    db_purchase = Purchase(user_id=purchase.user_id, item_id=purchase.item_id)
    db.add(db_purchase)
    db.commit()
    db.refresh(db_purchase)

    # Add purchase to Weaviate
    client.collections.get("Purchase").data.insert({
        "user_id": db_purchase.user_id,
        "item_id": db_purchase.item_id,
        "id": db_purchase.id
    })
    return db_purchase

# Search endpoint
@app.get('/search/')
def search(query: str):
    # Search across 'Item' and 'Manual' collections
    results = []

    for collection_name in ['Item', 'Manual']:
        collection = client.collections.get(collection_name)
        response = collection.query.near_text(
            query=query,
            limit=5,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        for obj in response.objects:
            results.append({
                "class": collection_name,
                "properties": obj.properties,
                "score": obj.metadata.distance
            })

    if not results:
        return {'answer': 'No relevant results found.'}

    # Prepare the context for the LLM
    context = ""
    for item in results:
        if item['class'] == 'Item':
            context += f"Item: {item['properties'].get('description', '')}\n"
        elif item['class'] == 'Manual':
            context += f"Manual Content: {item['properties'].get('content', '')}\n"

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