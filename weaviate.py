# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
import weaviate
import weaviate.classes.config as wc

from database import SessionLocal, engine
import models
import schemas

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")  # Adjust URL if necessary

# Ensure Weaviate classes exist
def create_weaviate_schema():
    # Check if 'Item' class exists
    if not client.schema.contains({"classes": [{"class": "Item"}]}):
        client.collections.create(
            name="Item",
            properties=[
                wc.Property(name="item_id", data_type=wc.DataType.TEXT, skip_vectorization=True),
                wc.Property(name="name", data_type=wc.DataType.TEXT),
                wc.Property(name="description", data_type=wc.DataType.TEXT, optional=True),
                wc.Property(name="box", data_type=wc.DataType.TEXT, skip_vectorization=True),
                wc.Property(name="workspace", data_type=wc.DataType.TEXT, skip_vectorization=True),
            ],
            vectorizer_config=wc.Configure.Vectorizer.text2vec_cohere(),
            generative_config=wc.Configure.Generative.openai(),
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=wc.VectorDistances.COSINE,
            ),
        )

create_weaviate_schema()

# CRUD Endpoints

# Create Item
@app.post("/items/", response_model=schemas.ItemInDB)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    # Check if item exists
    db_item = db.query(models.Item).filter(models.Item.item_id == item.item_id).first()
    if db_item:
        raise HTTPException(status_code=400, detail="Item already exists")

    # Add item to SQL database
    db_item = models.Item(
        item_id=item.item_id,
        name=item.name,
        description=item.description,
        box=item.box,
        workspace=item.workspace,
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    # Add item to Weaviate
    client.data_object.create(
        data_object={
            "item_id": item.item_id,
            "name": item.name,
            "description": item.description,
            "box": item.box,
            "workspace": item.workspace,
        },
        class_name="Item",
        uuid=str(uuid4()),  # Generate a UUID for Weaviate object
    )

    return db_item

# Read Item
@app.get("/items/{item_id}", response_model=schemas.ItemInDB)
def read_item(item_id: str, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

# Update Item
@app.put("/items/{item_id}", response_model=schemas.ItemInDB)
def update_item(item_id: str, item: schemas.ItemUpdate, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    # Update SQL database
    for var, value in vars(item).items():
        if value is not None:
            setattr(db_item, var, value)
    db.commit()
    db.refresh(db_item)

    # Update Weaviate object
    # Find Weaviate object UUID
    where_filter = {
        "path": ["item_id"],
        "operator": "Equal",
        "valueText": item_id,
    }
    result = client.query.get(
        class_name="Item",
        properties=["_additional { id }"]
    ).with_where(where_filter).do()

    if result and result['data']['Get']['Item']:
        weaviate_id = result['data']['Get']['Item'][0]['_additional']['id']
        client.data_object.update(
            data_object={
                "name": item.name or db_item.name,
                "description": item.description or db_item.description,
                "box": item.box or db_item.box,
                "workspace": item.workspace or db_item.workspace,
            },
            class_name="Item",
            uuid=weaviate_id,
        )
    else:
        raise HTTPException(status_code=404, detail="Item not found in Weaviate")

    return db_item

# Delete Item
@app.delete("/items/{item_id}")
def delete_item(item_id: str, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    # Delete from SQL database
    db.delete(db_item)
    db.commit()

    # Delete from Weaviate
    # Find Weaviate object UUID
    where_filter = {
        "path": ["item_id"],
        "operator": "Equal",
        "valueText": item_id,
    }
    result = client.query.get(
        class_name="Item",
        properties=["_additional { id }"]
    ).with_where(where_filter).do()

    if result and result['data']['Get']['Item']:
        weaviate_id = result['data']['Get']['Item'][0]['_additional']['id']
        client.data_object.delete(
            class_name="Item",
            uuid=weaviate_id,
        )
    else:
        raise HTTPException(status_code=404, detail="Item not found in Weaviate")

    return {"detail": "Item deleted successfully"}

# List Items
@app.get("/items/", response_model=list[schemas.ItemInDB])
def list_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = db.query(models.Item).offset(skip).limit(limit).all()
    return items
