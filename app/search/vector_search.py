import numpy as np
from redis.commands.search.query import Query

from app.storage.redis_client import r
from app.embeddings.embedding_router import generate_embedding
