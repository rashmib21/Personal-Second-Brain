#this file is created to store embeddings permanently
import redis #Connect to redis database
import numpy as np #convert the embedding (python list) into binary bytes because redis expects vectors in byte format
