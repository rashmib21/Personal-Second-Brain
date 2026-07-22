#it is mainly responsible for redisearch index that makes semantic (vector) search possible
import redis
from redis.commands.search.field import VectorField, TextField #(the paramaters are path, file type, embedding-vector - the first two is text and third one is vector)

