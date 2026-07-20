from kafka import KafkaProducer
import json

def serializer(v):
	return json.dumps(v).encode('utf-8')

	