from kafka import KafkaProducer
import json
from config import KAFKA_BOOTSTRAP_SERVERS


def value_serializer(v):
	return json.dumps(v).encode('utf-8')

def key_serializer(v):
	return json.dumps(v).encode('utf-8')


producer=KafkaProducer(
	bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
	value_serializer=value_serializer,
	key_serializer=key_serializer
	)	

def publish_file_event(event_type, path):
	#create event data
	event={
	"event":event_type,
	"path":path,
	"timestamp":time.time()
	}

	#Send event to kafka producer
	producer.send(
		"file-events", #kafka topic
		key=path,
		value=event
		)

	#check message is send
	producer.flush()
