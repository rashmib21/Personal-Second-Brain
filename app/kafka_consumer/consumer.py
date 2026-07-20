from kafka import KafkaConsumer
import json
from config import KAFKA_BOOTSTRAP_SERVERS

def value_deserializer(v):
	return json.loads(v).decode('utf-8')

consumer=KafkaConsumer(
	'file-events'
	bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
	value_deserializer=value_deserializer,
	group_id='second-brain-workers',
	auto_offset_reset='earliest'
	)

for message in consumer:
	event=message.value

	print("Received Event: ",event)		