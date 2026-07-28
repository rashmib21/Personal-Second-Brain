from kafka import KafkaConsumer
import json
import os
from config import KAFKA_BOOTSTRAP_SERVERS
from app.utils.file_types import FILE_TYPES

def value_deserializer(v):
	return json.loads(v.decode("utf-8"))

consumer=KafkaConsumer(
	'file-events',
	bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
	value_deserializer=value_deserializer,
	group_id='second-brain-workers',
	auto_offset_reset='earliest'
	)

print("Consumer started... Waiting for events...\n")

#Listen continously for new events
for message in consumer:
	event=message.value

	print("="*60)
	print("Received Event: ",event)		

	#Validate event contains path
	if "path" not in event:
		print("Invalid event. Missing 'path'")
		continue
	path=event['path']
	

	#Check whether file actually exists
	if not os.path.exists(path):
		print(f"File not found: {path}")	
		continue

	#Determine the file extension
	file_type=FILE_TYPES.get(extension)

	if file_type is None:
		print(f"Unsupported file type: {extension}")
		continue	

	#Add useful metadata to event
	event['extension']=extension
	event['file_type']=file_type
	event['file_size']=os.path.getsize(path)

	#Print final validated event
	print("\nValidated event: ")
	print(event)

	# TASKS = {
	#     "pdf": process_pdf,
	#     "document": process_document,
	#     "text": process_text,
	#     "image": process_image,
	#     "audio": process_audio,
	#     "video": process_video,
	# }

