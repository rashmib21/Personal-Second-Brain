from kafka import KafkaConsumer
import json
import os

from config import KAFKA_BOOTSTRAP_SERVERS
from app.utils.file_types import FILE_TYPES

# Celery Router
from app.celery_app.router import route_file


def value_deserializer(value):
    # Convert Kafka bytes into a Python dictionary

    return json.loads(value.decode("utf-8"))


consumer = KafkaConsumer(
    "file-events",
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=value_deserializer,
    group_id="second-brain-workers",
    auto_offset_reset="earliest",
)

print("Consumer started... Waiting for events...\n")


# Listen continuously for new events
for message in consumer:

    event = message.value

    print("=" * 70)
    print("Received Event:")
    print(event)

    # Validate Event
    if "path" not in event:
        print("Invalid event. Missing 'path'")
        continue

    path = event["path"]

    # Check whether the file exists

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    # Determine file extension
    extension = os.path.splitext(path)[1].lower()

    # Determine file type
    file_type = FILE_TYPES.get(extension)

    if file_type is None:
        print(f"Unsupported file type: {extension}")
        continue

    # Add useful metadata
    event["extension"] = extension
    event["file_type"] = file_type
    event["file_size"] = os.path.getsize(path)

    # Validated Event
    print("\nValidated Event:")
    print(event)

    # Send to Celery

    route_file.delay(event)

    print(f"\nTask submitted to Celery ({file_type})")