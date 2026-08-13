import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="kafka")

from kafka import KafkaProducer
import json
import time

from config import KAFKA_BOOTSTRAP_SERVERS



def value_serializer(v):
    return json.dumps(v).encode("utf-8")


def key_serializer(v):
    return str(v).encode("utf-8")


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=value_serializer,
    key_serializer=key_serializer,
)


def publish_file_event(event):
    # Add timestamp
    event["timestamp"] = time.time()

    try:
        future = producer.send(
            "file-events",
            key=event["path"],
            value=event,
        )

        metadata = future.get(timeout=10)

        print(
            f"Sent successfully -> "
            f"Topic: {metadata.topic}, "
            f"Partition: {metadata.partition}, "
            f"Offset: {metadata.offset}"
        )

    except Exception as e:
        print(f"Producer Error: {e}")

    producer.flush()