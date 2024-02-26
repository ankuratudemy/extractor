import sys
from google.api_core import retry
from google.cloud import pubsub_v1

# Example usage
project_id = "structhub-412620"
subscription_id = "structhub-credits-usage-subscriber-pull-test"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

NUM_MESSAGES = 3

# Wrap the subscriber in a 'with' block to automatically call close() to
# close the underlying gRPC channel when done.
with subscriber:
    # The subscriber pulls a specific number of messages. The actual
    # number of messages pulled may be smaller than max_messages.
    response = subscriber.pull(
        request={"subscription": subscription_path, "max_messages": NUM_MESSAGES},
        retry=retry.Retry(
            initial=1,  # Initial delay in seconds
            maximum=10,  # Maximum delay in seconds
            multiplier=2,  # Factor to increase delay
            deadline=300,  # Total time for retries
        ),
    )

    if len(response.received_messages) == 0:
        sys.exit("No messages received.")

    ack_ids = []
    for received_message in response.received_messages:
        print(f"Received: {received_message.message.data}.")
        ack_ids.append(received_message.ack_id)

    # Acknowledges the received messages so they will not be sent again.
    subscriber.acknowledge(
        request={"subscription": subscription_path, "ack_ids": ack_ids}
    )

    print(
        f"Received and acknowledged {len(response.received_messages)} messages from {subscription_path}."
    )
