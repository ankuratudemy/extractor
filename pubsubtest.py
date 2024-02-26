from google.cloud import pubsub_v1

def pull_messages(project_id: str, subscription_id: str):

    topic_name = 'projects/{project_id}/topics/{topic}'.format(
         project_id=project_id,
        topic='structhub-credit-usage-stage',  # Set this to something appropriate.
    )
    print(topic_name)
    subscription_name = 'projects/{project_id}/subscriptions/{sub}'.format(
        project_id=project_id,
        sub=subscription_id,  # Set this to something appropriate.
    )
    print(subscription_name)
    
    def callback(message):
        print(message.data)
        message.ack()

    with pubsub_v1.SubscriberClient() as subscriber:
        subscriber.create_subscription(
            name=subscription_name, topic=topic_name)
        future = subscriber.subscribe(subscription_name, callback)

# Example usage
project_id = "structhub-412620"
subscription_id = "structhub-credits-usage-subscriber-pull-test"

pull_messages(project_id, subscription_id)
