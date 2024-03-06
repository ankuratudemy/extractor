def publish_messages_with_retry_settings(project_id: str, topic_id: str, message, bearer_token) -> None:
    """Publishes messages with custom retry settings."""
    # [START pubsub_publisher_retry_settings]
    from google import api_core
    from google.cloud import pubsub_v1

    # TODO(developer)
    # project_id = "your-project-id"
    # topic_id = "your-topic-id"

    # Configure the retry settings. Defaults shown in comments are values applied
    # by the library by default, instead of default values in the Retry object.
    custom_retry = api_core.retry.Retry(
        initial=0.250,  # seconds (default: 0.1)
        maximum=90.0,  # seconds (default: 60.0)
        multiplier=1.45,  # default: 1.3
        deadline=300.0,  # seconds (default: 60.0)
        predicate=api_core.retry.if_exception_type(
            api_core.exceptions.Aborted,
            api_core.exceptions.DeadlineExceeded,
            api_core.exceptions.InternalServerError,
            api_core.exceptions.ResourceExhausted,
            api_core.exceptions.ServiceUnavailable,
            api_core.exceptions.Unknown,
            api_core.exceptions.Cancelled,
        ),
    )

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    data = message.encode("utf-8")
    future = publisher.publish(topic=topic_path, data=data, retry=custom_retry)
    print(future.result())
    print(f"Published messages with retry settings to {topic_path}.")
    # [END pubsub_publisher_retry_settings]