from types import SimpleNamespace

import pytest

from gpu_worker import ensure_exclusive_queue, process_message

MESSAGE = {"Body": "{}", "ReceiptHandle": "receipt", "MessageId": "message", "Attributes": {"ApproximateReceiveCount": "2"}}


def test_success_preserves_sqs_contract_then_acknowledges():
    deleted, events = [], []
    sqs = SimpleNamespace(delete_message=lambda **kw: deleted.append(kw))

    def handler(event, context):
        events.append(event)
        return {"status": "completed"}

    process_message(sqs, "queue", MESSAGE, handler)
    assert events[0]["Records"][0]["eventSource"] == "aws:sqs"
    assert events[0]["Records"][0]["attributes"]["ApproximateReceiveCount"] == "2"
    assert deleted == [{"QueueUrl": "queue", "ReceiptHandle": "receipt"}]


def test_failure_does_not_delete_message():
    sqs = SimpleNamespace(delete_message=lambda **kw: pytest.fail("Acknowledged failed job"))

    def handler(event, context):
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        process_message(sqs, "queue", MESSAGE, handler)


def test_active_lambda_consumer_blocks_gpu_cutover():
    sqs = SimpleNamespace(get_queue_attributes=lambda **kw: {"Attributes": {"QueueArn": "arn:test"}})
    paginator = SimpleNamespace(paginate=lambda **kw: [{"EventSourceMappings": [{"State": "Enabled"}]}])
    lambdas = SimpleNamespace(get_paginator=lambda name: paginator)
    with pytest.raises(RuntimeError, match="Lambda mapping"):
        ensure_exclusive_queue(sqs, lambdas, "queue")


def test_completed_handler_cannot_acknowledge_after_lease_failure():
    import threading

    attempted = threading.Event()

    def renew(**kwargs):
        attempted.set()
        raise RuntimeError("Expired receipt")

    def handler(event, context):
        assert attempted.wait(2)
        return {"status": "completed"}

    sqs = SimpleNamespace(change_message_visibility=renew,
                           delete_message=lambda **kwargs: pytest.fail("Deleted after lease failure"))
    with pytest.raises(RuntimeError, match="lease was lost"):
        process_message(sqs, "queue", MESSAGE, handler, heartbeat_seconds=.001)
