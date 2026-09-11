"""Serial GPU consumer preserving the existing Studio SQS/S3 execution contract."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading

LOG = logging.getLogger(__name__)


def ensure_exclusive_queue(sqs, lambdas, queue_url):
    queue = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
    arn = queue["Attributes"]["QueueArn"]
    mappings = lambdas.get_paginator("list_event_source_mappings").paginate(EventSourceArn=arn)
    for page in mappings:
        for mapping in page.get("EventSourceMappings", []):
            if mapping["State"] != "Disabled":
                raise RuntimeError("A Lambda mapping still consumes this queue; use a dedicated GPU test queue or complete cutover")
    return arn


def process_message(sqs, queue_url, message, handler, *, heartbeat_seconds=60, visibility_seconds=300):
    """Never acknowledge on handler failure or lease-renewal failure."""
    stop = threading.Event()
    lease_errors = []

    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            try:
                sqs.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"],
                                              VisibilityTimeout=visibility_seconds)
            except Exception as exc:
                lease_errors.append(exc)
                LOG.exception("Queue lease renewal failed")
                return

    lease = threading.Thread(target=heartbeat, daemon=True)
    lease.start()
    try:
        event = {"Records": [{"eventSource": "aws:sqs", "body": message["Body"],
                              "messageId": message["MessageId"], "attributes": message.get("Attributes", {})}]}
        result = handler(event, None)
        if result.get("status") != "completed":
            raise RuntimeError("Worker did not confirm completed processing")
        stop.set()
        lease.join()
        if lease_errors:
            raise RuntimeError("Queue lease was lost; retaining message for idempotent retry") from lease_errors[0]
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
        return result
    finally:
        stop.set()
        lease.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-url", default=os.environ.get("OMNIVOICE_QUEUE_URL"))
    parser.add_argument("--run", action="store_true", help="Start consuming; default only validates queue ownership")
    args = parser.parse_args(argv)
    if not args.queue_url:
        parser.error("Set OMNIVOICE_QUEUE_URL to the dedicated GPU test FIFO")
    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_REGION", "us-east-1")
    sqs = boto3.client("sqs", region_name=region, config=Config(read_timeout=30, connect_timeout=5))
    arn = ensure_exclusive_queue(sqs, boto3.client("lambda", region_name=region), args.queue_url)
    print(json.dumps({"queue": arn, "mode": "run" if args.run else "check"}), flush=True)
    if not args.run:
        return 0
    if not os.environ.get("OMNIVOICE_REFERENCE_MANIFEST"):
        raise ValueError("Mount a transcript manifest and set OMNIVOICE_REFERENCE_MANIFEST before consuming")
    os.environ["NARRATION_ENGINE"] = "native"
    os.environ.setdefault("OMNIVOICE_CACHE_DIR", "/cache/raw")
    os.environ.setdefault("OMNIVOICE_REPORT_DIR", "/cache/reports")
    os.environ.setdefault("NARRATION_VERIFY_TEXT", "required")
    from speech_engine import create_engine

    create_engine("native")  # Load once, before receiving a message.
    from lambda_function import lambda_handler

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stop.set())
    while not stop.is_set():
        response = sqs.receive_message(QueueUrl=args.queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20,
                                       VisibilityTimeout=300, MessageSystemAttributeNames=["All"])
        for message in response.get("Messages", []):
            try:
                process_message(sqs, args.queue_url, message, lambda_handler)
            except Exception:
                LOG.exception("Generation failed; message retained for queue retry/DLQ policy")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
