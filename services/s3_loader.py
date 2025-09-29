"""Utilities for streaming documents from S3."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


@dataclass
class S3Document:
    """Simple container for S3 document payloads."""

    key: str
    text: str


class S3DocumentLoader:
    """Stream text objects from an S3 bucket."""

    def __init__(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
        page_size: int = 1000,
    ) -> None:
        self.bucket = bucket_name
        self.prefix = prefix
        self.client = s3_client or boto3.client("s3")
        self.page_size = page_size

    def iter_documents(self) -> Generator[S3Document, None, None]:
        """Yield documents from S3, streaming the object body."""
        continuation_token: Optional[str] = None
        while True:
            try:
                params = {"Bucket": self.bucket, "MaxKeys": self.page_size}
                if self.prefix:
                    params["Prefix"] = self.prefix
                if continuation_token:
                    params["ContinuationToken"] = continuation_token
                response = self.client.list_objects_v2(**params)
            except (BotoCoreError, ClientError) as exc:
                logger.error("Failed to list objects in bucket %s: %s", self.bucket, exc)
                raise

            for obj in response.get("Contents", []):
                key = obj["Key"]
                try:
                    s3_obj = self.client.get_object(Bucket=self.bucket, Key=key)
                    body = s3_obj["Body"].read()
                    text = body.decode("utf-8")
                except (BotoCoreError, ClientError, UnicodeDecodeError) as exc:
                    logger.warning("Skipping %s due to read/decode error: %s", key, exc)
                    continue
                yield S3Document(key=key, text=text)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
