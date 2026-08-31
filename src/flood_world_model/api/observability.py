from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)


REQUEST_COUNT = Counter(
    "flood_api_requests_total",
    "Total number of API requests.",
    [
        "method",
        "path",
        "status",
    ],
)

REQUEST_LATENCY = Histogram(
    "flood_api_request_duration_seconds",
    "API request duration in seconds.",
    [
        "method",
        "path",
    ],
)

EVACUATION_COUNT = Counter(
    "flood_api_evacuations_total",
    "Total evacuation requests.",
    [
        "status",
    ],
)

ROUTE_COUNT = Counter(
    "flood_api_routes_total",
    "Total route requests.",
    [
        "status",
    ],
)

HAZARD_COUNT = Counter(
    "flood_api_hazard_queries_total",
    "Total hazard queries.",
    [
        "status",
    ],
)


class JsonFormatter(
    logging.Formatter
):
    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload = {
            "timestamp": self.formatTime(
                record,
                "%Y-%m-%dT%H:%M:%S%z",
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(
            record,
            "request_id",
            None,
        )

        if request_id is not None:
            payload[
                "request_id"
            ] = request_id

        return json.dumps(
            payload,
            ensure_ascii=False,
        )


def configure_logging() -> None:

    root = logging.getLogger()

    root.setLevel(
        logging.INFO
    )

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(
                JsonFormatter()
            )

        return

    handler = logging.StreamHandler()

    handler.setFormatter(
        JsonFormatter()
    )

    root.addHandler(
        handler
    )


async def request_metrics_middleware(
    request: Request,
    call_next: Callable,
) -> Response:

    request_id = request.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        request_id = str(
            uuid.uuid4()
        )

    start = time.perf_counter()

    status_code = 500

    try:
        response = await call_next(
            request
        )

        status_code = response.status_code

        return response

    finally:

        elapsed = (
            time.perf_counter()
            - start
        )

        path = request.url.path
        method = request.method

        REQUEST_COUNT.labels(
            method=method,
            path=path,
            status=str(
                status_code
            ),
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            path=path,
        ).observe(
            elapsed
        )

        response_headers = getattr(
            locals().get(
                "response",
                None
            ),
            "headers",
            None,
        )

        if response_headers is not None:
            response_headers[
                "X-Request-ID"
            ] = request_id

        logger = logging.getLogger(
            "flood_world_model.api.request"
        )

        logger.info(
            (
                "%s %s -> %s "
                "(%.4fs)"
            ),
            method,
            path,
            status_code,
            elapsed,
            extra={
                "request_id": request_id,
            },
        )


def metrics_response() -> Response:

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
