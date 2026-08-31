from __future__ import annotations

import asyncio


class PlanningLimiter:

    def __init__(
        self,
        max_concurrent: int,
    ) -> None:

        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1."
            )

        self._semaphore = asyncio.Semaphore(
            max_concurrent
        )

    async def __aenter__(
        self,
    ):
        await self._semaphore.acquire()

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self._semaphore.release()
