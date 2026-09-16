"""In-process asyncio pub/sub broadcast bus.

Any number of subscribers can await messages published by a producer.
Messages are delivered to all current subscribers; slow subscribers drop
old messages when their internal queue is full (bounded, drop-oldest policy).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

_log = logging.getLogger(__name__)

_QUEUE_MAX_SIZE: int = 256


class Broadcaster:
    """In-process asyncio broadcast bus.

    Usage
    -----
    Producer::

        await bus.publish({"key": "value"})

    Consumer (async context manager)::

        async with bus.subscribe() as queue:
            while True:
                msg = await queue.get()
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    async def publish(self, msg: dict[str, Any]) -> None:
        """Broadcast *msg* to all active subscribers.

        If a subscriber's queue is full the oldest item is discarded before
        enqueuing the new message.

        Parameters
        ----------
        msg:
            Any JSON-serialisable dictionary.
        """
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                _log.debug("Broadcaster: queue full after drain attempt, dropping.")
                dead.append(q)
        for q in dead:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue]:
        """Async context manager that registers a new subscriber queue.

        Yields
        ------
        asyncio.Queue
            A bounded queue.  Each :meth:`publish` call enqueues one item.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._subscribers.append(q)
        try:
            yield q
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)


# Module-level singleton used by both the API server and the consumer runner.
bus: Broadcaster = Broadcaster()
