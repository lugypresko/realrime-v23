from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Dict, List


EventPayload = dict


class EventBus:
    """Thread-safe, bounded pub/sub event bus that drops oldest events when full."""

    def __init__(self, max_queue_size: int = 128, max_workers: int = 8):
        self.max_queue_size = max_queue_size
        self._subscribers: Dict[str, List[Callable[[EventPayload], None]]] = {}
        self._queues: Dict[str, Deque[EventPayload]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="event-bus")

    def subscribe(self, event_type: str, handler: Callable[[EventPayload], None]) -> None:
        with self._lock:
            handlers = self._subscribers.setdefault(event_type, [])
            handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[EventPayload], None]) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type)
            if not handlers:
                return
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                self._subscribers.pop(event_type, None)

    def publish(self, event_type: str, payload: EventPayload) -> None:
        callbacks: List[Callable[[EventPayload], None]]
        with self._lock:
            queue = self._queues.setdefault(event_type, deque(maxlen=self.max_queue_size))
            if len(queue) == queue.maxlen:
                queue.popleft()
            queue.append(payload)
            callbacks = list(self._subscribers.get(event_type, []))
        for callback in callbacks:
            self._executor.submit(callback, payload)

    def get_queue_snapshot(self, event_type: str) -> List[EventPayload]:
        with self._lock:
            queue = self._queues.get(event_type, deque())
            return list(queue)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
