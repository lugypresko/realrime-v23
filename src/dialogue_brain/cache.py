from __future__ import annotations

import threading
from collections import OrderedDict, deque
from typing import Deque, Generic, Hashable, Optional, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple thread-safe LRU cache with bounded size."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._store: "OrderedDict[K, V]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._store:
                return None
            # Move to end to mark as recently used
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


class LastSuggestionRing:
    """Ring buffer to avoid repeating the last few suggestions."""

    def __init__(self, max_items: int = 5):
        self._max_items = max_items
        self._buffer: Deque[str] = deque(maxlen=max_items)

    def push(self, suggestion: str) -> None:
        self._buffer.append(suggestion)

    def contains(self, suggestion: str) -> bool:
        return suggestion in self._buffer

    def recent(self) -> list[str]:
        return list(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

