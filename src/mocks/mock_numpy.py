class MockArray(list):
    def __init__(self, data):
        super().__init__(data)

    def astype(self, dtype=None):
        return self

    def copy(self):
        return MockArray(self)


def zeros(length, dtype=None):
    return MockArray([0.0 for _ in range(length)])


def concatenate(sequences):
    combined = []
    for seq in sequences:
        combined.extend(seq)
    return MockArray(combined)


def array(values):
    return MockArray(list(values))


def percentile(values, percentile_value):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile_value / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def savez(path, **kwargs):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f)


def load(path):
    import json

    class _Loader(dict):
        def __getitem__(self, item):
            return super().__getitem__(item)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parsed = {}
    for k, v in data.items():
        if isinstance(v, str):
            parsed[k] = v
        else:
            parsed[k] = MockArray(v)
    return _Loader(parsed)


float32 = "float32"
ndarray = MockArray
