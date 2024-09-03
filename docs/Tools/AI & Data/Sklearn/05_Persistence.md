# Persistence

> Pickle is not safe

```python
import skops.io as sio
```

## Saving

```python
# from file
sio.dump(model, "model.skops")

# compression
from zipfile import ZIP_DEFLATED
sio.dump(model, "model.skops", compression=ZIP_DEFLATED, compresslevel=9)

# in-memory
serialized = sio.dumps(model)
```

```python
# from file
unknown_types = get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

# in-memory
unknown_types = get_untrusted_types(serialized)
model = sio.loads(serialized, trusted=unknown_types)
```