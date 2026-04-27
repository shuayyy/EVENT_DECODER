import numpy as np


EVENT_DTYPE = np.dtype(
    {
        "names": [
            "x",
            "y",
            "polarity",
            "sensor_timestamp",
            "witnessed_utc_ns",
        ],
        "formats": ["<u2", "<u2", "i1", "<i4", "<i8"],
        "offsets": [0, 2, 4, 5, 9],
        "itemsize": 17,
    }
)
