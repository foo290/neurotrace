"""
Long Term Memory (LTM) module for neurotrace.
"""
from sqlalchemy.testing.suite.test_reflection import metadata


class LongTermMemory:
    def __init__(self):
        pass

    msm_format = [
        {"role": "user", "content": "Hello", "timestamp": "...", 'metadata': {}},
        {"role": "ai", "content": "Hi!", "timestamp": "...", 'metadata': {}}
    ]
