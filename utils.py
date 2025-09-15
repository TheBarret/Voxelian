import os
import struct
import zlib
import numpy as np
import secrets
import string
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

def write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def read_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"File '{filename}' not found")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def generate_random_message(length: int = 512, table: str = "") -> str:
    """Generate pseudo-random text message of fixed size."""
    return ''.join(secrets.choice(table) for _ in range(length))
