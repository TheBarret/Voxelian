import os
import struct
import zlib
import numpy as np
import secrets
import string
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum

# Visualizer imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

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

def visualize_message(encoder, message: bytes, cube_ids):
    """
    Scientific visualization of binary data as 3D cube representations.
    Shows the spatial encoding of bytes through cube edge configurations.
    """
    # Enhanced figure setup
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Base64 encode the message for processing
    b64_string = message #base64.b64encode(message).decode('ascii')
    encoded_ids = cube_ids #encoder.encode_text(message)
    
    base_vertices = np.array(encoder.rotations.vertices)
    
    # Dynamic grid calculation for better layout
    total_chars = len(b64_string)
    cubes_per_row = min(12, max(6, int(np.sqrt(total_chars * 1.5))))
    num_rows = (total_chars + cubes_per_row - 1) // cubes_per_row
    
    # color schemes
    vertex_color = '#1f77b4'
    edge_colors = ['#ff7f0e', '#d62728', '#2ca02c', '#9467bd']  # Distinct colors for edge density
    
    for i, (b64_char, cube_id) in enumerate(zip(b64_string, encoded_ids)):
        cube = encoder.get_cube_by_id(cube_id)
        
        # grid position with enhanced spacing
        row = i // cubes_per_row
        col = i % cubes_per_row
        
        # spacing for better visual separation
        offset = np.array([col * 2.0, -row * 2.0, 0])
        
        # vertices with enhanced styling
        verts = base_vertices + offset
        ax.scatter(verts[:,0], verts[:,1], verts[:,2], 
                  color=vertex_color, s=5, alpha=0.8, edgecolors='black', linewidths=0.2)
        
        # interactive edges
        edge_count = len(cube.edges)
        edge_color = edge_colors[min(edge_count // 3, 3)]   # Color by density
        edge_width = 1.0 + (edge_count / 12.0) * 1.5        # Thickness by connectivity
        
        for edge_idx in cube.edges:
            a, b = encoder.rotations.edges[edge_idx] 
            v1, v2 = verts[a], verts[b]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                   color=edge_color, linewidth=edge_width, alpha=0.85)
    ax.set_xlabel('X Coordinate', fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontweight='bold') 
    ax.set_zlabel('Z Coordinate', fontweight='bold')
    ax.set_box_aspect([1, 1, 0.5])
    
    # Scientific title with statistics
    original_size = len(message)
    b64_size = len(b64_string)
    compression_info = f"{original_size}→{b64_size} chars" if b64_size != original_size else f"{original_size} bytes"
    ax.set_title(f'Data: {compression_info} | Cubes: {len(encoded_ids)}', fontsize=11, pad=20)
    stats_text = (f'Encoding Statistics:\n'
                 f'• Original bytes: {original_size}\n'
                 f'• Base64 chars: {b64_size}\n' 
                 f'• Cube instances: {len(encoded_ids)}\n'
                 f'• Unique cube IDs: {len(set(encoded_ids))}\n'
                 f'• Edge density range: {min(len(encoder.get_cube_by_id(cid).edges) for cid in encoded_ids)}-'
                 f'{max(len(encoder.get_cube_by_id(cid).edges) for cid in encoded_ids)} edges')
    
    # position statistics in corner
    fig.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=edge_colors[0], label='1-3 edges'),
        Rectangle((0, 0), 1, 1, facecolor=edge_colors[1], label='4-6 edges'),
        Rectangle((0, 0), 1, 1, facecolor=edge_colors[2], label='7-9 edges'), 
        Rectangle((0, 0), 1, 1, facecolor=edge_colors[3], label='10-12 edges')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # viewing angle for 3D perspective
    ax.view_init(elev=20, azim=15)
    
    plt.tight_layout()
    plt.savefig('output.png')

class SerializationError(Exception):
    # exception for serialization/deserialization errors
    pass

class CubeFormat(IntEnum):
    # two versions
    V1_BASIC = 1
    V2_EXTENDED = 2

@dataclass
class CubeMetadata:
    """Metadata container for cube serialization"""
    encoder_version: str
    original_size: int
    encoding_type: str
    cube_count: int
    format_version: int = CubeFormat.V1_BASIC
    compression: Optional[str] = None
    timestamp: Optional[int] = None
    custom_fields: Optional[Dict[str, Any]] = None

class CubeSerializer:
    """
    Binary serialization module for Sol DeWit's cube encoder.
    Provides compact, efficient storage and transmission format for cube data.
    """
    
    MAGIC_NUMBER = b'CUBE'
    CURRENT_VERSION = CubeFormat.V2_EXTENDED
    MAX_CUBE_ID = 65535
    
    def __init__(self):
        self.format_handlers = {
            CubeFormat.V1_BASIC: (self._serialize_v1, self._deserialize_v1),
            CubeFormat.V2_EXTENDED: (self._serialize_v2, self._deserialize_v2)
        }
    
    def serialize(self, cube_ids: List[int], metadata: CubeMetadata) -> bytes:
        """
        Serialize cube IDs and metadata to binary format.
        
        Args:
            cube_ids: List of cube IDs (0-65535)
            metadata: Serialization metadata
            
        Returns:
            Binary serialized data
            
        Raises:
            SerializationError: If cube IDs exceed limits or format unsupported
        """
        # Validate input
        if not cube_ids:
            raise SerializationError("Cannot serialize empty cube list")
        
        if max(cube_ids) > self.MAX_CUBE_ID:
            raise SerializationError(f"Cube ID exceeds maximum ({self.MAX_CUBE_ID})")
        
        if metadata.format_version not in self.format_handlers:
            raise SerializationError(f"Unsupported format version: {metadata.format_version}")
        
        # Update metadata
        metadata.cube_count = len(cube_ids)
        
        # Delegate to version-specific handler
        serializer, _ = self.format_handlers[metadata.format_version]
        return serializer(cube_ids, metadata)
    
    def deserialize(self, data: bytes) -> tuple[List[int], CubeMetadata]:
        """
        Deserialize binary data to cube IDs and metadata.
        
        Args:
            data: Binary serialized data
            
        Returns:
            Tuple of (cube_ids, metadata)
            
        Raises:
            SerializationError: If data corrupted or format unsupported
        """
        if len(data) < 8:
            raise SerializationError("Data too short for valid cube format")
        
        # Read magic number and version
        magic, version = struct.unpack('>4sB', data[:5])
        
        if magic != self.MAGIC_NUMBER:
            raise SerializationError(f"Invalid magic number: {magic}")
        
        if version not in self.format_handlers:
            raise SerializationError(f"Unsupported format version: {version}")
        
        # Delegate to version-specific handler
        _, deserializer = self.format_handlers[version]
        return deserializer(data)
    
    def _serialize_v1(self, cube_ids: List[int], metadata: CubeMetadata) -> bytes:
        """V1 Format: Basic header + cube data"""
        # Pack cube data
        cube_data = struct.pack(f'>{len(cube_ids)}H', *cube_ids)
        checksum = zlib.crc32(cube_data) & 0xffffffff
        
        # V1 Header: MAGIC(4) + VERSION(1) + COUNT(4) + CHECKSUM(4) = 13 bytes
        header = struct.pack('>4sBI I', 
                           self.MAGIC_NUMBER, 
                           metadata.format_version,
                           len(cube_ids),
                           checksum)
        
        return header + cube_data
    
    def _serialize_v2(self, cube_ids: List[int], metadata: CubeMetadata) -> bytes:
        """V2 Format: Extended header with metadata"""
        # Pack cube data
        cube_data = struct.pack(f'>{len(cube_ids)}H', *cube_ids)
        
        # Encode metadata strings
        encoder_version_bytes = metadata.encoder_version.encode('utf-8')
        encoding_type_bytes = metadata.encoding_type.encode('utf-8')
        
        # V2 Extended metadata block
        meta_block = struct.pack('>BB', 
                                len(encoder_version_bytes),
                                len(encoding_type_bytes))
        meta_block += encoder_version_bytes + encoding_type_bytes
        meta_block += struct.pack('>I', metadata.original_size)
        
        # Calculate checksums
        meta_checksum = zlib.crc32(meta_block) & 0xffffffff
        data_checksum = zlib.crc32(cube_data) & 0xffffffff
        
        # V2 Header: MAGIC(4) + VERSION(1) + COUNT(4) + META_LEN(2) + META_CRC(4) + DATA_CRC(4) = 19 bytes
        header = struct.pack('>4sBI H II',
                           self.MAGIC_NUMBER,
                           metadata.format_version, 
                           len(cube_ids),
                           len(meta_block),
                           meta_checksum,
                           data_checksum)
        
        return header + meta_block + cube_data
    
    def _deserialize_v1(self, data: bytes) -> tuple[List[int], CubeMetadata]:
        """Deserialize V1 format"""
        # Parse header
        magic, version, count, checksum = struct.unpack('>4sBI I', data[:13])
        
        # Extract cube data
        cube_data = data[13:13 + (count * 2)]
        if len(cube_data) != count * 2:
            raise SerializationError("Truncated cube data")
        
        # Verify checksum
        if zlib.crc32(cube_data) & 0xffffffff != checksum:
            raise SerializationError("Cube data checksum mismatch")
        
        # Unpack cube IDs
        cube_ids = list(struct.unpack(f'>{count}H', cube_data))
        
        # Create minimal metadata for V1
        metadata = CubeMetadata(
            encoder_version="unknown",
            original_size=0,
            encoding_type="unknown",
            cube_count=count,
            format_version=version
        )
        
        return cube_ids, metadata
    
    def _deserialize_v2(self, data: bytes) -> tuple[List[int], CubeMetadata]:
        """Deserialize V2 format"""
        # Parse header
        header_data = data[:19]
        magic, version, count, meta_len, meta_checksum, data_checksum = \
            struct.unpack('>4sBI H II', header_data)
        
        # Extract and verify metadata block
        meta_block = data[19:19 + meta_len]
        if len(meta_block) != meta_len:
            raise SerializationError("Truncated metadata block")
        
        if zlib.crc32(meta_block) & 0xffffffff != meta_checksum:
            raise SerializationError("Metadata checksum mismatch")
        
        # Parse metadata
        encoder_len, encoding_len = struct.unpack('>BB', meta_block[:2])
        offset = 2
        
        encoder_version = meta_block[offset:offset + encoder_len].decode('utf-8')
        offset += encoder_len
        
        encoding_type = meta_block[offset:offset + encoding_len].decode('utf-8')
        offset += encoding_len
        
        original_size, = struct.unpack('>I', meta_block[offset:offset + 4])
        
        # Extract and verify cube data
        cube_data_start = 19 + meta_len
        cube_data = data[cube_data_start:cube_data_start + (count * 2)]
        
        if len(cube_data) != count * 2:
            raise SerializationError("Truncated cube data")
        
        if zlib.crc32(cube_data) & 0xffffffff != data_checksum:
            raise SerializationError("Cube data checksum mismatch")
        
        # Unpack cube IDs
        cube_ids = list(struct.unpack(f'>{count}H', cube_data))
        
        # Create metadata object
        metadata = CubeMetadata(
            encoder_version=encoder_version,
            original_size=original_size,
            encoding_type=encoding_type,
            cube_count=count,
            format_version=version
        )
        
        return cube_ids, metadata
    
    def get_format_info(self, data: bytes) -> Dict[str, Any]:
        """
        Extract format information without full deserialization.
        
        Args:
            data: Binary serialized data
            
        Returns:
            Dictionary with format information
        """
        if len(data) < 8:
            return {"error": "Invalid data length"}
        
        magic, version = struct.unpack('>4sB', data[:5])
        
        if magic != self.MAGIC_NUMBER:
            return {"error": f"Invalid magic number: {magic}"}
        
        try:
            if version == CubeFormat.V1_BASIC:
                _, _, count, _ = struct.unpack('>4sBI I', data[:13])
                return {
                    "format": "Cube1",
                    "version": version,
                    "cube_count": count,
                    "size_bytes": len(data),
                    "has_metadata": False
                }
            elif version == CubeFormat.V2_EXTENDED:
                _, _, count, meta_len, _, _ = struct.unpack('>4sBI H II', data[:19])
                return {
                    "format": "Cube2", 
                    "version": version,
                    "cube_count": count,
                    "metadata_size": meta_len,
                    "size_bytes": len(data),
                    "has_metadata": True
                }
            else:
                return {"error": f"Unsupported version: {version}"}
        except struct.error:
            return {"error": "Corrupted header data"}

