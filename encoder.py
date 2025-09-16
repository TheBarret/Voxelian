# core imports
import sys
import os
import argparse
import unittest
import base64
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from itertools import chain, combinations

# Visualizer imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

# Uilitie imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import generate_random_message, write_file, read_file, VoxelianError

"""
Voxelian Encoder
Using Sol LeWitt's cube model for data encoding with 217 canonical cubes.

Mathematical Foundation:
- Cube: 8 vertices in {0,1}³, 12 edges, 24 rotational symmetries
- Edge subsets: 4095 non-empty configurations → 217 canonical forms
- Base64 encoding: 64 symbols mapped to most frequent canonical cubes
"""

# Version constants
VERSION             = "0.5.0"

# Encoding constants
DEFAULT_ENCODING    = "utf-8"
B64_CHARSET         = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="

# cube settings
CANONICAL_CUBES     = 217

class CubeRotations:
    """Handles generation and application of cube rotation mappings."""
    
    def __init__(self):
        # Cube vertices in integer coordinates
        self.vertices = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ])
        
        # Edges defined as vertex index pairs
        self.edges = [
            (0,1), (1,2), (2,3), (3,0),  # bottom face
            (4,5), (5,6), (6,7), (7,4),  # top face
            (0,4), (1,5), (2,6), (3,7)   # vertical edges
        ]
        
        self.rotations = self._generate_rotations()
        self._validate_rotations()

    def _generate_rotations(self) -> List[Dict[int, int]]:
        """Generate all 24 valid cube rotation mappings."""
        rotations = []
        seen_configurations = set()
        
        # Standard rotation matrices for cube symmetries
        axis_permutations = [
            [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]
        ]
        sign_combinations = [
            (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
        ]

        for perm in axis_permutations:
            for signs in sign_combinations:
                rotation_matrix = np.zeros((3,3), dtype=int)
                for i in range(3):
                    rotation_matrix[i, perm[i]] = signs[i]
                
                # Apply rotation around cube center
                transformed_vertices = np.dot(self.vertices - 0.5, rotation_matrix.T) + 0.5
                transformed_vertices = np.round(transformed_vertices).astype(int)
                
                # Check uniqueness
                config_key = tuple(transformed_vertices.flatten())
                if config_key in seen_configurations:
                    continue
                seen_configurations.add(config_key)
                
                # Create vertex mapping
                vertex_lookup = {tuple(v): i for i, v in enumerate(self.vertices)}
                vertex_mapping = {i: vertex_lookup[tuple(tv)] 
                                for i, tv in enumerate(transformed_vertices)}
                
                # Generate edge mapping
                edge_mapping = {}
                for edge_idx, (v1, v2) in enumerate(self.edges):
                    mapped_v1, mapped_v2 = vertex_mapping[v1], vertex_mapping[v2]
                    
                    # Find corresponding edge in original configuration
                    if (mapped_v1, mapped_v2) in self.edges:
                        edge_mapping[edge_idx] = self.edges.index((mapped_v1, mapped_v2))
                    else:
                        edge_mapping[edge_idx] = self.edges.index((mapped_v2, mapped_v1))
                
                rotations.append(edge_mapping)
                
                if len(rotations) == 24:
                    break
            if len(rotations) == 24:
                break

        return rotations

    def _validate_rotations(self):
        """Validate rotation correctness."""
        if len(self.rotations) != 24:
            raise VoxelianError(f"Invalid rotation count: {len(self.rotations)} (expected 24)")
        
        for i, rotation in enumerate(self.rotations):
            if len(rotation) != 12 or set(rotation.values()) != set(range(12)):
                raise VoxelianError(f"Invalid rotation mapping at index {i}")

    def apply_rotation(self, cube_edges: Set[int], rotation_idx: int) -> Set[int]:
        """Apply rotation transformation to cube edges."""
        if not 0 <= rotation_idx < len(self.rotations):
            raise VoxelianError(f"Invalid rotation index: {rotation_idx}")
        
        rotation_map = self.rotations[rotation_idx]
        return {rotation_map[edge] for edge in cube_edges}


class Cube:
    """Represents a cube with specific edge configuration."""
    
    def __init__(self, edges: Set[int], cube_id: Optional[int] = None):
        if not isinstance(edges, (set, frozenset)):
            edges = set(edges)
        
        # Validate edge indices
        invalid_edges = edges - set(range(12))
        if invalid_edges:
            raise VoxelianError(f"Invalid edge indices: {invalid_edges}")
        
        self.edges = frozenset(edges)
        self.id = cube_id

    def get_canonical_form(self, rotations: CubeRotations) -> "Cube":
        """Compute canonical (lexicographically minimal) form."""
        canonical_candidates = []
        
        for rotation_idx in range(len(rotations.rotations)):
            rotated_edges = rotations.apply_rotation(self.edges, rotation_idx)
            canonical_candidates.append(tuple(sorted(rotated_edges)))
        
        minimal_form = min(canonical_candidates)
        return Cube(frozenset(minimal_form), self.id)

    def __repr__(self) -> str:
        edge_list = sorted(list(self.edges))
        return f"Cube(id={self.id}, edges={edge_list})"


class CubeLibrary:
    """Manages generation and storage of canonical cube configurations."""
    
    def __init__(self, rotations: CubeRotations):
        self.rotations = rotations
        self.canonical_cubes: List[Cube] = []
        self._edge_to_cube_map: Dict[frozenset, int] = {}

    def generate_canonical_library(self) -> None:
        """Generate all rotationally distinct cube configurations."""
        if self.canonical_cubes:
            return  # Already generated
        
        seen_canonical_forms = set()
        cube_id = 0

        # Generate all non-empty edge subsets
        all_edges = list(range(12))
        for subset_size in range(1, 13):
            for edge_subset in combinations(all_edges, subset_size):
                cube = Cube(edge_subset)
                canonical_cube = cube.get_canonical_form(self.rotations)
                
                if canonical_cube.edges not in seen_canonical_forms:
                    canonical_cube.id = cube_id
                    self.canonical_cubes.append(canonical_cube)
                    self._edge_to_cube_map[canonical_cube.edges] = cube_id
                    seen_canonical_forms.add(canonical_cube.edges)
                    cube_id += 1

        if len(self.canonical_cubes) != CANONICAL_CUBES:
            raise VoxelianError(
                f"Canonical library size mismatch: {len(self.canonical_cubes)} "
                f"(expected {CANONICAL_CUBES})"
            )

    def get_cube_by_id(self, cube_id: int) -> Cube:
        """Retrieve cube by canonical ID."""
        if not 0 <= cube_id < len(self.canonical_cubes):
            raise VoxelianError(f"Invalid cube ID: {cube_id}")
        return self.canonical_cubes[cube_id]

    def get_canonical_id(self, edges: Set[int]) -> int:
        """Get canonical ID for edge configuration."""
        cube = Cube(edges)
        canonical_cube = cube.get_canonical_form(self.rotations)
        
        if canonical_cube.edges not in self._edge_to_cube_map:
            raise VoxelianError(f"Edge configuration not in canonical library: {edges}")
        
        return self._edge_to_cube_map[canonical_cube.edges]


class ChecksumValidator:
    """Handles checksum computation and validation for cube sequences."""
    
    def __init__(self, cube_library: CubeLibrary):
        self.cube_library = cube_library
        self._checksum_lut = self._build_checksum_lookup()

    def _build_checksum_lookup(self) -> List[int]:
        """Build lookup table for cube checksum values."""
        checksum_values = []
        
        for cube in self.cube_library.canonical_cubes:
            # Convert edge set to 12-bit bitmask
            edge_bitmask = 0
            for edge_idx in cube.edges:
                edge_bitmask |= (1 << edge_idx)
            
            # Use low byte for checksum
            checksum_values.append(edge_bitmask & 0xFF)
        
        return checksum_values

    def compute_checksum(self, cube_ids: List[int]) -> int:
        """Compute XOR checksum for cube ID sequence."""
        if not cube_ids:
            return 0
        
        checksum = 0
        for cube_id in cube_ids:
            if not 0 <= cube_id < len(self._checksum_lut):
                raise VoxelianError(f"Invalid cube ID for checksum: {cube_id}")
            checksum ^= self._checksum_lut[cube_id]
        
        return checksum

    def validate_checksum(self, cube_ids: List[int], expected_checksum: int) -> bool:
        """Validate cube sequence against expected checksum."""
        try:
            computed_checksum = self.compute_checksum(cube_ids)
            return computed_checksum == expected_checksum
        except VoxelianError:
            return False


class VoxelianEncoder:
    """Main encoder class for Voxelian data encoding/decoding."""
    
    def __init__(self):
        self.rotations = CubeRotations()
        self.library = CubeLibrary(self.rotations)
        self.library.generate_canonical_library()
        self.checksum_validator = ChecksumValidator(self.library)
        
        # Build Base64 symbol mapping (first 65 canonical cubes)
        self._symbol_to_cube = {}
        self._cube_to_symbol = {}
        
        for i, symbol in enumerate(B64_CHARSET):
            if i < len(self.library.canonical_cubes):
                cube = self.library.canonical_cubes[i]
                self._symbol_to_cube[symbol] = cube
                self._cube_to_symbol[cube.id] = symbol

    def encode_text(self, text: str, encoding: str = DEFAULT_ENCODING) -> List[int]:
        """Encode text string to cube ID sequence."""
        try:
            binary_data = text.encode(encoding)
            return self.encode_binary(binary_data, encoding)
        except UnicodeEncodeError as e:
            raise VoxelianError(f"Text encoding error: {e}")

    def encode_binary(self, data: bytes, encoding: str = 'ascii') -> List[int]:
        """Encode binary data via Base64 intermediate representation."""
        try:
            base64_string = base64.b64encode(data).decode(encoding)
            return [self._get_cube_id_for_symbol(char) for char in base64_string]
        except Exception as e:
            raise VoxelianError(f"Binary encoding error: {e}")

    def decode_to_binary(self, cube_ids: List[int], encoding: str = 'ascii') -> bytes:
        """Decode cube ID sequence to binary data."""
        try:
            # Validate all cube IDs first
            for cube_id in cube_ids:
                if cube_id not in self._cube_to_symbol:
                    raise VoxelianError(f"Cube ID not in Base64 mapping: {cube_id}")
            
            # Reconstruct Base64 string
            base64_string = ''.join(self._cube_to_symbol[cube_id] for cube_id in cube_ids)
            return base64.b64decode(base64_string.encode(encoding))
        except Exception as e:
            raise VoxelianError(f"Decoding error: {e}")

    def decode_to_text(self, cube_ids: List[int], encoding: str = DEFAULT_ENCODING) -> str:
        """Decode cube ID sequence to text string."""
        binary_data = self.decode_to_binary(cube_ids)
        try:
            return binary_data.decode(encoding)
        except UnicodeDecodeError as e:
            raise VoxelianError(f"Text decoding error: {e}")

    def _get_cube_id_for_symbol(self, symbol: str) -> int:
        """Get cube ID for Base64 symbol."""
        if symbol not in self._symbol_to_cube:
            raise VoxelianError(f"invalid base64 symbol: '{symbol}'")
        return self._symbol_to_cube[symbol].id

    def get_cube_by_id(self, cube_id: int) -> Cube:
        """Retrieve cube object by ID."""
        return self.library.get_cube_by_id(cube_id)

    def compute_checksum(self, cube_ids: List[int]) -> int:
        """Compute checksum for cube sequence."""
        return self.checksum_validator.compute_checksum(cube_ids)

    def validate_checksum(self, cube_ids: List[int], checksum: int) -> bool:
        """Validate cube sequence checksum."""
        return self.checksum_validator.validate_checksum(cube_ids, checksum)

class Visualizer():
    def __init__(self, encoder: VoxelianEncoder):
        self.encoder = encoder
        
    def visualize_message(self, message: bytes, cube_ids):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        b64_string = message
        encoded_ids = cube_ids
        
        base_vertices = np.array(self.encoder.rotations.vertices)
        
        # Dynamic grid calculation for better layout
        total_chars = len(b64_string)
        cubes_per_row = min(12, max(6, int(np.sqrt(total_chars * 1.5))))
        num_rows = (total_chars + cubes_per_row - 1) // cubes_per_row
        
        # color schemes
        vertex_color = '#1f77b4'
        edge_colors = ['#cc1414', '#2dc80f', '#141dcc', '#c914cc']
    
        for i, (b64_char, cube_id) in enumerate(zip(b64_string, encoded_ids)):
            cube = self.encoder.get_cube_by_id(cube_id)
            
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
                a, b = self.encoder.rotations.edges[edge_idx] 
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
                     f'• Edge density range: {min(len(self.encoder.get_cube_by_id(cid).edges) for cid in encoded_ids)}-'
                     f'{max(len(self.encoder.get_cube_by_id(cid).edges) for cid in encoded_ids)} edges')
        
        # position statistics in corner
        fig.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=edge_colors[0], label='1-3'),
            Rectangle((0, 0), 1, 1, facecolor=edge_colors[1], label='4-6'),
            Rectangle((0, 0), 1, 1, facecolor=edge_colors[2], label='7-9'), 
            Rectangle((0, 0), 1, 1, facecolor=edge_colors[3], label='10-12')
        ]
        ax.legend(handles=legend_elements, title='Edges', loc='upper right', bbox_to_anchor=(0.98, 0.98))
        ax.view_init(elev=20, azim=15)
        plt.tight_layout()
        plt.savefig('output.png')

class VoxelianTestSuite(unittest.TestCase):
    """Comprehensive test suite for Voxelian encoder."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.encoder = VoxelianEncoder()

    def test_rotation_generation(self):
        """Test cube rotation generation."""
        self.assertEqual(len(self.encoder.rotations.rotations), 24)
        
        # Test rotation uniqueness
        rotation_signatures = []
        for rotation in self.encoder.rotations.rotations:
            signature = tuple(sorted(rotation.items()))
            self.assertNotIn(signature, rotation_signatures)
            rotation_signatures.append(signature)

    def test_canonical_library_generation(self):
        """Test canonical cube library."""
        self.assertEqual(len(self.encoder.library.canonical_cubes), CANONICAL_CUBES)
        
        # Test cube ID uniqueness
        cube_ids = [cube.id for cube in self.encoder.library.canonical_cubes]
        self.assertEqual(len(cube_ids), len(set(cube_ids)))

    def test_basic_encoding_decoding(self):
        """Test basic string encoding/decoding."""
        test_strings = [
            "Hello, World!",                                # normal string
            "The quick brown fox jumps over the lazy dog",  # normal string
            "1234567890",                                   # numbers
            "!@#$%^&*()_+-=[]{}|;:,.<>?",                   # symbols
            " "                                             # single space
        ]
        
        for test_string in test_strings:
            with self.subTest(test_string=test_string):
                if test_string:  # Skip empty string for encoding
                    cube_ids = self.encoder.encode_text(test_string)
                    decoded_string = self.encoder.decode_to_text(cube_ids)
                    self.assertEqual(test_string, decoded_string)

    def test_checksum_functionality(self):
        """Test checksum computation and validation."""
        test_message = "Hello, World!"
        cube_ids = self.encoder.encode_text(test_message)
        
        # Compute checksum
        checksum = self.encoder.compute_checksum(cube_ids)
        self.assertIsInstance(checksum, int)
        self.assertTrue(0 <= checksum <= 255)
        
        # Validate correct checksum
        self.assertTrue(self.encoder.validate_checksum(cube_ids, checksum))
        
        # Validate incorrect checksum
        wrong_checksum = (checksum + 1) % 256
        self.assertFalse(self.encoder.validate_checksum(cube_ids, wrong_checksum))

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid cube ID
        with self.assertRaises(VoxelianError):
            self.encoder.get_cube_by_id(-1)
        
        with self.assertRaises(VoxelianError):
            self.encoder.get_cube_by_id(1000)
        
        # Test invalid cube ID in checksum
        with self.assertRaises(VoxelianError):
            self.encoder.compute_checksum([-1])

    def test_base64_symbol_coverage(self):
        """Test Base64 symbol to cube mapping."""
        # All Base64 symbols should be mappable
        for symbol in B64_CHARSET:
            cube_id = self.encoder._get_cube_id_for_symbol(symbol)
            self.assertIsInstance(cube_id, int)
            self.assertTrue(0 <= cube_id < len(self.encoder.library.canonical_cubes))

class Console:
    """Command-line interface for Voxelian encoder."""
    def __init__(self):
        self.encoder = VoxelianEncoder()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Voxelian Encoder",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""Examples:
                      {sys.argv[0]} --encode "Hello World"
                      {sys.argv[0]} --encode "Message" --checksum --visualize
                      {sys.argv[0]} --test
                      {sys.argv[0]} --encode "01234" --padding 32
                                """
        )
        
        # Input options
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            "--encode", "--string", type=str, metavar="TEXT", 
            dest="text_input", help="Text message to encode"
        )
        input_group.add_argument(
            "--file", type=str, metavar="PATH",
            help="Input file to encode"
        )
        input_group.add_argument(
            "--test", action="store_true",
            help="Run comprehensive unit tests"
        )
        
        # Processing options
        parser.add_argument(
            "--padding", type=int, metavar="N",
            help="Add N random padding characters"
        )
        parser.add_argument(
            "--checksum", action="store_true",
            help="Compute and validate checksum"
        )
        parser.add_argument(
            "--visualize", "--vis", action="store_true",
            help="Generate 3D visualization"
        )
        
        # Output options
        parser.add_argument(
            "--output", "-o", type=str, metavar="FILE",
            default="output.vox", help="Output file (default: output.vox)"
        )
        parser.add_argument(
            "--encoding", type=str, default=DEFAULT_ENCODING,
            help=f"Text encoding (default: {DEFAULT_ENCODING})"
        )
        
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute CLI with given arguments."""
        print("=" * 50)
        print("Running encoder...")
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.test:
                return self._run_tests()
            
            if not parsed_args.text_input and not parsed_args.file:
                # Default: encode random message
                message_text = generate_random_message(16, B64_CHARSET)
                print(f"No input specified, using random message: '{message_text}'")
            else:
                message_text = self._get_input_text(parsed_args)
            
            # Apply padding if requested
            if parsed_args.padding:
                padding_text = generate_random_message(parsed_args.padding, B64_CHARSET)
                message_text += padding_text
                print(f"        padding: {parsed_args.padding}+")
            return self._encode_and_process(message_text, parsed_args)
         
        except VoxelianError as e:
            print(f"Encoding error: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 2

    def _get_input_text(self, args) -> str:
        if args.text_input:
            return args.text_input
        elif args.file:
            return read_file(args.file)
        else:
            raise VoxelianError("no valid input specified")

    def _encode_and_process(self, text: str, args) -> int:
        """Encode text and process according to arguments."""
        # Encode the message
        original_data = text.encode(args.encoding)
        cube_ids = self.encoder.encode_text(text, args.encoding)
        
        # Verify encoding integrity
        decoded_data = self.encoder.decode_to_binary(cube_ids)
        if decoded_data != original_data:
            raise VoxelianError("Encoding failed [mismatch]")
        
        # Compute checksum if requested
        checksum = None
        if args.checksum:
            checksum = self.encoder.compute_checksum(cube_ids)
        
        # Generate visualization if requested
        if args.visualize:
            vis = Visualizer(self.encoder)
            vis.visualize_message(text, cube_ids)
        
        # Print summary
        print("=" * 50)
        print("Result           : success")
        print(f"Original size    : {len(original_data)} bytes")
        print(f"Encoded cubes    : {len(cube_ids)}")
        print(f"Unique IDs       : {len(set(cube_ids))}")
        if checksum is not None:
            print(f"Checksum         : {checksum}")
        print(f"Encoding         : {cube_ids}")
        return 0

    def _run_tests(self) -> int:
        """Execute unit test suite."""
        print("Running tests...")
        print("=" * 50)
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(VoxelianTestSuite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1


def main():
    return Console().run()


if __name__ == "__main__":

    sys.exit(main())

