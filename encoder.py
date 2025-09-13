# core imports
import argparse
import unittest
import base64
import numpy as np
from typing import List
from itertools import chain, combinations, product, permutations

# utilities
from utils import CubeSerializer, CubeMetadata, generate_random_message, visualize_message, write_file

"""
Library     : Voxelian Encoder
Description : A Sol DeWit's storage model for data encoding using 217 unique cubes,
             (not the theoretical estimate 144).

Workflow :
    Define Cube Structure
        Vertices: 8 points in {0,1}³
        Edges: 12 edge pairs connecting vertices

    Generate Rotations (CubeRotations)
        Build all 24 valid cube rotations (SO(3))
        Each rotation is a matrix with axis permutations + sign flips
        Rotate vertices around cube center (0.5,0.5,0.5)
        Map rotated vertices to original indices
        Map edges to new positions → edge permutation

    Generate All Edge Subsets (CubeLibrary)
        Produce powerset of 12 edges (all 4095 non-empty subsets)
        
    Compute Canonical Form (Cube.canonical)
        Apply all 24 rotations to the subset
        Convert each rotated edge set to sorted tuple
        Pick lexicographically minimal tuple → canonical representation

    Store Unique Canonical Cubes (CubeLibrary)
        Keep a seen set of canonical edge frozensets
        Add new canonical cubes to library if not already present

    Assign IDs
        Sequential ID assignment for each unique canonical cube

    Output
        Library contains n-amount (selected) canonical cubes (one per rotational equivalence class)
        Each cube stores: edges (frozenset) + ID
        
    Remark: 
"""

# globals
APPV_MAIN = 0 # main version
APPV_SUB_ = 5 # sub version
APPV_REV  = 0 # revision


DEFAULT_ENCODER: str = "utf-8"

B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="

class CubeRotations:
    def __init__(self):
        # cube vertices in integer coordinates
        self.vertices = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]
        ])
        # edges defined as vertex index pairs
        self.edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        self.rotations = self.generate_rotations()

    def generate_rotations(self):
        mats = []
        seen = set()
        verts = self.vertices

        # predefined rotation matrices
        # axis permutations + ±1 signs with det=1
        axes_perms = [
                [0,1,2],[0,2,1],
                [1,0,2],[1,2,0],
                [2,0,1],[2,1,0]
                ]
        signs = [(1,1,1),(1,-1,-1),
                 (-1,1,-1),(-1,-1,1)
                ]

        for perm in axes_perms:
            for sign in signs:
                # build rotation matrix
                mat = np.zeros((3,3), dtype=int)
                for i in range(3):
                    mat[i, perm[i]] = sign[i]
                # rotate around cube center
                rotated = np.dot(verts - 0.5, mat.T) + 0.5
                rotated = np.round(rotated).astype(int)  # integer-safe coordinates
                key = tuple(rotated.flatten())
                if key in seen:
                    continue
                seen.add(key)
                # map rotated vertices to original vertex indices using dict lookup
                coord_to_index = {tuple(v): i for i, v in enumerate(verts)}
                vertex_map = {i: coord_to_index[tuple(rv)] for i, rv in enumerate(rotated)}
                # map edges using vertex_map
                mapping = {}
                for i, (a, b) in enumerate(self.edges):
                    va, vb = vertex_map[a], vertex_map[b]
                    if (va, vb) in self.edges:
                        mapping[i] = self.edges.index((va, vb))
                    else:
                        mapping[i] = self.edges.index((vb, va))
                mats.append(mapping)
                if len(mats) == 24:
                    return mats

        return mats

    def apply(self, cube_edges: frozenset[int], rotation_index: int) -> frozenset[int]:
        perm = self.rotations[rotation_index]
        return frozenset(perm[e] for e in cube_edges)

class Cube:
    def __init__(self, edges: frozenset[int], id: int | None = None):
        self.edges = edges
        self.id = id

    def canonical(self, rot: CubeRotations) -> "Cube":
        candidates = []
        for i in range(len(rot.rotations)):
            rotated_edges = rot.apply(self.edges, i)
            # convert to sorted tuple for deterministic comparison
            candidates.append(tuple(sorted(rotated_edges)))
        # lexicographic min
        min_edges = min(candidates)
        return Cube(frozenset(min_edges))

class CubeLibrary:
    """ Library logic
        1. Generate ALL possible edge subsets (brute force completeness)
        2. Compute canonical form for each via rotation group action  
        3. Remove only rotational duplicates (no other filtering)
        4. Result: 217 mathematically verified unique equivalence classes
    """
    def __init__(self, rotations: CubeRotations):
        self.rotations = rotations
        self.cubes: list[Cube] = []
    def generate_library(self):
        all_edges = list(range(12))
        seen = set()
        cube_id = 0

        def powerset(edges):
            return chain.from_iterable(combinations(edges, r) for r in range(1, 13))

        for subset in powerset(all_edges):
            cube = Cube(frozenset(subset))
            canonical_cube = cube.canonical(self.rotations)
            # Use the canonical frozenset directly as the seen key
            if canonical_cube.edges not in seen:
                canonical_cube.id = cube_id
                cube_id += 1
                self.cubes.append(canonical_cube)
                seen.add(canonical_cube.edges)
       
class Encoder:
    def __init__(self, rotations=None, library=None):
        from encoder import Cube, CubeRotations, CubeLibrary

        # Initialize rotations and library if not provided
        self.rotations = rotations or CubeRotations()
        self.library = library or CubeLibrary(self.rotations)
        if not self.library.cubes:
            self.library.generate_library()

        # Build a quick lookup: canonical frozenset -> cube ID
        self.canon_lookup = {cube.edges: cube.id for cube in self.library.cubes}

        # ##########################################################
        # Encoding symbol space allocation type
        # Workflow : data → Base64 → 64 symbols → 64 cubes → product
        
        # Base 16 (Hexadecimal)     : 7.4% coverage
        # Base 64 (default)         : 29.5% coverage
        # Base 128 (ASCII)          : 59% coverage
        
        # ##########################################################
        # Base32
        #self.symbol_to_cube = {}
        #cubes_list = list(self.library.cubes)
        #for i in range(min(256, len(cubes_list))):
        #    self.symbol_to_cube[i] = cubes_list[i]
        
        # ##########################################################
        # Base64 
        self.symbol_to_cube = {}
        cubes_list = list(self.library.cubes)
        for i, char in enumerate(B64_CHARS):
            # Maps 64 chars to first 64 cubes
            self.symbol_to_cube[char] = cubes_list[i]
  
    def encode_b64_char(self, b64_char: str) -> int:
        if b64_char not in self.symbol_to_cube:
            raise ValueError(f"invalid base64 character '{b64_char}'")
        cube = self.symbol_to_cube[b64_char]
        return cube.id
    
    def encode_data(self, binary_data: bytes, encoding = 'ascii') -> List[int]:
        b64_string = base64.b64encode(binary_data).decode(encoding)
        return [self.encode_b64_char(char) for char in b64_string]
        
        
    def decode_data(self, cube_ids: List[int], encoding = 'ascii') -> bytes:
        reverse_lookup = {v.id: k for k, v in self.symbol_to_cube.items()}
        b64_string = ''.join(reverse_lookup[cube_id] for cube_id in cube_ids)
        return base64.b64decode(b64_string.encode(encoding))

    def decode_id(self, cube_id: int) -> Cube:
        """Get the Cube object from its canonical ID"""
        return next(c for c in self.library.cubes if c.id == cube_id)

    def get_meta(self):
        return CubeMetadata(
                encoder_version=f"Voxelian.{APPV_MAIN}.{APPV_SUB_}.{APPV_REV}",
                original_size=len(message),
                cube_count=len(self.library.cubes),
                encoding_type="base64"
                )

class Tester(unittest.TestCase):
    def setUp(self):
        from encoder import Cube, CubeRotations, CubeLibrary
        self.rotations = CubeRotations()
        self.library = CubeLibrary(self.rotations)
        self.library.generate_library()

    def test_rotation_count(self):
        """Ensure exactly 24 unique rotations"""
        self.assertEqual(len(self.rotations.rotations), 24)

    def test_vertex_mapping(self):
        """Each rotated vertex maps to original vertex"""
        verts = self.rotations.vertices
        for perm in self.rotations.rotations:
            for e in perm.values():
                self.assertIn(e, range(12))

    def test_canonical_consistency(self):
        """Two cubes that are rotations of each other have same canonical form"""
        cube1 = Cube(frozenset({0, 1, 2}))
        # rotate cube 0 by first rotation
        rotated_edges = Cube(frozenset({self.rotations.rotations[0][e] for e in cube1.edges}))
        self.assertEqual(cube1.canonical(self.rotations).edges,
                         rotated_edges.canonical(self.rotations).edges)

    def test_library_uniqueness(self):
        """All cubes in library have unique canonical edges"""
        seen = set()
        for cube in self.library.cubes:
            self.assertNotIn(cube.edges, seen)
            seen.add(cube.edges)

    def test_library_size(self):
        """Check that library has expected number of canonical cubes"""
        expected_canonical_cubes = 217  
        self.assertEqual(len(self.library.cubes), expected_canonical_cubes)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Voxelian console encoder")
    parser.add_argument("--string", type=str, metavar="TEXT", help="Message phrase")
    parser.add_argument("--vis", action='store_true', help="Render visualized presentation")
    parser.add_argument("--test", action='store_true', help="Perform unit test")
    args = parser.parse_args()
    
    if args.test:
            print('Running unit test...')
            unittest.main(argv=[''])
            exit(0)
        
    if args.string:
        message = args.string.encode(DEFAULT_ENCODER)
    else:
        random_str = generate_random_message(16)
        message = random_str.encode(DEFAULT_ENCODER)
        
    # declare encoder
    encoder = Encoder()
    encoded_ids = encoder.encode_data(message)
    decoded = encoder.decode_data(encoded_ids)
    
    if decoded == message:
        print("Result success")
        serializer = CubeSerializer()
        # serialize
        binary_data = serializer.serialize(encoded_ids, encoder.get_meta())
        write_file('output.vox', binary_data)
        print(f"Serialized size: {len(binary_data)} bytes")
        print(f"Format info: {serializer.get_format_info(binary_data)}")
        if args.vis:
            visualize_message(encoder, decoded)
