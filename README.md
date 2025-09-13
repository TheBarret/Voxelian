# Voxelian
A Sol DeWit's Open Cube encoder model


<img width="942" height="674" alt="image" src="https://github.com/user-attachments/assets/03fbc275-7238-4904-9a46-62a4b357be37" />

A Using Sol DeWit's open cube model for data encoding using 217 unique cubes, not the theoretical estimate 144.  
This encoder utilizes base64 to condense the character table as an intermediate layer and have a flexible input range,  
this does result in only 29.5% coverage(*) space.  

(*) Space coverage ratios: 
```
# Base 16 (Hexadecimal)     : 7.4% coverage
# Base 64 (default)         : 29.5% coverage
# Base 128 (ASCII)          : 59% coverage
```

# Workflow
- Define Cube Structure  
    Vertices: 8 points in {0,1}³  
    Edges: 12 edge pairs connecting vertices  

- Generate Rotations (CubeRotations)  
    Build all 24 valid cube rotations (SO(3))  
    Each rotation is a matrix with axis permutations + sign flips  
    Rotate vertices around cube center (0.5,0.5,0.5)  
    Map rotated vertices to original indices  
    Map edges to new positions → edge permutation  

- Generate All Edge Subsets (CubeLibrary)  
    Produce powerset of 12 edges (all 4095 non-empty subsets)  
    
- Compute Canonical Form (Cube.canonical)  
    Apply all 24 rotations to the subset  
    Convert each rotated edge set to sorted tuple  
    Pick lexicographically minimal tuple → canonical representation  

- Store Unique Canonical Cubes (CubeLibrary)  
    Keep a seen set of canonical edge frozensets  
    Add new canonical cubes to library if not already present  

- Assign IDs  
    Sequential ID assignment for each unique canonical cube  

- Output  
    Library contains n-amount (selected) canonical cubes (one per rotational equivalence class)  
    Each cube stores: edges (frozenset) + ID  

# Usage
```
python encoder.py -h
usage: encoder.py [-h] [--string TEXT] [--vis] [--test]

Voxelian console encoder

options:
  -h, --help     show this help message and exit
  --string TEXT  Message phrase
  --vis          Render visualized presentation
  --test         Perform unit test
```

## Encoding a string
```
python encoder.py --vis --string 'Hello, World!'
Result success
Serialized size: 53 bytes
Format info: {'format': 'Cube1', 'version': 1, 'cube_count': 20, 'size_bytes': 53, 'has_metadata': False}
```

Encoder saves 'output.vox' in local root folder, a serialized state of the 'encoded_ids' data for reversing the process.

