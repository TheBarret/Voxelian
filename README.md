# Voxelian
A Sol Lewitt's Open Cube encoder model


<img width="930" height="674" alt="image" src="https://github.com/user-attachments/assets/f6f2a905-ac3c-48d0-a63f-d6fa0bd41a52" />  

##
> Screenshot shows the equivalent of 'Hello, World!'  

Using Sol Lewitt's open cube model for data encoding using 217 unique cubes, not the theoretical estimate 144.  
This encoder utilizes base64 to condense the character table as an intermediate layer and have a flexible input range,  
this does result in only 29.5% coverage(*) space.  

# Encoding Density & Coverage

<img width="300" alt="image" src="https://github.com/user-attachments/assets/68aded04-b967-4282-8626-6de11c8c6eac" />  

The encoder's base principle is what determines the coverage length within the cube space available.  
You could technically introduce an encryption layer as well providing you with extra security.  

```
 Base 16 (Hexadecimal)     : 7.4% coverage
 Base 64 (default)         : 29.5% coverage
 Base 128 (ASCII)          : 59% coverage
```

## Logic In-Depth

```mermaid
flowchart TD
    A[Define Cube Structure<br>8 Vertices, 12 Edges] --> B[Generate 24 Rotations<br>CubeRotations: SO3 Group]

    B --> C[Generate All Edge Subsets<br>Powerset: 4095 non-empty sets]

    C --> D{For Each Subset}
    D --> E[Apply All 24 Rotations]
    E --> F[Compute Canonical Form<br>Lexicographically Min. Tuple]
    F --> G{Canonical Rep.<br>in Library Set?}

    G -- No --> H[Add to Library<br>Assign New ID]
    G -- Yes --> I[Discard as Duplicate]
    
    H --> J[Add to Master List]
    I --> J

    D --> K[All Subsets Processed?]
    J --> K
    K -- No --> D
    K -- Yes --> L[Output Library<br>N Unique Canonical Cubes]
```
# Usage
```
python encoder.py --h
==================================================
Running encoder...
usage: encoder.py [-h] [--encode TEXT | --file PATH | --test] [--padding N] [--checksum] [--visualize] [--output FILE] [--encoding ENCODING]

Voxelian Encoder - Professional Data Encoding System

options:
  -h, --help            show this help message and exit
  --encode TEXT, --string TEXT
                        Text message to encode
  --file PATH           Input file to encode
  --test                Run comprehensive unit tests
  --padding N           Add N random padding characters
  --checksum            Compute and validate checksum
  --visualize, --vis    Generate 3D visualization
  --output FILE, -o FILE
                        Output file (default: output.vox)
  --encoding ENCODING   Text encoding (default: utf-8)

Examples:
  encoder.py --encode "Hello World"
  encoder.py --encode "Message" --checksum --visualize
  encoder.py --test
  encoder.py --encode "Data" --padding 32 --output encoded.vox
```

*encoder saves all output in local root folder*

# Unit testing
```
python encoder.py --test
==================================================
Running encoder...
Running encoder test suite...
==================================================
test_base64_symbol_coverage (__main__.VoxelianTestSuite)
Test Base64 symbol to cube mapping. ... ok
test_basic_encoding_decoding (__main__.VoxelianTestSuite)
Test basic string encoding/decoding. ... ok
test_canonical_library_generation (__main__.VoxelianTestSuite)
Test canonical cube library. ... ok
test_checksum_functionality (__main__.VoxelianTestSuite)
Test checksum computation and validation. ... ok
test_error_handling (__main__.VoxelianTestSuite)
Test error handling for invalid inputs. ... ok
test_rotation_generation (__main__.VoxelianTestSuite)
Test cube rotation generation. ... ok

----------------------------------------------------------------------
Ran 6 tests in 2.189s

OK
```
