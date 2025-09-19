## Issue 1 — Improve Encoding Density / Coverage
**Purpose:** Current symbol→cube mapping wastes ~70% of cube library.  
**Scope:** Optimize mapping to increase usage. Do not change cube canonicalization rules.  
**Tasks:**  
- Analyze current mapping.  
- Implement improved symbol→cube mapping (variable-length, multi-symbol, or expanded set).  
- Add tests to validate improved coverage.  
**Acceptance Criteria:**  
- Coverage report shows ≥80% usage of cube IDs in UTF-8 and base64 modes.  

---

## Issue 2 — Add Error Detection & Correction (ECC)
**Purpose:** Current system only detects errors (checksums).  
**Scope:** Add lightweight ECC. Do not over-engineer (no heavy cryptography).  
**Tasks:**  
- Research small ECC (Hamming, Reed–Solomon, parity).  
- Implement optional ECC layer (`--with-ecc`).  
- Add tests with corrupted inputs.  
**Acceptance Criteria:**  
- Single-symbol errors are corrected.  
- Multi-symbol corruption is detected.  

---

## Issue 3 — Performance & Scalability Optimization
**Purpose:** Python implementation is slow for large data.  
**Scope:** Optimize performance without changing outputs.  
**Tasks:**  
- Profile with `cProfile`.  
- Optimize rotation/canonicalization (memoization, precomputed tables).  
- Add multiprocessing/parallelism.  
- Optional: Re-implement hot paths in Cython or Rust.  
**Acceptance Criteria:**  
- Encoding speed ≥2× faster on >10MB files.  
- Outputs remain identical to current version.  

---

## Issue 4 — Compression & Overhead Benchmarking
**Purpose:** Project lacks quantitative benchmarks.  
**Scope:** Build benchmarking only (no redesign for compression).  
**Tasks:**  
- Benchmark sample files (text, binary, media).  
- Compare size and speed with base64, hex, gzip, zstd.  
- Document results in `docs/benchmarks.md`.  
**Acceptance Criteria:**  
- Benchmarks reproducible via script.  
- Results documented with tables/graphs.  

---

## Issue 5 — Define a Standardized Output Format
**Purpose:** Encoded data lacks portability.  
**Scope:** Define `.vxl` container format with metadata. Do not break raw-mode output.  
**Tasks:**  
- Specify format: version, encoding mode, ECC flag, cube data.  
- Implement writer/reader.  
- Update CLI with import/export.  
- Document spec in `docs/spec.md`.  
**Acceptance Criteria:**  
- `.vxl` file can be exported/imported across systems using spec only.  

---

## Issue 6 — Cross-Language Decoder Implementations
**Purpose:** Python-only limits adoption.  
**Scope:** Write minimal reference decoders (not full encoder).  
**Tasks:**  
- Implement JS/TypeScript decoder.  
- Implement C++ or Rust decoder (optional).  
- Provide test vectors for cross-language checks.  
**Acceptance Criteria:**  
- `.vxl` files decode identically in Python and another language.  

---

## Issue 7 — Extended Documentation & Use-Case Guides
**Purpose:** Clarify real-world value (art, education, steganography).  
**Scope:** Documentation only. No feature creep.  
**Tasks:**  
- Add `docs/usecases.md`.  
- Update `README.md` with quickstart + diagrams.  
- Provide sample datasets and visual outputs.  
**Acceptance Criteria:**  
- Documentation explains use-cases clearly.  
- New users can follow examples to replicate results.  

---

## Issue 8 — Expand Tests & CI/CD
**Purpose:** Ensure reliability and automation.  
**Scope:** Improve test coverage and CI only.  
**Tasks:**  
- Add property-based tests (Hypothesis).  
- Add fuzz tests with random inputs.  
- Configure GitHub Actions for linting, tests, benchmarks.  
**Acceptance Criteria:**  
- CI runs automatically on PRs.  
- Coverage report ≥90%.  

---

## Issue 9 — Visualization & Debugging Tools
**Purpose:** Expand usability of visual output.  
**Scope:** Visualization tools only (no core encoder changes).  
**Tasks:**  
- Add CLI flag to render `.png` images or `.obj`/`.glb` 3D models.  
- Add interactive viewer (Three.js or Jupyter notebook).  
**Acceptance Criteria:**  
- Encoded data can be visualized as 2D/3D output.  
- Users can inspect cube structures interactively.  
