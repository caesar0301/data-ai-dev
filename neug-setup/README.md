# NeuG Setup Guide

This directory contains NeuG installation and test scripts.

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Run Working Tests

#### Test Basic Graph Functionality ✅
```bash
python3 test_basic_graph.py
```

This test demonstrates:
- Database creation
- Loading built-in datasets
- Cypher queries (pattern matching, filtering, aggregation)
- Triangle detection
- Graph analytics

**Expected output**: All tests should PASS ✓

## About Extensions

### JSON & Parquet Extensions
⚠️ **Status**: Source code exists but requires building from source

The JSON and Parquet extensions are present in `neug/extension/` but are NOT included in the pip wheel package. To use them:

1. Build NeuG from source:
```bash
cd neug
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_EXECUTABLES=ON -DBUILD_TEST=ON
make -j$(sysctl -n hw.ncpu)  # macOS
```

2. Use the built extensions:
```python
conn.execute("LOAD JSON;")
conn.execute("LOAD PARQUET;")
```

### Hypergraph Support
❌ **Not Supported**

NeuG implements standard property graphs with binary edges only. See `NEUG_SETUP_REPORT.md` for details and workarounds.

## Test Scripts

| Script | Status | Description |
|--------|--------|-------------|
| `test_basic_graph.py` | ✅ PASSING | Core graph database functionality |
| `test_json_extension.py` | ⚠️ Needs build | JSON import/export extension |
| `test_parquet_extension.py` | ⚠️ Needs build | Parquet import/export extension |

## Documentation

- `NEUG_SETUP_REPORT.md`: Comprehensive setup report and analysis
- `neug/README.md`: Official NeuG documentation
- `neug/dev_and_test.md`: Development and testing guide

## Next Steps

1. ✅ Core functionality ready - start using NeuG for graph operations
2. ⚠️ Build from source if you need JSON/Parquet extensions
3. 📖 Explore built-in datasets: tinysnb, modern_graph, lsqb
4. 🔬 Write custom Cypher queries for your use case

## Example Usage

```python
import neug

# Create database
db = neug.Database("/path/to/db")
db.load_builtin_dataset("tinysnb")

# Connect and query
conn = db.connect()
result = conn.execute("""
    MATCH (a:person)-[:knows]->(b:person)-[:knows]->(c:person),
          (a)-[:knows]->(c)
    RETURN a.fName, b.fName, c.fName
""")

for record in result:
    print(f"{record[0]}, {record[1]}, {record[2]} form a triangle")

conn.close()
```

## Resources

- Official Docs: https://graphscope.io/neug/en/overview/introduction/
- GitHub: https://github.com/alibaba/neug
- Discord: https://discord.gg/2S8344ew