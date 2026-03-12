# AI Agents in Data-AI-Dev Project

This document provides an overview of the AI agents, intelligent systems, and autonomous components within the data-ai-dev project.

## Overview

The data-ai-dev project is a comprehensive AI and data engineering playground that explores various technologies for data processing, storage, and analysis. While not a traditional multi-agent system, the project contains several intelligent components and AI-powered tools that can be considered as specialized agents for different tasks.

## Agent Components

### 1. PandasAI Agent (`pandasai/`)

**Location**: `pandasai/llm.py`

**Description**: A conversational AI agent for data analysis and visualization using natural language queries.

**Key Features**:
- Integrates with DashScope's Qwen models (qwen-plus, qwen-turbo, qwen-max)
- Provides natural language to pandas code conversion
- Supports automated data visualization with Plotly
- Custom environment setup for secure code execution
- Handles DataFrame operations through conversational interface

**Core Classes**:
- `PandasAILLMDashScope`: Custom OpenAI-compatible wrapper for DashScope API
- `create_pandasai_agent()`: Factory function to create analysis agents

**Usage**:
```python
from pandasai import Agent
from llm import setup_pandasai_llm, create_pandasai_agent

llm = setup_pandasai_llm()
agent = create_pandasai_agent(dataframe, llm)
response = agent.chat("Show me the distribution of values in column X")
```

### 2. Ray Distributed Compute Agents (`ray/`)

**Location**: `ray/my_app.py`, `ray/check_requests_ver.py`

**Description**: Distributed computing agents using Ray framework for parallel task execution.

**Key Features**:
- Parallel task execution using Ray's @ray.remote decorator
- Distributed computation across multiple nodes
- Scalable task processing for large datasets
- Kubernetes deployment support with Kuberay operator

**Core Components**:
- Remote task functions for distributed processing
- Cluster management and orchestration
- Request verification and validation agents

**Usage**:
```python
import ray

@ray.remote
def square(x):
    return x * x

# Launch parallel tasks
futures = [square.remote(i) for i in range(4)]
results = ray.get(futures)
```

### 3. Data Processing Agents (`lancedb/`)

**Location**: `lancedb/perf_comp_*.py`, `lancedb/read_*.py`

**Description**: Specialized agents for data compression, format conversion, and performance benchmarking.

**Key Features**:
- Automatic compression algorithm selection
- Performance benchmarking agents
- Data format conversion (CSV, Parquet, Lance, NPZ)
- Vector and text data processing specialists

**Agent Types**:
- **Text Compression Agent**: Optimizes text data storage using various compression algorithms
- **Vector Compression Agent**: Handles high-dimensional vector data compression
- **UBM Data Agent**: Specialized for binary key-value pair data processing

### 4. GPU Acceleration Agents (`nvidia-dali/`)

**Location**: `nvidia-dali/ho-01.ipynb`

**Description**: GPU-accelerated data processing pipeline agents using NVIDIA DALI.

**Key Features**:
- Parallel image decoding and processing
- GPU-based data augmentation
- High-performance data loading pipelines
- Batch processing optimization

**Pipeline Components**:
- File reading agents
- Image decoding agents
- Data transformation agents

### 5. Vector Database Agents (`weaviate/`)

**Description**: Vector similarity search and retrieval agents (implementation in progress).

**Intended Features**:
- Semantic search capabilities
- Vector embeddings management
- Similarity-based retrieval agents

## Agent Architecture

### Communication Patterns

1. **Direct Function Calls**: Most agents communicate through direct Python function calls
2. **Ray Remote Tasks**: Distributed agents use Ray's task-based communication
3. **API Integration**: External AI services (DashScope) for language understanding
4. **File-based Exchange**: Some agents use shared files for data exchange

### Data Flow

```
Raw Data → Processing Agents → Storage Agents → Analysis Agents → Visualization
    ↓              ↓               ↓              ↓              ↓
  Files/GPU    Ray Cluster    Lance/Parquet   PandasAI     Plotly Charts
```

### Agent Orchestration

- **Manual Orchestration**: Currently, agents are manually orchestrated through scripts
- **Future Enhancement**: Consider implementing a central orchestrator agent
- **Configuration**: Agents are configured through environment variables and config files

## Environment Setup

### Prerequisites

1. **Python 3.11+**: Required for all agent components
2. **GPU Support**: For NVIDIA DALI agents
3. **Docker/Kubernetes**: For Ray cluster deployment
4. **API Keys**: DashScope API key for PandasAI agent

### Installation

```bash
# Setup PandasAI Agent
cd pandasai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup Ray Agents
cd ray
python -m venv .venv
source .venv/bin/activate
pip install "ray[default,data]==2.44.1"

# Setup LanceDB Agents
cd lancedb
poetry install
# or
pip install -r requirements.txt

# Setup NVIDIA DALI Agents
cd nvidia-dali
pip install nvidia-dali[tensorflow]
```

## Usage Examples

### Running the PandasAI Agent

```python
import pandas as pd
from llm import setup_pandasai_llm, create_pandasai_agent

# Load data
df = pd.read_csv('data.csv')

# Setup agent
llm = setup_pandasai_llm()
agent = create_pandasai_agent(df, llm)

# Query data
response = agent.chat("What are the summary statistics for each column?")
print(response)
```

### Running Ray Distributed Agents

```python
import ray

# Initialize Ray
ray.init(address='ray://head-node:10001')

# Define distributed task
@ray.remote
def process_data_chunk(chunk):
    # Process data chunk
    return processed_chunk

# Execute in parallel
futures = [process_data_chunk.remote(chunk) for chunk in data_chunks]
results = ray.get(futures)
```

### Running Benchmarking Agents

```bash
cd lancedb

# Text compression benchmark
python perf_comp_text.py

# Vector compression benchmark
python perf_comp_vector.py

# UBM data processing
python read_ubm_data.py
```

## Future Enhancements

### Planned Agent Additions

1. **Central Orchestrator Agent**: Coordinate multiple specialized agents
2. **AutoML Agent**: Automated machine learning pipeline optimization
3. **Monitoring Agent**: Track agent performance and health
4. **Scheduling Agent**: Optimize task scheduling across agents
5. **Communication Agent**: Handle inter-agent messaging protocols

### Architecture Improvements

1. **Message Queue System**: Implement RabbitMQ or Kafka for agent communication
2. **Service Discovery**: Dynamic agent registration and discovery
3. **Fault Tolerance**: Automatic agent recovery and failover
4. **Scalability**: Auto-scaling based on workload
5. **Security**: Agent authentication and authorization

## Contributing

When adding new agents to the project:

1. Follow the existing directory structure patterns
2. Include comprehensive documentation
3. Add unit tests for agent functionality
4. Update this AGENTS.md file
5. Consider interoperability with existing agents

## Troubleshooting

### Common Issues

1. **PandasAI Agent**: API key errors - ensure DASHSCOPE_API_KEY is set
2. **Ray Agents**: Connection issues - check Ray cluster status
3. **LanceDB Agents**: Memory errors - reduce dataset size or increase memory
4. **NVIDIA DALI**: GPU errors - verify CUDA installation

### Debug Tips

- Enable verbose logging in agents
- Check agent logs in respective directories
- Use Ray dashboard for distributed agent monitoring
- Monitor GPU usage for DALI agents

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [PandasAI Documentation](https://github.com/gventuri/pandas-ai)
- [Lance Documentation](https://lancedb.github.io/lance/)
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/)
- [DashScope API Documentation](https://help.aliyun.com/zh/model-developer/)