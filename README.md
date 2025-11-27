# File1

A Python package for file analysis, summarization, and relationship visualization powered by Large Language Models (LLMs).

## Features

- **File Summarization**: Automatically generate summaries for files and directories
- **Duplicate Detection**: Identify duplicate files using LLM-based comparison
- **Simulated Data Detection**: Detect and remove simulated/mock data files
- **Relationship Visualization**: Create visual graphs showing file relationships
- **Vision Model Integration**: Extract and analyze content from images and PDFs
- **Reranking Support**: Use reranking models to improve relevance scoring

## Installation

You can install File1 using pip:

```bash
pip install file1
```

For development:

```bash
git clone https://github.com/file1/file1.git
cd file1
pip install -e .[dev]
```

## Quick Start

```python
from file1 import File1
from file1.config import File1Config

# Initialize with default configuration
config = File1Config()
file1 = File1(config)

# Clean repository (remove duplicates and simulated data)
file1.clean_repository("/path/to/your/project")

# Build file relationship graph
graph = file1.build_graph("/path/to/your/project")

# Visualize the graph
file1.visualize_graph(graph, save_path="file_relationship.png")
```

## Configuration

File1 uses a TOML configuration file to specify model settings and other parameters:

```toml
[model]
name = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[llm.chat]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[llm.vision]
model = "gpt-4o-mini"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"

[reranker]
model = "bge-reranker-v2-m3"
base_url = "https://api.bge-m3.com/v1"
api_key = "your-reranker-api-key"

[save_path]
path = "/path/to/save/directory"
```

## API Reference

### File1

The main class for file analysis and management.

#### Methods

- `clean_repository(directory)`: Remove duplicate files and simulated data
- `build_graph(directory)`: Build a file relationship graph
- `visualize_graph(graph, save_path)`: Visualize the file relationship graph

### FileSummary

A class for generating file and directory summaries.

#### Methods

- `summarize_file(file_path)`: Generate a summary for a single file
- `summarize_directory(directory_path)`: Generate a summary for a directory
- `get_file_tree_with_summaries(directory_path)`: Get a file tree with summaries

### FileManager

A class for managing files, detecting duplicates, and building relationships.

#### Methods

- `detect_file_duplication(file1, file2)`: Check if two files are duplicates
- `detect_simulated_data(file_path)`: Check if a file contains simulated data
- `find_duplicates_with_reranker(files)`: Find duplicates using reranking

## Examples

### Basic Usage

```python
from file1 import File1
from file1.config import File1Config

# Initialize with configuration
config = File1Config.from_file("config.toml")
file1 = File1(config)

# Analyze a directory
summary = file1.summarize_directory("/path/to/project")
print(summary)
```

### Custom Configuration

```python
from file1.config import ModelConfig, LLMConfig, RerankConfig, File1Config

# Create custom configuration
model_config = ModelConfig(
    name="gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key"
)

llm_config = LLMConfig(
    chat=model_config,
    vision=model_config
)

rerank_config = RerankConfig(
    model="bge-reranker-v2-m3",
    base_url="https://api.bge-m3.com/v1",
    api_key="your-reranker-api-key"
)

config = File1Config(
    llm=llm_config,
    reranker=rerank_config,
    save_path="/path/to/save/directory"
)

file1 = File1(config)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you have any questions or issues, please open an issue on [GitHub Issues](https://github.com/file1/file1/issues).