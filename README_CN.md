# File1

一个基于大语言模型（LLM）的Python包，用于文件分析、摘要生成和关系可视化。

## 功能特性

- **文件摘要**：自动为文件和目录生成摘要
- **重复文件检测**：使用基于LLM的比较识别重复文件
- **模拟数据检测**：检测并删除模拟/测试数据文件
- **关系可视化**：创建显示文件关系的可视化图表
- **视觉模型集成**：从图像和PDF中提取并分析内容
- **重排序支持**：使用重排序模型提高相关性评分

## 安装

您可以使用pip安装File1：

```bash
pip install file1
```

开发环境安装：

```bash
git clone https://github.com/file1/file1.git
cd file1
pip install -e .[dev]
```

## 快速开始

```python
from file1 import File1
from file1.config import File1Config

# 使用默认配置初始化
config = File1Config()
file1 = File1(config)

# 清理仓库（删除重复文件和模拟数据）
file1.clean_repository("/path/to/your/project")

# 构建文件关系图
graph = file1.build_graph("/path/to/your/project")

# 可视化图表
file1.visualize_graph(graph, save_path="file_relationship.png")
```

## 配置

File1使用TOML配置文件来指定模型设置和其他参数：

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

## API参考

### File1

用于文件分析和管理的主要类。

#### 方法

- `clean_repository(directory)`: 删除重复文件和模拟数据
- `build_graph(directory)`: 构建文件关系图
- `visualize_graph(graph, save_path)`: 可视化文件关系图

### FileSummary

用于生成文件和目录摘要的类。

#### 方法

- `summarize_file(file_path)`: 为单个文件生成摘要
- `summarize_directory(directory_path)`: 为目录生成摘要
- `get_file_tree_with_summaries(directory_path)`: 获取带有摘要的文件树

### FileManager

用于管理文件、检测重复项和构建关系的类。

#### 方法

- `detect_file_duplication(file1, file2)`: 检查两个文件是否重复
- `detect_simulated_data(file_path)`: 检查文件是否包含模拟数据
- `find_duplicates_with_reranker(files)`: 使用重排序查找重复文件

## 示例

### 基本用法

```python
from file1 import File1
from file1.config import File1Config

# 使用配置初始化
config = File1Config.from_file("config.toml")
file1 = File1(config)

# 分析目录
summary = file1.summarize_directory("/path/to/project")
print(summary)
```

### 自定义配置

```python
from file1.config import ModelConfig, LLMConfig, RerankConfig, File1Config

# 创建自定义配置
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

## 许可证

本项目采用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请随时提交Pull Request。对于重大更改，请先开issue讨论您想要更改的内容。

## 支持

如果您有任何问题或建议，请在[GitHub Issues](https://github.com/file1/file1/issues)上开issue。