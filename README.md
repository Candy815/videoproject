# 🎬 视频抄袭检测系统

基于多模态指纹检索与 CLIP 语义验证的智能侵权检测平台

## 📋 项目简介

视频抄袭检测系统是一个端到端的智能侵权检测解决方案，集成了视频处理、深度学习特征提取、向量相似性检索和语义验证技术。系统能够从海量视频库中快速识别抄袭内容，并提供详细的侵权分析报告。

## ✨ 核心特性

### 🔍 多模态检测技术
- **视觉特征匹配**：使用 ResNet50 提取深度视觉特征
- **PHash 粗筛**：快速过滤相似内容，提升检索效率
- **CLIP 语义验证**：基于 OpenAI CLIP 模型的语义级内容理解
- **向量检索**：Milvus 向量数据库支持大规模相似性搜索

### 🚀 智能处理流程
- **流式处理**：支持 URL 直链检测，无需下载视频文件
- **动态抽帧**：按需提取关键帧，减少存储开销
- **智能缓存**：CLIP 特征缓存机制，提升重复检测效率
- **多阈值调节**：可配置的相似度和语义匹配阈值

### 📊 可视化界面
- **Gradio Web 界面**：用户友好的交互式 Web 应用
- **实时进度显示**：检测过程可视化展示
- **帧级对比**：相似帧并排对比展示
- **详细报告**：HTML/JSON/TXT 格式的完整检测报告

## 🏗️ 系统架构

```
视频输入 (本地文件/URL)
    ↓
流式抽帧处理器 (StreamFrameExtractor)
    ↓
多模态特征提取
├── PHash 感知哈希 (快速粗筛)
├── ResNet50 视觉特征 (深度学习)
└── CLIP 语义特征 (语义理解)
    ↓
向量检索 (Milvus)
    ↓
相似度计算与排序
    ↓
CLIP 语义验证 (二次过滤)
    ↓
侵权分析报告生成
    ↓
可视化展示 (Gradio)
```

## 🛠️ 技术栈

- **后端框架**: Python 3.8+
- **深度学习**: PyTorch, Torchvision, CLIP
- **特征提取**: Towhee, ResNet50
- **向量数据库**: Milvus 2.3+
- **视频处理**: PyAV, OpenCV
- **图像处理**: Pillow, ImageHash
- **Web 界面**: Gradio
- **数据处理**: NumPy, Pandas
- **工具库**: Loguru, Tqdm, PyYAML

## 📦 安装指南

### 前置要求

1. **Python 环境**: Python 3.8 或更高版本
2. **Milvus 数据库**: 安装并运行 Milvus 2.3+ ([安装指南](https://milvus.io/docs/install_standalone-docker.md))
3. **CUDA (可选)**: 用于 GPU 加速的特征提取

### 步骤 1: 克隆项目

```bash
git clone <repository_url>
cd videoproject
```

### 步骤 2: 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 步骤 3: 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: CLIP 库将从 GitHub 源码安装，可能需要较长时间。

### 步骤 4: 环境配置

复制环境配置文件模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件，根据你的环境配置：

```env
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 大模型配置 (可选，用于智能报告生成)
LLM_PROVIDER=ollama  # 或 openai
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# OpenAI 配置 (如果需要)
OPENAI_API_KEY=your_api_key_here
```

### 步骤 5: 启动 Milvus

确保 Milvus 数据库正在运行：

```bash
# 使用 Docker Compose (推荐)
docker-compose up -d

# 或者直接启动 Milvus 容器
docker run -d --name milvus-standalone \
    -p 19530:19530 \
    -p 9091:9091 \
    milvusdb/milvus:v2.3.0
```

## ⚙️ 配置说明

### 主要配置文件

- **`config/settings.py`**: 系统核心配置
- **`.env`**: 环境变量配置

### 关键配置项

```python
# 视频处理
FRAME_EXTRACT_INTERVAL = 1.0      # 抽帧间隔（秒）
MAX_FRAMES_PER_VIDEO = 300        # 最大抽帧数

# 相似度阈值
SIMILARITY_THRESHOLD = 0.7        # 视觉特征相似度阈值
CLIP_SIMILARITY_THRESHOLD = 0.6   # CLIP 语义相似度阈值
PHASH_THRESHOLD = 10              # PHash 距离阈值

# Milvus 配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "video_features"
CLIP_COLLECTION_NAME = "clip_features"

# 流式处理
STREAM_TIMEOUT = 30               # 网络超时（秒）
STREAM_MAX_RETRIES = 3            # 最大重试次数
```

## 🚀 快速开始

### 方法 1: 启动 Web 界面（推荐）

```bash
python app.py
```

访问 `http://localhost:7860` 使用图形界面。

### 方法 2: 命令行批量处理

```bash
# 训练模式：构建视频特征库
python main.py --mode train --video_dir ./data/raw

# 检测模式：检测单个视频
python main.py --mode detect --video_path ./data/raw/test.mp4

# URL 检测模式
python main.py --mode detect --video_url "https://example.com/video.mp4"
```

### 方法 3: Python API 调用

```python
from pipeline.detection_pipeline import DetectionPipeline

# 初始化检测流水线
pipeline = DetectionPipeline()

# 检测本地视频
report = pipeline.detect_infringement(
    video_path="path/to/video.mp4",
    similarity_threshold=0.7,
    top_k=10
)

# 检测在线视频
report = pipeline.detect_from_url(
    video_url="https://example.com/video.mp4",
    user_keywords=["侵权", "抄袭"],
    top_k=10
)

# 打印结果
print(f"侵权等级: {report.infringement_level}")
print(f"匹配片段数: {report.total_matches}")
print(f"最高相似度: {report.max_similarity:.2%}")
```

## 🖥️ Web 界面使用指南

### 1. 输入源选择
- **上传文件**: 支持 MP4、AVI、MOV、MKV 格式
- **视频链接**: 支持 B站、抖音、YouTube 等平台 URL

### 2. 检测参数配置
- **相似度阈值**: 控制视觉特征的匹配严格度（0.5-0.95）
- **语义阈值**: 控制 CLIP 语义匹配的严格度（0.5-0.95）
- **自定义关键词**: 针对性的侵权关键词（英文逗号分隔）

### 3. 结果展示
- **检测摘要**: 侵权等级、匹配片段数、相似度统计
- **视频对比**: 生成并排对比视频，高亮相似片段
- **帧级对比**: 相似帧图片对比，显示时间戳和相似度
- **侵权报告**: 可下载的详细报告（JSON/TXT 格式）

## 📁 项目目录结构

```
videoproject/
├── app.py                          # Gradio Web 应用入口
├── main.py                         # 命令行入口
├── requirements.txt                # Python 依赖包
├── .env                           # 环境变量配置
├── .env.example                   # 环境变量模板
│
├── config/                        # 配置文件目录
│   ├── settings.py                # 系统配置
│   └── __init__.py
│
├── core/                          # 核心功能模块
│   ├── clip_cache.py              # CLIP 特征缓存管理
│   ├── clip_detector.py           # CLIP 语义检测
│   ├── extractor.py               # 视频抽帧
│   ├── feature_extractor.py       # 特征提取（Towhee）
│   ├── frame_comparator.py        # 帧对比数据组装
│   ├── milvus_manager.py          # Milvus 数据库管理
│   ├── phash_similarity.py        # PHash 相似度计算
│   ├── report_generator.py        # 报告生成器
│   ├── stream_extractor.py        # 流式帧提取
│   ├── stream_processor.py        # 流式处理器
│   ├── streaming_downloader.py    # 流式下载器
│   ├── video_compare.py           # 视频对比生成
│   └── __init__.py
│
├── models/                        # 数据模型
│   ├── data_models.py             # Pydantic 数据模型
│   └── __init__.py
│
├── pipeline/                      # 处理流水线
│   ├── detection_pipeline.py      # 侵权检测流水线
│   ├── training_pipeline.py       # 训练流水线
│   └── __init__.py
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始视频文件
│   ├── frames/                    # 抽取的帧图片
│   ├── features/                  # 提取的特征
│   ├── comparisons/               # 对比视频
│   └── reports/                   # 检测报告
│
├── utils/                         # 工具函数
│   ├── file_utils.py              # 文件操作工具
│   ├── logger.py                  # 日志配置
│   ├── video_utils.py             # 视频处理工具
│   └── __init__.py
│
├── tests/                         # 测试文件
│   └── test_detection.py
│
├── logs/                          # 系统日志
└── CLIP-main/                     # CLIP 模型源码（从 Git 安装）
```

## 🔧 高级功能

### 1. 批量视频库构建

```python
from pipeline.training_pipeline import TrainingPipeline

trainer = TrainingPipeline()
trainer.build_video_library(
    video_dir="./data/raw",
    batch_size=10,
    skip_existing=True
)
```

### 2. CLIP 语义特征缓存

```python
from core.clip_cache import CLIPCacheManager
from core.milvus_manager import MilvusManager

milvus = MilvusManager()
cache_manager = CLIPCacheManager(milvus)

# 缓存视频的 CLIP 特征
cache_manager.insert_features(
    video_id="video_123",
    timestamps=[0.0, 1.0, 2.0],
    features=[feature1, feature2, feature3]
)

# 查询缓存特征
cached_features = cache_manager.get_features("video_123")
```

### 3. 自定义侵权关键词库

编辑 `config/settings.py` 中的关键词配置：

```python
DEFAULT_CLIP_KEYWORDS = [
    "copyright infringement",
    "stolen content",
    "copied scene",
    "plagiarism"
]

DEFAULT_CLIP_KEYWORDS_CN = [
    "侵权内容",
    "盗用视频",
    "抄袭画面",
    "非法复制"
]
```

## 🐛 故障排除

### 常见问题

#### 1. Milvus 连接失败
```
Error: 连接 Milvus 失败
```
**解决方案**:
- 确认 Milvus 服务正在运行：`docker ps | grep milvus`
- 检查端口配置：默认 `localhost:19530`
- 验证防火墙设置

#### 2. CLIP 模型加载失败
```
Error: CLIP 模型加载失败
```
**解决方案**:
- 确保网络连接正常（需要从 GitHub 下载）
- 手动安装 CLIP：`pip install git+https://github.com/openai/CLIP.git`
- 检查 PyTorch 版本兼容性

#### 3. 视频处理错误
```
Error: 无法打开视频文件
```
**解决方案**:
- 安装 FFmpeg：`apt-get install ffmpeg` 或 `brew install ffmpeg`
- 检查视频文件格式是否支持
- 确保有足够的磁盘空间

#### 4. 内存不足
```
Error: CUDA out of memory
```
**解决方案**:
- 减小批量处理大小
- 启用 CPU 模式：设置 `device='cpu'`
- 减少 `MAX_FRAMES_PER_VIDEO` 配置

### 日志查看

```bash
# 查看系统日志
tail -f logs/system.log

# 启用调试日志
export LOG_LEVEL=DEBUG
python app.py
```

## 📈 性能优化建议

### 1. 硬件加速
- **GPU 支持**: 启用 CUDA 加速特征提取
- **内存优化**: 调整 `MAX_FRAMES_PER_VIDEO` 减少内存使用
- **存储优化**: 启用 `PERMANENTLY_KEEP_FRAMES=False` 减少磁盘占用

### 2. 检索优化
- **索引优化**: 根据视频库大小调整 Milvus 索引参数
- **缓存策略**: 启用 CLIP 特征缓存减少重复计算
- **分级检索**: 先 PHash 粗筛，再深度学习精筛

### 3. 部署优化
- **容器化**: 使用 Docker 容器化部署
- **负载均衡**: 多实例部署支持高并发
- **监控告警**: 集成 Prometheus + Grafana 监控

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境搭建

1. Fork 项目仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加适当的文档字符串（docstrings）
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- **OpenAI CLIP**: 提供强大的视觉-语言预训练模型
- **Milvus**: 提供高效的向量相似性检索
- **Towhee**: 提供便捷的特征提取框架
- **Gradio**: 提供易用的 Web 界面框架

## 📞 支持与反馈

如果您遇到任何问题或有改进建议：

1. 查看 [Issues](https://github.com/your-repo/issues) 页面
2. 提交新的 Issue 报告问题
3. 通过邮件联系我们

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**