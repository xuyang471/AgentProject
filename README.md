# 多模态图文智能分析助手

这是一个基于 `Streamlit + Python + Qwen API + LangChain` 的轻量级多模态文档分析系统，支持上传 `PDF / DOCX / PNG / JPG` 文件，完成图文内容提取、Markdown 总结报告生成，以及基于文档内容的问答交互。

当前版本已经不只是简单的 MVP 页面原型，而是具备以下工程能力的本地系统：

- 多模态图文解析
- RAG 检索问答
- LangChain + LangGraph 工具调用型 Agent 工作流
- SQLite 轻量持久化状态管理
- 历史会话手动保存与切换
- 本地 stdio MCP 风格接入层
- 答案校验与风险提示

## 1. 技术栈

- 前端界面：`Streamlit`
- 核心语言：`Python`
- PDF 解析：`pypdf`、`pdfplumber`、`PyMuPDF`
- DOCX 解析：`python-docx`
- 图片处理：`Pillow`
- 本地 OCR：`pytesseract`
- 云端模型：`Qwen`
  - 文本总结与问答：`qwen-plus`
  - 图片理解：`qwen3-vl-flash`
  - 向量模型：`text-embedding-v4`
- RAG 检索：`BM25 + FAISS + embedding`
- Agent 编排：`LangChain + LangGraph`
- 状态管理：`SQLite`
- MCP 风格接入：本地 `stdio server + client`

## 2. 当前已实现能力

### 2.1 文件处理

- 支持上传 `PDF / DOCX / PNG / JPG / JPEG`
- 单次最多上传 5 个文件，总大小不超过 20MB
- 支持提取 PDF 文本、DOCX 段落文本
- 支持抽取 PDF / DOCX 中的嵌入图片
- 支持从 PDF 中提取表格，并转成 Markdown 表格文本块
- 对论文类 PDF 增加了图注区域裁切与整页渲染回退，减少图像漏检和整页截图噪声

### 2.2 多模态理解

- 图片支持本地 OCR
- 图片支持 Qwen 多模态分析
- 图片块会保留来源、尺寸、OCR 文本、图片描述
- 表格块会进入统一内容块结构，参与报告生成和检索

### 2.3 报告生成

- 生成固定结构的 Markdown 总结报告
- 支持在线预览和下载
- 报告中可嵌入检测到的图片
- 报告中可嵌入表格预览图
- 报告会附带来源信息，降低幻觉风险

### 2.4 问答系统

- 支持 RAG 问答
- 检索层为 `BM25 + FAISS + embedding` 混合检索
- 支持 LangChain Retrieval Chain
- 支持工具调用型 Agent
- 支持 LangGraph 状态图编排 RAG / Agent 分支与回退
- 支持计算器工具与联网查询工具扩展
- 问答结果显示来源片段、图片说明和图片引用，并可直接渲染本地图片

### 2.5 状态管理

- 已接入 SQLite 轻量持久化状态管理
- 支持保存：
  - 分析会话
  - 文档记录
  - 文档块
  - 报告记录
  - 问答记录
  - Agent 运行记录
- 分析完成后不会自动写入 SQLite，需用户点击“保存到历史会话”后持久化
- 侧边栏支持查看并切换历史会话

### 2.6 MCP 进展

- 已实现本地 `MCP 风格资源 + 工具` 适配层
- 已实现 `stdio` 方式启动的本地 MCP server
- 已实现 Agent 侧 MCP client
- Agent 工具已优先通过 MCP client 访问资源和工具
- MCP 工具响应已统一为 `{ok, data, error, metadata}`
- 当前可挂接的外部工具包括文档检索、OCR、图片分析、计算器和联网查询

### 2.7 答案校验层

- 回答后会自动执行轻量校验
- 校验内容包括：
  - 是否存在证据支撑
  - 证据数量
  - 答案与证据的关键词重合度
  - 是否包含明显不确定性表达
  - 是否存在“答案较长但证据较少”的扩写风险
- 前端会展示：
  - 可信度
  - 是否有证据支撑
  - 支撑依据
  - 风险提示

## 3. 当前系统定位

当前系统更准确的定位是：

**“具备多模态解析、RAG、LangChain + LangGraph Agent 工作流、SQLite 状态管理、MCP 风格接入和答案校验层的文档分析 Agent 原型。”**

它已经明显超过了最初的页面级 MVP，但还没有达到真正的企业级 Agent。

## 4. 与正式方案的完成度

### 4.1 已基本完成

- 多格式文档上传
- 文本提取
- 图片抽取
- OCR + 图片理解基础链路
- Markdown 报告生成
- 单轮问答
- 在线预览与下载
- 来源保留与展示
- LangChain 化
- LangGraph 基础状态图编排
- Agent 原型
- SQLite 状态持久化
- 历史会话管理

### 4.2 部分完成

- 扫描版 PDF 支持
  - 已有图注感知和整页渲染回退
  - 但还不是完整的高质量整页 OCR 流程
- 表格理解
  - 已能抽取和进入检索
  - 但复杂跨行跨列表格仍有局限
- MCP
  - 已有本地 stdio server + client
  - 但还不是完全标准化的 MCP 协议实现
- Agent
  - 已能工具调用
  - 但规划、失败恢复和复杂编排能力仍偏轻量
- 答案校验
  - 已有规则化校验
  - 但还没有做 LLM 二次核验和更强一致性校验

### 4.3 尚未完成

- 扫描版 PDF 的高质量整页 OCR
- 更强的图像区域级定位
- 更细粒度的图表结构理解
- 检索 reranker
- 多文档联合分析与对比
- 问答来源高亮定位
- 更完整的工具治理、审计和安全层

## 5. 当前完成度评估

如果按原始正式方案的 MVP 目标衡量，当前系统大致可以认为：

- 核心主链路完成度：`85% 左右`
- 可演示程度：`高`
- 可用于课程设计 / 毕设原型展示：`可以`
- 与企业级 Agent 的距离：`仍有明显差距`

最明显的增强点在于：

- 状态可恢复
- Agent 工具调用更完整
- 引入了 MCP 风格接入层
- 增加了答案校验层

## 6. 项目结构

```text
project/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .env
├─ parsers/
├─ services/
├─ storage/
├─ langchain_app/
├─ mcp_server/
├─ output/
├─ prompts/
└─ tests/
```

## 7. 启动方式

### 7.1 安装依赖

```bash
pip install -r requirements.txt
```

说明：

- 如果启用本地 OCR，除了安装 `pytesseract` 之外，还需要在系统中安装 `Tesseract OCR`
- 如果当前环境缺少部分依赖，系统会尽量回退而不是直接崩溃

### 7.2 配置环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_TEXT_MODEL=qwen-plus
QWEN_VISION_MODEL=qwen3-vl-flash
QWEN_EMBEDDING_MODEL=text-embedding-v4
TAVILY_API_KEY=你的联网搜索APIKey
```

### 7.3 启动 Web 应用

```bash
streamlit run app.py
```

### 7.4 运行测试

```bash
python -m unittest discover -s tests -v
```

## 8. 当前局限

- 扫描版 PDF 的支持仍不够完整
- 本地 OCR 仍依赖系统安装的 Tesseract
- 中文 OCR 和复杂图表理解仍有波动
- MCP 目前是本地 stdio 版本，离更标准协议实现还有距离
- Agent 仍属于轻量原型，缺少复杂规划与失败恢复
- 检索层虽然已是混合检索，但还没有 rerank 和系统化评测
- 答案校验目前主要是规则层，还没有引入 LLM 二次核验

## 9. 下一步建议

- 为扫描版 PDF 增加更完整的整页 OCR 流程
- 将 `pytesseract` 升级为 `PaddleOCR`
- 为表格、图片、正文建立更细粒度的检索与 rerank
- 将答案校验升级为“规则 + LLM 二次核验”
- 为 `qa_record` 和 `agent_run` 增加可信度字段
- 继续推进 MCP 标准化
- 增强 Agent 的规划、失败恢复和工具治理能力
- 增加结构化日志、追踪与成本统计
