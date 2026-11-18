# SteelVision · 钢材表面缺陷智能检测系统

基于上传的自定义 YOLO `best.pt` 权重与 Streamlit 实现。老师通过公开网址即可上传 JPG/PNG 图片、获取标注结果与缺陷分类。**不需要老师安装软件**。

## 功能

- **系统首页**：模型支持的缺陷类别、系统状态和检测流程
- **智能检测**：上传 1–5 张图片、调节置信度/IoU、运行 YOLO、对比原图/检测图、显示结论与缺陷明细
- **数据统计**：当前浏览器会话内的检测图片、缺陷框统计和类别图表
- **历史记录**：本次会话最多保留 30 次，允许查看、导出、主动清空
- **系统设置**：读取权重中的实际类别名、显示任务类型和数据保护说明
- **下载**：PNG 检测标注图、JSON 检测报告和会话 CSV 概览

## 项目结构

```text
SteelDefectAI/
├── app.py                   # Streamlit 主入口
├── core/detection.py        # 模型加载、推理、标注、结果导出
├── model/best.pt            # 你的 YOLO 训练权重（已打包）
├── requirements.txt         # Python / CPU PyTorch 依赖
├── .python-version          # Python 3.11
├── .streamlit/config.toml   # 主题与上传配置
├── DATASET_README.md        # 原始 NEU-DET 数据集说明
├── tests/test_detection.py  # 无需模型推理的辅助功能测试
└── README.md
```

## 本地运行（建议先验证）

推荐 **Python 3.11**，进入项目目录执行：

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# macOS/Linux：source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。首次启动会加载权重，在 CPU 上推理可能比 Colab GPU 慢。

> 若在你的操作系统上无法安装 `+cpu` 轮子，可先在 Python 3.11 的 Linux 环境（与云端一致）安装运行，或者将 requirements 中 PyTorch 依赖调整成适配平台的版本；部署前要确认环境兼容。模型权重由 PyTorch 反序列化，请勿使用来源不明的 `.pt` 文件。

## 免费部署到 Streamlit Community Cloud

1. 登录 GitHub，**新建一个仓库**（如 `steel-defect-ai`）。将整个项目的文件、`core/`、`model/`、`.streamlit/` 等目录上传到仓库；隐藏文件 `.python-version` 和 `.streamlit/config.toml` 也要上传。
2. 确保 GitHub 中存在 `app.py`、`requirements.txt` 和 `model/best.pt`，且权重文件不是 Git LFS 指针。模型约 6.3 MB，可以直接上传。
3. 登录 https://share.streamlit.io/ ，选择 **Create app** → **I have an app**，选择你的 GitHub 仓库与分支。
4. Main file path 填 `app.py`，Python 版本选择 **3.11**（如界面提供运行时版本选项）；指定想要的子域名，点击 Deploy。
5. 构建完成后会得到 `https://你的子域名.streamlit.app`。把地址发给老师即可。若休眠，访问者按提示唤醒后可再使用。

**重要：** 代码交付不等于已经上线。公开网址需要你登录 GitHub、Streamlit Cloud 自行授权及部署。本项目不能保证免费云平台一直在线、永不调整政策或免于休眠。

## 数据保存说明

- 网站不接数据库，不把上传的原图/结果写入磁盘；图片及检测结果只在 `st.session_state` 内临时保留（服务器内存）。
- 同一个浏览器会话中切换页面，历史和统计仍可查看；浏览器会话断开、应用重启、休眠/唤醒，历史**不保证保留**，这符合当前设计要求。
- 多位老师/同学的会话数据相互独立。若更注重隐私，建议不要上传敏感/受保密限制的图像。

## 检测结论解释

- **检出缺陷**：当前置信度和 NMS IoU 设置下，模型返回了一个或多个检测框。
- **未检出缺陷**：当前阈值下没有检测框，**不能直接当作工业质量合格证明**。
- 图像缺陷框上的文字使用模型的原始英文标签（无需安装中文字体）；结果表格展示中文标签。
- 网站直接读取权重中的 `model.names`，按名称翻译已知的六类缺陷，**不会擅自假设类别 ID 与 README 列表顺序一致**。未匹配的标签显示原始名称。
- NEU-DET 数据集主要包含缺陷样本；建议单独用无缺陷图测试误检率及漏检率。

## 常见部署问题

- **构建失败 / pip 错误**：检查 Python 版本是否为 3.11；查看 Streamlit Cloud 的构建日志。若 CPU PyTorch 下载异常，确认镜像源和 PyTorch 版本能在当前平台安装。
- **模型加载失败**：检查文件路径 `model/best.pt`，并确保上传的是实际模型权重而非 LFS 指针。
- **休眠后历史消失**：符合当前会话临时存储设计，不影响模型文件和网站代码。
- **第一次检测慢**：云端是 CPU 推理，首轮模型预热及依赖加载可能有延迟。

## 项目依赖与参考

- Streamlit 部署：https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit 依赖：https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Ultralytics 推理：https://docs.ultralytics.com/modes/predict/
