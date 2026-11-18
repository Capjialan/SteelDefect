"""钢材表面缺陷智能检测系统 · Streamlit 公网演示版本。"""
from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.detection import (
    MAX_HISTORY, MAX_IMAGE_BYTES, MODEL_PATH, get_counts, infer, load_yolo,
    local_name, model_labels, result_json_bytes,
)

st.set_page_config(page_title="SteelVision | 钢材表面缺陷智能检测", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 80% 0%, #162b45 0, #0b1221 40%, #090f1b 100%); color: #e7eefb; }
  [data-testid="stSidebar"] { background: #101b2d; border-right: 1px solid #25344b; }
  h1, h2, h3 { color: #eff6ff !important; letter-spacing: -.03em; }
  .hero { padding: 26px 30px; border: 1px solid #334862; border-radius: 18px; background: linear-gradient(110deg, #172d49, #111d31); margin-bottom: 18px; }
  .eyebrow { color: #58d6d0; font-size: 12px; letter-spacing: .2em; font-weight: 800; }
  .hero h1 { margin: 8px 0; font-size: 34px; }
  .muted { color: #acbdd4; font-size: 14px; }
  .panel { border: 1px solid #2c425e; background: #122139; border-radius: 13px; padding: 16px; margin: 8px 0 16px; }
  .tag { display:inline-block; padding:4px 10px; border-radius:20px; background:#1a3746; color:#65e3d6; font-size:12px; font-weight:600; }
  .small-note { color: #aebfd5; font-size: 12px; }
  div[data-testid="stMetric"] { padding: 12px 16px; background: #122139; border: 1px solid #2c425e; border-radius: 12px; }
  div[data-testid="stMetricLabel"] { color: #b6cae2; }
  .stButton > button[kind="primary"] { background: #20b6b0; color: #081823; border: none; font-weight: 700; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "records" not in st.session_state:
    st.session_state.records = []
if "last_result_id" not in st.session_state:
    st.session_state.last_result_id = None

@st.cache_resource(show_spinner="正在加载 YOLO 模型，请稍候……")
def cached_model():
    return load_yolo()

try:
    model = cached_model()
    categories = model_labels(model)
    model_ready = True
except Exception as exc:
    model_ready = False
    categories = []
    model_error = str(exc)

with st.sidebar:
    st.markdown("### ◈ STEELVISION")
    st.caption("INDUSTRIAL DEFECT INSPECTION")
    st.divider()
    page = st.radio("导航", ["🏠 系统首页", "🔍 智能检测", "📊 数据统计", "🕘 历史记录", "⚙️ 系统设置"], label_visibility="collapsed")
    st.divider()
    st.markdown("**系统状态**")
    if model_ready:
        st.success("● 模型已就绪")
        st.caption(f"YOLO Detect · {len(categories)} 个类别 · CPU 推理")
    else:
        st.error("● 模型加载失败")
    st.caption("临时会话记录 · 不保存到云端数据库")
    st.caption("部署版本 · v1.0")

if not model_ready:
    st.error(f"模型无法加载：{model_error}")
    st.info("请确认 model/best.pt 已上传，并根据 README 中的部署指南检查依赖及 Python 版本。")
    st.stop()


def header(title: str, subtitle: str, eyebrow: str = "AI QUALITY INSPECTION"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><div class="muted">{subtitle}</div></div>', unsafe_allow_html=True)


def show_result(entry: dict, key_prefix: str):
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown("#### 原始图像")
        st.image(entry["original_png"], use_container_width=True)
    with cols[1]:
        st.markdown("#### AI 检测结果")
        st.image(entry["marked_png"], use_container_width=True)
    st.markdown("#### 判别结论")
    if entry["count"]:
        st.error(f"⚠️ 检出 {entry['count']} 处疑似缺陷，建议人工复核")
    else:
        st.success("✅ 在当前阈值下未检出缺陷（不等于保证无缺陷）")
    c1, c2, c3 = st.columns(3)
    c1.metric("检测框数量", entry["count"])
    c2.metric("推理耗时", f"{entry['elapsed']:.2f} s")
    c3.metric("图片分辨率", f"{entry['size'][0]} × {entry['size'][1]}")
    if entry["detections"]:
        rows = [
            {"编号": i, "缺陷类别": d["class_zh"], "模型标签": d["class_en"], "置信度": f"{d['confidence']:.1%}", "位置(x1,y1,x2,y2)": str(d["bbox"])}
            for i, d in enumerate(entry["detections"], start=1)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ 下载标注图片 PNG", data=entry["marked_png"], file_name=f"result_{entry['id']}.png", mime="image/png", key=f"img_{key_prefix}_{entry['id']}", use_container_width=True)
    with d2:
        st.download_button("⬇️ 下载检测报告 JSON", data=result_json_bytes(entry), file_name=f"report_{entry['id']}.json", mime="application/json", key=f"json_{key_prefix}_{entry['id']}", use_container_width=True)
    st.caption(f"识别参数：置信度阈值 {entry['threshold']['confidence']:.2f}，IoU 阈值 {entry['threshold']['iou']:.2f}。检测结果仅供教学演示，不构成工业质量保证。")


if page == "🏠 系统首页":
    header("钢材表面缺陷智能检测系统", "基于 YOLO 深度学习 · 一键上传钢材图片 · 自动定位并识别疑似缺陷", "STEELVISION / DEEP LEARNING")
    a, b, c = st.columns(3)
    a.metric("可识别类别", len(categories))
    b.metric("当前会话检测", len(st.session_state.records))
    c.metric("推理环境", "CPU 云端")
    st.markdown("### 识别类别")
    for start in range(0, len(categories), 3):
        cols = st.columns(3)
        for col, (index, en, zh) in zip(cols, categories[start:start + 3]):
            col.markdown(f'<div class="panel"><div class="tag">CLASS {index:02d}</div><h3>{zh}</h3><div class="muted">{en}</div></div>', unsafe_allow_html=True)
    st.info("使用左侧「智能检测」进入上传和识别页面。检测类别由模型权重自动读取；如类别与预期不符，请检查训练数据标签。")
    st.markdown("### 检测流程")
    st.markdown("**上传图片**　→　**YOLO 推理**　→　**缺陷框标注**　→　**查看分类与置信度**　→　**下载检测报告**")

elif page == "🔍 智能检测":
    header("智能图像检测", "支持 JPG / PNG 格式，最多一次检测 5 张图片；结果保留在当前浏览器会话中。")
    u1, u2 = st.columns([3, 2])
    with u1:
        uploads = st.file_uploader("上传待检测钢材表面图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="单张图片最大 12 MB，最多 5 张。")
    with u2:
        conf = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU 阈值（NMS）", 0.10, 0.90, 0.45, 0.05)
    if uploads:
        st.caption(f"已选择 {len(uploads)} 张图片。点击「开始 AI 检测」后执行推理。")
    if st.button("🚀 开始 AI 检测", type="primary", use_container_width=True, disabled=not uploads):
        if len(uploads) > 5:
            st.error("单次最多上传 5 张图片。")
        else:
            added = 0
            progress = st.progress(0, text="正在检测……")
            for index, upload in enumerate(uploads):
                try:
                    raw = upload.getvalue()
                    if len(raw) > MAX_IMAGE_BYTES:
                        raise ValueError("图片超过 12 MB")
                    entry = infer(model, raw, upload.name, conf, iou)
                    entry["id"] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.records.insert(0, entry)
                    st.session_state.records = st.session_state.records[:MAX_HISTORY]
                    added += 1
                except Exception as exc:
                    st.warning(f"{upload.name} 检测失败：{exc}")
                progress.progress((index + 1) / len(uploads), text=f"已完成 {index + 1} / {len(uploads)} 张")
            if added:
                st.session_state.last_result_id = st.session_state.records[0]["id"]
                st.success(f"检测完成！成功处理 {added} 张图片。")
    if st.session_state.last_result_id:
        record = next((r for r in st.session_state.records if r["id"] == st.session_state.last_result_id), None)
        if record:
            st.divider()
            st.markdown(f"### 最近检测：{record['filename']}")
            show_result(record, "last")
    st.caption("提示：当前会话内的历史可在左侧「历史记录」中查看；应用休眠、重启或会话断开后不保证保留。")

elif page == "📊 数据统计":
    header("当前会话数据统计", "数据只统计你本次浏览器会话里的检测记录，不包含其他访问者。")
    records = st.session_state.records
    counts = get_counts(records)
    a, b, c, d = st.columns(4)
    a.metric("累计检测图片", len(records))
    b.metric("检出缺陷图片", sum(r["count"] > 0 for r in records))
    c.metric("未检出缺陷图片", sum(r["count"] == 0 for r in records))
    d.metric("累计缺陷框", sum(r["count"] for r in records))
    if counts:
        st.markdown("### 各类缺陷检出次数")
        chart = pd.DataFrame([{"缺陷类别": name, "检出数量": count} for name, count in counts.most_common()]).set_index("缺陷类别")
        st.bar_chart(chart, color="#31c4bd")
        st.dataframe(chart.reset_index(), hide_index=True, use_container_width=True)
    else:
        st.info("还没有缺陷统计数据，先到「智能检测」上传图片吧。")
    st.caption("同一图片有多个缺陷框时，每个框均计入类别统计。")

elif page == "🕘 历史记录":
    header("会话检测历史", f"本页面最多保留最近 {MAX_HISTORY} 次检测，应用或浏览器会话结束后不保证保留。")
    records = st.session_state.records
    if not records:
        st.info("当前会话还没有检测历史。")
    else:
        df = pd.DataFrame([{"检测时间": r["timestamp"], "图片名称": r["filename"], "判别结果": r["status"], "缺陷框数量": r["count"], "推理耗时(s)": r["elapsed"]} for r in records])
        st.dataframe(df, use_container_width=True, hide_index=True)
        options = {f"{idx + 1}. {r['filename']} · {r['timestamp']}": r for idx, r in enumerate(records)}
        selected = st.selectbox("查看历史结果", list(options))
        show_result(options[selected], "history")
        c1, c2 = st.columns(2)
        with c1:
            overview = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载历史汇总 CSV", overview, "session_history.csv", "text/csv", use_container_width=True)
        with c2:
            if st.button("清空当前会话历史", use_container_width=True):
                st.session_state.records = []
                st.session_state.last_result_id = None
                st.rerun()

elif page == "⚙️ 系统设置":
    header("系统设置与模型信息", "模型随网站部署，访问者无需额外下载。")
    st.markdown("### 模型信息")
    st.write(f"**权重文件：** `{MODEL_PATH.relative_to(Path(__file__).parent)}`")
    st.write(f"**模型任务：** `{model.task}`")
    st.write(f"**类别数量：** `{len(categories)}`")
    st.write("**推理设备：** `CPU`（适配免费云端，无需 GPU）")
    st.markdown("### 权重中的类别名称")
    st.dataframe(pd.DataFrame([{"类别ID": n, "原始模型标签": en, "中文展示": zh} for n, en, zh in categories]), hide_index=True, use_container_width=True)
    st.markdown("### 隐私与数据")
    st.info("图片和检测结果仅保存在当前会话的服务器内存中用于演示，不写入永久数据库；上传内容仍需经服务器处理，因此不要上传敏感或保密图像。")
    st.markdown("### 使用限制")
    st.caption("采用 12 MB 单图 / 5 张单批 / 最近 30 条会话记录的限制，以控制免费 CPU 云环境的内存与推理负载。")
