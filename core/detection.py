"""YOLO 推理、中文缺陷标签与结果导出工具。"""
from __future__ import annotations

import io
import json
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best.pt"
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_HISTORY = 30
PREDICTION_LOCK = threading.Lock()

# 只对能匹配的模型标签翻译；未匹配时保留真实标签，避免错误映射。
NAME_MAP = {
    "crazing": "裂纹", "crack": "裂纹", "cracks": "裂纹", "cr": "裂纹",
    "inclusion": "夹杂物", "inclusions": "夹杂物", "in": "夹杂物",
    "patches": "斑点", "patch": "斑点", "pa": "斑点",
    "pitted surface": "麻面", "pitted_surface": "麻面", "pitted": "麻面", "ps": "麻面",
    "rolled in scale": "轧入氧化皮", "rolled-in scale": "轧入氧化皮",
    "rolled_in_scale": "轧入氧化皮", "rs": "轧入氧化皮",
    "scratches": "划痕", "scratch": "划痕", "sc": "划痕",
}
COLORS = ["#36cfc9", "#60a5fa", "#f59e0b", "#fb7185", "#a78bfa", "#34d399", "#f97316"]


@dataclass
class Detection:
    class_id: int
    class_en: str
    class_zh: str
    confidence: float
    bbox: tuple[int, int, int, int]


def local_name(name: Any) -> str:
    value = str(name).strip()
    key = re.sub(r"\s+", " ", value.lower().replace("_", " "))
    return NAME_MAP.get(key, value)


def class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"类别 {class_id}")))
    return str(names[class_id]) if 0 <= class_id < len(names) else f"类别 {class_id}"


def open_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("文件为空或超过 12 MB，请更换图片。")
    try:
        with Image.open(io.BytesIO(raw)) as original:
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError("图片像素过大（超过 2000 万像素）。")
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("无法读取图片，请上传有效的 JPG 或 PNG 文件。") from error


def load_yolo():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    if model.task != "detect":
        raise RuntimeError(f"当前模型任务为 {model.task}，网站需要 YOLO detect 检测模型。")
    return model


def model_labels(model: Any) -> list[tuple[int, str, str]]:
    names = model.names
    indices = sorted(int(i) for i in names) if isinstance(names, dict) else range(len(names))
    return [(index, class_name(names, index), local_name(class_name(names, index))) for index in indices]


def annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=max(12, min(22, image.width // 35)))
    line_width = max(2, image.width // 300)
    for det in detections:
        color = COLORS[det.class_id % len(COLORS)]
        x1, y1, x2, y2 = det.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        # 英文来自权重模型，确保部署环境无需中文字体也能绘制文字。
        label = f"{det.class_en} {det.confidence:.1%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(0, y1 - text_h - 8)
        tx = max(0, min(x1, image.width - text_w - 10))
        draw.rectangle((tx, ty, min(image.width, tx + text_w + 10), ty + text_h + 7), fill=color)
        draw.text((tx + 5, ty + 2), label, font=font, fill="#0c1728", stroke_width=0)
    return output


def image_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def infer(model: Any, raw: bytes, filename: str, conf: float, iou: float) -> dict:
    image = open_image(raw)
    start = time.perf_counter()
    # 在多会话中共用缓存模型时串行推理，防止共享模型的并发冲突。
    with PREDICTION_LOCK:
        result = model.predict(source=np.array(image), conf=conf, iou=iou, device="cpu", verbose=False)[0]
    elapsed = time.perf_counter() - start
    detections: list[Detection] = []
    names = result.names
    if result.boxes is not None:
        for coords, score, cls in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            index = int(cls)
            english = class_name(names, index)
            x1, y1, x2, y2 = (int(round(v)) for v in coords)
            x1, x2 = sorted((max(0, min(x1, image.width - 1)), max(0, min(x2, image.width - 1))))
            y1, y2 = sorted((max(0, min(y1, image.height - 1)), max(0, min(y2, image.height - 1))))
            detections.append(Detection(index, english, local_name(english), float(score), (x1, y1, x2, y2)))
    marked = annotate(image, detections)
    return {
        "filename": filename,
        "detections": [asdict(d) for d in detections],
        "count": len(detections),
        "status": "检出缺陷" if detections else "未检出缺陷",
        "elapsed": round(elapsed, 3),
        "size": [image.width, image.height],
        "original_png": image_png_bytes(image),
        "marked_png": image_png_bytes(marked),
        "threshold": {"confidence": conf, "iou": iou},
    }


def result_json_bytes(entry: dict) -> bytes:
    payload = {key: val for key, val in entry.items() if key not in ("original_png", "marked_png")}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def get_counts(records: list[dict]) -> Counter:
    return Counter(d["class_zh"] for rec in records for d in rec["detections"])
