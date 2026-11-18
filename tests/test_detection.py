from io import BytesIO

from PIL import Image

from core.detection import Detection, annotate, class_name, get_counts, local_name, open_image, result_json_bytes


def test_names():
    assert local_name("Pitted_Surface") == "麻面"
    assert local_name("Rolled-in Scale") == "轧入氧化皮"
    assert local_name("Other") == "Other"
    assert class_name({0: "Crazing", 1: "Scratches"}, 1) == "Scratches"


def test_open_and_draw():
    buffer = BytesIO()
    Image.new("RGB", (200, 120), "gray").save(buffer, "PNG")
    image = open_image(buffer.getvalue())
    result = annotate(image, [Detection(0, "Crazing", "裂纹", .9, (5, 10, 100, 70))])
    assert result.size == (200, 120)
    assert image.getpixel((5, 10)) != result.getpixel((5, 10))


def test_counts_and_export():
    rows = [{"detections": [{"class_zh": "划痕"}, {"class_zh": "划痕"}]}]
    assert get_counts(rows)["划痕"] == 2
    payload = result_json_bytes({"id": "x", "marked_png": b"123", "count": 2})
    assert b"marked_png" not in payload
    assert b'"count": 2' in payload
