import argparse
from pathlib import Path
from collections import Counter
from src.data.common import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="reports/predictions")
    a = p.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    results = model.predict(
        source=a.source, conf=a.confidence, device=a.device, imgsz=640, verbose=False
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        name = f"{Path(r.path).stem}_{i}"
        r.save(filename=str(out / f"{name}.jpg"))
        detections = [
            {
                "class_id": int(c),
                "name": r.names[int(c)],
                "confidence": float(s),
                "xyxy": b,
            }
            for c, s, b in zip(
                r.boxes.cls.cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
                r.boxes.xyxy.cpu().tolist(),
            )
        ]
        data = {
            "detections": detections,
            "total": len(detections),
            "per_class": dict(Counter(d["name"] for d in detections)),
            "speed_ms": r.speed,
        }
        save_json(out / f"{name}.json", data)
        print(data)


if __name__ == "__main__":
    main()
