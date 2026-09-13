"""Create the faculty progress PDF from saved, executed project evidence.

Requires reportlab (optional documentation dependency, not needed to train).
"""

from pathlib import Path
import csv
import json
import re

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/UVH26_Faculty_Progress_Report.pdf"
NAVY = colors.HexColor("#102E46")
TEAL = colors.HexColor("#007D80")
GRAY = colors.HexColor("#526575")
LIGHT = colors.HexColor("#EEF4F6")
AMBER = colors.HexColor("#FFF4D9")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="Cover",
        fontName="Helvetica-Bold",
        fontSize=32,
        leading=37,
        textColor=NAVY,
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="Kicker",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=TEAL,
        spaceAfter=13,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=NAVY,
        spaceAfter=16,
    )
)
styles.add(
    ParagraphStyle(
        name="Copy",
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=NAVY,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallCopy",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=GRAY,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell", fontName="Helvetica", fontSize=9, leading=12, textColor=NAVY
    )
)
styles.add(
    ParagraphStyle(
        name="TableHead",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
    )
)
styles.add(
    ParagraphStyle(
        name="CodeCopy",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=NAVY,
        spaceAfter=9,
    )
)


def P(text, style="Copy"):
    return Paragraph(str(text), styles[style])


def load(path):
    return json.loads((ROOT / path).read_text())


def table(rows, widths, header=True):
    cells = [
        [P(str(c), "TableHead" if header and i == 0 else "TableCell") for c in row]
        for i, row in enumerate(rows)
    ]
    t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5E0E5")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]
    else:
        commands += [("BACKGROUND", (0, 0), (-1, -1), LIGHT)]
    t.setStyle(TableStyle(commands))
    return t


def callout(text, color=LIGHT):
    t = Table([[P(text)]], colWidths=[503])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5E0E5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 13),
                ("RIGHTPADDING", (0, 0), (-1, -1), 13),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.line(46, h - 35, w - 46, h - 35)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(46, 27, "UVH-26 | Faculty progress review | 11 September 2026")
    canvas.drawRightString(w - 46, 27, f"{doc.page} / 5")
    canvas.restoreState()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    eda = load("reports/tables/eda_summary.json")
    run = load("reports/tables/yolov8n_uvh26_mv_smoke_seed42_provenance.json")
    metrics = load("reports/tables/smoke_validation_seed42_metrics.json")
    passed = re.search(
        r"(\d+) passed", (ROOT / "reports/audit/pytest.txt").read_text()
    ).group(1)
    with (ROOT / "reports/tables/class_mapping.csv").open() as f:
        mapping = list(csv.DictReader(f))
    with (
        ROOT / "reports/tables/uvh26_mv_baseline_subset_v1_distribution.csv"
    ).open() as f:
        dist = list(csv.DictReader(f))
    max_delta = max(abs(float(r["share_difference_pp"])) for r in dist)
    story = []
    story += [
        Spacer(1, 25),
        P("DEEP LEARNING PROJECT / PHASE 1", "Kicker"),
        P("Vehicle Detection<br/>for Indian<br/>Urban Roads", "Cover"),
        P("YOLOv8 + IISc AIM UVH-26", "SectionTitle"),
        P("Manav Shah | Fourth-year undergraduate project", "Copy"),
        P("Faculty progress deliverable - 11 September 2026", "SmallCopy"),
        Spacer(1, 13),
    ]
    story += [
        table(
            [
                ["26,646", "316,220", "14", passed],
                ["Images listed", "MV object boxes", "Vehicle classes", "Tests passed"],
            ],
            [126, 126, 125, 126],
            False,
        ),
        Spacer(1, 19),
    ]
    story += [
        callout(
            "<b>Verified progress, with a clear experimental boundary.</b><br/>Dataset annotation audit, EDA, data-pipeline implementation and a genuine one-epoch training smoke test are complete. Full image acquisition, full image integrity checks and the proper baseline remain in progress.",
            AMBER,
        ),
        Spacer(1, 19),
    ]
    story += [
        P("The problem", "Kicker"),
        P(
            "Indian traffic combines dense scenes, occlusion and fine-grained vehicle types. This project prepares a reproducible detector for categories such as two-wheelers, three-wheelers, LCVs and different passenger-car types."
        ),
        P(
            "<b>Pipeline:</b> pinned dataset -&gt; audit -&gt; COCO-to-YOLO conversion -&gt; validation -&gt; transfer learning -&gt; evaluation."
        ),
        P(
            "Real-time analytics is the project objective. End-to-end real-time performance and final detection accuracy have not yet been established.",
            "SmallCopy",
        ),
    ]
    story += [
        PageBreak(),
        P("01 / Dataset evidence", "Kicker"),
        P("Inspect first. Preserve the splits.", "SectionTitle"),
    ]
    story += [
        table(
            [
                ["Majority Voting split", "Images listed", "Object boxes"],
                ["Training", "21,349", "252,723"],
                ["Validation", "5,297", "63,497"],
                ["Total", "26,646", "316,220"],
            ],
            [253, 125, 125],
        ),
        Spacer(1, 12),
    ]
    story += [
        P(
            "These are counts from the downloaded annotation JSON files. They do not imply that all image files are already present. MV is the selected ground truth; STAPLE is kept separate because its inspected files list 21,726 images and 283,402 objects."
        ),
        P(
            "All 14 original category names are preserved. Sorted original IDs 1-14 map to YOLO IDs 0-13.",
            "SmallCopy",
        ),
    ]
    rows = [["YOLO ID", "Category", "MV instances"]]
    rows += [
        [c["yolo_id"], c["name"], f"{eda['classes'][c['name']]:,}"] for c in mapping
    ]
    story += [
        table(rows, [72, 306, 125]),
        Spacer(1, 12),
        P(
            "<b>Annotation-only checks:</b> zero invalid boxes, missing mandatory fields, duplicate IDs within each split or shared train-validation filenames. Full image decoding, dimension checks and content-hash leakage checks are still required.",
            "SmallCopy",
        ),
    ]
    story += [
        PageBreak(),
        P("02 / Exploratory analysis", "Kicker"),
        P("Imbalance and crowded scenes", "SectionTitle"),
    ]
    story += [
        Image(
            str(ROOT / "reports/figures/class_distribution.png"),
            width=503,
            height=251.5,
        ),
        P(
            "Two-wheelers: <b>149,730 instances</b>. Others: <b>352 instances</b>. Per-class reporting is essential; an aggregate score can hide poor rare-class performance.",
            "Copy",
        ),
        Image(
            str(ROOT / "reports/figures/object_center_heatmap.png"),
            width=463,
            height=231.5,
        ),
        P(
            f"Objects per image: mean <b>{eda['objects_per_image']['mean']:.2f}</b>, median <b>10</b>, maximum <b>66</b>. At original resolution, 4,041 boxes are smaller than 32 x 32 pixels in area. The heatmap shows where annotated object centres occur; it is not a traffic-density prediction.",
            "SmallCopy",
        ),
    ]
    story += [
        PageBreak(),
        P("03 / Executed engineering work", "Kicker"),
        P("A real training smoke test", "SectionTitle"),
    ]
    story += [
        P(
            "The 96-image pilot was independently decoded, dimension-checked and hashed across splits. Ultralytics scanned it with zero corrupt images. Training losses were finite, validation completed, and best/last checkpoints were saved."
        ),
        table(
            [
                ["Configuration", "Executed value"],
                ["Hardware", "Apple M5 MacBook Pro, 24 GiB memory"],
                ["Environment", "Python 3.12.14; PyTorch 2.14.0; Ultralytics 8.4.146"],
                ["Model / device", "COCO-pretrained YOLOv8n / MPS"],
                [
                    "Pilot / epochs",
                    "64 training images + 32 validation images / 1 epoch",
                ],
                ["Image size / batch", "640 / 8"],
                ["Optimizer", "AdamW; initial learning rate 0.000556"],
                ["Total wall time", f"{run['duration_seconds']:.2f} seconds"],
            ],
            [160, 343],
        ),
        Spacer(1, 13),
    ]
    story += [
        P("Standalone smoke validation - scores on a 0-to-1 scale", "Kicker"),
        table(
            [
                ["Precision", "Recall", "Macro F1", "mAP50", "mAP50-95"],
                [
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['macro_f1']:.6f}",
                    f"{metrics['map50']:.6f}",
                    f"{metrics['map50_95']:.6f}",
                ],
            ],
            [100, 100, 101, 101, 101],
        ),
        Spacer(1, 12),
        callout(
            "<b>This is not the proper baseline.</b> The smoke checkpoint has poor detection quality. All 32 saved predictions had zero detections at confidence 0.01. The run proves that the training/evaluation pipeline executes; it does not establish final project accuracy.",
            AMBER,
        ),
        Spacer(1, 12),
        P(
            f"<b>{passed} tests passed.</b> Synthetic tests cover geometry, class mapping, invalid inputs, empty labels, split manifests, reproducible selection and leakage blocking. MPS reported nondeterministic operations despite deterministic settings; exact numerical replay is not guaranteed.",
            "SmallCopy",
        ),
    ]
    story += [
        PageBreak(),
        P("04 / Demonstration and next milestone", "Kicker"),
        P("What can be shown today", "SectionTitle"),
    ]
    story += [
        table(
            [
                ["Show", "Evidence"],
                [
                    "Data analysis",
                    "Audited annotation counts, class chart and spatial heatmap",
                ],
                [
                    "Engineering",
                    "Versioned conversion code, immutable raw-data policy and tests",
                ],
                [
                    "Executed training",
                    "Smoke epoch CSV, checkpoint checksum and per-class evaluation",
                ],
                [
                    "Honest limitations",
                    "Missing/loose source labels, redacted imagery, unfinished full audit",
                ],
            ],
            [132, 371],
        ),
        Spacer(1, 13),
        P("A short offline demonstration", "Kicker"),
        P("python3 scripts/show_progress.py", "CodeCopy"),
        P(
            'Then open the EDA figures and walk through the source code. With dependencies installed, run <font face="Courier">python -m pytest -q</font>. The portable pack includes saved evidence; raw images and weights remain local.'
        ),
        P("Next: the proper baseline", "Kicker"),
        P(
            f"The deterministic 8,000/2,000 train/validation selection covers all 14 classes. The maximum class-share deviation from the corresponding full split is <b>{max_delta:.3f} percentage points</b>. Its annotations contain 94,484 training objects and 24,342 validation objects. Acquisition and preparation are in progress."
        ),
        P(
            "Planned settings: YOLOv8n, 30 epochs, image size 640, batch 8, seed 42, MPS. Finish full-image audit, inspect the prepared subset, run the proper baseline, evaluate its best checkpoint, review errors and then complete the Git checkpoint."
        ),
        P("Boundaries", "Kicker"),
        P(
            "No final accuracy or useful real-time deployment claim. No proper baseline commit or GitHub push has been completed. This deliverable is suitable for a progress review, not final project completion.",
            "SmallCopy",
        ),
        P("Sources and reproducibility", "Kicker"),
        P(
            'Dataset: <link href="https://huggingface.co/datasets/iisc-aim/UVH-26" color="#007D80">IISc AIM UVH-26</link>, CC BY 4.0.<br/>Revision: 59f82c57821e8a54dc40bc1f42e83909dbad0b70.<br/>Citation: Sharma et al., IISc Technical Report, November 2025, <link href="https://doi.org/10.48550/arXiv.2511.02563" color="#007D80">arXiv:2511.02563</link>.<br/>All project numbers above come from included audit, EDA, test and smoke-run artifacts.',
            "SmallCopy",
        ),
    ]
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=46,
        leftMargin=46,
        topMargin=51,
        bottomMargin=46,
        title="UVH-26 Faculty Progress Report",
        author="Manav Shah",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
