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
    canvas.drawString(46, 27, "UVH-26 | Faculty progress review | 13 September 2026")
    canvas.drawRightString(w - 46, 27, f"{doc.page} / 5")
    canvas.restoreState()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    name = "yolov8n_uvh26_mv_baseline_seed42_v1"
    run = load(f"reports/tables/{name}_provenance.json")
    metrics = load(f"reports/tables/{name}_validation_metrics.json")
    timing = load(f"reports/tables/{name}_latency.json")
    perclass = list(
        csv.DictReader(
            (ROOT / f"reports/tables/{name}_validation_per_class.csv").open()
        )
    )
    passed = re.search(
        r"(\d+) passed", (ROOT / "reports/audit/closeout_pytest.txt").read_text()
    ).group(1)
    story = [
        Spacer(1, 20),
        P("PHASE 1 / MEASURED SUBSET BASELINE", "Kicker"),
        P("Vehicle Detection<br/>for Indian<br/>Urban Roads", "Cover"),
        P("YOLOv8n + UVH-26 Majority Voting", "SectionTitle"),
        P("Manav Shah | Fourth-year undergraduate project"),
        P("Faculty progress review | 13 September 2026", "SmallCopy"),
        Spacer(1, 15),
        table(
            [
                ["30 / 30", "30", "8,000 / 2,000", passed],
                [
                    "Completed epochs",
                    "Best epoch",
                    "Train / validation",
                    "Tests passed",
                ],
            ],
            [120, 110, 160, 113],
            False,
        ),
        Spacer(1, 18),
        callout(
            "<b>Proper baseline training and evaluation are complete.</b><br/>Exit code 0; no early stopping. Fresh validation of best.pt and synchronized Apple MPS timing are recorded. Phase 2 training has not started."
        ),
        Spacer(1, 18),
        P("What this result means", "Kicker"),
        P(
            "A reproducible reference detector for 14 vehicle categories, including two-wheelers, three-wheelers and fine-grained car and commercial-vehicle types. The next research step is controlled improvement against this frozen reference."
        ),
        P(
            "<b>Scope:</b> subset validation, not full-dataset or independent test-set accuracy. Unselected-image acquisition and integrity remain incomplete. No production-readiness or real-time video claim.",
            "SmallCopy",
        ),
        PageBreak(),
        P("01 / DATA AND TRAINING EVIDENCE", "Kicker"),
        P("Two different audit scopes", "SectionTitle"),
        table(
            [
                ["Evidence", "Images", "Boxes"],
                ["Full MV annotation catalogue", "26,646", "316,220"],
                ["Frozen local training subset", "8,000", "94,609"],
                ["Frozen local validation subset", "2,000", "24,342"],
            ],
            [283, 110, 110],
        ),
        Spacer(1, 13),
        P(
            "The catalogue audit checks JSON records and box metadata. The separate local audit decoded all 10,000 selected images, checked dimensions, paths, labels and SHA-256 hashes, and found no train-validation content overlap. All 14 classes are present. Full unselected-image pixel verification is unfinished."
        ),
        P(
            "Recovery preserved original files: a 1620-versus-1920 width mismatch in 803489.png could not be safely repaired. It and two visually degraded candidates were quarantined. Deterministic replacements were audited before freezing v2; no validation labels were changed after training."
        ),
        P("The actual completed run", "Kicker"),
        table(
            [
                ["Setting", "Executed value"],
                ["Model / initialization", "YOLOv8n / COCO pretrained"],
                ["Epochs / best / stopping", "30 / 30 / no early stop; exit 0"],
                ["Image size / batch / seed", "640 / 8 / 42"],
                ["Optimizer", "AdamW; lr0 0.000556; weight decay 0.0005"],
                ["Warm-up / close mosaic", "3 epochs / final 5 epochs"],
                ["Device / effective AMP / workers", "Apple M5 MPS / false / 0"],
                [
                    "Training wall duration",
                    f"{run['duration_seconds']:.3f} s (4 h 27 min 10 s)",
                ],
                ["Runtime", "Python 3.12.14; torch 2.14.0; Ultralytics 8.4.146"],
            ],
            [194, 309],
        ),
        Spacer(1, 10),
        P(
            "Both 6,228,714-byte checkpoints load with finite tensors and the correct taxonomy. Hashes and frozen configuration are recorded in the Phase 1 report. MPS nondeterminism warnings prevent a bitwise replay guarantee. Original run files remain unchanged.",
            "SmallCopy",
        ),
        PageBreak(),
        P("02 / FRESH BEST-CHECKPOINT VALIDATION", "Kicker"),
        P("Measured detection accuracy", "SectionTitle"),
        table(
            [
                ["Precision", "Recall", "F1*", "Macro F1", "mAP50", "mAP50-95"],
                [
                    f"{metrics[k]:.4f}"
                    for k in [
                        "precision",
                        "recall",
                        "f1",
                        "macro_f1",
                        "map50",
                        "map50_95",
                    ]
                ],
            ],
            [83, 84, 84, 84, 84, 84],
        ),
        Spacer(1, 10),
        P(
            "2,000 validation images / 24,342 objects; best checkpoint from epoch 30. Fresh MPS evaluation: imgsz 640, batch 8, confidence floor 0.001, NMS IoU 0.7, max_det 300. AP spans IoU 0.50:0.05:0.95.",
            "SmallCopy",
        ),
        P(
            "*F1 is 2PR/(P+R) using macro P/R. Macro F1 averages 14 class F1 scores. Common confidence 0.309309 maximizes smoothed mean class F1 at IoU 0.5. Neither is micro-F1.",
            "SmallCopy",
        ),
    ]
    rows = [["Class", "Precision", "Recall", "AP50", "AP50-95"]]
    rows += [
        [r["name"]]
        + [f"{float(r[k]):.4f}" for k in ["precision", "recall", "ap50", "ap50_95"]]
        for r in perclass
    ]
    story += [
        table(rows, [171, 83, 83, 83, 83]),
        Spacer(1, 12),
        P(
            "Three-wheeler is strongest; Others and Mini-bus are weakest. Others has only 31 validation instances and zero recall at the F1 operating point; precision 1 is an empty-prediction interpolation convention. Mini-bus has only 58 instances. No test-set generalization or confidence intervals are established.",
            "SmallCopy",
        ),
        PageBreak(),
        P("03 / SPEED AND FAILURE ANALYSIS", "Kicker"),
        P("File-to-result timing", "SectionTitle"),
    ]
    rows = [["Stage", "Mean ms", "Median ms", "p95 ms"]]
    for k, v in timing["summary"].items():
        rows.append(
            [k.replace("_ms", "").replace("_", " ")]
            + [f"{v[x]:.3f}" for x in ["mean_ms", "median_ms", "p95_ms"]]
        )
    story += [
        table(rows, [230, 91, 91, 91]),
        Spacer(1, 12),
        P(
            f"<b>Inference-only: {timing['summary']['inference_ms']['fps_from_total_time']:.2f} FPS.</b> <b>End-to-end still images: {timing['summary']['end_to_end_ms']['fps_from_total_time']:.2f} FPS.</b> Rates are 1000 / mean milliseconds, not reciprocal median."
        ),
        P(
            "Apple M5, 24 GiB RAM, MPS float32, batch 1, imgsz 640 with rectangular letterboxing. Ten warm-ups excluded; 100 seed42 validation images timed. Each stage explicitly synchronizes MPS. Preprocessing converts BGR to RGB and normalizes; postprocessing uses class-aware NMS at conf 0.25 / IoU 0.7."
        ),
        P(
            "End-to-end includes local file read/decode, all processing, API overhead and synchronization barriers. It excludes model loading, drawing, video capture and UI. OS caching may affect file I/O. The library's unsynchronized MPS validation profiler is not used for these latency claims.",
            "SmallCopy",
        ),
        P("Ground-truth comparisons", "Kicker"),
        table(
            [
                ["Validation scene", "Observed behavior"],
                [
                    "4232 / correct detections",
                    "10 correct-class matches, including truck and two-wheelers; two extra predictions remain.",
                ],
                [
                    "10518 / small vehicle",
                    "Distant annotated two-wheeler missed; its box is about 65 square pixels after scaling to 640.",
                ],
                [
                    "1364 / dense and occluded",
                    "48 correct-class matches out of 56 GT; 3 unmatched GT and many duplicate/extra low-confidence boxes. Pedestrian predicted as Bicycle.",
                ],
                [
                    "4711 and 954 / rare classes",
                    "Mini-bus predicted Bus; construction vehicle (Others) predicted Truck. Unlabelled partial auto in 4711 is a separate source limitation.",
                ],
            ],
            [155, 348],
        ),
        Spacer(1, 10),
        P(
            "101 diagnostic predictions generated; six GT/prediction pairs manually reviewed at conf 0.10. Matching is diagnostic IoU 0.5, not AP. Unmatched predictions can include unlabelled real vehicles. Occlusion-stratified or small-object AP has not been measured.",
            "SmallCopy",
        ),
        PageBreak(),
        P("04 / REPRODUCIBILITY AND NEXT STEP", "Kicker"),
        P("A reference for comparison", "SectionTitle"),
        Image(
            str(ROOT / f"reports/figures/{name}_validation_BoxPR_curve.png"),
            width=440,
            height=293,
        ),
        P(
            "Standalone best-checkpoint precision-recall curves. Full per-class CSV, PR/F1/P/R plots and numeric confusion matrix accompany this PDF. The matrix uses confidence 0.001 and matching IoU 0.45, distinct from the selected F1 operating point.",
            "SmallCopy",
        ),
        P("Accuracy-speed trade-off", "Kicker"),
        P(
            "This compact detector is fast on still images, but rare-class recall, small vehicles and dense-scene duplicates limit usefulness. File/decode and API overhead dominate beyond inference. Larger models or inputs may improve accuracy but need controlled comparison; no such improvement is claimed yet."
        ),
        P("Demonstrate the saved evidence", "Kicker"),
        P("python3 scripts/show_progress.py", "CodeCopy"),
        P(
            "With dependencies installed: python -m streamlit run app.py<br/>Run tests: python -m pytest -q",
            "SmallCopy",
        ),
        P(
            "The portable ZIP contains source, configs, summarized results, tests, plots, this PDF and a presentation script. Dataset images, labels, checkpoints, full runs and local secrets are excluded. Read the saved subset Phase 2 gate and Git delivery status under reports/audit. Controlled Phase 2 work requires a Git checkpoint; no Phase 2 training was started."
        ),
        P(
            'Source: <link href="https://huggingface.co/datasets/iisc-aim/UVH-26">IISc AIM UVH-26</link>, CC BY 4.0. Revision 59f82c57821e8a54dc40bc1f42e83909dbad0b70. Sharma et al., IISc Technical Report, November 2025, arXiv:2511.02563. All project metrics above come from included measured artifacts.',
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
        title="UVH-26 Phase 1 Baseline Results",
        author="Manav Shah",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
