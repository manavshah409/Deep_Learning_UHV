# Phase 3A licensed video sources — 18 September 2026

Disk checked before download:715 GiB available. Three source files downloaded, below2GB combined; one resolution per source. All video/image derivatives are ignored. No YouTube/social-media downloader, restricted login or arbitrary dataset archive was used.

## Search procedure

Searched Wikimedia Commons first for Indian urban traffic and road videos. Also searched Zenodo while assessing alternative sources. Commons supplied two directly authored CC BY-SA clips, so later-ranked Internet Archive, university, government and stock sources were not needed. A terminal page request returned403; the ordinary public browser displayed the licence and original-file links without login or protection bypass. Public original-file downloads succeeded with curl TLS verification and resume enabled.

## Accepted sources

1. [Ramana Maharshi Road, Bangalore (2025) 01](https://commons.wikimedia.org/wiki/File:Ramana_Maharshi_Road,_Bangalore_(2025)_01.webm), creator/uploader **Gpkp**, own work,11October2025. [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/).
2. [Ramana Maharshi Road, Bangalore (2025) 02](https://commons.wikimedia.org/wiki/File:Ramana_Maharshi_Road,_Bangalore_(2025)_02.webm), creator/uploader **Gpkp**, own work,11October2025. [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Attribution for each: title, Gpkp, Wikimedia Commons, linked source and CC BY-SA4.0; disclose excerpt, audio removal, re-encoding and added annotations. If distributed, adapted video must retain the same or compatible share-alike licence. No endorsement is implied. Source copyright is separate from project-code licensing.

Both are elevated views of the same urban road at night, with opposing traffic and distant vehicles. Sampled frames show a mostly stable view, headlight glare, occlusion and minor handheld movement. They are distinct recordings, not duplicate resolutions. Limited location/lighting diversity is explicit. Accepted originals decode sequentially;30-second MP4 derivatives are separate, with source and derivative hashes. OpenCV reports VP8 originals,1920×1080,30FPS. Pilot uses01;02 supplies an additional validation candidate, not a second accuracy result.

Exact byte sizes, hashes, original URLs, metadata and attribution: `reports/audit/phase3_video_sources.json`. Local provenance: ignored `data/videos/external/PROVENANCE.local.yaml`. Derived provenance: `reports/audit/phase3_derived_clips.json`. Preparation command `.venv/bin/python scripts/prepare_phase3_clips.py` refuses existing derivative files. It fully decodes each accepted source and re-encodes its first30seconds using mp4v, without audio. No source is modified.

## Rejected / not selected

- [Moving vehicles in Link road, Cuttack, Odisha](https://commons.wikimedia.org/wiki/File:Moving_vehicles_in_Link_road,_Cuttack,_Odisha.webm), Subhashish Panigrahi, CC BY-SA3.0: downloaded35,099,101bytes, but decoder reported keyframe-index warnings and3342 decoded vs3368 advertised frames. Excluded from pilot; original and failed preparation log preserved. No silent timing repair.
- Commons New BEL Road: moving-camera driving view; not selected for fixed-line pilot.
- Commons VIP Road Kolkata: travel footage candidate; not downloaded because more suitable elevated views became available.
- Rohtang mountain-road travel and historical1906street footage: unsuitable setting/era; not downloaded.
- [Zenodo two-wheeler seepage study](https://zenodo.org/records/18955993): restricted files and no included raw video; not downloaded.
- [IDS-JODHA](https://zenodo.org/records/17258045): large archives exceed resource scope; not downloaded.
- [Indian Traffic VQA](https://zenodo.org/records/17300841): still-image question-answer dataset, not video; not downloaded.

No face/plate identification feature is implemented. Raw and annotated footage stays local. Representative-frame review establishes candidate suitability, not exhaustive proof of no camera motion or a population benchmark.

## Decode verification addendum

Clip01 decoded1224 frames; clip02 decoded1232 vs1237 estimated by OpenCV. Unlike the rejected Cuttack source, both accepted clips have strictly monotonic timestamps with33–34ms spacing throughout and no decode warnings in sequential validation. Clip02's reported duration/frame estimate is not the decoded stream length. The first900 frames (30seconds) are complete in each derived v2 file. Initial derivative v1 files from the strict metadata-count check are preserved but excluded; v2 provenance records actual counts and timestamp ranges. This is an explicit metadata discrepancy, not a hidden repair. Sampled0/10/20/29second frames in clip02 show dense traffic, the opposite view from the same location, with glare and modest tilt; no sampled scene cut. Clip01 is the frozen pilot. All three downloads total69,815,481bytes, well below2GB.
