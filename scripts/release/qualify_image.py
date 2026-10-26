"""Evaluate an immutable candidate container before Docker Hub publication."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import cv2
import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    subprocess.run(args, check=True)


def read(path):
    return json.loads(path.read_text())


def detection_checks(summary, policy):
    return (
        summary["cases"] == policy["cases"]
        and summary["true_positive"] >= policy["true_positive_min"]
        and summary["false_positive"] <= policy["false_positive_max"]
        and summary["false_negative"] <= policy["false_negative_max"]
        and summary["manifest_sha256"] == policy["manifest_sha256"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".evaluation-local/image-gate"
    )
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    policy = read(ROOT / "evaluation/release_policy.json")
    metadata = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image], text=True)
    )[0]
    assert (
        metadata["Config"]["Labels"]["org.opencontainers.image.revision"]
        == args.source_sha
    )
    assert metadata["Os"] == "linux" and metadata["Architecture"] == "amd64"
    base = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={folder},dst=/app/output/evaluation",
        "--mount",
        f"type=bind,src={ROOT / 'input'},dst=/app/input,readonly",
    ]
    command(
        [*base, args.image, "python", "-m", "unittest", "discover", "-s", "tests", "-q"]
    )
    for name, manifest in [
        ("development", "coco-validation-64"),
        ("validation", "coco-holdout-36"),
    ]:
        command(
            [
                *base,
                args.image,
                "python",
                "-m",
                "scripts.evaluation.evaluate_detector",
                "--backend",
                "onnx",
                "--device",
                "cuda",
                "--manifest",
                f"input/{manifest}/manifest.json",
                "--output",
                f"output/evaluation/{name}",
            ]
        )
    command(
        [
            *base,
            args.image,
            "python",
            "-m",
            "scripts.evaluation.evaluate_tracking",
            "--output",
            "output/evaluation/tracking.json",
        ]
    )
    checks = {
        "development_detection": detection_checks(
            read(folder / "development/summary.json"), policy["detection"]
        ),
        "validation_detection": detection_checks(
            read(folder / "validation/summary.json"), policy["holdout_detection"]
        ),
    }
    tracking = read(folder / "tracking.json")
    checks["tracking_complete"] = tracking["cases"] == len(policy["tracking"])
    checks["tracking_quality"] = all(
        r["idf1"] >= policy["tracking"][r["case_id"]] for r in tracking["results"]
    )
    name = "attention-qualification"
    data = folder / "api-data"
    data.mkdir()
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    api = []
    browser = {}
    base_url = "http://127.0.0.1:8771"
    try:
        command(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--gpus",
                "all",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--name",
                name,
                "--mount",
                f"type=bind,src={data},dst=/app/data",
                "-p",
                "127.0.0.1:8771:8770",
                "-e",
                "ATTENTION_API_MAX_CLIP_SECONDS=2",
                args.image,
            ]
        )
        for _ in range(180):
            try:
                with urllib.request.urlopen(
                    base_url + "/health", timeout=2
                ) as response:
                    health = json.load(response)
                assert (
                    health["person_device"] == "cuda"
                    and health["landmark_device"] == "cpu"
                )
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Container readiness deadline")
        with httpx.Client(base_url=base_url, timeout=30) as client:
            for view, expected in policy["workflow"].items():
                source = ROOT / expected["fixture"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == expected["source_sha256"]
                )
                with source.open("rb") as stream:
                    response = client.post(
                        "/jobs?max_seconds=2",
                        files={"video": (source.name, stream, "video/mp4")},
                    )
                response.raise_for_status()
                job = response.json()["id"]
                started = time.monotonic()
                while True:
                    status = client.get("/jobs/" + job).json()
                    if status["status"] != "running":
                        break
                    if time.monotonic() - started > 180:
                        raise TimeoutError("Video job deadline")
                    time.sleep(0.25)
                assert status["status"] == "completed", status
                report = client.get("/jobs/" + job + "/result")
                report.raise_for_status()
                report = report.json()
                observations = client.get("/jobs/" + job + "/observations")
                observations.raise_for_status()
                rows = [json.loads(line) for line in observations.text.splitlines()]
                labels = {}
                for person in report["people"]:
                    for label, count in person["label_counts"].items():
                        labels[label] = labels.get(label, 0) + count
                checks[f"workflow_{view}"] = (
                    report["frames"] == expected["frames"]
                    and len(report["people"]) == expected["tracks"]
                    and labels == expected["label_counts"]
                )
                checks[f"observations_{view}"] = len(rows) == sum(
                    labels.values()
                ) and all(
                    row["face_bbox"] is None
                    for row in rows
                    if row["label"] == "no_face"
                )
                video = client.get("/jobs/" + job + "/video")
                video.raise_for_status()
                path = folder / f"api-{view}.mp4"
                path.write_bytes(video.content)
                capture = cv2.VideoCapture(str(path))
                decoded = 0
                while capture.read()[0]:
                    decoded += 1
                capture.release()
                checks[f"video_{view}"] = decoded == expected["frames"]
                api.append(
                    {
                        "view": view,
                        "status": status["status"],
                        "job_seconds": status["seconds"],
                        "frames": report["frames"],
                        "tracks": len(report["people"]),
                        "label_counts": labels,
                        "performance": report["performance"],
                    }
                )
        with sync_playwright() as playwright:
            chromium = playwright.chromium.launch(args=["--no-sandbox"])
            page = chromium.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base_url)
            page.locator("#video").set_input_files(
                str(ROOT / policy["workflow"]["L"]["fixture"])
            )
            page.locator("#submit").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent==='completed'",
                timeout=180000,
            )
            page.wait_for_function(
                "document.querySelector('#preview').readyState>=2", timeout=15000
            )
            page.locator("#preview").evaluate("v=>{v.muted=true;return v.play()}")
            page.wait_for_function(
                "document.querySelector('#preview').currentTime>.4", timeout=15000
            )
            page.locator("#preview").evaluate("v=>v.pause()")
            checks["browser_h264_playback"] = True
            links = page.locator("#results a")
            checks["browser_artifacts"] = links.count() == 3
            for index in range(links.count()):
                assert page.request.get(
                    base_url + links.nth(index).get_attribute("href")
                ).ok
            browser = {
                "completed": True,
                "artifact_links": links.count(),
                "javascript_errors": errors,
            }
            checks["browser_javascript"] = not errors
            chromium.close()
        (folder / "container.log").write_bytes(
            subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT)
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    summary = {
        "source_sha": args.source_sha,
        "image": args.image,
        "policy": policy,
        "passed": all(checks.values()),
        "checks": checks,
        "development_detection": read(folder / "development/summary.json"),
        "validation_detection": read(folder / "validation/summary.json"),
        "tracking": tracking,
        "api_workflows": api,
        "browser": browser,
        "scope": "Actual candidate models, human-box detection metrics, synthetic tracker association, model-derived AMI workflow regression and real browser upload; not measured behavioral attention accuracy.",
    }
    (folder / "gate-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "checks": checks}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
