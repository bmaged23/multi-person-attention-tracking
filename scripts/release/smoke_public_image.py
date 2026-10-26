"""Verify normal startup and a real API video job from the public immutable image."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    name = "attention-public-" + args.device
    command = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "-p",
        "127.0.0.1::8770",
        "-e",
        "ATTENTION_PERSON_DEVICE=" + args.device,
    ]
    if args.device == "cuda":
        command += ["--gpus", "all"]
    subprocess.run([*command, args.image], check=True)
    try:
        port = (
            subprocess.check_output(["docker", "port", name, "8770/tcp"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        with httpx.Client(base_url="http://127.0.0.1:" + port, timeout=30) as client:
            for _ in range(180):
                try:
                    health = client.get("/health")
                    health.raise_for_status()
                    assert health.json()["person_device"] == args.device
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise TimeoutError("Public container readiness deadline")
            assert "Multi-person visual attention" in client.get("/").text
            with (ROOT / "evaluation/fixtures/ami-L.mkv").open("rb") as source:
                response = client.post(
                    "/jobs?max_seconds=2",
                    files={"video": ("clip.mkv", source, "video/x-matroska")},
                )
            response.raise_for_status()
            job = response.json()["id"]
            started = time.monotonic()
            while True:
                row = client.get("/jobs/" + job).json()
                if row["status"] != "running":
                    break
                if time.monotonic() - started > 180:
                    raise TimeoutError("Public inference deadline")
                time.sleep(0.25)
            assert row["status"] == "completed", row
            report = client.get("/jobs/" + job + "/result").json()
            assert report["frames"] == 50 and report["people"]
            assert client.get("/jobs/" + job + "/video").status_code == 200
            print(
                json.dumps(
                    {
                        "normal_public_startup": "passed",
                        "actual_video_job": "passed",
                        "person_device": args.device,
                        "frames": report["frames"],
                        "image": args.image,
                    }
                )
            )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()
