import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from channelhub.config import RunConfig
from channelhub.server import create_app


def test_public_api_and_strict_boundary(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        assert client.get("/api/config").json()["defaults"]["engine"] == "statistical"
        assert client.get("/api/jobs").json() == []
        bad = {"config": {"model_checkpoint": "unpublished"}}
        assert client.post("/api/jobs", json=bad).status_code == 422
        assert (
            client.post(
                "/api/jobs", json={"config": {}}, headers={"Origin": "https://example.com"}
            ).status_code
            == 403
        )
        assert client.get("/api/jobs/invalid").status_code == 404
        assert client.get("/api/models").status_code == 404
        assert client.get("/api/processing").status_code == 404
        assert client.get("/.env").status_code == 404


def test_safe_custom_upload(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        for x, code in [
            (np.ones((14, 1, 48), complex), 201),
            (np.ones((14, 1, 48)), 422),
            (np.array([object()]), 422),
        ]:
            data = io.BytesIO()
            np.save(data, x)
            response = client.post("/api/uploads/x", content=data.getvalue())
            assert response.status_code == code
        assert client.post("/api/uploads/x", content=b"not numpy").status_code == 422


@pytest.mark.statistical
def test_job_subprocess_download_and_preview(tmp_path):
    pytest.importorskip("sionna.phy")
    with TestClient(create_app(tmp_path)) as client:
        cfg = RunConfig(num_rb=4, num_slots=1, profile="TDL-A").model_dump(mode="json")
        response = client.post("/api/jobs", json={"config": cfg})
        assert response.status_code == 202
        job_id = response.json()["id"]
        until = time.monotonic() + 60
        while time.monotonic() < until:
            state = client.get(f"/api/jobs/{job_id}").json()
            if state["status"] not in ("queued", "running"):
                break
            time.sleep(0.1)
        assert state["status"] == "completed", state
        result = client.get(f"/api/jobs/{job_id}/download/result.zip")
        assert result.status_code == 200 and result.content[:2] == b"PK"
        preview = client.get(f"/api/jobs/{job_id}/preview").json()
        assert len(preview["frequency_mhz"]) == 48
        assert preview["metadata"]["schema"] == "channelhub-observation-v1"
        assert client.get(f"/api/jobs/{job_id}/download/secrets.txt").status_code == 404


def test_restart_marks_unfinished_job_interrupted(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"id": "a" * 32, "status": "running", "percent": 5}))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/jobs/" + ("a" * 32)).json()["status"] == "interrupted"
