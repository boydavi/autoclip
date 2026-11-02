"""Cross-site request protection for the state-changing API."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import paths
from autoclip.app import ENV_ALLOWED_ORIGINS, create_app, origin_allowed
from fastapi.testclient import TestClient

EVIL = {"Origin": "https://evil.example"}


def make_client(monkeypatch: pytest.MonkeyPatch, **env: str) -> TestClient:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    monkeypatch.delenv(ENV_ALLOWED_ORIGINS, raising=False)
    monkeypatch.delenv(app_module.ENV_API_KEY, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return TestClient(create_app(), base_url="http://localhost")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with make_client(monkeypatch) as test_client:
        yield test_client


def upload(client: TestClient, headers: dict | None = None):
    return client.post(
        "/api/music",
        files={"file": ("bed.mp3", b"audio", "application/octet-stream")},
        headers=headers or {},
    )


class TestOriginAllowed:
    ALLOWED = frozenset({"http://localhost:5173"})

    def test_same_origin_is_allowed(self) -> None:
        assert origin_allowed("http://localhost:8000", "localhost:8000", self.ALLOWED)

    def test_other_site_is_refused(self) -> None:
        assert not origin_allowed("https://evil.example", "localhost:8000", self.ALLOWED)

    def test_same_host_different_port_is_refused(self) -> None:
        assert not origin_allowed("http://localhost:9999", "localhost:8000", self.ALLOWED)

    def test_lookalike_host_is_refused(self) -> None:
        assert not origin_allowed("http://localhost.evil.example", "localhost", self.ALLOWED)

    def test_null_and_empty_origins_are_refused(self) -> None:
        assert not origin_allowed("null", "null", self.ALLOWED)
        assert not origin_allowed("", "", self.ALLOWED)

    def test_listed_origin_is_allowed_and_case_insensitive(self) -> None:
        assert origin_allowed("HTTP://LOCALHOST:5173", "localhost:8000", self.ALLOWED)

    def test_dns_rebinding_page_is_refused_despite_matching_headers(self) -> None:
        # evil.example re-pointed at 127.0.0.1 sends Origin and Host that agree.
        assert not origin_allowed("http://evil.example:8000", "evil.example:8000", self.ALLOWED)

    @pytest.mark.parametrize(
        ("origin", "host"),
        [
            ("http://127.0.0.1:8000", "127.0.0.1:8000"),
            ("http://192.168.1.20:8000", "192.168.1.20:8000"),
            ("http://[::1]:8000", "[::1]:8000"),
            ("http://localhost", "localhost"),
        ],
    )
    def test_loopback_and_ip_literal_hosts_are_trusted(self, origin: str, host: str) -> None:
        assert origin_allowed(origin, host, self.ALLOWED)

    def test_a_hostname_works_once_its_origin_is_listed(self) -> None:
        listed = frozenset({"http://mybox.local:8000"})

        assert origin_allowed("http://mybox.local:8000", "mybox.local:8000", listed)
        assert not origin_allowed("http://mybox.local:8000", "mybox.local:8000", self.ALLOWED)


class TestStateChangingRequests:
    def test_cross_site_upload_is_blocked_and_writes_nothing(self, client: TestClient) -> None:
        response = upload(client, EVIL)

        assert response.status_code == 403
        assert "Cross-origin" in response.json()["detail"]
        assert not (paths.music_dir() / "bed.mp3").exists()

    def test_cross_site_put_and_delete_are_blocked(self, client: TestClient) -> None:
        assert client.put("/api/settings", json={}, headers=EVIL).status_code == 403
        assert client.delete("/api/jobs/anything", headers=EVIL).status_code == 403

    def test_cross_site_job_creation_is_blocked(self, client: TestClient) -> None:
        response = client.post("/api/jobs", json={"source_id": "x"}, headers=EVIL)

        assert response.status_code == 403

    def test_null_origin_is_blocked(self, client: TestClient) -> None:
        assert upload(client, {"Origin": "null"}).status_code == 403

    def test_same_origin_upload_works(self, client: TestClient) -> None:
        assert upload(client, {"Origin": "http://localhost"}).status_code == 201

    def test_rebinding_style_request_is_blocked_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with make_client(monkeypatch) as client:
            client.base_url = "http://evil.example"

            response = upload(client, {"Origin": "http://evil.example"})

        assert response.status_code == 403
        assert not (paths.music_dir() / "bed.mp3").exists()

    def test_requests_without_an_origin_header_work(self, client: TestClient) -> None:
        assert upload(client).status_code == 201

    def test_vite_dev_origin_is_allowed_by_default(self, client: TestClient) -> None:
        assert upload(client, {"Origin": "http://localhost:5173"}).status_code == 201

    def test_reads_are_never_blocked(self, client: TestClient) -> None:
        assert client.get("/api/music", headers=EVIL).status_code == 200
        assert client.get("/api/health", headers=EVIL).status_code == 200


class TestConfiguredOrigins:
    def test_listed_origin_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = {ENV_ALLOWED_ORIGINS: "https://tool.example/, http://10.0.0.5:3000"}
        with make_client(monkeypatch, **env) as client:
            assert upload(client, {"Origin": "https://tool.example"}).status_code == 201
            assert upload(client, {"Origin": "http://10.0.0.5:3000"}).status_code == 201
            assert upload(client, EVIL).status_code == 403

    def test_block_happens_before_auth_is_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with make_client(monkeypatch, **{app_module.ENV_API_KEY: "s3cret"}) as client:
            blocked = upload(client, EVIL)
            unauthorised = upload(client)
            allowed = upload(client, {"Authorization": "Bearer s3cret"})

        assert blocked.status_code == 403
        assert unauthorised.status_code == 401
        assert allowed.status_code == 201
