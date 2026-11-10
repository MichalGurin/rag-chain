from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.ingestion import storage


@pytest.fixture
def s3_settings():
    return Settings(
        _env_file=None,
        s3_endpoint_url="https://s3.example.test",
        s3_bucket="rag-test-bucket",
        s3_access_key_id=SecretStr("test-access"),
        s3_secret_access_key=SecretStr("test-secret"),
    )


def test_s3_upload_deletes_local_file_only_after_success(tmp_path, monkeypatch, s3_settings):
    source = tmp_path / "policy.txt"
    source.write_text("policy", encoding="utf-8")
    client = Mock()
    monkeypatch.setattr(storage, "s3_client", lambda _settings: client)

    object_path = storage.upload_path(source, "document-id", ".txt", s3_settings)

    assert object_path == "s3://rag-test-bucket/documents/document-id.txt"
    assert client.upload_file.call_args.args == (
        str(source),
        "rag-test-bucket",
        "documents/document-id.txt",
    )
    assert not source.exists()


def test_s3_upload_failure_preserves_local_file(tmp_path, monkeypatch, s3_settings):
    source = tmp_path / "policy.txt"
    source.write_text("policy", encoding="utf-8")
    client = Mock()
    client.upload_file.side_effect = OSError("storage unavailable")
    monkeypatch.setattr(storage, "s3_client", lambda _settings: client)

    with pytest.raises(OSError, match="storage unavailable"):
        storage.upload_path(source, "document-id", ".txt", s3_settings)

    assert source.read_text(encoding="utf-8") == "policy"


def test_s3_download_uses_a_temporary_file_and_cleans_it(
    monkeypatch,
    s3_settings,
):
    client = Mock()

    def download_file(_bucket, _key, destination):
        Path(destination).write_text("policy", encoding="utf-8")

    client.download_file.side_effect = download_file
    monkeypatch.setattr(storage, "s3_client", lambda _settings: client)

    with storage.local_document_path(
        "s3://rag-test-bucket/documents/document-id.pdf",
        "document-id",
        s3_settings,
    ) as local_path:
        temporary_file = Path(local_path)
        assert temporary_file.suffix == ".pdf"
        assert temporary_file.read_text(encoding="utf-8") == "policy"
        assert temporary_file.exists()

    assert not temporary_file.exists()


@pytest.mark.parametrize(
    "object_path",
    [
        "s3://rag-test-bucket",
        "s3://other-bucket/documents/policy.txt",
    ],
)
def test_s3_download_rejects_malformed_or_unconfigured_bucket(
    object_path,
    monkeypatch,
    s3_settings,
):
    monkeypatch.setattr(storage, "s3_client", lambda _settings: Mock())

    with pytest.raises((ValueError, RuntimeError)):
        with storage.local_document_path(object_path, "id", s3_settings):
            pytest.fail("A bad object path must not be yielded.")


def test_s3_delete_rejects_objects_from_a_different_bucket(
    monkeypatch,
    s3_settings,
):
    monkeypatch.setattr(storage, "s3_client", lambda _settings: Mock())

    with pytest.raises(RuntimeError, match="bucket is not configured"):
        storage.delete_upload("s3://other-bucket/documents/policy.txt", s3_settings)


def test_local_delete_is_idempotent(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, upload_dir=str(tmp_path))

    storage.delete_upload(str(tmp_path / "already-deleted.txt"), settings)
