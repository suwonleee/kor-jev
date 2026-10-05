import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kor_jev import cli


@pytest.fixture
def checkpoint(tmp_path):
    for name in cli.REQUIRED_FILES:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else 'test-weight-bytes')
    decision = {'format_version': 1, 'base_model': 'Qwen/Qwen3.5-0.8B',
                'revision': 'a' * 40, 'codes': [str(i) for i in range(255)],
                'token_ids': list(range(255)), 'temperature': 0.9728202031484768}
    (tmp_path / 'decision_config.json').write_text(json.dumps(decision))
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'template'}))
    (tmp_path / 'processor_config.json').write_text('{}')
    return tmp_path


def manifest(checkpoint):
    return {'published': True, 'repo_id': 'test/kor-jev', 'revision': 'b' * 40,
            'files': {p.name: {'size': p.stat().st_size, 'sha256': cli.file_sha256(p)}
                      for p in checkpoint.iterdir()}}


def test_unpublished_model_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kw: pytest.fail('Must not download an unrelated model')))
    with pytest.raises(ValueError, match='아직 공개되지'):
        cli.fetch_checkpoint(tmp_path, {'published': False})


def test_fetch_is_anonymous_pinned_and_checks_files(monkeypatch, checkpoint):
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kw: calls.append(kw)))
    assert cli.fetch_checkpoint(checkpoint, manifest(checkpoint)) == checkpoint
    assert calls[0]['token'] is False
    assert calls[0]['revision'] == 'b' * 40
    assert set(calls[0]['allow_patterns']) == set(p.name for p in checkpoint.iterdir())


def test_corrupt_download_is_rejected(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kw: (checkpoint / 'readout.safetensors').write_text('corrupt')))
    with pytest.raises(ValueError, match='integrity check failed'):
        cli.fetch_checkpoint(checkpoint, release)


@pytest.mark.parametrize('revision', ['main', None, 'a' * 39])
def test_mutable_or_missing_hub_revision_rejected(checkpoint, revision):
    release = manifest(checkpoint)
    release['revision'] = revision
    with pytest.raises(ValueError, match='immutable Hub revision'):
        cli.fetch_checkpoint(checkpoint, release)


def test_manifest_path_traversal_rejected_before_download(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    release['files']['../token'] = {'sha256': '0' * 64, 'size': 1}
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kw: pytest.fail('Invalid manifest must not reach the Hub')))
    with pytest.raises(ValueError, match='flat'):
        cli.fetch_checkpoint(checkpoint, release)


def test_incomplete_checkpoint_cannot_start_server(monkeypatch, checkpoint):
    (checkpoint / 'readout.safetensors').unlink()
    monkeypatch.setitem(sys.modules, 'jeff.server', SimpleNamespace(
        main=lambda: pytest.fail('Must not start with incomplete weights')))
    with pytest.raises(SystemExit) as error:
        cli.main(['serve', '--checkpoint', str(checkpoint), '--backend', 'pytorch'])
    assert error.value.code == 1


def test_local_checkpoint_starts_correct_backend_without_fetch(monkeypatch, checkpoint):
    calls = []
    monkeypatch.setattr(cli, 'fetch_checkpoint', lambda *args: pytest.fail('Local path needs no download'))
    monkeypatch.setenv('JEFF_CHECKPOINT', 'invalid-previous-setting')
    monkeypatch.setitem(sys.modules, 'jeff.server', SimpleNamespace(main=lambda: calls.append(True)))
    cli.main(['serve', '--checkpoint', str(checkpoint), '--backend', 'pytorch', '--device', 'cpu'])
    assert calls == [True]
    assert cli.os.environ['JEFF_CHECKPOINT'] == str(checkpoint.resolve())
    assert cli.os.environ['JEFF_BACKEND'] == 'pytorch'
    assert cli.os.environ['JEFF_DEVICE'] == 'cpu'
    assert cli.os.environ['JEFF_HOST'] == '127.0.0.1'
    assert cli.os.environ['PORT'] == '8765'


@pytest.mark.parametrize('field,value', [('base_model', 'Qwen/Qwen3.8-27B'),
                                        ('temperature', 0), ('temperature', float('nan')),
                                        ('token_ids', [])])
def test_wrong_checkpoint_configuration_rejected(checkpoint, field, value):
    path = checkpoint / 'decision_config.json'
    decision = json.loads(path.read_text())
    decision[field] = value
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError):
        cli.validate_checkpoint(checkpoint)


def test_missing_chat_template_rejected(checkpoint):
    (checkpoint / 'tokenizer_config.json').write_text('{}')
    with pytest.raises(ValueError, match='chat template'):
        cli.validate_checkpoint(checkpoint)


def test_legacy_processor_config_supported(checkpoint):
    (checkpoint / 'processor_config.json').rename(checkpoint / 'preprocessor_config.json')
    assert cli.validate_checkpoint(checkpoint)['format_version'] == 1


def test_missing_processor_config_rejected(checkpoint):
    (checkpoint / 'processor_config.json').unlink()
    with pytest.raises(ValueError, match='processor configuration'):
        cli.validate_checkpoint(checkpoint)


def test_publish_dry_run_excludes_optimizer_and_training_state(monkeypatch, checkpoint, capsys):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts/publish_model.py'
    spec = importlib.util.spec_from_file_location('publish_model', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (checkpoint / 'resume.pt').write_text('optimizer must remain local')
    (checkpoint / 'private-training.json').write_text('training metadata must remain local')
    monkeypatch.setattr(sys, 'argv', ['publish_model', '--checkpoint', str(checkpoint), '--dry-run'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: pytest.fail('Dry run must not access credentials or upload')))
    module.main()
    output = json.loads(capsys.readouterr().out)
    assert 'model.safetensors' in output['files']
    assert {'LICENSE', 'NOTICE', 'README.md'}.issubset(output['files'])
    assert 'resume.pt' not in output['files']
    assert 'private-training.json' not in output['files']


def test_api_serializes_three_types_without_loading_weights(monkeypatch):
    from fastapi.testclient import TestClient
    from jeff.server import app, service

    class FakeModel:
        backend = 'mlx'

        def decide(self, rows):
            return [([0.75, 0.25], 10) for row in rows]

    monkeypatch.setenv('JEFF_ANY_MODEL', '1')
    monkeypatch.delenv('JEFF_API_KEY', raising=False)
    monkeypatch.setattr(service, 'model', FakeModel())
    payload = {'model': 'kor-jev', 'state': '검증용 입력', 'questions': {
        'choice': {'type': 'choice', 'criteria': {'a': '첫 후보', 'b': '둘째 후보'}},
        'noul': {'type': 'noul'}, 'score': {'type': 'score', 'criteria': ['낮음', '높음']}}}
    # No TestClient context: the real model-loading lifespan must not run.
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ready'
    response = client.post('/v1/systemone', json=payload)
    assert response.status_code == 200
    answers = response.json()['answers']
    assert answers['choice']['choice'] == 'a'
    assert answers['choice']['confidence'] == 0.5
    assert answers['noul']['noul'] == 0.25
    assert answers['score']['score'] == 0.25
    monkeypatch.setenv('JEFF_API_KEY', 'test-local-key')
    assert client.post('/v1/systemone', json=payload).status_code == 401
    assert client.post('/v1/systemone', json=payload,
                       headers={'Authorization': 'Bearer test-local-key'}).status_code == 200
    monkeypatch.setattr(service, 'model', None)
    assert client.post('/v1/systemone', json=payload,
                       headers={'Authorization': 'Bearer test-local-key'}).status_code == 503
