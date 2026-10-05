import io
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
    return {'published': True, 'provider': 'github', 'repository': 'test/kor-jev',
            'tag': 'ko-v7-preview', 'source_commit': 'b' * 40,
            'files': {p.name: {'size': p.stat().st_size, 'sha256': cli.file_sha256(p),
                               'url': f'https://github.com/test/kor-jev/releases/download/ko-v7-preview/{p.name}'}
                      for p in checkpoint.iterdir()}}


def fake_download(monkeypatch, checkpoint, corrupt=None):
    calls = []
    data = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    def open_url(request, **kwargs):
        calls.append(request)
        name = request.full_url.rsplit('/', 1)[-1]
        return io.BytesIO(b'corrupt' if name == corrupt else data[name])
    monkeypatch.setattr(cli.urllib.request, 'urlopen', open_url)
    return calls


def test_unpublished_model_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.urllib.request, 'urlopen',
                        lambda *a, **kw: pytest.fail('Must not download an unrelated model'))
    with pytest.raises(ValueError, match='아직 공개되지'):
        cli.fetch_checkpoint(tmp_path, {'published': False})


def test_github_download_is_anonymous_and_checks_files(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    calls = fake_download(monkeypatch, checkpoint)
    destination = checkpoint / 'downloaded'
    assert cli.fetch_checkpoint(destination, release) == destination
    assert len(calls) == len(release['files'])
    assert all(call.get_header('Authorization') is None for call in calls)
    assert all('/releases/download/ko-v7-preview/' in call.full_url for call in calls)
    assert (destination / 'model.safetensors').read_bytes() == (checkpoint / 'model.safetensors').read_bytes()


def test_verified_checkpoint_works_offline(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    monkeypatch.setattr(cli.urllib.request, 'urlopen',
                        lambda *a, **kw: pytest.fail('Verified local files must need no network'))
    assert cli.fetch_checkpoint(checkpoint, release) == checkpoint


def test_corrupt_download_is_rejected_and_temporary_file_removed(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    fake_download(monkeypatch, checkpoint, corrupt='readout.safetensors')
    destination = checkpoint / 'downloaded'
    with pytest.raises(ValueError, match='integrity check failed'):
        cli.fetch_checkpoint(destination, release)
    assert not (destination / 'readout.safetensors').exists()
    assert not list(destination.glob('*.download'))


@pytest.mark.parametrize('field,value', [('source_commit', None), ('source_commit', 'a' * 39),
                                        ('tag', ''), ('repository', 'invalid')])
def test_invalid_github_release_identity_rejected(checkpoint, field, value):
    release = manifest(checkpoint)
    release[field] = value
    with pytest.raises(ValueError, match='Invalid'):
        cli.fetch_checkpoint(checkpoint, release)


def test_manifest_path_traversal_rejected_before_download(monkeypatch, checkpoint):
    release = manifest(checkpoint)
    release['files']['../token'] = {'sha256': '0' * 64, 'size': 1}
    monkeypatch.setattr(cli.urllib.request, 'urlopen',
                        lambda *a, **kw: pytest.fail('Invalid manifest must not reach GitHub'))
    with pytest.raises(ValueError, match='flat'):
        cli.fetch_checkpoint(checkpoint, release)


def test_foreign_download_host_rejected(checkpoint):
    release = manifest(checkpoint)
    release['files']['model.safetensors']['url'] = 'https://example.com/weights'
    with pytest.raises(ValueError, match='asset URL'):
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
    monkeypatch.setattr(module, 'gh', lambda *a, **kw: pytest.fail('Dry run must not contact GitHub'))
    module.main()
    output = json.loads(capsys.readouterr().out)
    assert 'model.safetensors' in output['files']
    assert {'LICENSE', 'NOTICE', 'README.md'}.issubset(output['files'])
    assert 'resume.pt' not in output['files']
    assert 'private-training.json' not in output['files']


def test_github_asset_digest_mismatch_rejected():
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts/publish_model.py'
    spec = importlib.util.spec_from_file_location('publish_model', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='digest verification failed'):
        module.check_assets([{'name': 'model.safetensors', 'size': 10, 'digest': 'sha256:' + '0' * 64}],
                            {'model.safetensors': {'size': 10, 'sha256': '1' * 64}})


def test_published_release_never_overwritten(monkeypatch, checkpoint):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts/publish_model.py'
    spec = importlib.util.spec_from_file_location('publish_model', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', ['publish_model', '--checkpoint', str(checkpoint)])
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='b' * 40))
    calls = []
    def github(*args, **kwargs):
        calls.append(args)
        assert args[:2] == ('release', 'view'), 'An existing public release must only be read'
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'isDraft': False, 'databaseId': 1, 'targetCommitish': 'b' * 40}))
    monkeypatch.setattr(module, 'gh', github)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert len(calls) == 1


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
