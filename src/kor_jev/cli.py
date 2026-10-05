"""Download a pinned Korean checkpoint and serve it through the local API."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path


REQUIRED_FILES = (
    'decision_config.json', 'config.json', 'model.safetensors',
    'readout.safetensors', 'tokenizer.json', 'tokenizer_config.json',
)


def release_info() -> dict:
    return json.loads(files('kor_jev').joinpath('release.json').read_text())


def validate_checkpoint(directory: Path) -> dict:
    missing = [name for name in REQUIRED_FILES
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    if missing:
        raise ValueError(f'Incomplete checkpoint {directory}: {", ".join(missing)}')
    if not any((directory / name).is_file() for name in ('processor_config.json', 'preprocessor_config.json')):
        raise ValueError('Checkpoint processor configuration is missing.')
    decision = json.loads((directory / 'decision_config.json').read_text())
    if decision.get('format_version') != 1:
        raise ValueError('Unsupported decision checkpoint format.')
    if not str(decision.get('base_model', '')).lower().endswith('qwen3.5-0.8b'):
        raise ValueError('This release requires a Qwen3.5-0.8B decision checkpoint.')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', str(decision.get('revision', ''))):
        raise ValueError('Checkpoint base revision must be a pinned commit.')
    if len(decision.get('codes', [])) != 255 or len(decision.get('token_ids', [])) != 255:
        raise ValueError('Checkpoint must contain 255 decision answer codes.')
    temperature = decision.get('temperature')
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Checkpoint temperature must be finite and positive.')
    tokenizer = json.loads((directory / 'tokenizer_config.json').read_text())
    if not tokenizer.get('chat_template') and not (directory / 'chat_template.jinja').is_file():
        raise ValueError('Checkpoint chat template is missing.')
    return decision


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(directory: Path, release: dict | None = None) -> Path:
    release = release_info() if release is None else release
    if not release.get('published'):
        raise ValueError('한국어 가중치가 아직 공개되지 않았습니다. 기존 가중치가 있으면 serve --checkpoint 경로를 지정하세요.')
    if release.get('provider') != 'github':
        raise ValueError('This runtime downloads Korean weights from GitHub Releases.')
    repository = str(release.get('repository', ''))
    tag = str(release.get('tag', ''))
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', tag):
        raise ValueError('Invalid GitHub release repository or version tag.')
    if not re.fullmatch(r'[0-9a-f]{40}', str(release.get('source_commit', ''))):
        raise ValueError('Invalid model release: source commit required.')
    expected = release.get('files', {})
    if not all(name in expected for name in REQUIRED_FILES):
        raise ValueError('Invalid model release: required file checksums missing.')
    if not any(name in expected for name in ('processor_config.json', 'preprocessor_config.json')):
        raise ValueError('Invalid model release: processor configuration missing.')
    for name, metadata in expected.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Model release filenames must be flat.')
        if not re.fullmatch(r'[0-9a-f]{64}', str(metadata.get('sha256', ''))) or not isinstance(metadata.get('size'), int) or metadata['size'] <= 0:
            raise ValueError(f'Invalid model release checksum or size: {name}')
        expected_url = f'https://github.com/{repository}/releases/download/{tag}/{name}'
        if metadata.get('url') != expected_url:
            raise ValueError(f'Invalid GitHub release asset URL: {name}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, metadata in expected.items():
        path = directory / name
        if path.is_file() and path.stat().st_size == metadata['size'] and file_sha256(path) == metadata['sha256']:
            continue  # An already verified checkpoint works without network access.
        temporary = directory / f'.{name}.download'
        digest, count = hashlib.sha256(), 0
        try:
            request = urllib.request.Request(metadata['url'], headers={'User-Agent': 'kor-jev/0.1'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    count += len(chunk)
                    if count > metadata['size']:
                        raise ValueError(f'Model file exceeds expected size: {name}')
                    digest.update(chunk)
                    stream.write(chunk)
            if count != metadata['size'] or digest.hexdigest() != metadata['sha256']:
                raise ValueError(f'Model file integrity check failed: {name}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    validate_checkpoint(directory)
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='한국어 로컬 판정 모델 다운로드 및 실행')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch', help='공개된 한국어 가중치 다운로드 및 검증')
    fetch.add_argument('--output', type=Path, default=Path('checkpoints/ko-v7'))
    serve = commands.add_parser('serve', help='로컬 Jev 호환 API 실행')
    serve.add_argument('--checkpoint', type=Path)
    serve.add_argument('--backend', choices=['auto', 'mlx', 'pytorch'], default='auto')
    serve.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == 'fetch':
            print(fetch_checkpoint(args.output))
            return
        checkpoint = args.checkpoint or fetch_checkpoint(Path('checkpoints/ko-v7'))
        validate_checkpoint(checkpoint)
        backend = args.backend
        if backend == 'auto':
            backend = 'mlx' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'pytorch'
        if backend == 'mlx':
            if args.device is not None:
                raise ValueError('--device is only used by the pytorch backend.')
            from importlib.util import find_spec
            if find_spec('mlx_lm') is None:
                raise ValueError('MLX 실행에는 uv sync --extra mac이 필요합니다.')
        os.environ.update(JEFF_CHECKPOINT=str(checkpoint.resolve()), JEFF_BACKEND=backend,
                          JEFF_ANY_MODEL='1', JEFF_HOST=args.host, PORT=str(args.port))
        if args.device:
            os.environ['JEFF_DEVICE'] = args.device
        from jeff.server import main as serve_main
        serve_main()
    except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        parser.exit(1, f'kor-jev: {error}\n')


if __name__ == '__main__':
    main()
