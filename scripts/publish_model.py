"""Publish ko-v7 checkpoint files to GitHub Releases and pin their checksums.

Requires GitHub CLI (gh) login only. No Hugging Face account is used.
--dry-run validates files without creating or uploading a release.
"""

import argparse
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from kor_jev.cli import REQUIRED_FILES, file_sha256, validate_checkpoint


ALLOWED_FILES = set(REQUIRED_FILES) | {
    'chat_template.jinja', 'chat_template.json', 'processor_config.json',
    'video_preprocessor_config.json', 'generation_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'merges.txt', 'vocab.txt',
}


def gh(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(['gh', *arguments], text=True, capture_output=True)
    if check and result.returncode:
        raise ValueError(result.stderr.strip() or 'GitHub CLI command failed.')
    return result


def prepare_files(checkpoint: Path, root: Path) -> dict[str, Path]:
    decision = validate_checkpoint(checkpoint)
    if abs(decision['temperature'] - 0.9728202031484768) > 1e-10:
        raise ValueError('This publisher is for selected ko-v7; calibration temperature does not match.')
    candidates = {p.name: p for p in checkpoint.iterdir() if p.is_file() and p.name in ALLOWED_FILES}
    candidates.update({
        'README.md': root / 'models/ko-v7/README.md',
        'LICENSE': root / 'models/ko-v7/LICENSE',
        'NOTICE': root / 'models/ko-v7/NOTICE',
    })
    for name, path in candidates.items():
        if not 0 < path.stat().st_size < 2 * 1024**3:
            raise ValueError(f'GitHub Release asset must be nonempty and under 2 GiB: {name}')
    return candidates


def check_assets(assets: list[dict], expected: dict) -> None:
    actual = {asset['name']: asset for asset in assets}
    for name, metadata in expected.items():
        asset = actual.get(name)
        if asset is None or asset['size'] != metadata['size'] or asset.get('digest') != f"sha256:{metadata['sha256']}":
            raise ValueError(f'GitHub asset digest verification failed; keep release as draft: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-id', default='suwonleee/kor-jev', help='GitHub owner/repository')
    parser.add_argument('--tag', default='ko-v7-preview')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', args.tag):
            raise ValueError('Invalid GitHub repository or release tag.')
        candidates = prepare_files(args.checkpoint, root)
        metadata = {name: {'size': p.stat().st_size, 'sha256': file_sha256(p),
                          'url': f'https://github.com/{args.repo_id}/releases/download/{args.tag}/{name}'}
                    for name, p in candidates.items()}
        print(json.dumps({'repository': args.repo_id, 'tag': args.tag, 'files': metadata}, indent=2))
        if args.dry_run:
            return
        source_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                       check=True, text=True, capture_output=True).stdout.strip()
        existing = gh('release', 'view', args.tag, '--repo', args.repo_id,
                      '--json', 'isDraft,databaseId,targetCommitish', check=False)
        if existing.returncode == 0:
            draft = json.loads(existing.stdout)
            if not draft['isDraft'] or draft['targetCommitish'] != source_commit:
                raise ValueError('Existing release/tag must not be replaced. Use a new --tag.')
        else:
            with tempfile.TemporaryDirectory(prefix='kor-jev-release-notes-') as temp:
                notes = Path(temp) / 'notes.md'
                notes.write_text('ko-v7 experimental Korean decision checkpoint.\n\n'
                                 'Full checkpoint, tokenizer, readout and calibration settings. '
                                 'Not adopted as a general-purpose production model. '
                                 'See repository README for evaluation scope and limitations.\n')
                gh('release', 'create', args.tag, '--repo', args.repo_id, '--draft', '--prerelease',
                   '--target', source_commit, '--title', 'ko-v7 experimental Korean decision model',
                   '--notes-file', str(notes))
        draft = json.loads(gh('release', 'view', args.tag, '--repo', args.repo_id,
                              '--json', 'databaseId').stdout)
        api_path = f"repos/{args.repo_id}/releases/{draft['databaseId']}"
        state = json.loads(gh('api', api_path).stdout)
        previous = {asset['name']: asset for asset in state['assets']}
        for name, path in candidates.items():
            if name in previous:
                check_assets([previous[name]], {name: metadata[name]})
                continue  # Resume matching draft assets; never clobber files or tags.
            print(f'Uploading {name}: {metadata[name]["size"] / 1024**2:.1f} MiB', flush=True)
            gh('release', 'upload', args.tag, str(path), '--repo', args.repo_id)
        state = json.loads(gh('api', api_path).stdout)
        check_assets(state['assets'], metadata)
        gh('release', 'edit', args.tag, '--repo', args.repo_id, '--draft=false', '--prerelease')
        # Confirm anonymous access before advertising availability to clone users.
        with urllib.request.urlopen(metadata['decision_config.json']['url'], timeout=120) as response:
            published_config = response.read()
        import hashlib
        if hashlib.sha256(published_config).hexdigest() != metadata['decision_config.json']['sha256']:
            raise ValueError('Public checkpoint configuration could not be verified.')
        release = {'version': 'ko-v7', 'provider': 'github', 'published': True,
                   'repository': args.repo_id, 'tag': args.tag,
                   'source_commit': source_commit, 'release_id': state['id'], 'files': metadata}
        (root / 'src/kor_jev/release.json').write_text(json.dumps(release, indent=2) + '\n')
        print(f'Public model verified: https://github.com/{args.repo_id}/releases/tag/{args.tag}')
        print('Next: verify a fresh download and inference, then commit release.json and update README status.')
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'publish_model: {error}\n')


if __name__ == '__main__':
    main()
