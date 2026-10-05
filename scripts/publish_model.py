"""Publish the complete Korean checkpoint, then pin the public runtime manifest.

Run with --dry-run to validate files without creating or uploading a Hub repository.
Authentication uses the normal Hugging Face CLI login; credentials are never printed.
"""

import argparse
import json
from pathlib import Path

from kor_jev.cli import REQUIRED_FILES, file_sha256, validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-id', help='Default: authenticated HF account/kor-jev-ko-v7')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        decision = validate_checkpoint(args.checkpoint)
        if abs(decision['temperature'] - 0.9728202031484768) > 1e-10:
            raise ValueError('This publisher is for the selected ko-v7 checkpoint; temperature does not match.')
        allowed = set(REQUIRED_FILES) | {
            'chat_template.jinja', 'chat_template.json', 'processor_config.json',
            'video_preprocessor_config.json', 'generation_config.json',
            'special_tokens_map.json', 'added_tokens.json', 'merges.txt', 'vocab.txt',
        }
        candidates = {p.name: p for p in args.checkpoint.iterdir()
                      if p.is_file() and p.name in allowed}
        # Do not publish training state, optimizer state, logs, or local paths.
        candidates.update({
            'README.md': root / 'models/ko-v7/README.md',
            'LICENSE': root / 'models/ko-v7/LICENSE',
            'NOTICE': root / 'models/ko-v7/NOTICE',
        })
        if not all(name in candidates for name in REQUIRED_FILES):
            raise ValueError('Required checkpoint files missing.')
        metadata = {name: {'size': p.stat().st_size, 'sha256': file_sha256(p)}
                    for name, p in candidates.items()}
        print(json.dumps({'repo_id': args.repo_id, 'files': metadata}, indent=2))
        if args.dry_run:
            return
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        account = api.whoami()['name']
        repo_id = args.repo_id or f'{account}/kor-jev-ko-v7'
        print(f'Publishing as {account}; repository {repo_id}')
        api.create_repo(repo_id, repo_type='model', private=True, exist_ok=True)
        commit = api.create_commit(
            repo_id=repo_id,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                        for name, path in candidates.items()],
            commit_message='Release ko-v7 experimental Korean decision checkpoint',
        )
        uploaded = set(api.list_repo_files(repo_id, revision=commit.oid))
        if not set(candidates).issubset(uploaded):
            raise ValueError('Uploaded model repository is incomplete; keep it private.')
        api.update_repo_settings(repo_id, private=False)
        public = HfApi(token=False).model_info(repo_id, revision=commit.oid)
        if public.sha != commit.oid:
            raise ValueError('Public model revision could not be verified.')
        release = {'version': 'ko-v7', 'published': True, 'repo_id': repo_id,
                   'revision': commit.oid, 'files': metadata}
        (root / 'src/kor_jev/release.json').write_text(json.dumps(release, indent=2) + '\n')
        print(f'Public model verified: https://huggingface.co/{repo_id}/tree/{commit.oid}')
        print('The runtime release.json is now pinned. Verify a fresh clone before publishing the GitHub update.')
    except (ValueError, OSError) as error:
        parser.exit(1, f'publish_model: {error}\n')


if __name__ == '__main__':
    main()
