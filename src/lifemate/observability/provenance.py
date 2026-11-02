"""Small, offline provenance collector. Never traverses data/model weights."""
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(repo):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)
    try:
        top = Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve()
        if top != repo:
            return {'commit': None, 'dirty': None, 'unavailable': True,
                    'reason': 'source directory is not the Git root; do not claim an ancestor repository'}
        commit = git('rev-parse', 'HEAD').decode().strip()
        status = git('status', '--porcelain=v1', '-z')
        diff = git('diff', '--binary', 'HEAD')
        names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode('utf-8').split('\0')
        hashes = {}
        for name in sorted(set(names)):
            rel = Path(name)
            if not name or rel.suffix not in {'.py', '.toml', '.txt'}: continue
            path = (repo / rel).resolve()
            if not path.is_relative_to(repo) or any(p.lower().endswith('_environment') for p in path.parts):
                raise ValueError('unsafe source path')
            if path.is_file(): hashes[rel.as_posix()] = file_hash(path)
        return {'commit': commit, 'dirty': bool(status),
                'status_sha256': sha256(status).hexdigest(), 'diff_sha256': sha256(diff).hexdigest(),
                'source_files_sha256': hashes}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None, 'unavailable': True}


def collect_provenance(repo, *, config, manifest=None, model_path=None, prompt=None):
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'numpy', 'huggingface-hub'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    model = None
    if model_path is not None:
        root = Path(model_path).resolve()
        # Metadata only, never weight shards. Unknown revision remains unknown.
        files = {name: file_hash(root/name) for name in
                 ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json')
                 if (root/name).is_file()}
        revision = None
        if (root/'config.json').is_file():
            revision = json.loads((root/'config.json').read_text(encoding='utf-8')).get('_commit_hash')
        if revision is None and root.parent.name == 'snapshots': revision = root.name
        local_metadata = root/'.cache/huggingface/download/config.json.metadata'
        if revision is None and local_metadata.is_file():
            candidate = local_metadata.read_text(encoding='utf-8').splitlines()[0]
            if len(candidate) == 40 and all(c in '0123456789abcdef' for c in candidate):
                revision = candidate
        model = {'path': str(root), 'revision': revision, 'metadata_sha256': files,
                 'revision_note': 'Local config/snapshot/download metadata; not a full weight-integrity certification. Null means unknown.'}
    return {'schema_version': 1, 'source': source_provenance(repo),
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'config': config, 'model': model,
            'manifest_sha256': file_hash(manifest) if manifest else None,
            'prompt_sha256': sha256(prompt.encode()).hexdigest() if prompt is not None else None}
