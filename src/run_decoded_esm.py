"""Derive restored sequences separately and extract source-bound ESM features."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.residue_cipher import ALPHABET, decode_sequence


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_inputs(cfg, root):
    """Preserve raw bytes and derive only protein IDs and restored sequences."""
    root = Path(root).resolve()
    receipt_path = root / cfg['mapping_path']
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('full_sequences_verified') is not True
            or receipt.get('fit_ids_only') is not True
            or receipt.get('labels_used') is not False):
        raise ValueError('Mapping provenance must be fit-only, full-sequence, and label-free')
    mapping = receipt['mapping']
    if set(mapping) != ALPHABET or set(mapping.values()) != ALPHABET:
        raise ValueError('All 20 standard residues require a complete bijection')
    sources = {key: root / cfg[key + '_path'] for key in ('train', 'test')}
    hashes = receipt['source_hashes']
    for key, path in sources.items():
        expected = hashes.get(cfg[key + '_path'])
        if expected is None or file_sha256(path) != expected:
            raise ValueError(f'{key} input hash differs from mapping receipt')
    output_dir = root / cfg['output_dir'] / 'inputs'
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {}
    for key, source in sources.items():
        frame = pd.read_csv(source, usecols=['protein_id', 'sequence'], dtype=str)
        if frame.isna().any().any() or frame['protein_id'].duplicated().any():
            raise ValueError(f'{key} IDs and sequences must be present and unique')
        frame['sequence'] = frame['sequence'].map(lambda s: decode_sequence(s, mapping))
        path = output_dir / (key + '-sequences.csv')
        frame[['protein_id', 'sequence']].to_csv(path, index=False)
        paths[key] = path
    provenance = dict(
        mapping_path=cfg['mapping_path'], mapping_sha256=file_sha256(receipt_path),
        source_hashes={key: file_sha256(path) for key, path in sources.items()},
        derived_hashes={key: file_sha256(path) for key, path in paths.items()},
        raw_csv_modified=False, labels_copied=False,
    )
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return paths


def verified_existing_inputs(cfg, root):
    """Resume extraction only when every raw, mapping, and derived hash agrees."""
    root = Path(root).resolve()
    inputs = root / cfg['output_dir'] / 'inputs'
    stored = json.loads((inputs / 'provenance.json').read_text(encoding='utf-8'))
    if stored['mapping_sha256'] != file_sha256(root / cfg['mapping_path']):
        raise ValueError('Resume mapping hash mismatch')
    paths = {}
    for key in ('train', 'test'):
        if stored['source_hashes'][key] != file_sha256(root / cfg[key + '_path']):
            raise ValueError('Resume input hash mismatch')
        path = inputs / (key + '-sequences.csv')
        if stored['derived_hashes'][key] != file_sha256(path):
            raise ValueError('Resume derived hash mismatch')
        paths[key] = path
    return paths


class ProgressBackend:
    def __init__(self, backend):
        self.backend = backend
        self.windows = 0
        self.started = time.monotonic()
        self.last_report = self.started

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def embed_windows(self, sequences):
        results = self.backend.embed_windows(sequences)
        self.windows += len(sequences)
        now = time.monotonic()
        if now - self.last_report >= 30:
            print(json.dumps(dict(windows=self.windows, elapsed_seconds=round(now-self.started),
                                  windows_per_second=round(self.windows/(now-self.started), 2))), flush=True)
            self.last_report = now
        return results


def extraction_settings(cfg, paths, root):
    root = Path(root)
    settings = {**cfg, 'train_path': str(paths['train']), 'test_path': str(paths['test']),
                'output_dir': str(root / cfg['output_dir'] / 'embeddings')}
    if cfg.get('test_screen_rows') is not None:
        rows = int(cfg['test_screen_rows'])
        if rows < 1:
            raise ValueError('test_screen_rows must be positive')
        expected = pd.read_csv(paths['test'], dtype=str).iloc[:rows]
        screen = Path(paths['test']).with_name('test-screen.csv')
        if screen.exists():
            if not pd.read_csv(screen, dtype=str).equals(expected):
                raise ValueError('Existing screen subset differs from source')
        else:
            expected.to_csv(screen, index=False)
        settings['test_path'] = str(screen)
    return settings


def main():
    from src.esm_embeddings import TransformersEsmBackend, extract_embedding_shards
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    root = Path.cwd()
    paths = verified_existing_inputs(cfg, root) if args.resume else prepare_inputs(cfg, root)
    settings = extraction_settings(cfg, paths, root)
    print('Validated separate decoded inputs; loading CUDA ESM', flush=True)
    if cfg.get('native_checkpoint'):
        from src.native_esm_features import NativeEsmBackend
        underlying = NativeEsmBackend(settings)
    elif cfg.get('residue_mean_max', False):
        from src.esm_token_features import TokenPoolingBackend
        underlying = TokenPoolingBackend(settings)
    else:
        underlying = TransformersEsmBackend(settings)
    backend = ProgressBackend(underlying)
    manifest = extract_embedding_shards(settings, backend=backend)
    print(str(manifest), flush=True)


if __name__ == '__main__':
    main()
