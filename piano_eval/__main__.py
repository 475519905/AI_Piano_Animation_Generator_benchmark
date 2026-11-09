"""Offline CLI. No network, credentials, model loading, or pickle decoding."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from .report import evaluate, aggregate, check_splits
from . import __version__


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--prediction', required=True); e.add_argument('--reference', required=True); e.add_argument('--metadata', required=True); e.add_argument('--output', required=True)
    a = sub.add_parser('aggregate'); a.add_argument('reports', nargs='+'); a.add_argument('--output', required=True)
    s = sub.add_parser('check-splits'); s.add_argument('manifest'); s.add_argument('--hold-out-performers', action='store_true'); s.add_argument('--output', required=True)
    args = p.parse_args(argv); output = Path(args.output)
    if output.exists():
        p.error('output exists; refusing overwrite')
    try:
        if args.command == 'evaluate':
            files = {'prediction': args.prediction, 'reference': args.reference, 'metadata': args.metadata}
            before = {k: digest(v) for k, v in files.items()}
            result = evaluate(load_npz(args.prediction), load_npz(args.reference), json.loads(Path(args.metadata).read_text(encoding='utf-8')))
            after = {k: digest(v) for k, v in files.items()}
            if before != after:
                raise ValueError('input changed during evaluation')
            result['input_sha256'] = before
        elif args.command == 'aggregate':
            before_reports = [digest(f) for f in args.reports]
            result = aggregate([json.loads(Path(f).read_text(encoding='utf-8')) for f in args.reports])
            if before_reports != [digest(f) for f in args.reports]:
                raise ValueError('report changed during aggregation')
            result['input_report_sha256'] = before_reports
        else:
            before_manifest = digest(args.manifest)
            result = check_splits(json.loads(Path(args.manifest).read_text(encoding='utf-8')), args.hold_out_performers)
            if before_manifest != digest(args.manifest):
                raise ValueError('manifest changed during validation')
            result['manifest_sha256'] = before_manifest
        result['tool_version'] = __version__
        result['report_kind'] = args.command
        result['evaluator_sha256'] = {name: digest(Path(__file__).parent / name) for name in ('__init__.py', '__main__.py', 'metrics.py', 'report.py')}
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as f:
            f.write(encoded)
    except (ValueError, KeyError, TypeError, OSError) as error:
        p.error(str(error))
    print(str(output))
    return 0

if __name__ == '__main__':
    sys.exit(main())
