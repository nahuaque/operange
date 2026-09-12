"""Run ``python -m operange.verify result.json`` without any solver calls."""

import argparse
from pathlib import Path
import sys

from .certificate_verification import (
    CertificateVerification,
    VerificationCheck,
    verify_result,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Result JSON file, or - for standard input")
    parser.add_argument(
        "--expected-contract-id", help="Pin the original contract content identity"
    )
    parser.add_argument("--max-bytes", type=int, default=32_000_000)
    args = parser.parse_args(argv)
    try:
        if args.max_bytes < 1:
            raise ValueError("max-bytes must be positive")
        if args.file == "-":
            document = sys.stdin.buffer.read(args.max_bytes + 1)
        else:
            with Path(args.file).open("rb") as stream:
                document = stream.read(args.max_bytes + 1)
        report = verify_result(
            document,
            expected_contract_id=args.expected_contract_id,
            max_bytes=args.max_bytes,
        )
    except (OSError, ValueError) as exc:
        report = CertificateVerification(
            None, (VerificationCheck("input", "invalid", str(exc)),)
        )
    print(report.to_json())
    return 0 if report.verified else 1 if report.status == "failed" else 2


if __name__ == "__main__":
    sys.exit(main())
