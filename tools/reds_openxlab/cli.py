#!/usr/bin/env python3
"""Run the installed OpenXLab CLI with an explicit, process-local TLS policy.

No SDK files, global SSL settings, sitecustomize, or proxy credentials are written.
Proxy variables are inherited from the caller. Credentials belong in the SDK's
interactive login, never in command arguments or source code.
"""
import argparse
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
import sys
from typing import Iterator, Optional, Sequence, Union

TLSVerify = Union[bool, str, None]


@contextmanager
def requests_tls(verify: TLSVerify) -> Iterator[None]:
    """Override Requests at send-time so an inherited CA env cannot undo it."""
    if verify is None:
        yield
        return
    import requests

    original = requests.sessions.Session.send

    def task_send(self, request, **kwargs):
        kwargs["verify"] = verify
        return original(self, request, **kwargs)

    requests.sessions.Session.send = task_send
    try:
        yield
    finally:
        requests.sessions.Session.send = original


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenXLab CLI wrapper; TLS changes affect this process only."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ca-bundle", type=Path, help="Trusted PEM CA bundle for Requests."
    )
    group.add_argument(
        "--insecure", action="store_true",
        help="Disable Requests TLS verification for this invocation only."
    )
    parser.add_argument("cli_args", nargs=argparse.REMAINDER,
                        help="Use -- followed by normal OpenXLab arguments.")
    args = parser.parse_args(argv)
    if args.cli_args[:1] == ["--"]:
        args.cli_args = args.cli_args[1:]
    if not args.cli_args:
        parser.error("Missing OpenXLab command; example: -- dataset info -r OpenDataLab/REDS")
    if args.ca_bundle is not None:
        args.ca_bundle = args.ca_bundle.expanduser().resolve()
        if not args.ca_bundle.is_file():
            parser.error("The specified CA bundle is not a readable file.")
        try:
            with args.ca_bundle.open("rb") as stream:
                if not stream.read(1):
                    parser.error("The CA bundle is empty.")
        except OSError:
            parser.error("The CA bundle cannot be read.")
    return args


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    try:
        dist = metadata.distribution("openxlab")
    except metadata.PackageNotFoundError:
        print("OPENXLAB_NOT_INSTALLED: install openxlab in the isolated download environment.",
              file=sys.stderr)
        return 2
    entries = [ep for ep in dist.entry_points
               if ep.group == "console_scripts" and ep.name == "openxlab"]
    if len(entries) != 1:
        print("OPENXLAB_ENTRYPOINT_ERROR: expected one installed openxlab console entry point.",
              file=sys.stderr)
        return 2
    verify = None
    if args.insecure:
        verify = False
        print("SSL_METHOD=VERIFY_DISABLED: this process only; TLS server identity is NOT verified.",
              file=sys.stderr)
    elif args.ca_bundle is not None:
        verify = str(args.ca_bundle)
        print("SSL_METHOD=CA_BUNDLE", file=sys.stderr)
    else:
        print("SSL_METHOD=SDK_DEFAULT", file=sys.stderr)
    original_argv = sys.argv
    try:
        sys.argv = ["openxlab", *args.cli_args]
        with requests_tls(verify):
            return entries[0].load()()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
