"""Offline tests: no OpenXLab account, network, or credentials needed."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests

spec = importlib.util.spec_from_file_location("reds_task_cli", Path(__file__).with_name("cli.py"))
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def test_args_forwarded_without_separator(self):
        self.assertEqual(cli.parse_args(["dataset", "info", "-r", "OpenDataLab/REDS"]).cli_args,
                         ["dataset", "info", "-r", "OpenDataLab/REDS"])

    def test_insecure_and_separator(self):
        args = cli.parse_args(["--insecure", "--", "login"])
        self.assertTrue(args.insecure)
        self.assertEqual(args.cli_args, ["login"])

    def test_empty_command_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args([])

    def test_missing_ca_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--ca-bundle", "/nonexistent/reds-ca.pem", "--", "login"])

    def test_conflicting_tls_modes_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--insecure", "--ca-bundle", "/x", "--", "login"])

    def test_insecure_overrides_ca_kwarg_and_restores(self):
        with patch.object(requests.sessions.Session, "send", return_value="ok") as send:
            with cli.requests_tls(False):
                self.assertEqual(requests.Session().send(object(), verify="old.pem"), "ok")
            self.assertIs(requests.sessions.Session.send, send)
            self.assertIs(send.call_args.kwargs["verify"], False)

    def test_ca_override(self):
        with patch.object(requests.sessions.Session, "send", return_value="ok") as send:
            with cli.requests_tls("/trusted.pem"):
                requests.Session().send(object(), verify=False)
            self.assertEqual(send.call_args.kwargs["verify"], "/trusted.pem")

    def test_default_does_not_patch(self):
        before = requests.sessions.Session.send
        with cli.requests_tls(None):
            self.assertIs(requests.sessions.Session.send, before)

    def test_restores_on_exception(self):
        before = requests.sessions.Session.send
        with self.assertRaises(RuntimeError):
            with cli.requests_tls(False):
                raise RuntimeError("test")
        self.assertIs(requests.sessions.Session.send, before)

    def test_missing_distribution(self):
        with patch.object(cli.metadata, "distribution", side_effect=cli.metadata.PackageNotFoundError), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--", "login"]), 2)

    def test_real_entrypoint_semantics_and_argv_restored(self):
        before = sys.argv
        def fake_cli():
            self.assertEqual(sys.argv, ["openxlab", "dataset", "info", "-r", "OpenDataLab/REDS"])
            return 7
        ep = SimpleNamespace(group="console_scripts", name="openxlab", load=lambda: fake_cli)
        dist = SimpleNamespace(entry_points=[ep])
        with patch.object(cli.metadata, "distribution", return_value=dist), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--", "dataset", "info", "-r", "OpenDataLab/REDS"]), 7)
        self.assertIs(sys.argv, before)

    def test_sdk_system_exit_restores_process_state(self):
        before_argv = sys.argv
        before_send = requests.sessions.Session.send
        def fake_cli():
            raise SystemExit(9)
        ep = SimpleNamespace(group="console_scripts", name="openxlab", load=lambda: fake_cli)
        dist = SimpleNamespace(entry_points=[ep])
        with patch.object(cli.metadata, "distribution", return_value=dist), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main(["--insecure", "--", "login"])
        self.assertEqual(result.exception.code, 9)
        self.assertIs(sys.argv, before_argv)
        self.assertIs(requests.sessions.Session.send, before_send)


if __name__ == "__main__":
    unittest.main(verbosity=2)
