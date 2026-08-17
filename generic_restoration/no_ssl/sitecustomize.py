"""Trusted-internal-network SSL bypass for setup/download commands only.

This module is injected through PYTHONPATH by the setup scripts. It contains no
proxy address or credential and must not be used outside the controlled server.
"""

from __future__ import annotations

import ssl


ssl._create_default_https_context = ssl._create_unverified_context

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass
try:
    import requests.sessions

    _request = requests.sessions.Session.request
    if not getattr(_request, "_generic_restoration_no_ssl", False):

        def request_without_verification(self, method, url, **kwargs):
            kwargs["verify"] = False
            return _request(self, method, url, **kwargs)

        request_without_verification._generic_restoration_no_ssl = True
        requests.sessions.Session.request = request_without_verification
except Exception:
    pass

try:
    import httpx

    _client_init = httpx.Client.__init__
    _async_client_init = httpx.AsyncClient.__init__
    if not getattr(_client_init, "_generic_restoration_no_ssl", False):

        def client_init_without_verification(self, *args, **kwargs):
            kwargs["verify"] = False
            return _client_init(self, *args, **kwargs)

        client_init_without_verification._generic_restoration_no_ssl = True
        httpx.Client.__init__ = client_init_without_verification
    if not getattr(_async_client_init, "_generic_restoration_no_ssl", False):

        def async_client_init_without_verification(self, *args, **kwargs):
            kwargs["verify"] = False
            return _async_client_init(self, *args, **kwargs)

        async_client_init_without_verification._generic_restoration_no_ssl = True
        httpx.AsyncClient.__init__ = async_client_init_without_verification
except Exception:
    pass
