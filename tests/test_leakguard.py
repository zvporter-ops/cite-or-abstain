"""Leak guard tests: the fail-closed rule for private material."""

from __future__ import annotations

import unittest

from coa.leakguard import CLOUD, LOCAL, classify, is_local_url, redaction_notice


class TestLocality(unittest.TestCase):
    def test_loopback_is_local(self):
        for url in ("http://localhost:8888/v1", "http://127.0.0.1:8000/v1",
                    "http://[::1]:1234/v1"):
            self.assertTrue(is_local_url(url), url)

    def test_rfc1918_is_local(self):
        for url in ("http://192.168.1.188:8000/v1", "http://10.0.0.5/v1",
                    "http://172.16.4.2:8080/v1"):
            self.assertTrue(is_local_url(url), url)

    def test_tailscale_cgnat_is_local(self):
        # ipaddress does NOT report 100.64/10 as private; this must still be local.
        self.assertTrue(is_local_url("http://100.106.18.71:8888/v1"))

    def test_mdns_names_are_local(self):
        for url in ("http://spark.local/v1", "http://box.lan:8000/v1",
                    "http://server.internal/v1"):
            self.assertTrue(is_local_url(url), url)

    def test_hosted_providers_are_cloud(self):
        for url in ("https://api.openai.com/v1", "https://api.anthropic.com/v1",
                    "https://openrouter.ai/api/v1"):
            self.assertFalse(is_local_url(url), url)

    def test_missing_url_is_cloud(self):
        # Fail closed: an unclassified endpoint must not receive private material.
        self.assertFalse(is_local_url(None))
        self.assertFalse(is_local_url(""))
        self.assertFalse(is_local_url("not a url"))

    def test_classify(self):
        self.assertEqual(classify("http://localhost:8888/v1"), LOCAL)
        self.assertEqual(classify("https://api.openai.com/v1"), CLOUD)


class TestNotice(unittest.TestCase):
    def test_notice_names_the_count_and_the_rule(self):
        text = redaction_notice(3)
        self.assertIn("3 passage", text)
        self.assertIn("NOT been included", text)
        self.assertIn("corpus_ask", text)


if __name__ == "__main__":
    unittest.main()
