"""Ingestion tests: "anything you give it" must actually be readable."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from coa import ingest


class IngestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, data: bytes | str) -> Path:
        path = self.root / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return path


class TestPlainText(IngestTestCase):
    def test_markdown_is_read(self):
        path = self.write("a.md", "# Title\n\nThe rollback takes four minutes.")
        self.assertIn("four minutes", ingest.extract_text(path))

    def test_binary_with_nul_is_skipped(self):
        path = self.write("blob.md", b"\x00\x01\x02binary")
        self.assertIsNone(ingest.extract_text(path))

    def test_html_is_stripped(self):
        path = self.write("a.html", "<html><style>x{}</style><p>Real text</p></html>")
        text = ingest.extract_text(path)
        self.assertIn("Real text", text)
        self.assertNotIn("<p>", text)
        self.assertNotIn("x{}", text)

    def test_unsupported_suffix_returns_none(self):
        path = self.write("a.bin", "whatever")
        self.assertIsNone(ingest.extract_text(path))

    def test_supported(self):
        self.assertTrue(ingest.supported(Path("notes.md")))
        self.assertTrue(ingest.supported(Path("contract.pdf")))
        self.assertFalse(ingest.supported(Path("image.psd")))


class TestOffice(IngestTestCase):
    def test_docx_text_is_extracted(self):
        path = self.root / "memo.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml",
                             "<w:document><w:body><w:p><w:t>Quarterly revenue</w:t></w:p>"
                             "<w:p><w:t>was up 12 percent</w:t></w:p></w:body></w:document>")
        text = ingest.extract_text(path)
        self.assertIn("Quarterly revenue", text)
        self.assertIn("12 percent", text)

    def test_xlsx_shared_strings_are_extracted(self):
        path = self.root / "sheet.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/sharedStrings.xml",
                             "<sst><si><t>Project Falcon</t></si><si><t>budget 250000</t></si></sst>")
        text = ingest.extract_text(path)
        self.assertIn("Project Falcon", text)
        self.assertIn("250000", text)

    def test_corrupt_zip_returns_none(self):
        path = self.write("broken.docx", b"not a zip")
        self.assertIsNone(ingest.extract_text(path))


if __name__ == "__main__":
    unittest.main()
