"""Corpus index tests.

Each test encodes a defect this module exists to prevent: a retriever that misses
a word form ("roll back" vs "rollback") makes the gate decorative, because the
pipeline can never retrieve what it needs to answer.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coa.corpus import CorpusIndex, content_words, related


class CorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "corpus"
        self.root.mkdir()
        self.db = Path(self.tmp.name) / "index.db"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def index(self) -> CorpusIndex:
        return CorpusIndex([self.root], self.db)


class TestWordForms(unittest.TestCase):
    def test_related_matches_prefix_variants(self):
        self.assertTrue(related("roll", "rollback"))
        self.assertTrue(related("deploy", "deployment"))

    def test_related_requires_four_characters(self):
        # A three-character stem is too weak to claim two words are the same form.
        self.assertFalse(related("spa", "sparse"))
        # Four characters is the floor: a 4-char prefix is a word form.
        self.assertTrue(related("spar", "sparse"))

    def test_related_rejects_different_stems(self):
        self.assertFalse(related("sparse", "spark"))

    def test_content_words_drop_stopwords(self):
        self.assertEqual(content_words("How do I roll back the deploy"),
                         ["roll", "back", "deploy"])


class TestBuild(CorpusTestCase):
    def test_build_counts_documents_and_passages(self):
        self.write("a.md", "# A\n\n" + "alpha beta gamma. " * 60)
        self.write("b.md", "# B\n\n" + "delta epsilon. " * 60)
        stats = self.index().build()
        self.assertEqual(stats.documents, 2)
        self.assertGreaterEqual(stats.passages, 2)

    def test_build_skips_binary_files(self):
        self.write("notes.md", "plain text about rollback")
        (self.root / "blob.md").write_bytes(b"\x00\x01\x02binary")
        stats = self.index().build()
        self.assertEqual(stats.documents, 1)
        self.assertEqual(stats.skipped, 1)

    def test_build_is_incremental(self):
        self.write("a.md", "alpha beta gamma " * 40)
        index = self.index()
        index.build()
        first = index.stats()
        index.build()  # nothing changed
        second = index.stats()
        self.assertEqual(first["passages"], second["passages"])
        self.assertEqual(first["documents"], second["documents"])

    def test_changed_file_is_reindexed(self):
        path = self.write("a.md", "alpha beta gamma " * 40)
        index = self.index()
        index.build()
        path.write_text("zeta eta theta " * 40, encoding="utf-8")
        index.build()
        hits = index.retrieve("zeta", k=3)
        self.assertTrue(hits, "re-indexed content must be retrievable")
        self.assertIn("zeta", hits[0].text)

    def test_long_document_is_chunked(self):
        self.write("long.md", ("paragraph about retrieval. " * 40 + "\n\n") * 12)
        stats = self.index().build()
        self.assertGreater(stats.passages, 2)

    def test_empty_root_is_not_an_error(self):
        index = CorpusIndex([Path(self.tmp.name) / "missing"], self.db)
        stats = index.build()
        self.assertEqual(stats.documents, 0)
        self.assertEqual(index.retrieve("anything"), [])


class TestRetrieve(CorpusTestCase):
    def test_retrieve_finds_exact_term(self):
        self.write("deploy.md", "The deployment pipeline runs on every merge.")
        index = self.index()
        index.build()
        hits = index.retrieve("deployment")
        self.assertEqual(len(hits), 1)
        self.assertIn("pipeline", hits[0].text)

    def test_retrieve_matches_word_form(self):
        # "roll back" must find a document that only says "rollback".
        self.write("rb.md", "Rollback is a staging migration procedure.")
        index = self.index()
        index.build()
        hits = index.retrieve("how do I roll back a bad deploy")
        self.assertTrue(hits, "word-form match must retrieve the document")

    def test_retrieve_returns_nothing_for_absent_topic(self):
        self.write("rb.md", "Rollback is a staging migration procedure.")
        index = self.index()
        index.build()
        self.assertEqual(index.retrieve("tomato soil mix"), [])

    def test_retrieve_respects_k(self):
        for i in range(6):
            self.write(f"d{i}.md", f"retrieval passage number {i} about indexing")
        index = self.index()
        index.build()
        self.assertLessEqual(len(index.retrieve("retrieval indexing", k=2)), 2)

    def test_passages_carry_citation(self):
        self.write("src.md", "citation anchor text about evidence")
        index = self.index()
        index.build()
        hit = index.retrieve("evidence")[0]
        self.assertTrue(hit.path.endswith("src.md"))
        self.assertIn("src.md", hit.cite())


class TestCrossThread(CorpusTestCase):
    """The gateway calls hooks on one thread and tools on another.

    A sqlite connection is bound to the thread that opened it, so a single cached
    connection raises ProgrammingError the moment the index is shared -- which in
    practice meant pre_llm_call failed and the gate never ran at all.
    """

    def test_query_from_another_thread(self):
        self.write("memo.md", "Project Falcon budget is 250,000 approved.")
        index = self.index()
        index.build()

        out = {}

        def worker():
            out["hits"] = index.retrieve("Project Falcon budget")
            out["stats"] = index.stats()

        import threading
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertNotIn("error", out)
        self.assertTrue(out["hits"], "a second thread must be able to query the index")
        self.assertEqual(out["stats"]["documents"], 1)

    def test_concurrent_queries(self):
        self.write("memo.md", "Helios ingest API slipped twice in Q4.")
        index = self.index()
        index.build()

        import threading
        errors, hits = [], []

        def worker():
            try:
                hits.append(len(index.retrieve("Helios ingest API")))
            except Exception as exc:  # noqa: BLE001 - the failure is the assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"concurrent queries raised: {errors[:2]}")
        self.assertEqual(len(hits), 8)
        self.assertTrue(all(h > 0 for h in hits))

    def test_build_then_query_across_threads(self):
        self.write("a.md", "rollback window definition for staging migrations")
        index = self.index()

        import threading
        result = {}

        def worker():
            try:
                index.build()
                result["hits"] = index.retrieve("rollback window")
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertNotIn("error", result)
        self.assertTrue(result["hits"])


if __name__ == "__main__":
    unittest.main()
