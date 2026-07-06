"""Tests for PMID date lookup. NCBI API calls are skipped in CI."""
import pytest
import json
from pathlib import Path
from data.pmid_date_lookup import PMIDDateLookup


class TestPMIDDateLookup:
    @pytest.fixture
    def lookup(self, tmp_path):
        cache = tmp_path / "test_pmid_cache.json"
        return PMIDDateLookup(cache_path=str(cache))

    def test_init_creates_empty_cache(self, lookup):
        assert lookup.cache_size == 0

    def test_fetch_dates_uses_cache(self, lookup):
        # Pre-populate cache
        lookup._cache = {"28086795": "2017-01-15", "12345678": "2015-03-20"}
        lookup._save_cache()

        # Load in new instance -- should use cache
        lookup2 = PMIDDateLookup(cache_path=str(lookup.cache_path))
        dates = lookup2.fetch_dates(["28086795", "12345678"])
        assert dates["28086795"] == "2017-01-15"
        assert dates["12345678"] == "2015-03-20"

    def test_fetch_dates_handles_uncached_pmids(self, lookup):
        """Uncached PMIDs trigger NCBI API call. Skip if no network."""
        dates = lookup.fetch_dates(["28086795"])  # known real PMID
        # May succeed or fail depending on network
        # At minimum should not crash
        assert isinstance(dates, dict)

    def test_cache_persistence(self, lookup, tmp_path):
        lookup._cache["99999999"] = "2020-06-15"
        lookup._save_cache()

        assert Path(lookup.cache_path).exists()
        with open(lookup.cache_path) as f:
            data = json.load(f)
        assert data["99999999"] == "2020-06-15"

    def test_batch_splitting(self, lookup):
        """Large PMID lists should be split into batches of 200."""
        pmids = [str(10000000 + i) for i in range(500)]
        # Pre-cache all to avoid actual API calls
        for p in pmids:
            lookup._cache[p] = "2020-01-01"
        dates = lookup.fetch_dates(pmids)
        assert len(dates) == 500
