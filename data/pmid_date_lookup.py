"""PMID-to-publication_date lookup via NCBI E-utilities API.

Enriches the VTE KG with temporal metadata for time-split validation.
NCBI E-utilities rate limit: 3 requests/sec without API key, 10/sec with.
"""
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from collections import defaultdict


class PMIDDateLookup:
    """Batch lookup publication dates for PMIDs via NCBI E-utilities.

    Usage:
        lookup = PMIDDateLookup(cache_path="data/pmid_dates.json")
        dates = lookup.fetch_dates(["28086795", "12345678"])
        # dates = {"28086795": "2017-01-15", "12345678": "2015-03-20"}
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BATCH_SIZE = 200  # NCBI recommends <= 200 PMIDs per efetch request

    def __init__(self, cache_path: str = "data/pmid_dates.json",
                 api_key: Optional[str] = None):
        self.cache_path = Path(cache_path)
        self.api_key = api_key
        self._cache = self._load_cache()
        self._request_count = 0

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _rate_limit(self):
        """Respect NCBI rate limits: 3/sec without key, 10/sec with."""
        self._request_count += 1
        if self.api_key:
            if self._request_count % 10 == 0:
                time.sleep(1.1)
        else:
            if self._request_count % 3 == 0:
                time.sleep(1.1)

    def fetch_dates(self, pmids: List[str]) -> Dict[str, str]:
        """Fetch publication dates for a list of PMIDs.

        Returns: {pmid: "YYYY-MM-DD"} for found PMIDs.
        Unresolved PMIDs are silently skipped (can retry later).
        """
        # Filter out already cached
        uncached = [p for p in pmids if p not in self._cache]

        if not uncached:
            return {p: self._cache[p] for p in pmids if p in self._cache}

        # Batch fetch uncached PMIDs
        for i in range(0, len(uncached), self.BATCH_SIZE):
            batch = uncached[i:i + self.BATCH_SIZE]
            self._fetch_batch(batch)

        self._save_cache()
        return {p: self._cache[p] for p in pmids if p in self._cache}

    def _fetch_batch(self, pmids: List[str]):
        """Fetch one batch of PMIDs from NCBI."""
        self._rate_limit()

        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "abstract",
            "id": ",".join(pmids),
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/efetch.fcgi?{urlencode(params)}"

        try:
            req = Request(url, headers={"User-Agent": "VTE-GNN-Research/1.0"})
            with urlopen(req, timeout=30) as response:
                xml_data = response.read().decode("utf-8")
            self._parse_efetch_response(xml_data)
        except Exception as e:
            print(f"  Warning: NCBI efetch failed for batch (first PMID: {pmids[0]}): {e}")
            # Don't crash -- uncached PMIDs will remain unresolved

    def _parse_efetch_response(self, xml_data: str):
        """Parse NCBI efetch XML response, extract PMID -> pub date."""
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None:
                continue

            pmid = pmid_elem.text

            # Try to get the most specific date available
            date = None
            # Prefer ArticleDate (structured: Year/Month/Day)
            article_date = article.find(".//ArticleDate")
            if article_date is not None:
                year = article_date.findtext("Year", "")
                month = article_date.findtext("Month", "01")
                day = article_date.findtext("Day", "01")
                if year:
                    date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Fallback to PubMedPubDate
            if date is None:
                pub_date = article.find(".//PubMedPubDate[@PubStatus='pubmed']")
                if pub_date is None:
                    pub_date = article.find(".//PubMedPubDate")
                if pub_date is not None:
                    year = pub_date.findtext("Year", "")
                    month = pub_date.findtext("Month", "01")
                    day = pub_date.findtext("Day", "01")
                    if year:
                        date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Fallback to Journal Issue PubDate
            if date is None:
                journal_date = article.find(".//Journal/JournalIssue/PubDate")
                if journal_date is not None:
                    year = journal_date.findtext("Year", "")
                    month = journal_date.findtext("Month", "01")
                    day = journal_date.findtext("Day", "01")
                    if year:
                        date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            if date:
                self._cache[pmid] = date

    def extract_pmids_from_neo4j(self, uri: str, user: str, password: str,
                                   database: str = "neo4j") -> Dict[Tuple[str, str, str], List[str]]:
        """Extract all unique PMIDs from the KG.

        Returns: {edge_type_tuple: [pmid_list]} for edges with PMIDs.
        """
        from neo4j import GraphDatabase

        # Collect all PMIDs from edges
        edge_pmids = defaultdict(set)

        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session(database=database) as session:
                # Find all edges that have pmids property
                query = """
                MATCH (a)-[r]->(b)
                WHERE r.pmids IS NOT NULL
                RETURN labels(a) AS src_labels, type(r) AS rel_type,
                       labels(b) AS dst_labels, r.pmids AS pmids
                """
                result = session.run(query)
                for record in result:
                    src_labels = record["src_labels"]
                    dst_labels = record["dst_labels"]
                    rel_type = record["rel_type"]
                    pmids = record["pmids"]

                    # Use the first label for each node type
                    src_t = src_labels[0] if src_labels else "Entity"
                    dst_t = dst_labels[0] if dst_labels else "Entity"

                    key = (src_t, rel_type, dst_t)
                    for pmid in pmids:
                        edge_pmids[key].add(pmid)
        finally:
            driver.close()

        # Convert sets to lists
        return {k: list(v) for k, v in edge_pmids.items()}

    def build_date_map_for_edges(self, uri: str, user: str, password: str,
                                   database: str = "neo4j") -> Dict[Tuple, List[Optional[str]]]:
        """Full pipeline: extract PMIDs -> fetch dates -> build edge date map.

        Returns: {edge_type: [date_string for each edge]} ready for inject_edge_dates().
        """
        from neo4j import GraphDatabase

        print("Extracting PMIDs from Neo4j...")
        edge_pmids = self.extract_pmids_from_neo4j(uri, user, password, database)

        # Collect all unique PMIDs across all edge types
        all_pmids = set()
        for pmids in edge_pmids.values():
            all_pmids.update(pmids)

        print(f"  Found {len(all_pmids)} unique PMIDs across {len(edge_pmids)} edge types")

        # Fetch dates
        print(f"Fetching publication dates from NCBI for {len(all_pmids)} PMIDs...")
        all_pmids_list = sorted(all_pmids)
        pmid_to_date = self.fetch_dates(all_pmids_list)
        resolved = len([p for p in all_pmids_list if p in pmid_to_date])
        print(f"  Resolved {resolved}/{len(all_pmids_list)} PMIDs to dates")

        # Build per-edge date maps
        # Need to align dates with the actual edge order in HeteroData
        # We do this by re-querying the edges in the same order as the exporter
        driver = GraphDatabase.driver(uri, auth=(user, password))
        edge_dates = {}

        try:
            with driver.session(database=database) as session:
                for (src_t, rel, dst_t) in edge_pmids:
                    query = f"""
                    MATCH (a:{src_t})-[r:{rel}]->(b:{dst_t})
                    WHERE r.pmids IS NOT NULL
                    RETURN r.pmids AS pmids
                    """
                    result = session.run(query)
                    dates_for_type = []
                    for record in result:
                        pmids = record["pmids"]
                        # Use the earliest date among all PMIDs for this edge
                        edge_dates_list = [pmid_to_date.get(p) for p in pmids]
                        edge_dates_list = [d for d in edge_dates_list if d is not None]
                        if edge_dates_list:
                            dates_for_type.append(min(edge_dates_list))
                        else:
                            dates_for_type.append(None)

                    if dates_for_type:
                        edge_dates[(src_t, rel, dst_t)] = dates_for_type

        finally:
            driver.close()

        self._save_cache()
        return edge_dates

    @property
    def cache_size(self) -> int:
        return len(self._cache)
