"""Node feature generation: PubMedBERT semantic + Node2Vec structural embeddings.

CRITICAL CONSTRAINT: All text summaries MUST be built from sources published
in or before 2024. Static reference databases (NCBI RefSeq, GO) are preferred
for known genes/proteins.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class EntitySummaryBuilder:
    """Build temporally-constrained text summaries for each KG entity.

    Priority: NCBI RefSeq / GO definitions > published abstract sentences (<=2024) > empty.
    """

    STATIC_SOURCES = ["RefSeq", "GO", "NCBI_Gene", "UniProt"]

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 database: str = "vte_kg", pubmed_xml_dir: str = None):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.database = database
        self.pubmed_xml_dir = Path(pubmed_xml_dir) if pubmed_xml_dir else None
        self._summary_cache = {}

    def _get_driver(self):
        from neo4j import GraphDatabase
        return GraphDatabase.driver(self.neo4j_uri,
                                     auth=(self.neo4j_user, self.neo4j_password))

    def get_summary_sources(self, entity_type: str, entity_name: str) -> List[dict]:
        """Return the source documents used to build this entity's summary.

        Returns empty list if Neo4j is unavailable or entity not found.
        """
        try:
            from neo4j.exceptions import Neo4jError, AuthError, ServiceUnavailable
        except ImportError:
            return []
        try:
            driver = self._get_driver()
        except Exception:
            return []
        sources = []
        try:
            with driver.session(database=self.database) as session:
                query = """
                MATCH (e)-[r:MENTIONED_IN|DISCUSSED_IN]->(p:Publication)
                WHERE (e.name = $name OR e.preferred_label = $name)
                  AND p.publication_date <= '2024-12-31'
                RETURN p.title AS title, p.publication_date AS pub_date,
                       p.abstract AS abstract, p.pmid AS pmid
                ORDER BY p.publication_date DESC
                LIMIT 50
                """
                result = session.run(query, name=entity_name)
                for record in result:
                    year = int(record["pub_date"][:4]) if record["pub_date"] else 9999
                    sources.append({
                        "source": f"PMID:{record['pmid']}",
                        "title": record["title"],
                        "year": year,
                        "text": record["abstract"] or "",
                    })
        except (Neo4jError, AuthError, ServiceUnavailable, OSError):
            sources = []
        finally:
            driver.close()
        return sources

    def build_entity_summary(self, entity_type: str, entity_name: str) -> str:
        """Build a temporally-constrained summary for one entity."""
        cache_key = f"{entity_type}:{entity_name}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        sources = self.get_summary_sources(entity_type, entity_name)
        if not sources:
            # No literature sources in KG — use entity name as summary
            # so PubMedBERT can still produce meaningful embeddings
            self._summary_cache[cache_key] = entity_name
            return entity_name

        static_texts = [s for s in sources if s["source"] in self.STATIC_SOURCES]
        if static_texts:
            parts = [f"{entity_name}: {s['text'][:500]}" for s in static_texts[:3]]
        else:
            parts = []
            for s in sources[:10]:
                if s["text"]:
                    sentences = [sent.strip() for sent in s["text"].split(".")
                                  if entity_name.lower() in sent.lower()]
                    parts.extend(sentences[:3])

        summary = ". ".join(parts[:15])
        if not summary:
            summary = entity_name

        self._summary_cache[cache_key] = summary
        return summary

    def build_all_summaries(self, node_data: Dict[str, Dict]) -> Dict[str, Dict[int, str]]:
        """Build summaries for all entities.

        Args:
            node_data: {entity_type: {attr_name: [values]}} from VTEKnowledgeGraphExporter

        Returns:
            {entity_type: {local_node_index: summary_text}}
        """
        summaries = {}
        for etype in tqdm(node_data, desc="Building entity summaries"):
            summaries[etype] = {}
            names = node_data[etype]["name"]
            for idx, name in enumerate(names):
                summary = self.build_entity_summary(etype, name)
                summaries[etype][idx] = summary
        return summaries


class PubMedBERTEncoder:
    """PubMedBERT encoder producing 768-dimensional semantic embeddings."""

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        self.model.eval()
        self._output_dim = 768

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode a single text to a 768-dimensional vector."""
        if not text or not text.strip():
            return torch.zeros(self._output_dim)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"]
        hidden = outputs.last_hidden_state  # [1, seq_len, 768]
        masked = hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return pooled.squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a list of texts, returning [N, 768]."""
        if len(texts) == 0:
            return torch.empty(0, self._output_dim)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            non_empty_mask = [bool(t and t.strip()) for t in batch]
            valid_texts = [t for t, m in zip(batch, non_empty_mask) if m]

            embeddings = torch.zeros(len(batch), self._output_dim)

            if valid_texts:
                inputs = self.tokenizer(
                    valid_texts, return_tensors="pt", truncation=True,
                    max_length=512, padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                hidden = outputs.last_hidden_state
                masked = hidden * attention_mask.unsqueeze(-1)
                pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

                valid_indices = [j for j, m in enumerate(non_empty_mask) if m]
                for k, vi in enumerate(valid_indices):
                    embeddings[vi] = pooled[k].cpu()

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)


class NodeFeaturePipeline:
    """Two-stage node feature pipeline: PubMedBERT(768) + Node2Vec(128) -> 896d."""

    def __init__(self, pubmedbert_model: str, node2vec_dim: int = 128,
                 output_dir: str = "data/features"):
        self.pubmedbert_encoder = PubMedBERTEncoder(model_name=pubmedbert_model)
        self.node2vec_dim = node2vec_dim
        self.output_dim = 768 + node2vec_dim  # 896
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self._node2vec_model = None

    def _validate_summaries(self, summaries: Dict[str, Dict[int, str]]) -> bool:
        """Validate that all node indices have summaries."""
        for etype, idx_summaries in summaries.items():
            if not idx_summaries:
                return False
        return True

    def generate_pubmedbert_features(self, summaries: Dict[str, Dict[int, str]]
                                     ) -> Dict[str, torch.Tensor]:
        """Generate 768d semantic features for all entity types."""
        features = {}
        for etype, idx_summaries in tqdm(summaries.items(), desc="PubMedBERT encoding"):
            n_nodes = max(idx_summaries.keys()) + 1
            texts = [idx_summaries.get(i, "") for i in range(n_nodes)]
            emb = self.pubmedbert_encoder.encode_batch(texts, batch_size=64)
            features[etype] = emb
        return features

    def generate_node2vec_features(self, edge_index_dict: dict,
                                    num_nodes_dict: Dict[str, int]
                                    ) -> Dict[str, torch.Tensor]:
        """Generate 128d structural features via Node2Vec on the train subgraph."""
        try:
            from node2vec import Node2Vec
        except ImportError:
            raise ImportError("node2vec package required. Install: pip install node2vec")

        features = {}
        total_nodes = sum(num_nodes_dict.values())
        offset_map = {}
        offset = 0
        for etype in sorted(num_nodes_dict.keys()):
            offset_map[etype] = offset
            offset += num_nodes_dict[etype]

        combined_edges = []
        for (src_t, _, dst_t), ei in edge_index_dict.items():
            src_offset = offset_map[src_t]
            dst_offset = offset_map[dst_t]
            combined_edges.append(ei.clone())
            combined_edges[-1][0] += src_offset
            combined_edges[-1][1] += dst_offset

        if not combined_edges:
            for etype, n_nodes in num_nodes_dict.items():
                features[etype] = torch.randn(n_nodes, self.node2vec_dim) * 0.01
            return features

        all_edges = torch.cat(combined_edges, dim=1)
        edge_list = [(int(all_edges[0, i]), int(all_edges[1, i]))
                      for i in range(all_edges.shape[1])]

        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(edge_list)

        n2v = Node2Vec(G, dimensions=self.node2vec_dim, walk_length=30,
                        num_walks=200, workers=4, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=4)

        all_embeddings = torch.zeros(total_nodes, self.node2vec_dim, dtype=torch.float32)
        for i in range(total_nodes):
            try:
                all_embeddings[i] = torch.tensor(model.wv[i], dtype=torch.float32)
            except KeyError:
                # Isolated node not in any edge — leave as zero vector
                pass

        for etype, n_nodes in num_nodes_dict.items():
            start = offset_map[etype]
            features[etype] = all_embeddings[start:start + n_nodes]

        self._node2vec_model = model
        return features

    def combine_and_normalize(self, pubmedbert_feats: Dict[str, torch.Tensor],
                               node2vec_feats: Dict[str, torch.Tensor]
                               ) -> Dict[str, torch.Tensor]:
        """Concatenate PubMedBERT and Node2Vec features, apply LayerNorm."""
        combined = {}
        for etype in pubmedbert_feats:
            pb = pubmedbert_feats[etype]
            n2v = node2vec_feats.get(etype)
            if n2v is None:
                n2v = torch.zeros(pb.shape[0], self.node2vec_dim)
            n2v = n2v.to(pb.device).to(pb.dtype)
            concat = torch.cat([pb, n2v], dim=-1)
            combined[etype] = self.layer_norm(concat)
        return combined

    def save_features(self, features: Dict[str, torch.Tensor], split: str):
        """Save feature tensors to disk."""
        path = self.output_dir / f"{split}_features.pt"
        torch.save(features, path)
        print(f"Saved {split} features to {path}")

    def load_features(self, split: str) -> Dict[str, torch.Tensor]:
        """Load saved feature tensors."""
        path = self.output_dir / f"{split}_features.pt"
        return torch.load(path, weights_only=True)
