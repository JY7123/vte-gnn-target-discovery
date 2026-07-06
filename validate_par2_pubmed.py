#!/usr/bin/env python3
"""PAR-2 (F2RL1) multi-dimensional evidence validation for VTE target discovery.
Queries PubMed, quantifies publication landscape, builds evidence matrix.
"""
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

def log(msg):
    print(msg, flush=True)

def pubmed_search(term, retmax=100):
    """Search PubMed via E-utilities, return PMIDs and counts."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Search
    params = urllib.parse.urlencode({
        'db': 'pubmed', 'term': term, 'retmax': retmax,
        'sort': 'relevance', 'retmode': 'json'
    })
    url = f"{base}/esearch.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get('esearchresult', {})
    except Exception as e:
        return {'error': str(e), 'term': term}

def pubmed_fetch_details(pmids, retmax=20):
    """Fetch article details (title, year, journal, abstract) for PMIDs."""
    if not pmids:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = urllib.parse.urlencode({
        'db': 'pubmed', 'id': ','.join(pmids[:retmax]),
        'retmode': 'xml', 'rettype': 'abstract'
    })
    url = f"{base}/efetch.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            tree = ET.parse(resp)
        articles = []
        for article in tree.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle')
            year = article.find('.//PubDate/Year')
            journal = article.find('.//Journal/Title')
            abstract = article.find('.//Abstract/AbstractText')
            pmid_elem = article.find('.//PMID')
            articles.append({
                'pmid': pmid_elem.text if pmid_elem is not None else '',
                'title': title.text[:200] if title is not None and title.text else '',
                'year': year.text if year is not None else '',
                'journal': journal.text if journal is not None else '',
                'abstract': abstract.text[:500] if abstract is not None and abstract.text else '',
            })
        return articles
    except Exception as e:
        return [{'error': str(e)}]

log("=" * 70)
log("PAR-2 (F2RL1) VTE Literature & Evidence Validation")
log(f"{datetime.now():%H:%M:%S}")
log("=" * 70)

# ── 1. PubMed literature quantification ──
log("\n[1/3] Quantifying PAR-2 publication landscape...")

# Comparison queries
queries = {
    'PAR-2 in VTE/DVT': '("PAR-2" OR F2RL1 OR "protease-activated receptor 2") AND (venous thrombosis OR deep vein thrombosis OR VTE OR venous thromboembolism)',
    'PAR-2 total': '("PAR-2" OR F2RL1 OR "protease-activated receptor 2")',
    'PAR-2 in inflammation': '("PAR-2" OR F2RL1) AND (inflammation OR inflammatory)',
    'PAR-2 in cancer': '("PAR-2" OR F2RL1) AND (cancer OR tumor OR neoplasm)',
    'PAR-2 cardiovascular': '("PAR-2" OR F2RL1) AND (cardiovascular OR atherosclerosis OR hypertension OR vascular)',
    'VTE total': 'venous thromboembolism OR deep vein thrombosis OR pulmonary embolism',
    'Coagulation in VTE': '(factor V OR factor VIII OR thrombin OR fibrinogen) AND (venous thrombosis OR VTE)',
}

results = {}
for label, term in queries.items():
    r = pubmed_search(term)
    count = int(r.get('count', 0))
    pmids = r.get('idlist', [])
    results[label] = {'count': count, 'pmids': pmids[:5]}
    log(f"  {label}: {count:,} publications")
    time.sleep(0.35)  # NCBI rate limit

# ── 2. Fetch top PAR-2 VTE articles ──
log("\n[2/3] Fetching top PAR-2 VTE articles...")
vte_pmids = results.get('PAR-2 in VTE/DVT', {}).get('pmids', [])
vte_articles = pubmed_fetch_details(vte_pmids) if vte_pmids else []

# Also get top cardiovascular PAR-2 articles
cv_pmids = results.get('PAR-2 cardiovascular', {}).get('pmids', [])
cv_articles = pubmed_fetch_details(cv_pmids) if cv_pmids else []
time.sleep(0.35)

# ── 3. Build evidence report ──
log("\n[3/3] Building evidence report...")

report = {
    'title': 'PAR-2 (F2RL1) Multi-Dimensional Evidence Validation for VTE Target Discovery',
    'timestamp': datetime.now().isoformat(),
    'publication_landscape': {
        query_label: {'count': r['count']}
        for query_label, r in results.items()
    },
    'top_vte_articles': vte_articles,
    'top_cv_articles': cv_articles,
    'neglect_ratio': None,
}

# Calculate neglect ratio
total_par2 = results.get('PAR-2 total', {}).get('count', 1)
vte_par2 = results.get('PAR-2 in VTE/DVT', {}).get('count', 0)
total_vte = results.get('VTE total', {}).get('count', 1)
coag_vte = results.get('Coagulation in VTE', {}).get('count', 0)

if total_par2 > 0:
    report['neglect_ratio'] = {
        'par2_vte_pct': round(vte_par2 / total_par2 * 100, 2),
        'coag_vte_ratio': round(coag_vte / vte_par2, 1) if vte_par2 > 0 else float('inf'),
        'interpretation': (
            f'Only {vte_par2:,} of {total_par2:,} PAR-2 papers ({vte_par2/total_par2*100:.2f}%) '
            f'mention VTE/DVT. Coagulation factors have {coag_vte:,} VTE papers '
            f'({coag_vte/vte_par2:.0f}x more). PAR-2 is systematically understudied in VTE.'
        ) if vte_par2 > 0 else 'Zero PAR-2 publications in VTE context.'
    }

# Print summary
log(f"\n{'='*70}")
log(f"PUBLICATION LANDSCAPE")
log(f"{'='*70}")
for label, r in results.items():
    log(f"  {label:<30}: {r['count']:>8,}")

if report['neglect_ratio']:
    nr = report['neglect_ratio']
    log(f"\n  PAR-2 VTE neglect ratio: {nr['par2_vte_pct']}% of PAR-2 literature")
    if vte_par2 > 0:
        log(f"  Coagulation factors have {nr['coag_vte_ratio']:.0f}x more VTE publications than PAR-2")
    log(f"  Interpretation: {nr['interpretation'][:200]}")

if vte_articles:
    log(f"\n  Top PAR-2 VTE articles:")
    for a in vte_articles[:5]:
        log(f"    [{a.get('year','?')}] {a.get('title','')[:120]}")

# ── Export ──
import os
os.makedirs("figures/pca_hidden", exist_ok=True)
out_path = "figures/pca_hidden/par2_evidence_validation.json"
with open(out_path, "w", encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
log(f"\nEvidence report saved: {out_path}")
