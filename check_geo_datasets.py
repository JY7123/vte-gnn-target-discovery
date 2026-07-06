#!/usr/bin/env python3
"""Search GEO for F2RL1/PAR-2 expression in VTE-related datasets."""
import json, time, urllib.request, urllib.parse

def log(msg):
    print(msg, flush=True)

def geo_search(term, retmax=20):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = urllib.parse.urlencode({
        'db': 'gds', 'term': term, 'retmax': retmax, 'retmode': 'json'
    })
    url = f"{base}/esearch.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get('esearchresult', {})
    except Exception as e:
        return {'error': str(e)}

def geo_summary(gds_ids):
    """Get summaries for GEO datasets."""
    if not gds_ids:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = urllib.parse.urlencode({
        'db': 'gds', 'id': ','.join(gds_ids[:15]), 'retmode': 'json'
    })
    url = f"{base}/esummary.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        results = []
        for uid in gds_ids[:15]:
            r = data.get('result', {}).get(uid, {})
            results.append({
                'gds_id': uid,
                'title': r.get('title', ''),
                'summary': r.get('summary', '')[:300],
                'organism': r.get('taxon', ''),
                'type': r.get('gdsType', ''),
                'samples': r.get('n_samples', ''),
            })
        return results
    except Exception as e:
        return [{'error': str(e)}]

log("=" * 60)
log("GEO Dataset Search: VTE + F2RL1/PAR-2")
log("=" * 60)

# Search 1: VTE expression datasets
log("\n[1] VTE/DVT expression datasets...")
r1 = geo_search('(venous thromboembolism OR deep vein thrombosis) AND (expression OR RNA-seq OR microarray) AND human')
log(f"  Count: {r1.get('count', 0)}")
ids1 = r1.get('idlist', [])
summaries1 = geo_summary(ids1)
for s in summaries1:
    log(f"  {s['gds_id']}: {s['title'][:100]} [{s['type']}, {s['samples']} samples]")
time.sleep(0.35)

# Search 2: F2RL1-specific datasets
log("\n[2] F2RL1/PAR-2 expression datasets...")
r2 = geo_search('F2RL1 OR PAR-2 OR "protease activated receptor 2"')
log(f"  Count: {r2.get('count', 0)}")
ids2 = r2.get('idlist', [])
summaries2 = geo_summary(ids2)
for s in summaries2:
    log(f"  {s['gds_id']}: {s['title'][:100]}")
time.sleep(0.35)

# Search 3: DVT miRNA-mRNA GSE196751 (known from web search)
log("\n[3] GSE196751 DVT RNA-seq...")
r3 = geo_search('GSE196751')
log(f"  Count: {r3.get('count', 0)}")
summaries3 = geo_summary(r3.get('idlist', []))
for s in summaries3:
    log(f"  {s['gds_id']}: {s['title'][:150]}")
    log(f"    Summary: {s.get('summary','')[:200]}")

# Search 4: GSE19151 VTE microarray
log("\n[4] GSE19151 VTE microarray...")
r4 = geo_search('GSE19151')
log(f"  Count: {r4.get('count', 0)}")
summaries4 = geo_summary(r4.get('idlist', []))
for s in summaries4:
    log(f"  {s['gds_id']}: {s['title'][:150]}")
    log(f"    Summary: {s.get('summary','')[:200]}")

log("\nDone.")
