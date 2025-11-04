import re
from typing import Dict, List, Tuple, Optional


def normalize_feedback_keys(comments: Dict[str, str]) -> Dict[str, str]:
    """Normalize judge feedback keys to lowercase canonical names."""
    if not comments:
        return {}
    out = {}
    for k, v in comments.items():
        lk = (k or '').strip().lower()
        # Map known variants to canonical
        if 'cohesion' in lk:
            out['cohesion'] = v
        elif 'coverage' in lk:
            out['coverage'] = v
        elif 'redundancy' in lk:
            out['redundancy'] = v
        elif 'practical' in lk:
            out['practicality'] = v
        else:
            out[lk] = v
    return out


_TOKEN = re.compile(r"[A-Za-z0-9가-힣_]{2,}")
_STOP = set([
    # English
    'the','and','for','with','that','this','from','into','your','have','has','are','was','were','will','shall','should','must','may','not','but','can','could','to','of','in','on','by','as','at','it','or','be','is','a','an','we','you','they','their','our','its','over','under','more','less','than','such',
    # Generic placeholders/common nouns to avoid vague queries
    'spec','specs','some','any','many','various','thing','things','item','items','example','examples','policy','policies','requirement','requirements','group','groups','text','texts',
    # Korean (small, extend as needed)
    '및','그리고','또는','그러나','하지만','등','이','가','을','를','은','는','에','의','로','으로','에서','까지','마다','하여','하고','한다','수','있는','없는','대한','대한','관련','위한'
])


def _tokens(s: str) -> List[str]:
    if not s:
        return []
    toks = [t.lower() for t in _TOKEN.findall(s)]
    out: List[str] = []
    for t in toks:
        if t in _STOP:
            continue
        # drop pure digits and very short tokens
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def extract_keywords(text: str, max_k: int = 8) -> List[str]:
    """Very light keyword extractor via frequency."""
    freq: Dict[str, int] = {}
    for t in _tokens(text):
        freq[t] = freq.get(t, 0) + 1
    # sort by freq desc then alphabet
    return [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_k]]


def _missing_profile_terms(
    group_specs: Optional[List[Dict[str, str]]],
    domain_profile: str,
    task_profile: str,
    max_terms: int = 8
) -> List[str]:
    """Pick profile terms that are absent from current group's text."""
    profile_tokens = extract_keywords(domain_profile, 32) + extract_keywords(task_profile, 32)
    profile_tokens = list(dict.fromkeys(profile_tokens))
    group_tokens: set = set()
    if group_specs:
        for s in group_specs:
            for t in _tokens(s.get('text', '')):
                group_tokens.add(t)
    miss = [t for t in profile_tokens if t not in group_tokens]
    return miss[:max_terms]


def build_retrieval_queries_from_feedback(
    feedback: Dict[str, str],
    domain_profile: str,
    task_profile: str,
    max_queries: int = 8,
    group_specs: Optional[List[Dict[str, str]]] = None
) -> List[str]:
    """Create short, diversified English queries from feedback, profiles, and group text.

    - English only (per user preference)
    - If feedback is sparse/empty, fall back to missing profile terms in group
    """
    norm = normalize_feedback_keys(feedback or {})
    q: List[str] = []

    cov_kw = extract_keywords(norm.get('coverage', ''), 4)
    red_kw = extract_keywords(norm.get('redundancy', ''), 4)
    prac_kw = extract_keywords(norm.get('practicality', ''), 4)
    coh_kw = extract_keywords(norm.get('cohesion', ''), 4)

    prof_kw = (extract_keywords(domain_profile, 6) + extract_keywords(task_profile, 6))[:8]
    prof_kw = list(dict.fromkeys(prof_kw))

    # Missing terms from profiles w.r.t. group text
    missing_terms = _missing_profile_terms(group_specs, domain_profile, task_profile, max_terms=8)

    # Facet-oriented queries (English only)
    for w in cov_kw[:2]:
        q.append(f"coverage gaps for {w}")
        q.append(f"requirements to include {w}")
    for w in red_kw[:2]:
        q.append(f"deduplicate overlapping requirements {w}")
        q.append(f"merge similar policies {w}")
    for w in prac_kw[:2]:
        q.append(f"practical constraints and feasibility {w}")
        q.append(f"operational constraints {w}")
    for w in coh_kw[:2]:
        q.append(f"ensure group cohesion around {w}")
        q.append(f"align terminology for {w}")

    # Missing profile terms become coverage-targeted queries
    for t in missing_terms[:4]:
        q.append(f"coverage missing {t}")
        q.append(f"best practices for {t}")

    # Profile + feedback hybrid queries
    mix_src = (cov_kw or red_kw or prac_kw or coh_kw or missing_terms)
    if not mix_src:
        mix_src = extract_keywords(" ".join(norm.values()), 4)
    for a in prof_kw[:3]:
        for b in mix_src[:2]:
            q.append(f"{a} {b} requirements")

    # Deduplicate, filter vague queries, and cap
    GENERIC = {'spec','specs','some','item','items','example','examples','policy','policies','requirement','requirements'}
    q_dedup: List[str] = []
    seen = set()
    for item in q:
        s = item.strip()
        if not s:
            continue
        # filter if query tokens are all generic or too short
        qt = _tokens(s)
        if not qt:
            continue
        if all(t in GENERIC for t in qt):
            continue
        if len(qt) < 2:  # enforce at least two meaningful tokens
            continue
        if s in seen:
            continue
        q_dedup.append(s)
        seen.add(s)
        if len(q_dedup) >= max_queries:
            break
    return q_dedup


def _score_text_by_query_tokens(text: str, query: str) -> int:
    if not text or not query:
        return 0
    t = text.lower()
    score = 0
    for tok in _tokens(query):
        if tok and tok in t:
            score += 1
    return score


def select_specs_by_queries(
    specs: List[Dict[str, str]],
    queries: List[str],
    top_k: int = 5
) -> List[Dict[str, str]]:
    """Naive ranking by token overlap; returns top_k specs."""
    if not specs or not queries:
        return []
    scored: List[Tuple[int, Dict[str, str]]] = []
    for s in specs:
        txt = s.get('text', '')
        total = 0
        for q in queries:
            total += _score_text_by_query_tokens(txt, q)
        if total > 0:
            scored.append((total, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]
