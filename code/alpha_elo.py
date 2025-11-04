from __future__ import annotations
import os
import json
import re
import time
import copy
import random
import hashlib
import uuid
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 외부 라이브러리 의존 (미설치/미키 설정 시 런타임에서만 실패)
try:
    import anthropic
except Exception:
    anthropic = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# Elo constants & helpers
# =========================
INITIAL_ELO: float = 1500.0

# 전역 캐시: 페어와이즈 심판 결과 (A vs B 순서 포함)
_PAIR_CACHE: Dict[Tuple[str, str], str] = {}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# === numbering stripper for pretty outputs ===
_NUM_PREFIX = re.compile(r'^\s*(?:\(?\d{1,3}\)?\s*(?:[.)]|[-–—:])\s+)')
def strip_leading_numbering(s: str) -> str:
    return _NUM_PREFIX.sub('', s or '')


def _pair_cache_key(text_a: str, text_b: str, constitution: str, domain_profile: str, task_profile: str) -> Tuple[str, str]:
    # 컨텍스트(헌법/도메인/태스크)에 따라 결과가 달라질 수 있으므로 포함
    ctx = _sha1("||".join([constitution, domain_profile, task_profile]))
    return (_sha1("A|" + ctx + "|" + text_a), _sha1("B|" + ctx + "|" + text_b))


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_elo(r_a: float, r_b: float, outcome_a: float, k: float = 24.0) -> Tuple[float, float]:
    """Return new (r_a, r_b) after a single match.
    outcome_a: 1.0 (A win), 0.5 (draw), 0.0 (A loss)
    """
    e_a = _elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    r_a_new = r_a + k * (outcome_a - e_a)
    r_b_new = r_b + k * ((1.0 - outcome_a) - e_b)
    return r_a_new, r_b_new


# PATCH: 동적 K (난이도/경기수 기반 약한 스케일링)
def effective_k(base_k: float, r_a: float, r_b: float, games_a: int, games_b: int, min_k: float = 8.0) -> float:
    delta = abs(r_a - r_b)
    # 델타가 작을수록(모델이 비슷할수록) 더 크게, 경기수 많을수록 점차 작게
    scale_delta = 1.0 / (1.0 + delta / 100.0)
    scale_games = 1.0 / (1.0 + max(games_a, games_b) / 10.0)
    k = base_k * 0.5 * (scale_delta + scale_games)
    return max(min_k, k)


@dataclass
class EvolverConfig:
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    generator_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "gpt-4o-mini"

    generations: int = 5
    population_per_gen: int = 8
    parallel_workers: int = 3

    judge_weights: Dict[str, int] = field(default_factory=lambda: {
        "constitution": 40,
        "domain": 30,
        "task": 30,
    })

    output_dir: str = "single_spec_result_ver1"
    use_timestamp_suffix: bool = False  # 폴더명에 타임스탬프 추가 여부
    random_seed: int = 1234

    # === Task-diverse parent sampling ===
    use_task_diversity: bool = True
    task_pool_size: int = 30
    parent_selection_size: int = 10
    diverse_parent_mix: Dict[str, float] = field(default_factory=lambda: {"top": 0.8, "low": 0.2})

    # === Pairwise Elo 설정 ===
    use_pairwise_elo: bool = True
    elo_k: float = 24.0
    use_ab_ba: bool = True
    elo_initial: float = 1500.0
    elo_dynamic_k: bool = True
    elo_min_k: float = 8.0
    significance_gap: int = 30
    use_score_normalization: bool = True
    parent_selection_metric: str = "score_norm"

    # === 비용 절감 설정 ===
    pairwise_top_m: int = 64
    elo_neighbor_k: int = 8
    matches_floor: int = 1
    matches_ceil: int = 4


# ===== Utility IDs =====

def make_id(text: str, suffix: str = "") -> str:
    # 과거 호환을 위해 유지(내용 기반 해시), 가능하면 make_unique_id 사용 권장
    base_text = f"{text}{suffix}"
    return hashlib.sha1(base_text.encode('utf-8')).hexdigest()[:12]


def make_unique_id(text: str, generation: int = 0, index: int = 0) -> str:
    # 충돌 방지 및 가독성: uuid4 사용
    return uuid.uuid4().hex[:12]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Archive (정렬을 Elo 우선)
# =========================
class Archive:
    # PATCH: elo_initial 인자 추가
    def __init__(self, max_capacity: int = 100, low_task_reservoir_capacity: int = 30, elo_initial: float = INITIAL_ELO):
        self.max_capacity = max_capacity
        self.elo_initial = elo_initial
        self.specs: List[Dict[str, Any]] = []
        self.low_task_reservoir_capacity = low_task_reservoir_capacity
        self.low_task_reservoir: List[Dict[str, Any]] = []

    def _dedup(self, lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for s in lst:
            sid = s.get('id')
            if sid and sid not in seen:
                seen.add(sid)
                out.append(s)
        return out

    def _update_low_task_reservoir(self, spec: Dict[str, Any]):
        if 'scores' not in spec:
            return
        self.low_task_reservoir = [s for s in self.low_task_reservoir if s.get('id') != spec.get('id')]
        self.low_task_reservoir.append(spec)
        # 기본은 task 점수 기준(참고용), 정규화 점수가 있으면 그걸로 보조
        self.low_task_reservoir.sort(key=lambda x: x.get('scores', {}).get('task', 0))
        if len(self.low_task_reservoir) > self.low_task_reservoir_capacity:
            self.low_task_reservoir = self.low_task_reservoir[:self.low_task_reservoir_capacity]

    @staticmethod
    def _rank_key(x: Dict[str, Any]) -> Tuple[float, float]:
        # PATCH: Elo 우선, 타이브레이크는 score_norm -> score
        score_norm = float(x.get('score_norm', x.get('score', 0.0)))
        return (float(x.get('elo', INITIAL_ELO)), score_norm)

    def _merge_stats(self, dst: Dict[str, Any], src: Dict[str, Any]):
        """동일 ID 업데이트 시 in-place 병합 (경기수/점수 유실 방지)."""
        stat_keys = [
            'elo', 'games', 'wins', 'losses', 'draws',
            'score', 'score_norm', 'scores', 'scores_norm', 'score_weighted',
            'provenance', 'evaluated_at', 'text', 'meta'
        ]
        for k in stat_keys:
            if k in src:
                dst[k] = src[k]

    def add(self, spec: Dict[str, Any]):
        # Elo 필드 기본값 보장 (cfg 기반 초기값)
        spec.setdefault('elo', self.elo_initial)
        spec.setdefault('games', 0)
        spec.setdefault('wins', 0)
        spec.setdefault('losses', 0)
        spec.setdefault('draws', 0)

        existing_ids = {s['id'] for s in self.specs}
        if spec['id'] in existing_ids:
            for i, s in enumerate(self.specs):
                if s['id'] == spec['id']:
                    # PATCH: 순위 우열과 무관하게 상태 in-place 병합
                    self._merge_stats(s, spec)
                    self.specs.sort(key=self._rank_key, reverse=True)
                    self._update_low_task_reservoir(s)
                    return

        self.specs.append(spec)
        self.specs.sort(key=self._rank_key, reverse=True)
        if len(self.specs) > self.max_capacity:
            self.specs = self.specs[:self.max_capacity]
        self._update_low_task_reservoir(spec)

    def sample_parents(self, n: int) -> List[Dict[str, Any]]:
        if not self.specs:
            return []
        top_30_specs = min(30, len(self.specs))
        candidates = self.specs[:top_30_specs]
        return random.sample(candidates, min(n, len(candidates)))

    # PATCH: 부모 샘플링에 사용할 메트릭을 외부에서 지정 가능
    def sample_parents_task_diverse(self, n: int, mix: Dict[str, float], task_pool_size: int,
                                    metric: str = "task") -> List[Dict[str, Any]]:
        if not self.specs:
            return []
        total = max(1, n)
        top_ratio = max(0.0, min(1.0, mix.get("top", 0.8)))
        n_top = max(1, int(round(total * top_ratio)))
        n_low = total - n_top

        def metric_value(s: Dict[str, Any]) -> float:
            if metric == "score_norm":
                return float(s.get('score_norm', 0.0))
            elif metric == "score":
                return float(s.get('score', 0.0))
            else:
                return float(s.get('scores', {}).get('task', 0.0))

        by_metric_desc = sorted(self.specs, key=metric_value, reverse=True)
        by_metric_asc = list(reversed(by_metric_desc))
        top_pool = by_metric_desc[:min(task_pool_size, len(by_metric_desc))]
        low_pool_main = by_metric_asc[:min(task_pool_size, len(by_metric_asc))]
        low_pool = self._dedup(low_pool_main + self.low_task_reservoir)

        sel: List[Dict[str, Any]] = []
        if top_pool and n_top > 0:
            chosen_top = random.sample(top_pool, min(n_top, len(top_pool)))
            sel += [dict(x, **{"_tier": "top"}) for x in chosen_top]  # ✅ 라벨링
        if low_pool and n_low > 0:
            chosen_low = random.sample(low_pool, min(n_low, len(low_pool)))
            sel += [dict(x, **{"_tier": "low"}) for x in chosen_low]  # ✅ 라벨링

        sel = self._dedup(sel)
        if len(sel) < total:
            fallback = [s for s in by_metric_desc if s.get('id') not in {x.get('id') for x in sel}]
            # 부족분은 기본적으로 TOP에서 채우되 라벨 부여
            fallback = [dict(x, **{"_tier": "top"}) for x in fallback]
            sel += fallback[:total - len(sel)]
        return sel[:total]
    

    def all_elites(self) -> List[Dict[str, Any]]:
        return self.specs.copy()


# =========================
# 프롬프트 파일 로드 함수들
# =========================

def load_text_prompt(file_path: str) -> str:
    """텍스트 프롬프트 파일 로드"""
    # 조용히 로드 (verbose 메시지 제거)
    if not os.path.exists(file_path):
        return ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
    except Exception as e:
        return ""

def load_json_prompt(file_path: str) -> List[Dict[str, Any]]:
    """JSON 프롬프트 파일 로드 (조용히)"""
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



# =========================
# 프롬프트 매니저 - 한 번만 로드하고 재사용
# =========================
class PromptManager:
    def __init__(self):
        self.generator_prompt = None
        self.pairwise_referee_prompt = None
        self.improvement_approaches = None
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """모든 프롬프트를 한 번에 로드 (조용히)"""
        # Generator 프롬프트 로드
        self.generator_prompt = load_text_prompt("prompts/generator_prompt.txt")
        if not self.generator_prompt:
            raise FileNotFoundError("Generator 프롬프트 파일을 찾을 수 없습니다. prompts/generator_prompt.txt 파일을 확인해주세요.")

        # Pairwise Referee 프롬프트 로드
        self.pairwise_referee_prompt = load_text_prompt("prompts/pairwise_referee_prompt.txt")
        if not self.pairwise_referee_prompt:
            raise FileNotFoundError("Pairwise Referee 프롬프트 파일을 찾을 수 없습니다. prompts/pairwise_referee_prompt.txt 파일을 확인해주세요.")

        # Improvement Approaches 로드
        self.improvement_approaches = load_json_prompt("prompts/improvement_approaches.json")
        if not self.improvement_approaches:
            raise FileNotFoundError("improvement_approaches.json 파일을 찾을 수 없습니다. prompts/improvement_approaches.json 파일을 확인해주세요.")

# 전역 프롬프트 매니저 인스턴스
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """프롬프트 매니저 싱글톤 인스턴스 반환"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager

# =========================
# Model wrappers
# =========================
def call_with_retry(fn, *, tries: int = 3, backoff: float = 0.8):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(backoff * (2 ** i))


class AnthropicClientWrapper:
    def __init__(self, api_key: str, model: str = None):
        if anthropic is None:
            raise ImportError("anthropic package not available")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.3) -> str:
        def _call():
            resp = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return resp.content[0].text
            except Exception:
                try:
                    return getattr(resp, 'completion', '')
                except Exception:
                    return str(resp)
        return call_with_retry(_call)


class OpenAIClientWrapper:
    def __init__(self, api_key: str, model: str = None):
        if OpenAI is None:
            raise ImportError("openai package not available")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.3) -> str:
        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        return call_with_retry(_call)


# =========================
# Generation (multi-parent refine)
# =========================
def llm_refine_multi_parent(
    parent_specs: List[Dict[str, Any]],
    generator,
    constitution: str,
    domain_profile: str,
    task_profile: str,
    generation: int = 0,
    generator_log_dir: str = None,
    domain_name: str = None,
    task_name: str = None,
    domain_concepts: List[str] = None,
    task_concepts: List[str] = None,
    forbidden_topics: str = None
) -> Dict[str, Any]:
    """여러 부모 spec들을 조합해서 개선된 spec 생성 (TOP/LOW 구분 반영)"""
    # 프롬프트 매니저에서 개선 방향들 가져오기
    pm = get_prompt_manager()
    selected_approach = random.choice(pm.improvement_approaches)

    # --- TOP/LOW 라벨 정규화 (기본: TOP) ---
    norm_parent_specs = []
    for p in parent_specs[:10]:
        tier = p.get('_tier', 'top')
        if tier not in ('top', 'low'):
            tier = 'top'
        q = dict(p)  # 얕은 복사: 원본 오염 방지
        q['_tier'] = tier
        norm_parent_specs.append(q)

    tops = [p for p in norm_parent_specs if p['_tier'] == 'top']
    lows = [p for p in norm_parent_specs if p['_tier'] == 'low']

    def _fmt_examples(lst: List[Dict[str, Any]], tag: str) -> str:
        out = ""
        for i, parent in enumerate(lst, 1):
            score = parent.get('score', 0)
            elo = parent.get('elo', INITIAL_ELO)
            text = parent.get('text', '')
            out += f"\n[{tag}] Example {i} (Score: {score}/100, Elo: {elo:.1f}):\n{text}\n"
        return out if out else f"\n[{tag}] (none)\n"

    # NEW: 프롬프트에서 사용하는 변수들을 명시적으로 정의
    approach_focus = selected_approach['focus']
    approach_description = selected_approach['description']
    good_examples = _fmt_examples(tops, "TOP")
    bad_examples = _fmt_examples(lows, "LOW")
    approach_focus_lower = approach_focus.lower()

    # 프롬프트 매니저에서 미리 로드된 템플릿 사용
    prompt_template = pm.generator_prompt
    
    # 동적 변수들 준비
    task_type = task_name or "general task"
    domain_type = domain_name or "general domain"
    task_concepts_str = ", ".join(task_concepts or []) or "general task concepts"
    domain_concepts_str = ", ".join(domain_concepts or []) or "general domain concepts"
    
    # Forbidden topics 추가 (다양성 강제)
    forbidden_topics_str = forbidden_topics or "None identified - encourage diverse coverage."
    
    prompt = prompt_template.format(
        domain_profile=domain_profile,
        task_profile=task_profile,
        constitution=constitution,
        approach_focus=approach_focus,
        approach_description=approach_description,
        approach_focus_lower=approach_focus_lower,
        good_examples=good_examples,
        bad_examples=bad_examples,
        task_type=task_type,
        domain_type=domain_type,
        task_concepts=task_concepts_str,
        domain_concepts=domain_concepts_str,
        forbidden_topics=forbidden_topics_str
    )

    try:
        refined = generator.generate(prompt, max_tokens=2000, temperature=0.4)
        if generator_log_dir:
            timestamp = int(time.time() * 1000)
            log_file = os.path.join(generator_log_dir, f"gen{generation:03d}_multi_parent_{timestamp}.json")
            log_data = {
                'generation': generation,
                'timestamp': timestamp,
                'type': 'multi_parent_refinement',
                'parent_count': len(norm_parent_specs),
                'parent_ids': [p.get('id') for p in norm_parent_specs],
                'parent_tiers': [p.get('_tier') for p in norm_parent_specs],
                'prompt': prompt,
                'response': refined,
                'temperature': 0.4,
                'max_tokens': 2000
            }
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"🚨 Generator 호출 실패: {e}")
        print(f"🔧 Fallback으로 첫 번째 부모 spec 사용")
        fallback_spec = copy.deepcopy(norm_parent_specs[0]) if norm_parent_specs else {'text': '// LLM-refine-failed: ' + str(e)}
        fallback_spec['id'] = make_unique_id(fallback_spec.get('text', 'fallback'))
        return fallback_spec

    new_spec = {
        'text': refined,
        'id': make_unique_id(refined, generation, 0),
        'meta': {
            'origin': 'multi_parent_refinement',
            'parent_count': len(norm_parent_specs),
            'parent_tiers': [p.get('_tier') for p in norm_parent_specs]
        }
    }
    return new_spec

# =========================
# Split LLM output → individual specs
# =========================

def split_llm_response_to_specs(llm_response: str, parent_id: str) -> List[Dict[str, Any]]:
    import re
    lines = [line.strip() for line in llm_response.splitlines() if line.strip()]
    rfc2119_keywords = [
        'MUST', 'MUST NOT', 'MUST_NOT',
        'SHOULD', 'SHOULD NOT', 'SHOULD_NOT',
        'SHALL', 'SHALL NOT', 'SHALL_NOT',
        'MAY', 'REQUIRED', 'RECOMMENDED', 'OPTIONAL'
    ]

    # 보다 강건한 헤더 매칭 (불릿/콜론/대소문자 허용)
    KW = r"(MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|SHALL(?:\s+NOT)?|MAY|REQUIRED|RECOMMENDED|OPTIONAL)"
    BUL = r"(?:\d+\.\s*|[-*•–—]\s*)"
    import re as _re
    pat_head = _re.compile(rf"^\s*(?:{BUL})?\s*{KW}\b[:\-]?\s*(.*)", _re.I)

    spec_lines: List[str] = []
    current_spec = ""

    for line in lines:
        m = pat_head.match(line)
        if m:
            # 새 스펙 시작
            content = m.group(0).strip()
            if current_spec:
                spec_lines.append(current_spec.strip())
            current_spec = content
        else:
            # 이어지는 줄 합치기 (불릿 제거)
            cleaned = _re.sub(r'^\s*(?:\d+\.\s*|[-*•–—]\s*)', '', line).strip()
            if current_spec:
                current_spec += " " + cleaned

    if current_spec:
        spec_lines.append(current_spec.strip())
    print(f"    🔍 필터링 결과: {len(lines)}줄 → {len(spec_lines)}개 RFC2119 spec")

    individual_specs: List[Dict[str, Any]] = []
    for i, spec_text in enumerate(spec_lines):
        spec = {
            'id': make_unique_id(spec_text, generation=0, index=i),
            'text': strip_leading_numbering(spec_text),
            'meta': {'origin': 'llm_split', 'index': i, 'parent_id': parent_id},
            'provenance': [{'op': 'llm_refine_split', 'parent': parent_id}],
            'elo': INITIAL_ELO,
            'games': 0, 'wins': 0, 'losses': 0, 'draws': 0,
        }
        individual_specs.append(spec)
    return individual_specs


# =========================
# Variation wrapper
# =========================

def apply_variation_multi_parent(parent_specs: List[Dict[str, Any]], generator, constitution: str, domain_profile: str, task_profile: str, generation: int = 0, generator_log_dir: str = None, domain_name: str = None, task_name: str = None, domain_concepts: List[str] = None, task_concepts: List[str] = None, forbidden_topics: str = None) -> List[Dict[str, Any]]:
    refined_response = llm_refine_multi_parent(parent_specs, generator, constitution, domain_profile, task_profile, generation, generator_log_dir, domain_name, task_name, domain_concepts, task_concepts, forbidden_topics)
    individual_specs = split_llm_response_to_specs(refined_response['text'], 'multi_parent')
    parent_ids = [p['id'] for p in parent_specs[:5]]
    parent_tiers = refined_response.get('meta', {}).get('parent_tiers', [])
    for spec in individual_specs:
        spec.setdefault('provenance', []).append({
            'op': 'multi_parent_refine_split',
            'parent_count': len(parent_specs),
            'parent_ids': parent_ids,
            'parent_tiers': parent_tiers  # ✅ TOP/LOW 라벨 보존
        })
    if not individual_specs:
        fallback = copy.deepcopy(parent_specs[0]) if parent_specs else {'text': '// Multi-parent failed', 'id': make_unique_id('fallback')}
        fallback.setdefault('provenance', []).append({'op': 'multi_parent_fallback'})
        fallback.setdefault('elo', INITIAL_ELO)
        fallback.setdefault('games', 0)
        fallback.setdefault('wins', 0)
        fallback.setdefault('losses', 0)
        fallback.setdefault('draws', 0)
        return [fallback]
    return individual_specs


# =========================
# Judges (Pointwise) & Pairwise Referee
# =========================
class Judge:
    def __init__(self, name: str, client):
        self.name = name
        self.client = client

        # 프롬프트를 한 번만 로드하고 저장 (조용히)
        if self.name == "domain":
            self.prompt_template = load_text_prompt("prompts/domain_judge_prompt.txt")
        elif self.name == "task":
            self.prompt_template = load_text_prompt("prompts/task_judge_prompt.txt")
        else:  # constitution
            self.prompt_template = load_text_prompt("prompts/constitution_judge_prompt.txt")

        if not self.prompt_template:
            raise FileNotFoundError(f"{self.name} Judge 프롬프트 파일을 찾을 수 없습니다. prompts/{self.name}_judge_prompt.txt 파일을 확인해주세요.")

    def score(self, spec_text: str, constitution: str, domain_profile: str, task_profile: str, max_tokens: int = 300) -> Tuple[float, str, str]:
        # 저장된 프롬프트 템플릿 사용
        if not hasattr(self, 'prompt_template') or not self.prompt_template:
            print(f"🚨 {self.name} Judge: 프롬프트 템플릿이 로드되지 않았습니다")
            raise RuntimeError(f"{self.name} Judge: 프롬프트 템플릿이 로드되지 않았습니다. 초기화를 확인해주세요.")
        
        # 프롬프트 포맷팅
        if self.name == "domain":
            prompt = self.prompt_template.format(
                domain_profile=domain_profile,
                spec_text=spec_text
            )
        elif self.name == "task":
            prompt = self.prompt_template.format(
                task_profile=task_profile,
                spec_text=spec_text
            )
        else:  # constitution
            prompt = self.prompt_template.format(
                constitution=constitution,
                spec_text=spec_text
            )
        try:
            raw = self.client.generate(prompt, max_tokens=max_tokens, temperature=0.1)
            if self.name in ("domain", "task"):
                total = None
                for line in raw.split('\n'):
                    if line.startswith('TOTAL='):
                        try:
                            total = int(line.split('=')[1].strip())
                            break
                        except:
                            pass
                if total is None:
                    return 0, prompt, raw
                return max(0, min(30, total)), prompt, raw
            elif self.name == "constitution":
                total = None
                for line in raw.split('\n'):
                    if line.startswith('TOTAL='):
                        try:
                            total = int(line.split('=')[1].strip())
                            break
                        except:
                            pass
                if total is None:
                    return 0, prompt, raw
                return max(0, min(40, total)), prompt, raw
            else:
                return 0.0, prompt, raw
        except Exception:
            return 0.0, "", ""


def pairwise_referee_decision(client, text_a: str, text_b: str, constitution: str, domain_profile: str, task_profile: str, max_tokens: int = 200) -> str:
    """승/패/무만 판단. Returns 'A', 'B', or 'TIE'"""
    # 프롬프트 매니저에서 미리 로드된 템플릿 사용
    pm = get_prompt_manager()
    prompt_template = pm.pairwise_referee_prompt
    
    prompt = prompt_template.format(
        constitution=constitution,
        domain_profile=domain_profile,
        task_profile=task_profile,
        text_a=text_a,
        text_b=text_b
    )
    try:
        raw = client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception:
        return 'TIE'
    ans = 'TIE'
    for line in reversed(raw.splitlines()):
        ls = line.strip().upper()
        if ls.startswith('ANSWER:'):
            token = ls.split(':', 1)[1].strip()
            if token in ('A', 'B', 'TIE'):
                ans = token
            break
    return ans


def pairwise_referee_decision_cached(client, text_a: str, text_b: str, constitution: str, domain_profile: str, task_profile: str, max_tokens: int = 200) -> str:
    """캐시 사용 버전"""
    k = _pair_cache_key(text_a, text_b, constitution, domain_profile, task_profile)
    if k in _PAIR_CACHE:
        return _PAIR_CACHE[k]
    ans = pairwise_referee_decision(client, text_a, text_b, constitution, domain_profile, task_profile, max_tokens=max_tokens)
    _PAIR_CACHE[k] = ans
    return ans


def pairwise_ab_ba_conditional(client, text_a: str, text_b: str, constitution: str, domain_profile: str, task_profile: str, use_ab_ba: bool = True) -> float:
    """AB/BA 역전으로 위치 편향 상쇄. 반환값은 A 관점 outcome(1/0.5/0), 첫 결과가 TIE면 BA 생략."""
    res_ab = pairwise_referee_decision_cached(client, text_a, text_b, constitution, domain_profile, task_profile)
    if (not use_ab_ba) or (res_ab == 'TIE'):
        return 1.0 if res_ab == 'A' else (0.0 if res_ab == 'B' else 0.5)
    res_ba = pairwise_referee_decision_cached(client, text_b, text_a, constitution, domain_profile, task_profile)
    res_ba_mapped = 'B' if res_ba == 'A' else ('A' if res_ba == 'B' else 'TIE')
    if res_ab == res_ba_mapped:
        return 1.0 if res_ab == 'A' else (0.0 if res_ab == 'B' else 0.5)
    else:
        return 0.5


# =========================
# Pointwise evaluation (참고용 점수)
# =========================

def evaluate_spec_with_judges(spec: Dict[str, Any], judges: Dict[str, Judge], weights: Dict[str, int],
                              constitution: str, domain_profile: str, task_profile: str, 
                              generation: int = 0, quality_threshold: int = 3, judges_log_dir: str = None, is_top10: bool = False,
                              initial_elo: float = INITIAL_ELO) -> Dict[str, Any]:
    scores = {'constitution': 0, 'domain': 0, 'task': 0}
    spec_text = spec.get('text', '').strip()
    if not spec_text:
        out = copy.deepcopy(spec)
        out['scores'] = scores
        out['score'] = 0
        out['score_weighted'] = 0.0
        out['evaluated_at'] = time.time()
        out.setdefault('elo', initial_elo)
        out.setdefault('games', 0)
        out.setdefault('wins', 0)
        out.setdefault('losses', 0)
        out.setdefault('draws', 0)
        return out

    judge_responses = {}
    for k, j in judges.items():
        try:
            sc, prompt, raw_response = j.score(spec_text, constitution, domain_profile, task_profile)
            scores[k] = sc
            if is_top10 and judges_log_dir:
                judge_responses[k] = {
                    'judge_type': k,
                    'prompt': prompt,
                    'raw_response': raw_response,
                    'parsed_score': sc,
                    'temperature': 0.1,
                    'max_tokens': 300
                }
        except Exception:
            scores[k] = 0

    if is_top10 and judges_log_dir and len(judge_responses) > 0:
        timestamp = int(time.time() * 1000)
        log_file = os.path.join(judges_log_dir, f"gen{generation:03d}_{spec.get('id', 'unknown')}_all_judges.json")
        log_data = {
            'generation': generation,
            'timestamp': timestamp,
            'spec_id': spec.get('id', 'unknown'),
            'spec_text': spec_text[:500],
            'judges': judge_responses,
            'final_scores': scores,
            'total_score': 0
        }
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Judge 응답 저장 실패 {log_file}: {e}")

    constitution_score = scores.get('constitution', 0)
    domain_score = scores.get('domain', 0)
    task_score = scores.get('task', 0)
    total = constitution_score + domain_score + task_score

    # 가중 합산(실제 weights 반영)
    w = weights or {"constitution": 40, "domain": 30, "task": 30}
    score_weighted = (
        (constitution_score / 40.0) * w.get('constitution', 40) +
        (domain_score / 30.0) * w.get('domain', 30) +
        (task_score / 30.0) * w.get('task', 30)
    )

    if is_top10 and judges_log_dir and len(judge_responses) > 0:
        log_file = os.path.join(judges_log_dir, f"gen{generation:03d}_{spec.get('id', 'unknown')}_all_judges.json")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                log_data['total_score'] = total
                log_data['score_weighted'] = score_weighted
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: total_score 업데이트 실패 {log_file}: {e}")

    out = copy.deepcopy(spec)
    out['scores'] = scores
    out['score'] = total
    out['score_weighted'] = score_weighted
    out['evaluated_at'] = time.time()
    out.setdefault('elo', initial_elo)
    out.setdefault('games', 0)
    out.setdefault('wins', 0)
    out.setdefault('losses', 0)
    out.setdefault('draws', 0)
    return out


# =========================
# Score normalization (편향 완화)
# =========================
def normalize_judge_scores_for_pool(specs: List[Dict[str, Any]]):
    import statistics as st
    keys = ['constitution', 'domain', 'task']
    stats = {}
    for k in keys:
        vals = [float(s.get('scores', {}).get(k, 0.0)) for s in specs]
        mu = st.mean(vals) if vals else 0.0
        sd = st.pstdev(vals) if len(vals) > 1 else 1.0
        stats[k] = (mu, sd or 1.0)
    for s in specs:
        s.setdefault('scores_norm', {})
        for k in keys:
            mu, sd = stats[k]
            s['scores_norm'][k] = (float(s.get('scores', {}).get(k, 0.0)) - mu) / sd
        # 가중 합: constitution 0.4, domain 0.3, task 0.3
        s['score_norm'] = (
            s['scores_norm']['constitution'] * 0.4 +
            s['scores_norm']['domain'] * 0.3 +
            s['scores_norm']['task'] * 0.3
        )


# === PII masking + embedding-based (or shingle) deduplication ===
PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\+?\d[\d\s\-()]{7,}\d"),
    re.compile(r"\b\d{2,}[-/\s]?\d{2,}[-/\s]?\d{2,}\b"),
]
def mask_pii(text: str) -> str:
    t = text or ""
    for p in PII_PATTERNS:
        t = p.sub("[REDACTED]", t)
    t = re.sub(r"\d{4,}", "[NUMBER]", t)
    return t

def _l2_normalize(M):
    import numpy as _np
    M = _np.asarray(M, dtype=float)
    norms = _np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

_EMBED_BACKEND = None

def extract_semantic_core_generic(text):
    """RFC2119 키워드와 일반적인 구조어만 제거 (도메인 무관)"""
    import re
    
    # 1. RFC2119 키워드 제거
    text = re.sub(r'^\s*(MUST|SHOULD|SHALL|MAY|REQUIRED|RECOMMENDED|OPTIONAL)(\s+NOT)?\s+', '', text, flags=re.IGNORECASE)
    
    # 2. 매우 일반적인 구조어만 제거 (도메인 특화 단어는 모두 보존)
    generic_stop_words = {
        'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'when', 'where', 'how', 'that', 'this', 'these', 'those', 'all', 'any', 'each', 'every'
    }
    
    # 3. 단어 분리 후 매우 기본적인 불용어만 제거
    words = []
    for word in text.split():
        clean_word = re.sub(r'[^\w\-]', '', word).lower()  # 구두점 제거
        if clean_word and clean_word not in generic_stop_words and len(clean_word) > 1:
            words.append(clean_word)
    
    return ' '.join(words)

# 의미적 개념 추출 함수들 제거됨 - 단순한 키워드 추출만 사용

# =========================
# 개선된 클러스터링 시스템
# =========================

# 클러스터링 함수들 제거됨 - 우선순위 기반 지배 관계 중복 제거로 대체

def keyword_enrich(texts, topk=None):
    """Lightweight TF-IDF keyword boosting - 도메인 무관한 기본 불용어만 사용
    - RFC2119와 매우 일반적인 단어만 제거하여 도메인 독립성 보장
    - 도메인 특화 키워드는 모두 보존
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import re

    if not texts:
        return texts

    # 매우 기본적인 불용어만 (RFC2119 + 일반 구조어)
    stop = {
        "must","should","shall","may","not","never","always",
        "mustn","shouldn","shan","can","cannot","can't","cant",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "this", "that", "these", "those", "when", "where", "how", "what", "why", "who", "which"
    }

    try:
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]+\b",
            stop_words=list(stop),
            min_df=2,
            max_df=0.7,  # 더 관대하게 조정
            max_features=8000,
            sublinear_tf=True,
            norm="l2",
            smooth_idf=True,
        )
        X = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())

        # Fall back if vocabulary collapsed
        if X.shape[1] == 0:
            return texts

        A = X.toarray()
        enriched = []
        for i, t in enumerate(texts):
            # dynamic top-k: ~1 per 8 words, clamp [2,5] (더 보수적으로)
            if topk is None:
                wc = max(1, len(re.findall(r"[A-Za-z]+", t)))
                import numpy as _np
                k = int(_np.clip(round(wc / 8), 2, 5))
            else:
                k = topk
            # pick top-k unique tokens
            idx = A[i].argsort()[-k:]
            toks = [vocab[j] for j in idx if A[i, j] > 0]
            # dedupe and keep order by score
            toks_scored = sorted(((tok, A[i, j]) for tok, j in zip(toks, idx[::-1])), key=lambda x: -x[1])
            seen = set()
            kws = []
            for tok, _ in toks_scored:
                if tok not in seen:
                    seen.add(tok)
                    kws.append(tok)
                if len(kws) >= k:
                    break
            # 키워드 강조 (임베딩 품질 향상용)
            if kws:
                enriched.append(t + "\n\n### KEY: " + ", ".join(f"[{kw}]" for kw in kws))
            else:
                enriched.append(t)
        return enriched
    except Exception as e:
        # TF-IDF 키워드 추출 실패 시 원본 반환 (예외 발생 방지)
        print(f"⚠️  키워드 추출 실패, 원본 사용: {e}")
        return texts


def embed_texts(texts, enrich=False, model_name="paraphrase-mpnet-base-v2"):
    """Return L2-normalized embeddings for texts.
    Priority:
      1) sentence-transformers 'paraphrase-mpnet-base-v2' (더 정확한 의미 유사성)
      2) Fallback: 'all-MiniLM-L6-v2' (속도 우선)
    Optionally enrich texts with lightweight TF-IDF keywords to better separate near-duplicates.
    """
    import numpy as _np
    from sklearn.preprocessing import normalize as _normalize
    
    global _EMBED_BACKEND
    
    if not texts:
        return _np.zeros((0, 16), dtype=_np.float32)

    # Optional enrichment to emphasize discriminative terms
    proc = keyword_enrich(texts) if enrich else texts

    # sentence-transformers 사용 (개선된 모델 우선)
    if _EMBED_BACKEND is None:
        from sentence_transformers import SentenceTransformer
        try:
            _EMBED_BACKEND = SentenceTransformer(model_name)
            print(f"✅ 임베딩 모델 로드: {model_name}")
        except Exception as e:
            print(f"⚠️  {model_name} 로드 실패, fallback to all-MiniLM-L6-v2: {e}")
            _EMBED_BACKEND = SentenceTransformer("all-MiniLM-L6-v2")
    
    X = _EMBED_BACKEND.encode(proc, show_progress_bar=False, normalize_embeddings=True)
    return X

def priority_hierarchical_dedup(specs, score_key='elo', keep_ratio=0.4, similarity_threshold=0.70):
    """우선순위 기반 지배 관계 중복 제거 (AutoPolicy 방식 참조)"""
    if not specs or len(specs) <= 3:
        return specs
    
    import numpy as np
    
    print(f"  🎯 우선순위 기반 중복 제거 시작: {len(specs)}개 spec 분석...")
    
    # 1. 품질 순으로 정렬 (높은 것부터) - 디버깅 정보 추가
    sorted_specs = sorted(specs, key=lambda s: float(s.get(score_key, 0)), reverse=True)
    
    # 디버깅: 상위 5개 spec의 점수 분포 확인
    print(f"     📊 상위 5개 점수 분포:")
    for i in range(min(5, len(sorted_specs))):
        s = sorted_specs[i]
        elo = s.get('elo', 0)
        judge = s.get('score', 0) 
        norm = s.get('score_norm', 'N/A')
        print(f"       #{i}: Elo={elo:.1f}, Judge={judge:.1f}, Norm={norm}")
    
    # 유사도 및 품질 차이 통계
    similar_pairs = 0
    quality_diff_pairs = 0
    
    # 2. 모든 텍스트에 대해 임베딩 생성 (한 번만)
    texts = [mask_pii(s.get('text', '')) for s in sorted_specs]
    embeddings = embed_texts(texts)
    
    # 3. 지배 관계 분석 및 중복 제거
    dominated = set()
    dominance_relations = []
    
    for i, spec_a in enumerate(sorted_specs):
        if i in dominated:  # 이미 지배당한 것은 건너뛰기
            continue
            
        for j, spec_b in enumerate(sorted_specs[i+1:], i+1):
            if j in dominated:  # 이미 지배당한 것은 건너뛰기
                continue
                
            # 유사도 체크
            similarity = np.dot(embeddings[i], embeddings[j])
            if similarity >= similarity_threshold:
                similar_pairs += 1
                
                # 품질 차이 체크
                elo_a, elo_b = spec_a.get('elo', 0), spec_b.get('elo', 0)
                if 'score_norm' in spec_a and 'score_norm' in spec_b:
                    qual_a, qual_b = spec_a.get('score_norm', 0), spec_b.get('score_norm', 0)
                elif abs(elo_a - elo_b) < 10:
                    qual_a, qual_b = spec_a.get('score', 0), spec_b.get('score', 0)
                else:
                    qual_a, qual_b = elo_a, elo_b
                
                if qual_a > qual_b:
                    quality_diff_pairs += 1
                
            # 지배 관계 판단
            is_dominant, dominance_score = calculate_dominance_relationship(
                spec_a, spec_b, embeddings[i], embeddings[j], similarity_threshold
            )
            
            if is_dominant:
                dominated.add(j)
                dominance_relations.append((i, j, dominance_score))
    
    # 디버깅 통계 출력
    total_pairs = len(sorted_specs) * (len(sorted_specs) - 1) // 2
    print(f"     📊 분석 통계: 총 {total_pairs}쌍 중")
    print(f"       - 유사도 {similarity_threshold:.2f} 이상: {similar_pairs}쌍 ({similar_pairs/total_pairs*100:.1f}%)")
    print(f"       - 품질 차이 있음: {quality_diff_pairs}쌍 ({quality_diff_pairs/max(similar_pairs,1)*100:.1f}%)")
    print(f"       - 지배 관계 성립: {len(dominance_relations)}쌍")
    
    # 4. 지배당하지 않은 spec들 선택
    survivors = [spec for i, spec in enumerate(sorted_specs) if i not in dominated]
    
    # 5. 지배 관계 출력 (점수 타입 정보 포함)
    if dominance_relations:
        print(f"     🎯 발견된 지배 관계: {len(dominance_relations)}개")
        for dominator, dominated_idx, score in dominance_relations[:5]:  # 상위 5개만 출력
            # 사용된 점수 타입 확인 (동일한 로직으로)
            spec_dom = sorted_specs[dominator]
            spec_sub = sorted_specs[dominated_idx]
            
            if 'score_norm' in spec_dom and 'score_norm' in spec_sub:
                score_type = "Norm"
                qual_dom = spec_dom.get('score_norm', 0)
                qual_sub = spec_sub.get('score_norm', 0)
            elif abs(spec_dom.get('elo', 0) - spec_sub.get('elo', 0)) < 10:
                score_type = "Judge"
                qual_dom = spec_dom.get('score', 0)
                qual_sub = spec_sub.get('score', 0)
            else:
                score_type = "Elo"
                qual_dom = spec_dom.get('elo', 0)
                qual_sub = spec_sub.get('elo', 0)
            
            print(f"       #{dominator} → #{dominated_idx} (지배도: {score:.2f}, {score_type}: {qual_dom:.1f}>{qual_sub:.1f})")
        if len(dominance_relations) > 5:
            print(f"       ... 외 {len(dominance_relations)-5}개")
    
    # 6. 최종 정렬 (최소 보존율 강제 로직 제거)
    survivors.sort(key=lambda s: float(s.get(score_key, 0)), reverse=True)
    
    retention_ratio = len(survivors) / len(specs) * 100
    print(f"  🎯 우선순위 기반 중복 제거: {len(specs)}개 → {len(survivors)}개 선택 (보존율: {retention_ratio:.1f}%)")
    
    return survivors


def calculate_dominance_relationship(spec_a, spec_b, emb_a, emb_b, similarity_threshold=0.70):
    """두 spec 간의 지배 관계 판단 - 도메인 무관한 간단한 방식"""
    import numpy as np
    
    # 1. 기본 조건: 의미적 유사도가 높아야 함
    similarity = np.dot(emb_a, emb_b)  # 이미 정규화된 임베딩이므로 내적이 코사인 유사도
    if similarity < similarity_threshold:
        return False, 0.0
    
    # 2. 품질 차이 확인 (A가 B보다 같거나 높아야 지배 가능)
    # 우선순위: score_norm > score > elo (정규화된 점수가 가장 공정)
    elo_a = float(spec_a.get('elo', 0))
    elo_b = float(spec_b.get('elo', 0))
    
    # 정규화된 점수가 있으면 우선 사용
    if 'score_norm' in spec_a and 'score_norm' in spec_b:
        quality_a = float(spec_a.get('score_norm', 0))
        quality_b = float(spec_b.get('score_norm', 0))
        score_type = 'norm'
    # Elo 차이가 작으면 judge 점수 사용
    elif abs(elo_a - elo_b) < 10:
        quality_a = float(spec_a.get('score', 0))
        quality_b = float(spec_b.get('score', 0))
        score_type = 'judge'
    # Elo 차이가 크면 Elo 점수 사용
    else:
        quality_a = elo_a
        quality_b = elo_b
        score_type = 'elo'
    
    quality_gap = quality_a - quality_b
    
    # 품질 차이 조건: A가 B보다 낮으면 지배 불가
    if quality_gap < 0:
        return False, 0.0
    
    # 3. 간단한 지배도 계산 (키워드 의존성 제거)
    text_a = spec_a.get('text', '')
    text_b = spec_b.get('text', '')
    
    # 길이 기반 상세도 (더 자세한 spec이 간단한 spec을 지배)
    length_a = len(text_a.strip())
    length_b = len(text_b.strip())
    length_ratio = length_a / max(length_b, 1)
    length_advantage = min(length_ratio, 2.0)  # 최대 2배까지만 인정
    
    # 품질 우위도 정규화
    if score_type == 'norm':  # 정규화된 점수 (z-score 기반)
        quality_advantage = min(quality_gap * 0.5 + 1.0, 2.0)  # 1.0을 기준점으로
    elif score_type == 'judge':  # judge 점수 (0-100)
        quality_advantage = min(quality_gap / 20.0 + 1.0, 2.0)  # 20점 차이를 1.0으로 정규화
    else:  # Elo 점수
        quality_advantage = min(quality_gap / 30.0 + 1.0, 2.0)  # 30점 차이를 1.0으로 정규화
    
    # 4. 간단한 지배도 계산 (3가지 요소만)
    dominance_factors = {
        'similarity': similarity,           # 의미적 유사도 (0.7~1.0)
        'length_advantage': length_advantage,  # 길이 우위 (1.0~2.0) 
        'quality_advantage': quality_advantage # 품질 우위 (1.0~2.0)
    }
    
    # 가중치 (의미적 유사도 중심, 품질과 길이는 보조)
    weights = {
        'similarity': 0.6,          # 의미적 유사도가 핵심 (60%)
        'length_advantage': 0.2,    # 길이 우위 (20%)
        'quality_advantage': 0.2    # 품질 우위 (20%)
    }
    
    dominance_score = sum(
        dominance_factors[key] * weights[key] 
        for key in weights
    )
    
    # 5. 지배 판단 (임계값: 0.85로 조정)
    # similarity 0.7 * 0.6 + length_adv 1.0 * 0.2 + quality_adv 1.0 * 0.2 = 0.82 (최소값)
    # similarity 1.0 * 0.6 + length_adv 2.0 * 0.2 + quality_adv 2.0 * 0.2 = 1.4 (최대값)
    is_dominant = dominance_score > 0.85
    
    return is_dominant, dominance_score


# 키워드 기반 함수들 제거됨 - 도메인 무관한 지배도 측정을 위해

def dedupe_by_embeddings_greedy_fallback(specs, score_key='elo', sim_threshold=0.92):
    """기존 greedy 방식 (클러스터링 실패 시 fallback용)"""
    if not specs:
        return []
    texts = [mask_pii(s.get('text','')) for s in specs]
    X = embed_texts(texts)
    import numpy as _np
    order = _np.argsort([-float(s.get(score_key, 0.0)) for s in specs])
    kept, used = [], _np.zeros(len(specs), dtype=bool)
    for idx in order:
        if used[idx]: continue
        kept.append(idx); used[idx]=True
        sims = X @ X[idx]
        used |= (sims >= sim_threshold)
    return [specs[i] for i in kept]




# =========================
# (개선) Random Pairwise Elo with TOP-M, 캐싱, Elo-근접, 적응형 매치
# =========================

def pick_opponent_near_elo(pool: List[Dict[str, Any]], target: Dict[str, Any], k: int = 8) -> Optional[Dict[str, Any]]:
    others = [p for p in pool if p.get('id') != target.get('id')]
    if not others:
        return None
    sorted_by_gap = sorted(others, key=lambda s: abs(s.get('elo', INITIAL_ELO) - target.get('elo', INITIAL_ELO)))
    topk = sorted_by_gap[:min(k, len(sorted_by_gap))]
    return random.choice(topk) if topk else None


def matches_for_candidate(cand: Dict[str, Any], base: int = 2, floor: int = 1, ceil: int = 4) -> int:
    g = int(cand.get('games', 0))
    # 게임이 많을수록 점점 줄이고, 최소/최대 범위 클램프
    # 예: 0~9게임: base, 10~19: base-1, 20~: base-2 ...
    adj = base - (g // 10)
    return max(floor, min(ceil, adj))


# =========================
# Dueling Bandit Pairwise Elo (UCB scheduling)
# =========================
def _ucb(score, games, total, c=300.0):
    import math
    return float(score) + c * math.sqrt(max(0.0, math.log(1.0 + total)) / (1.0 + games))

def run_pairwise_elo_dueling_bandit(evaluated_candidates: List[Dict[str, Any]], archive: Archive, client_judge,
                                    constitution: str, domain_profile: str, task_profile: str,
                                    cfg: EvolverConfig, generation: int, judges_dir: str):
    if not evaluated_candidates:
        return
    log_path = os.path.join(judges_dir, f"gen{generation:03d}_pairwise_elo.jsonl")

    for s in evaluated_candidates:
        s.setdefault('elo', cfg.elo_initial)
        s.setdefault('games', 0); s.setdefault('wins', 0); s.setdefault('losses', 0); s.setdefault('draws', 0)

    pool = sorted(evaluated_candidates, key=lambda s: s.get('elo', cfg.elo_initial), reverse=True)[:max(1, cfg.pairwise_top_m)]
    total_duels = 0
    min_games = max(1, getattr(cfg, 'db_min_games', 2))
    max_duels = max(50, getattr(cfg, 'db_max_duels', 3000))
    stop_gap = max(10, getattr(cfg, 'db_stop_gap', 50))
    c_ucb = float(getattr(cfg, 'db_ucb_c', 300.0))

    while total_duels < max_duels:
        under = [s for s in pool if int(s.get('games',0)) < min_games]
        if len(under) >= 2:
            A = max(under, key=lambda s: s.get('elo', cfg.elo_initial))
            others = [x for x in under if x['id'] != A['id']]
            if not others:
                others = [x for x in pool if x['id'] != A['id']]
            B = min(others, key=lambda s: abs(s.get('elo', cfg.elo_initial) - A.get('elo', cfg.elo_initial)))
        else:
            total = sum(int(s.get('games',0)) for s in pool) + 1
            scores = [(s, _ucb(s.get('elo', cfg.elo_initial), int(s.get('games',0)), total, c_ucb)) for s in pool]
            scores.sort(key=lambda t: t[1], reverse=True)
            A = scores[0][0]
            candidates = [t[0] for t in scores[1:8]] or [t[0] for t in scores[1:]]
            B = min(candidates, key=lambda s: abs(s.get('elo', cfg.elo_initial) - A.get('elo', cfg.elo_initial)))

        outcome_a = pairwise_ab_ba_conditional(client_judge, A['text'], B['text'],
                                               constitution, domain_profile, task_profile,
                                               use_ab_ba=cfg.use_ab_ba)
        r_a_old, r_b_old = A['elo'], B['elo']
        k_val = cfg.elo_k
        if cfg.elo_dynamic_k:
            k_val = effective_k(cfg.elo_k, r_a_old, r_b_old, A['games'], B['games'], cfg.elo_min_k)
        r_a_new, r_b_new = update_elo(r_a_old, r_b_old, outcome_a, k=k_val)
        A['elo'] = r_a_new; B['elo'] = r_b_new
        A['games'] += 1; B['games'] += 1
        if outcome_a == 1.0:
            A['wins'] += 1; B['losses'] += 1
        elif outcome_a == 0.0:
            A['losses'] += 1; B['wins'] += 1
        else:
            A['draws'] += 1; B['draws'] += 1

        total_duels += 1

        try:
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({
                    'generation': generation,
                    'timestamp': time.time(),
                    'mode': 'DuelingBandit-UCB',
                    'duel': total_duels,
                    'A': A.get('id'), 'B': B.get('id'),
                    'A_elo_before': r_a_old, 'B_elo_before': r_b_old,
                    'A_elo_after': A['elo'], 'B_elo_after': B['elo'],
                    'outcome_a': outcome_a, 'k': k_val,
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"(warn) dueling bandit log write failed: {e}")

        pool.sort(key=lambda s: s.get('elo', cfg.elo_initial), reverse=True)
        if len(pool) >= 2 and (pool[0]['elo'] - pool[1]['elo']) >= stop_gap and total_duels >= len(pool):
            break

    archive.specs.sort(key=archive._rank_key, reverse=True)



# Evolution loop
# =========================

def run_task_evolution(task_name: str, initial_specs: List[Dict[str, Any]], constitution: str, domain_profile: str, task_profile: str, cfg: EvolverConfig, base_output_dir: str = None, domain_name: str = None, domain_concepts: List[str] = None, task_concepts: List[str] = None):
    # 모델 선택
    if cfg.generator_model.startswith('gpt'):
        client_gen = OpenAIClientWrapper(api_key=cfg.openai_api_key, model=cfg.generator_model)
    else:
        client_gen = AnthropicClientWrapper(api_key=cfg.anthropic_api_key, model=cfg.generator_model)

    if cfg.judge_model.startswith('gpt'):
        client_judge = OpenAIClientWrapper(api_key=cfg.openai_api_key, model=cfg.judge_model)
    else:
        client_judge = AnthropicClientWrapper(api_key=cfg.anthropic_api_key, model=cfg.judge_model)

    judges = {
        'constitution': Judge('constitution', client_judge),
        'domain': Judge('domain', client_judge),
        'task': Judge('task', client_judge),
    }

    archive = Archive(max_capacity=100, elo_initial=cfg.elo_initial)
    random.seed(cfg.random_seed)

    # 출력 디렉터리 (base_output_dir이 전달되지 않으면 기본 설정 사용)
    if base_output_dir is None:
        base_output_dir = cfg.output_dir
    
    # 디렉터리 구조: base_output_dir/domain/task/
    if domain_name:
        # 도메인과 태스크를 분리해서 저장
        safe_domain = domain_name.replace(' & ', '_and_').replace(' ', '_')
        safe_task = task_name.replace(' & ', '_and_').replace(' ', '_')
        out_dir = os.path.join(base_output_dir, safe_domain, safe_task)
    else:
        # 기존 방식 (호환성)
        out_dir = os.path.join(base_output_dir, task_name.replace(' ', '_'))
    ensure_dir(out_dir)
    best_dir = os.path.join(out_dir, 'best')
    generator_dir = os.path.join(out_dir, 'generator')
    judges_dir = os.path.join(out_dir, 'judges')
    ensure_dir(best_dir); ensure_dir(generator_dir); ensure_dir(judges_dir)

    population = []
    for s in initial_specs:
        seed_text = s.get('text', '')
        individual_seed_specs = split_llm_response_to_specs(seed_text, 'seed')
        print(f"💡 초기 seed를 {len(individual_seed_specs)}개 개별 spec으로 분리")
        print(f"  ⚖️ 개별 spec 평가 중... (총 {len(individual_seed_specs)}개)")
        for i, individual_spec in enumerate(individual_seed_specs):
            individual_spec['id'] = make_unique_id(individual_spec['text'], generation=-1, index=i)
            individual_spec['meta'] = {'origin': 'seed_split', 'index': i}
            print(f"    평가 중: {i+1}/{len(individual_seed_specs)} - \"{individual_spec['text'][:50]}...\"")
            is_top10 = i < 10
            evaluated = evaluate_spec_with_judges(
                individual_spec, judges, cfg.judge_weights,
                constitution, domain_profile, task_profile,
                -1, 3, judges_dir if is_top10 else None, is_top10,
                initial_elo=cfg.elo_initial
            )
            archive.add(evaluated)
            population.append(evaluated)
        print(f"  ✅ 초기 평가 완료! Archive에 {len(population)}개 spec 저장됨")
        if archive.specs:
            top_elo = archive.specs[0].get('elo', cfg.elo_initial)
            bottom_elo = archive.specs[-1].get('elo', cfg.elo_initial)
            print(f"  📊 Archive 저장: {len(archive.specs)}개 (최고 Elo:{top_elo:.1f}, 최저 Elo:{bottom_elo:.1f})")

    history_path = os.path.join(out_dir, 'history.jsonl')

    with open(history_path, 'a', encoding='utf-8') as hf:
        for gen in range(cfg.generations):
            print(f"\n🚀 [{task_name}] Generation {gen + 1}/{cfg.generations}")
            print("=" * 60)

            # 부모 샘플링 후 변이 생성
            all_archive_specs = archive.all_elites()
            print(f"  🧬 변이 생성 중... (Archive {len(all_archive_specs)}개, parents {cfg.parent_selection_size}개씩 → {cfg.population_per_gen}번)")
            all_candidate_specs: List[Dict[str, Any]] = []
            for i in range(cfg.population_per_gen):
                if cfg.use_task_diversity:
                    selected_parents = archive.sample_parents_task_diverse(
                        n=cfg.parent_selection_size,
                        mix=cfg.diverse_parent_mix,
                        task_pool_size=cfg.task_pool_size,
                        metric=(cfg.parent_selection_metric or "task")
                    )
                else:
                    top_30_specs = all_archive_specs[:30]
                    selected_parents = random.sample(top_30_specs, min(cfg.parent_selection_size, len(top_30_specs))) if top_30_specs else []
                print(f"    변이 {i+1}/{cfg.population_per_gen}: 부모 {len(selected_parents)}개 조합 중...")
                child_specs = apply_variation_multi_parent(selected_parents, client_gen, constitution, domain_profile, task_profile, gen, generator_dir, domain_name, task_name, domain_concepts, task_concepts)
                # 초기 Elo 세팅
                for cs in child_specs:
                    cs.setdefault('elo', cfg.elo_initial)
                    cs.setdefault('games', 0); cs.setdefault('wins', 0); cs.setdefault('losses', 0); cs.setdefault('draws', 0)
                all_candidate_specs.extend(child_specs)
                print(f"    → {len(child_specs)}개 spec 생성됨")
            print(f"  📊 총 {len(all_candidate_specs)}개 후보 spec 생성 완료")
            
            # 포인트와이즈 평가(참고·로깅용) - 먼저 평가!
            print(f"  ⚖️ 평가 중... ({len(all_candidate_specs)}개 spec → {cfg.parallel_workers}개 병렬)")
            evaluated_candidates: List[Dict[str, Any]] = []
            completed_count = 0
            with ThreadPoolExecutor(max_workers=cfg.parallel_workers) as ex:
                futures = {}
                for i, candidate in enumerate(all_candidate_specs):
                    is_top10 = i < 10
                    future = ex.submit(
                        evaluate_spec_with_judges, candidate, judges, cfg.judge_weights,
                        constitution, domain_profile, task_profile, gen, 3, judges_dir, is_top10,
                        cfg.elo_initial
                    )
                    futures[future] = candidate
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        evaluated_candidates.append(res)
                        completed_count += 1
                        if completed_count % max(1, len(all_candidate_specs) // 4) == 0 or completed_count == len(all_candidate_specs):
                            print(f"    평가 진행: {completed_count}/{len(all_candidate_specs)} 완료 ({completed_count/len(all_candidate_specs)*100:.1f}%)")
                    except Exception as e:
                        print(f'    ⚠️ 평가 오류: {e}')
            print(f"  ✅ 평가 완료! {len(evaluated_candidates)}개 spec 평가됨")
            
            # === Judge 점수로 우선순위 기반 지배 관계 중복 제거 ===
            before = len(evaluated_candidates)
            evaluated_candidates = priority_hierarchical_dedup(
                evaluated_candidates, 
                score_key='score',  # Judge 점수 사용!
                keep_ratio=0.4,  # 40% 보존
                similarity_threshold=0.70  # 개선된 임베딩 모델에 맞는 엄격한 기준
            )

            # PATCH: 심판 점수 정규화 (아카이브+후보 풀에 적용) → 편향 완화
            if cfg.use_score_normalization:
                pool_for_norm = archive.all_elites() + evaluated_candidates
                normalize_judge_scores_for_pool(pool_for_norm)

            # === Pairwise Elo 라운드 ===
            if cfg.use_pairwise_elo:
                print(f"  🥊 Pairwise-Elo 시작... (mode=DuelingBandit, dynK={cfg.elo_dynamic_k})")
                run_pairwise_elo_dueling_bandit(
                    evaluated_candidates, archive, client_judge,
                    constitution, domain_profile, task_profile,
                    cfg, generation=gen, judges_dir=judges_dir
                )
                print(f"  ✅ Pairwise-Elo 완료")

            # 아카이브에 후보 추가 (정렬은 Elo 우선, 타이브레이크=score_norm)
            print(f"  🎯 선별/추가 중... (Elo 우선 정렬)")
            for candidate in evaluated_candidates:
                archive.add(candidate)

            elites = archive.all_elites()
            best_elo = elites[0].get('elo', cfg.elo_initial) if elites else cfg.elo_initial
            
            # Elo dispersion metrics (no averages)
            elos = [e.get('elo', cfg.elo_initial) for e in evaluated_candidates]
            import statistics as st
            elo_std = st.pstdev(elos) if len(elos) > 1 else 0.0
            elo_min = min(elos, default=cfg.elo_initial)
            elo_max = max(elos, default=cfg.elo_initial)
            elo_p90 = float(np.percentile(elos, 90)) if elos else cfg.elo_initial
            elo_p10 = float(np.percentile(elos, 10)) if elos else cfg.elo_initial
            # elos, elo_std, elo_min/max, elo_p90/p10 계산 바로 아래에 추가
            avg_elo = float(np.mean(elos)) if elos else cfg.elo_initial

            record = {
                'generation': gen,
                'best_elo': best_elo,
                'elo_std': elo_std,
                'elo_min': elo_min,
                'elo_max': elo_max,
                'elo_range': elo_max - elo_min,
                'elo_p90': elo_p90,
                'elo_p10': elo_p10,
                'archive_size': len(archive.specs),
                'timestamp': time.time(),
            }
            hf.write(json.dumps(record, ensure_ascii=False) + "\n")
            hf.flush()


            # 상위 30개 저장 (Elo 포함, 유의성 마크)
            fname = os.path.join(best_dir, f"gen{gen:03d}_top30.md")
            with open(fname, 'w', encoding='utf-8') as wf:
                wf.write(f"# Generation {gen} - Top 30 Specs\n\n")
                wf.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                wf.write(f"Archive size: {len(archive.specs)}\n")
                wf.write(f"Best Elo: {best_elo:.1f}\n")
                wf.write(f"Elo p90 / p10: {elo_p90:.1f} / {elo_p10:.1f}\n")
                wf.write(f"Elo std: {elo_std:.1f} | range: {elo_min:.1f}–{elo_max:.1f} (Δ{(elo_max-elo_min):.1f})\n\n")
                wf.write("=" * 80 + "\n\n")
                prev_elo = None
                for i, s in enumerate(elites[:30]):
                    elo = s.get('elo', cfg.elo_initial)
                    gap_flag = ""
                    if prev_elo is not None and (prev_elo - elo) < cfg.significance_gap:
                        gap_flag = " (≈tie)"  # 유의성 부족
                    prev_elo = elo
                    wf.write(f"## Rank #{i+1}{gap_flag}\n\n")
                    wf.write(f"**ID:** {s['id']}\n")
                    wf.write(f"**Elo:** {elo:.1f} (W-D-L: {s.get('wins',0)}-{s.get('draws',0)}-{s.get('losses',0)}, Games: {s.get('games',0)})\n")
                    wf.write(f"**Score(Ref):** {s.get('score',0)}/100  |  **Score(norm):** {s.get('score_norm', 0.0):+.2f}\n")
                    wf.write(f"**Breakdown:** Constitution: {s.get('scores',{}).get('constitution',0)}/40, Domain: {s.get('scores',{}).get('domain',0)}/30, Task: {s.get('scores',{}).get('task',0)}/30\n")
                    provenance = s.get('provenance', [])
                    if provenance:
                        wf.write(f"**Evolution:** {' → '.join([p['op'] for p in provenance])}\n")
                    else:
                        wf.write(f"**Evolution:** [Original Seed]\n")
                    wf.write(f"\n**Spec Text:**\n")
                    wf.write(f"```\n{strip_leading_numbering(s['text'])}\n```\n\n")
                    wf.write("-" * 60 + "\n\n")

            if domain_name:
                task_display = f"{domain_name}/{task_name}"
            else:
                task_display = task_name
            print(f" [{task_display}] Gen {gen}: archive_best_elo={best_elo:.1f} avg_elo={avg_elo:.1f} archive_size={len(archive.specs)} candidates={len(evaluated_candidates)} added={len(evaluated_candidates)} mode=DuelingBandit")

    if domain_name:
        task_display = f"{domain_name}/{task_name}"
    else:
        task_display = task_name
    print(f"[{task_display}] Evolution finished. Results in {out_dir}")
    return archive


# =========================
# IO helpers
# =========================

def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning empty string.")
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_tasks_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning empty dict.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 호환성: 기존 문자열 형식을 새 형식으로 변환
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = {"description": value, "core_concepts": []}
    return data


def load_domains_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning empty dict.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 호환성: 기존 문자열 형식을 새 형식으로 변환
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = {"description": value, "core_concepts": []}
    return data


# =========================
# Entrypoint (auto-load one task)
# =========================

def run_domain_tasks_auto_load(cfg: EvolverConfig, target_domain: str = "Legal_and_Regulatory", few_shot_folder: str = "few_shot_examples"):
    """특정 도메인의 모든 태스크에 대해 SPEC 진화를 실행"""
    constitution = load_text_file("constitution.txt") or "Basic constitution for safe AI."
    domains = load_domains_json("domains.json")
    if not domains:
        domains = {"General": {"description": "General purpose domain for testing", "core_concepts": []}}
    tasks = load_tasks_json("tasks.json")
    if not tasks:
        tasks = {"Test Task": {"description": "A minimal test task for SPEC evolution", "core_concepts": []}}

    ensure_dir(few_shot_folder)

    # 도메인 데이터 확인
    domain_name = target_domain.replace('_and_', ' & ').replace('_', ' ')
    if domain_name not in domains:
        print(f"❌ 도메인을 찾을 수 없습니다: {domain_name}")
        print(f"📋 사용 가능한 도메인들: {list(domains.keys())}")
        return

    domain_data = domains[domain_name]
    domain_desc = domain_data.get("description", str(domain_data))
    domain_concepts = domain_data.get("core_concepts", [])

    print(f"🚀 도메인 '{domain_name}'의 모든 태스크 실행 시작")
    print(f"📊 도메인 설명: {domain_desc}")

    # 해당 도메인의 few_shot 폴더에서 모든 태스크 찾기
    safe_domain = target_domain
    domain_folder = os.path.join(few_shot_folder, safe_domain)
    ensure_dir(domain_folder)

    # 해당 도메인 폴더의 모든 .txt 파일 찾기 (각각이 태스크)
    if not os.path.exists(domain_folder):
        print(f"❌ 도메인 폴더를 찾을 수 없습니다: {domain_folder}")
        return

    task_files = [f for f in os.listdir(domain_folder) if f.endswith('.txt')]
    if not task_files:
        print(f"⚠️ {domain_folder}에 태스크 파일이 없습니다.")
        return

    print(f"📋 발견된 태스크들: {task_files}")

    # 각 태스크에 대해 진화 실행
    for task_file in task_files:
        task_name = task_file.replace('.txt', '').replace('_and_', ' & ').replace('_', ' ')
        print(f"\n🎯 태스크 실행: {task_name}")

        # 태스크 데이터 확인
        task_data = tasks.get(task_name, {})
        task_desc = task_data.get("description", f"Task: {task_name}")
        task_concepts = task_data.get("core_concepts", [])

        safe_task = task_file.replace('.txt', '')
        combination_name = f"{safe_domain}__{safe_task}"

        file_name = os.path.join(domain_folder, task_file)

        task_profile = f"### Task: {task_name}\n- Description: {task_desc}\n"
        domain_profile = f"### Domain: {domain_name}\n- Description: {domain_desc}\n"

        initial_specs: List[Dict[str, Any]] = []
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as f:
                txt = f.read()
                initial_specs.append({'text': txt})
            print(f"  ✅ Found existing spec file: {file_name}")
        else:
            minimal = f"## SPEC DRAFT for {combination_name}\n\nDomain: {domain_name}\nTask: {task_name}\n\nRULES:\n- MUST: follow instructions.\n"
            initial_specs.append({'text': minimal})
            print(f"  ⚠️ No spec file found, using minimal spec: {file_name}")

        # 타임스탬프가 포함된 출력 디렉터리 생성
        if cfg.use_timestamp_suffix:
            timestamp_suffix = time.strftime("_%Y%m%d_%H%M%S")
            base_output_dir = cfg.output_dir + timestamp_suffix
        else:
            base_output_dir = cfg.output_dir

        print(f"  📁 Output directory: {base_output_dir}")
        archive = run_task_evolution(task_name, initial_specs, constitution, domain_profile, task_profile, cfg, base_output_dir, domain_name, domain_concepts, task_concepts)

        elites = archive.all_elites()
        evolved_elites = [spec for spec in elites if spec.get('provenance', [])]

        if evolved_elites:
            best_evolved_spec = evolved_elites[0]
            print(f"    🎉 Best Evolved Spec (Seed 제외)")
            print(f"       Elo: {best_evolved_spec.get('elo', cfg.elo_initial):.1f}")
            print(f"       Score: {best_evolved_spec.get('score',0)}/100")
        else:
            print("    ⚠️ No evolved specs found.")

        if elites:
            print(f"    📊 Total specs: {len(elites)}, Evolved specs: {len(evolved_elites)}")
            print(f"    🏆 Highest Elo: {elites[0].get('elo', cfg.elo_initial):.1f}")

            # 출력 디렉터리 계산 (run_task_evolution과 동일한 로직)
            safe_domain = domain_name.replace(' & ', '_and_').replace(' ', '_')
            safe_task = task_name.replace(' & ', '_and_').replace(' ', '_')
            task_out_dir = os.path.join(base_output_dir, safe_domain, safe_task)
            ensure_dir(task_out_dir)
            archive_file_json = os.path.join(task_out_dir, 'top100_archive.json')
            archive_file_txt = os.path.join(task_out_dir, 'top100_archive.txt')

            # === 최종 출력 전 우선순위 기반 중복 제거 ===
            elites = priority_hierarchical_dedup(
                elites,
                score_key='elo',
                keep_ratio=0.6,  # 최종 아카이브는 60% 보존
                similarity_threshold=0.65  # 최종에서는 약간 관대한 기준 (세대별보다 완화)
            )
            archive_data = []
            for i, spec in enumerate(elites):
                archive_data.append({
                    'rank': i + 1,
                    'id': spec['id'],
                    'elo': spec.get('elo', cfg.elo_initial),
                    'games': spec.get('games', 0),
                    'wins': spec.get('wins', 0),
                    'losses': spec.get('losses', 0),
                    'draws': spec.get('draws', 0),
                    'score': spec.get('score', 0),
                    'score_norm': spec.get('score_norm', 0.0),
                    'scores': spec.get('scores', {}),
                    'provenance': spec.get('provenance', []),
                    'text': strip_leading_numbering(spec['text']),
                    'evaluated_at': spec.get('evaluated_at', ''),
                    'is_evolved': len(spec.get('provenance', [])) > 0
                })
            with open(archive_file_json, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, ensure_ascii=False, indent=2)

            with open(archive_file_txt, 'w', encoding='utf-8') as f:
                f.write(f"=== Top {len(elites)} Archive Specs ===\n")
                f.write(f"Domain: {domain_name}\n")
                f.write(f"Task: {task_name}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total specs: {len(elites)}, Evolved specs: {len(evolved_elites)}\n\n")
                prev_elo = None
                for i, spec in enumerate(elites):
                    elo = spec.get('elo', cfg.elo_initial)
                    gap_flag = ""
                    if prev_elo is not None and (prev_elo - elo) < cfg.significance_gap:
                        gap_flag = " (≈tie)"
                    prev_elo = elo
                    f.write(f"{'='*60}\n")
                    f.write(f"RANK #{i+1}{gap_flag}\n")
                    f.write(f"ID: {spec['id']}\n")
                    f.write(f"ELO: {elo:.1f} (W-D-L: {spec.get('wins',0)}-{spec.get('draws',0)}-{spec.get('losses',0)}, Games: {spec.get('games',0)})\n")
                    f.write(f"SCORE(REF): {spec.get('score',0)}/100  |  SCORE(NORM): {spec.get('score_norm', 0.0):+.2f}\n")
                    f.write(f"  - Constitution: {spec.get('scores',{}).get('constitution',0)}/40\n")
                    f.write(f"  - Domain: {spec.get('scores',{}).get('domain',0)}/30\n")
                    f.write(f"  - Task: {spec.get('scores',{}).get('task',0)}/30\n")
                    provenance = spec.get('provenance', [])
                    if provenance:
                        f.write(f"EVOLUTION: {' → '.join([p['op'] for p in provenance])}\n")
                    else:
                        f.write(f"EVOLUTION: [Original Seed]\n")
                    f.write(f"\nSPEC TEXT:\n")
                    f.write(f"{strip_leading_numbering(spec['text'])}\n\n")
        print(f"📁 Archive 저장 완료:")
        print(f"  - JSON: {archive_file_json}")
        print(f"  - TXT: {archive_file_txt}")


def run_single_task_evolution(target_domain: str, target_task: str):
    """특정 도메인의 특정 태스크에 대해서만 단일 스펙 진화를 실행"""
    import os
    
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    config = EvolverConfig(
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        use_timestamp_suffix=False
    )

    constitution = load_text_file("constitution.txt") or "Basic constitution for safe AI."
    domains = load_domains_json("domains.json")
    if not domains:
        domains = {"General": {"description": "General purpose domain for testing", "core_concepts": []}}
    tasks = load_tasks_json("tasks.json")
    if not tasks:
        tasks = {"Test Task": {"description": "A minimal test task for SPEC evolution", "core_concepts": []}}

    few_shot_folder = "few_shot_examples"
    ensure_dir(few_shot_folder)

    # 도메인 데이터 확인
    domain_name = target_domain.replace('_and_', ' & ').replace('_', ' ')
    if domain_name not in domains:
        print(f"❌ 도메인을 찾을 수 없습니다: {domain_name}")
        print(f"📋 사용 가능한 도메인들: {list(domains.keys())}")
        return

    domain_data = domains[domain_name]
    domain_desc = domain_data.get("description", str(domain_data))
    domain_concepts = domain_data.get("core_concepts", [])

    # 태스크 데이터 확인
    task_name = target_task.replace('_and_', ' & ').replace('_', ' ')
    task_data = tasks.get(task_name, {})
    task_desc = task_data.get("description", f"Task: {task_name}")
    task_concepts = task_data.get("core_concepts", [])

    print(f"🚀 단일 태스크 스펙 진화: {domain_name} / {task_name}")

    safe_domain = target_domain
    safe_task = target_task
    combination_name = f"{safe_domain}__{safe_task}"

    domain_folder = os.path.join(few_shot_folder, safe_domain)
    ensure_dir(domain_folder)
    file_name = os.path.join(domain_folder, f"{safe_task}.txt")

    task_profile = f"### Task: {task_name}\n- Description: {task_desc}\n"
    domain_profile = f"### Domain: {domain_name}\n- Description: {domain_desc}\n"

    initial_specs = []
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            txt = f.read()
            initial_specs.append({'text': txt})
        print(f"  ✅ Found existing spec file: {file_name}")
    else:
        minimal = f"## SPEC DRAFT for {combination_name}\n\nDomain: {domain_name}\nTask: {task_name}\n\nRULES:\n- MUST: follow instructions.\n"
        initial_specs.append({'text': minimal})
        print(f"  ⚠️ No spec file found, using minimal spec: {file_name}")

    base_output_dir = config.output_dir
    print(f"  📁 Output directory: {base_output_dir}")
    
    archive = run_task_evolution(task_name, initial_specs, constitution, domain_profile, task_profile, config, base_output_dir, domain_name, domain_concepts, task_concepts)

    elites = archive.all_elites()
    evolved_elites = [spec for spec in elites if spec.get('provenance', [])]

    if evolved_elites:
        best_evolved_spec = evolved_elites[0]
        print(f"    🎉 Best Evolved Spec (Seed 제외)")
        print(f"       Elo: {best_evolved_spec.get('elo', config.elo_initial):.1f}")
        print(f"       Score: {best_evolved_spec.get('score',0)}/100")
    else:
        print("    ⚠️ No evolved specs found.")

    if elites:
        print(f"    📊 Total specs: {len(elites)}, Evolved specs: {len(evolved_elites)}")
        print(f"    🏆 Highest Elo: {elites[0].get('elo', config.elo_initial):.1f}")

        # 출력 디렉터리 계산 (run_task_evolution과 동일한 로직)
        safe_domain = domain_name.replace(' & ', '_and_').replace(' ', '_')
        safe_task = task_name.replace(' & ', '_and_').replace(' ', '_')
        task_out_dir = os.path.join(base_output_dir, safe_domain, safe_task)
        ensure_dir(task_out_dir)
        archive_file_json = os.path.join(task_out_dir, 'top100_archive.json')
        archive_file_txt = os.path.join(task_out_dir, 'top100_archive.txt')

        # === 최종 출력 전 우선순위 기반 중복 제거 ===
        elites = priority_hierarchical_dedup(
            elites,
            score_key='elo',
            keep_ratio=0.6,  # 최종 아카이브는 60% 보존
            similarity_threshold=0.65  # 최종에서는 약간 관대한 기준 (세대별보다 완화)
        )
        archive_data = []
        for i, spec in enumerate(elites):
            archive_data.append({
                'rank': i + 1,
                'id': spec['id'],
                'elo': spec.get('elo', config.elo_initial),
                'games': spec.get('games', 0),
                'wins': spec.get('wins', 0),
                'losses': spec.get('losses', 0),
                'draws': spec.get('draws', 0),
                'score': spec.get('score', 0),
                'score_norm': spec.get('score_norm', 0.0),
                'scores': spec.get('scores', {}),
                'provenance': spec.get('provenance', []),
                'text': strip_leading_numbering(spec['text']),
                'evaluated_at': spec.get('evaluated_at', ''),
                'is_evolved': len(spec.get('provenance', [])) > 0
            })
        with open(archive_file_json, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2)

        with open(archive_file_txt, 'w', encoding='utf-8') as f:
            f.write(f"=== Top {len(elites)} Archive Specs ===\n")
            f.write(f"Domain: {domain_name}\n")
            f.write(f"Task: {task_name}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total specs: {len(elites)}, Evolved specs: {len(evolved_elites)}\n\n")
            prev_elo = None
            for i, spec in enumerate(elites):
                elo = spec.get('elo', config.elo_initial)
                gap_flag = ""
                if prev_elo is not None and (prev_elo - elo) < config.significance_gap:
                    gap_flag = " (≈tie)"
                prev_elo = elo
                f.write(f"{'='*60}\n")
                f.write(f"RANK #{i+1}{gap_flag}\n")
                f.write(f"ID: {spec['id']}\n")
                f.write(f"ELO: {elo:.1f} (W-D-L: {spec.get('wins',0)}-{spec.get('draws',0)}-{spec.get('losses',0)}, Games: {spec.get('games',0)})\n")
                f.write(f"SCORE(REF): {spec.get('score',0)}/100  |  SCORE(NORM): {spec.get('score_norm', 0.0):+.2f}\n")
                f.write(f"  - Constitution: {spec.get('scores',{}).get('constitution',0)}/40\n")
                f.write(f"  - Domain: {spec.get('scores',{}).get('domain',0)}/30\n")
                f.write(f"  - Task: {spec.get('scores',{}).get('task',0)}/30\n")
                provenance = spec.get('provenance', [])
                if provenance:
                    f.write(f"EVOLUTION: {' → '.join([p['op'] for p in provenance])}\n")
                else:
                    f.write(f"EVOLUTION: [Original Seed]\n")
                f.write(f"\nSPEC TEXT:\n")
                f.write(f"{strip_leading_numbering(spec['text'])}\n\n")

        print(f"📁 Archive 저장 완료:")
        print(f"  - JSON: {archive_file_json}")
        print(f"  - TXT: {archive_file_txt}")


if __name__ == '__main__':
    import sys

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    config = EvolverConfig(
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        use_timestamp_suffix=False  # 타임스탬프 폴더명 사용 안함
    )

    if len(sys.argv) >= 3:
        # 도메인과 태스크 둘 다 지정된 경우
        target_domain = sys.argv[1]
        target_task = sys.argv[2]
        print(f"🎯 Target: {target_domain} / {target_task}")
        run_single_task_evolution(target_domain, target_task)
    elif len(sys.argv) >= 2:
        # 도메인만 지정된 경우 (기존 방식)
        target_domain = sys.argv[1]
        print(f"🎯 Target Domain: {target_domain} (모든 태스크)")
        run_domain_tasks_auto_load(config, target_domain=target_domain)
    else:
        # 아무것도 지정하지 않은 경우
        print("🎯 Default: Legal_and_Regulatory (모든 태스크)")
        run_domain_tasks_auto_load(config, target_domain="Legal_and_Regulatory")