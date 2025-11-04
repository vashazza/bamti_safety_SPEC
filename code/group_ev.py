"""
Group-based Specification Evolution Pipeline
집단 spec 평가 및 진화를 위한 새로운 파이프라인

주요 특징:
- 개별 spec이 아닌 spec 집단(그룹) 단위로 평가
- 유기적 동작성, 리스크 커버리지, 중복성, 실용성 등을 종합 평가
- 강력한 클러스터링을 통해 의미적으로 coherent한 그룹 형성
- 집단 수준의 진화 및 최적화
"""

from __future__ import annotations
import os
import json
import time
import copy
import random
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# alpha_elo.py에서 핵심 기능들 import
from alpha_elo import (
    EvolverConfig, Archive, Judge, AnthropicClientWrapper, OpenAIClientWrapper,
    load_text_prompt, get_prompt_manager, make_unique_id, ensure_dir,
    apply_variation_multi_parent, normalize_judge_scores_for_pool,

)

# RAG helpers
from rag_utils import (
    normalize_feedback_keys,
)



@dataclass
class GroupEvolverConfig(EvolverConfig):
    """집단 진화를 위한 확장된 설정"""
    
    # 통합 집단 평가 (총 100점)
    max_group_score: int = 100
    
    # 집단 크기 설정
    min_group_size: int = 10
    max_group_size: int = 20
    target_group_size: int = 5
    
    
    # 집단 진화 설정
    group_crossover_rate: float = 0.3  # 그룹간 교배 확률
    group_mutation_rate: float = 0.4   # 그룹 내 변이 확률

    # RAG 옵션
    use_rag: bool = True
    rag_top_k: int = 5


class UnifiedGroupJudge:
    """통합 집단 spec 평가를 위한 Judge 클래스"""
    
    def __init__(self, client, max_points: int = 100):
        self.client = client
        self.max_points = max_points
        
        # 통합 프롬프트 로드 (조용히)
        prompt_file = "prompts/unified_group_judge_prompt.txt"
        self.prompt_template = load_text_prompt(prompt_file)

        if not self.prompt_template:
            raise FileNotFoundError(f"Unified Group Judge 프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
    
    def score_group(self, spec_group: List[Dict[str, Any]], domain_profile: str,
                   task_profile: str, max_tokens: int = 800) -> Tuple[Dict[str, int], int, str, str, Dict[str, str]]:
        """집단 spec들을 통합 평가하여 세부 점수와 총점 반환"""
        
        # 집단을 하나의 텍스트로 포맷팅
        group_text = self._format_spec_group(spec_group)
        
        prompt = self.prompt_template.format(
            domain_profile=domain_profile,
            task_profile=task_profile,
            spec_group=group_text
        )
        
        try:
            raw = self.client.generate(prompt, max_tokens=max_tokens, temperature=0.1)

            # 세부 점수들과 comment 파싱
            scores = {}
            comments = {}
            total = None

            lines = raw.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # 점수 파싱
                if line.startswith('COHESION_SCORE='):
                    try:
                        scores['cohesion'] = int(line.split('=')[1].strip())
                        # 다음 줄에 comment가 있는지 확인
                        if i + 1 < len(lines) and 'Comment:' in lines[i + 1]:
                            comments['cohesion'] = lines[i + 1].split('Comment:', 1)[1].strip()
                            i += 1  # comment 줄도 건너뜀
                    except:
                        scores['cohesion'] = 0
                elif line.startswith('COVERAGE_SCORE='):
                    try:
                        scores['coverage'] = int(line.split('=')[1].strip())
                        if i + 1 < len(lines) and 'Comment:' in lines[i + 1]:
                            comments['coverage'] = lines[i + 1].split('Comment:', 1)[1].strip()
                            i += 1
                    except:
                        scores['coverage'] = 0
                elif line.startswith('REDUNDANCY_SCORE='):
                    try:
                        scores['redundancy'] = int(line.split('=')[1].strip())
                        if i + 1 < len(lines) and 'Comment:' in lines[i + 1]:
                            comments['redundancy'] = lines[i + 1].split('Comment:', 1)[1].strip()
                            i += 1
                    except:
                        scores['redundancy'] = 0
                elif line.startswith('PRACTICALITY_SCORE='):
                    try:
                        scores['practicality'] = int(line.split('=')[1].strip())
                        if i + 1 < len(lines) and 'Comment:' in lines[i + 1]:
                            comments['practicality'] = lines[i + 1].split('Comment:', 1)[1].strip()
                            i += 1
                    except:
                        scores['practicality'] = 0
                elif line.startswith('TOTAL='):
                    try:
                        total = int(line.split('=')[1].strip())
                    except:
                        pass

                i += 1
            
            # 총점이 없으면 세부 점수 합계로 계산
            if total is None:
                total = sum(scores.values())
            
            # 점수 범위 검증
            scores['cohesion'] = max(0, min(30, scores.get('cohesion', 0)))
            scores['coverage'] = max(0, min(25, scores.get('coverage', 0)))
            scores['redundancy'] = max(0, min(25, scores.get('redundancy', 0)))
            scores['practicality'] = max(0, min(20, scores.get('practicality', 0)))
            total = max(0, min(self.max_points, total))
            
            return scores, total, prompt, raw, comments
            
        except Exception as e:
            print(f"  ⚠️ 통합 그룹 평가 실패: {e}")
            return {'cohesion': 0, 'coverage': 0, 'redundancy': 0, 'practicality': 0}, 0, "", "", {}
    
    def _format_spec_group(self, spec_group: List[Dict[str, Any]]) -> str:
        """집단 spec들을 평가용 텍스트로 포맷팅 - 평가 품질 향상"""
        formatted = []
        for i, spec in enumerate(spec_group, 1):
            spec_text = spec.get('text', '').strip()
            spec_id = spec.get('id', f'spec_{i}')

            score = spec.get('score', 'N/A')

            # 평가 정보 포함
            formatted.append(f"[SPEC {i}] (ID: {spec_id}, Score: {score})\n{spec_text}\n")

        # 그룹 통계 추가
        total_specs = len(spec_group)

        avg_score = sum(float(s.get('score', 0)) for s in spec_group) / total_specs if spec_group else 0

        header = f"GROUP OVERVIEW: {total_specs} specs, Avg Score: {avg_score:.1f}\n"
        header += "="*80 + "\n"

        return header + "\n".join(formatted) + "\n" + "="*80

    def _analyze_group_feedback(self, comments: Dict[str, str]) -> Dict[str, str]:
        """원본 judge 코멘트들을 그대로 반환"""
        if not comments:
            return {
                'cohesion': 'No cohesion feedback available',
                'coverage': 'No coverage feedback available',
                'redundancy': 'No redundancy feedback available',
                'practicality': 'No practicality feedback available'
            }

        return {
            'cohesion': comments.get('cohesion', 'No cohesion feedback'),
            'coverage': comments.get('coverage', 'No coverage feedback'),
            'redundancy': comments.get('redundancy', 'No redundancy feedback'),
            'practicality': comments.get('practicality', 'No practicality feedback')
        }



    def _format_examples_with_feedback(self, groups: List[Dict[str, Any]], tag: str) -> str:
        """그룹들을 평가 피드백과 함께 포맷팅"""
        if not groups:
            return f"\n[{tag}] (none)\n"

        formatted = []
        for i, group in enumerate(groups[:3], 1):  # 최대 3개 그룹만
            specs = group.get('specs', [])
            group_score = group.get('group_score', 0)
            comments = group.get('group_comments', {})

            # 원본 코멘트 사용
            feedback = self._analyze_group_feedback(comments)

            formatted.append(f"[{tag}] Group {i} (Score: {group_score}/100)")
            formatted.append(f"💬 Cohesion: {feedback['cohesion'][:100]}{'...' if len(feedback['cohesion']) > 100 else ''}")
            formatted.append(f"💬 Coverage: {feedback['coverage'][:100]}{'...' if len(feedback['coverage']) > 100 else ''}")
            formatted.append("")

            # 대표 spec들 (최대 2개)
            for j, spec in enumerate(specs[:2], 1):
                spec_text = spec.get('text', '').strip()
                spec_score = spec.get('score', 'N/A')
                formatted.append(f"  • Spec {j} (Score: {spec_score}): {spec_text[:120]}...")

            formatted.append("")

        return "\n".join(formatted)


class GroupArchive:
    """집단 spec들을 관리하는 Archive"""
    
    def __init__(self, max_capacity: int = 50):
        self.max_capacity = max_capacity
        self.groups: List[Dict[str, Any]] = []  # 각 그룹은 specs 리스트와 메타데이터 포함
    
    def add_group(self, group: Dict[str, Any]):
        """새로운 집단을 추가"""
        group.setdefault('id', make_unique_id(str(group.get('specs', []))))
        
        # 기존 그룹과 ID 중복 확인
        existing_ids = {g.get('id') for g in self.groups}
        if group['id'] in existing_ids:
            for i, g in enumerate(self.groups):
                if g['id'] == group['id']:
                    # 기존 그룹 업데이트
                    self.groups[i] = group
                    break
        else:
            self.groups.append(group)
        
        # 그룹 점수 우선 정렬
        self.groups.sort(key=lambda x: x.get('group_score', 0), reverse=True)
        
        # 용량 초과 시 제거
        if len(self.groups) > self.max_capacity:
            self.groups = self.groups[:self.max_capacity]
    
    def sample_parent_groups(self, n: int) -> List[Dict[str, Any]]:
        """부모 집단 샘플링"""
        if not self.groups:
            return []
        
        # 상위 30개 중에서 중복 허용 샘플링
        top_groups = self.groups[:min(30, len(self.groups))]
        return random.choices(top_groups, k=min(n, len(top_groups)))
    
    def all_groups(self) -> List[Dict[str, Any]]:
        """모든 집단 반환"""
        return self.groups.copy()


def create_spec_groups_from_clustering(specs: List[Dict[str, Any]], 
                                     cfg: GroupEvolverConfig) -> List[List[Dict[str, Any]]]:
    """클러스터링을 통해 의미적으로 coherent한 spec 그룹들을 생성"""
    
    if len(specs) < cfg.min_group_size:
        return [specs]  # 너무 적으면 하나의 그룹으로
    
    try:
        # 중복 허용 랜덤 샘플링으로 그룹 생성 (15-20개씩)
        import random

        # 그룹 크기 설정 (15-20개씩)
        group_size = random.randint(15, 20)

        # 그룹 개수 계산 (specs 길이 기반)
        num_groups = max(1, len(specs) // group_size)
        if len(specs) % group_size > 0:
            num_groups += 1

        groups = []
        cluster_descriptions = []

        for i in range(num_groups):
            # 중복 허용하여 랜덤 선택 (복원 추출)
            group_specs = random.choices(specs, k=min(group_size, len(specs) * 2))  # 최대 2배까지 허용
            
            if group_specs:  # 빈 그룹이 아니면
                groups.append(group_specs)
                cluster_descriptions.append(f"랜덤 그룹 {i+1} (크기: {len(group_specs)}, 중복허용)")

        print(f"  🎲 중복 허용 랜덤 샘플링으로 {len(groups)}개 그룹 생성 (평균 크기: {sum(len(g) for g in groups)/len(groups):.1f})")
        return groups
        
    except Exception as e:
        print(f"  ⚠️ 클러스터링 기반 그룹 생성 실패: {e}")
        # Fallback: 순차적 그룹 분할
        groups = []
        for i in range(0, len(specs), cfg.target_group_size):
            group = specs[i:i + cfg.target_group_size]
            if len(group) >= cfg.min_group_size:
                groups.append(group)
        return groups


def evaluate_spec_group(group_specs: List[Dict[str, Any]],
                       unified_judge: UnifiedGroupJudge,
                       domain_profile: str,
                       task_profile: str,
                       generation: int = 0,
                       judges_log_dir: str = None) -> Dict[str, Any]:
    """통합 judge로 집단 spec들을 평가"""
    
    if not group_specs:
        return {
            'specs': group_specs,
            'group_scores': {'cohesion': 0, 'coverage': 0, 'redundancy': 0, 'practicality': 0},
            'group_score': 0,
            'evaluated_at': time.time(),


            'id': make_unique_id('empty_group')
        }
    
    try:
        # 통합 평가 수행
        group_scores, total_score, prompt, raw_response, comments = unified_judge.score_group(
            group_specs, domain_profile, task_profile
        )
        
        # 로그 저장
        if judges_log_dir:
            timestamp = int(time.time() * 1000)
            group_id = make_unique_id(str([s.get('id') for s in group_specs]))
            log_file = os.path.join(judges_log_dir, f"gen{generation:03d}_group_{group_id}_unified.json")
            
            log_data = {
                'generation': generation,
                'timestamp': timestamp,
                'group_id': group_id,
                'group_size': len(group_specs),
                'spec_ids': [s.get('id') for s in group_specs],
                'unified_judge': {
                    'prompt': prompt,
                    'raw_response': raw_response,
                    'parsed_scores': group_scores,
                    'comments': comments,
                    'total_score': total_score,
                    'temperature': 0.1,
                    'max_tokens': 800
                },
                'final_group_scores': group_scores,
                'total_group_score': total_score
            }
            
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Unified group judge 응답 저장 실패 {log_file}: {e}")
        
        return {
            'specs': group_specs,
            'group_scores': group_scores,
            'group_score': total_score,
            'group_comments': comments,
            'evaluated_at': time.time(),


            'id': make_unique_id(str([s.get('id') for s in group_specs]))
        }
        
    except Exception as e:
        print(f"  ⚠️ 통합 그룹 평가 실패: {e}")
        return {
            'specs': group_specs,
            'group_scores': {'cohesion': 0, 'coverage': 0, 'redundancy': 0, 'practicality': 0},
            'group_score': 0,
            'group_comments': {},
            'evaluated_at': time.time(),


            'id': make_unique_id(str([s.get('id') for s in group_specs]))
        }


def group_crossover(parent_groups: List[Dict[str, Any]],
                   cfg: GroupEvolverConfig) -> List[Dict[str, Any]]:
    """집단 간 교배 (spec 교환) - 전략적 개선"""

    if len(parent_groups) < 2:
        return parent_groups

    offspring_groups = []

    for i in range(0, len(parent_groups), 2):
        parent1 = parent_groups[i]
        parent2 = parent_groups[i + 1] if i + 1 < len(parent_groups) else parent_groups[0]

        if random.random() < cfg.group_crossover_rate:
            # 전략적 교배 수행
            specs1 = parent1['specs'].copy()
            specs2 = parent2['specs'].copy()

            # 1. 각 그룹의 강점/약점 분석
            score1 = parent1.get('group_score', 0)
            score2 = parent2.get('group_score', 0)

            # 2. 교환할 spec 개수 결정 (더 낮은 점수의 그룹이 더 많이 교환)
            base_exchange = min(len(specs1), len(specs2)) // 4
            score_diff = abs(score1 - score2)
            exchange_bonus = int(score_diff / 20)  # 점수 차이가 클수록 더 많이 교환
            exchange_count = min(base_exchange + exchange_bonus, min(len(specs1), len(specs2)) // 2)

            if exchange_count > 0:
                # 3. 전략적 교환: 낮은 점수의 그룹에서 더 좋은 spec을 가져옴
                if score1 < score2:
                    # parent1이 낮은 점수이므로 parent2에서 좋은 spec 가져오기
                    specs2_sorted = sorted(specs2, key=lambda s: float(s.get('score', 0)), reverse=True)
                    to_remove2 = specs2_sorted[:exchange_count]  # parent2의 최고 spec들

                    specs1_sorted = sorted(specs1, key=lambda s: float(s.get('score', 0)))
                    to_remove1 = specs1_sorted[:exchange_count]  # parent1의 최저 spec들
                else:
                    # parent2가 낮은 점수이므로 parent1에서 좋은 spec 가져오기
                    specs1_sorted = sorted(specs1, key=lambda s: float(s.get('score', 0)), reverse=True)
                    to_remove1 = specs1_sorted[:exchange_count]  # parent1의 최고 spec들

                    specs2_sorted = sorted(specs2, key=lambda s: float(s.get('score', 0)))
                    to_remove2 = specs2_sorted[:exchange_count]  # parent2의 최저 spec들

                # 4. 교환 수행
                new_specs1 = [s for s in specs1 if s not in to_remove1] + to_remove2
                new_specs2 = [s for s in specs2 if s not in to_remove2] + to_remove1

                # 5. 크기 조정
                new_specs1 = new_specs1[:cfg.max_group_size]
                new_specs2 = new_specs2[:cfg.max_group_size]

                offspring_groups.append({
                    'specs': new_specs1,
                    'meta': {'origin': 'crossover', 'strategy': 'selective_exchange',
                            'parents': [parent1.get('id'), parent2.get('id')]}
                })
                offspring_groups.append({
                    'specs': new_specs2,
                    'meta': {'origin': 'crossover', 'strategy': 'selective_exchange',
                            'parents': [parent1.get('id'), parent2.get('id')]}
                })
            else:
                offspring_groups.extend([parent1, parent2])
        else:
            offspring_groups.extend([parent1, parent2])

    return offspring_groups


def deduplicate_specs_with_llm(specs: List[Dict[str, Any]], client, domain_profile: str, task_profile: str) -> List[Dict[str, Any]]:
    """
    LLM을 사용하여 중복된 spec들을 제거
    의미적으로 유사하거나 중복되는 spec을 통합
    """
    if not specs:
        return []
    
    # Spec 텍스트만 추출
    spec_texts = [spec.get('text', '') for spec in specs]
    
    # LLM에게 중복 제거 요청
    dedup_prompt = f"""You are a specification deduplication expert.

[Context]
{domain_profile}
{task_profile}

[Task]
Remove duplicate or highly redundant specifications from the following list.
Keep only unique, non-overlapping specifications.

Guidelines:
1. Remove specs that are semantically identical (even if worded differently)
2. If two specs cover 80%+ of the same content, keep only the more comprehensive one
3. Keep specs that address different aspects or add unique value
4. Maintain RFC2119 keywords (MUST/SHOULD/MAY/MUST NOT/SHOULD NOT)
5. Preserve the exact wording of kept specs (do not rewrite)

[Input Specifications]
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(spec_texts)])}

[Output Format]
Return ONLY the numbers of specifications to KEEP, separated by commas.
Example: 1,3,5,7,9,12,15

Numbers to keep:"""

    try:
        response = client.generate(dedup_prompt, max_tokens=500, temperature=0.1)
        
        # 응답 파싱: 숫자들만 추출
        import re
        numbers = re.findall(r'\d+', response)
        keep_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(specs)]  # 1-based to 0-based
        
        # 유효성 검사
        if not keep_indices:
            print("  ⚠️ LLM 중복 제거 실패, 모든 spec 유지")
            return specs
        
        # 선택된 spec만 반환
        deduplicated = [specs[i] for i in keep_indices if i < len(specs)]
        
        print(f"  🔍 중복 제거: {len(specs)}개 → {len(deduplicated)}개")
        return deduplicated
        
    except Exception as e:
        print(f"  ⚠️ 중복 제거 중 오류: {e}, 원본 반환")
        return specs


def extract_forbidden_topics(specs: List[Dict[str, Any]], top_n: int = 8) -> str:
    """
    기존 spec들에서 자주 나오는 주제/키워드를 추출하여 금지 목록 생성
    도메인 무관하게 작동하는 자동화 함수
    """
    from collections import Counter
    import re
    
    # 모든 spec 텍스트 수집
    all_text = " ".join([spec.get('text', '') for spec in specs])
    
    # 주요 명사구/개념 추출 (대문자로 시작하는 단어, 2-4 단어 연속)
    # 예: "pharmaceutical compound synthesis", "DEA numbers", "prompt injection"
    patterns = [
        r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,3}\b',  # 대문자 시작 구문
        r'\b(?:MUST|SHOULD|MAY)\s+(?:NOT\s+)?[a-z]+\s+[a-z]+(?:\s+[a-z]+){0,3}',  # 규칙 패턴
    ]
    
    phrases = []
    for pattern in patterns:
        phrases.extend(re.findall(pattern, all_text))
    
    # 빈도 계산
    phrase_counts = Counter(phrases)
    
    # 상위 N개 선택
    top_phrases = [phrase for phrase, count in phrase_counts.most_common(top_n) if count >= 2]
    
    # 추가: 자주 나오는 단일 키워드 추출 (4글자 이상, 일반적이지 않은 단어)
    words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
    common_words = {'must', 'should', 'generate', 'detect', 'include', 'provide', 'ensure', 'verify', 'maintain', 'with', 'from', 'that', 'this', 'when', 'while', 'before', 'after'}
    word_counts = Counter([w for w in words if w not in common_words])
    top_words = [word for word, count in word_counts.most_common(top_n) if count >= 3]
    
    # 포맷팅
    forbidden_list = []
    
    if top_phrases:
        forbidden_list.append("📌 Overused phrases/concepts:")
        for phrase in top_phrases[:5]:
            forbidden_list.append(f"  - {phrase}")
    
    if top_words:
        forbidden_list.append("📌 Overused keywords:")
        forbidden_list.append(f"  - {', '.join(top_words[:10])}")
    
    if not forbidden_list:
        return "None identified - encourage diverse coverage."
    
    return "\n".join(forbidden_list)


def group_mutation(groups: List[Dict[str, Any]],
                  generator,
                  unified_judge,
                  best_groups_history: List[Dict[str, Any]],
                  worst_groups_history: List[Dict[str, Any]],
                  constitution: str,
                  domain_profile: str,
                  task_profile: str,
                  cfg: GroupEvolverConfig,
                  generation: int = 0,
                  generator_log_dir: str = None,
                  domain_name: str = None,
                  task_name: str = None,
                  single_spec_pool: Optional[List[Dict[str, Any]]] = None,
                  rag_log_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """집단 내 변이 (새로운 spec 생성 또는 기존 spec 개선) - 전략적 개선"""

    mutated_groups = []

    for group in groups:
        if random.random() < cfg.group_mutation_rate:
            # 변이 수행
            specs = group['specs'].copy()
            group_score = group.get('group_score', 0)

            # 평가 피드백 기반 good/bad examples 활용
            if hasattr(unified_judge, '_format_examples_with_feedback'):
                good_examples_text = unified_judge._format_examples_with_feedback(best_groups_history, "SUCCESSFUL")
                bad_examples_text = unified_judge._format_examples_with_feedback(worst_groups_history, "IMPROVEMENT_NEEDED")
            else:
                good_examples_text = "Successful groups: High cohesion and coverage"
                bad_examples_text = "Areas for improvement: Better redundancy and practicality"

            # 전략적 변이 선택
            if len(specs) < cfg.max_group_size and (group_score < 70 or random.random() < 0.7):
                # 1. 그룹 점수가 낮거나 확률적으로 새 spec 추가 (그룹 다양성 증대)
                try:
                    # 그룹 내 최고 성능 spec들을 부모로 선택
                    parent_specs = sorted(specs, key=lambda s: float(s.get('score', 0)), reverse=True)[:3]
                    if len(parent_specs) < 2:
                        parent_specs = specs[:min(3, len(specs))]

                    # 그룹 컨텍스트 정보 + 평가 피드백 활용
                    if hasattr(unified_judge, '_analyze_group_feedback'):
                        feedback = unified_judge._analyze_group_feedback(group.get('group_comments', {}))
                    else:
                        feedback = {'cohesion': 'Good group structure', 'coverage': 'Good coverage', 'redundancy': 'Balanced redundancy', 'practicality': 'Practical implementation'}
                    group_context = f"Group Context: This is part of a {len(specs)}-spec group. " \
                                  f"Average group score: {group_score:.1f}/100. " \
                                  f"Cohesion: {feedback['cohesion']}. " \
                                  f"Coverage: {feedback['coverage']}. " \
                                  f"Redundancy: {feedback['redundancy']}. " \
                                  f"Practicality: {feedback['practicality']}."

                    # 평가 피드백 기반 examples 추가 + RAG 보강
                    learning_context = f"\n\nLEARNING FROM PAST GROUPS:\n{good_examples_text}\n{bad_examples_text}"

                    # 간단한 피드백 직접 전달 (RAG 대신)
                    if getattr(cfg, 'use_rag', False) and single_spec_pool:
                        try:
                            # 피드백을 직접 컨텍스트에 추가
                            feedback_text = "\n".join([
                                f"• Cohesion: {feedback.get('cohesion', 'N/A')}",
                                f"• Coverage: {feedback.get('coverage', 'N/A')}",
                                f"• Redundancy: {feedback.get('redundancy', 'N/A')}",
                                f"• Practicality: {feedback.get('practicality', 'N/A')}"
                            ])

                            # 가장 유사한 기존 spec들 몇 개만 골라서 컨텍스트 추가
                            similar_specs = []
                            if single_spec_pool:
                                try:
                                    # 간단한 유사도 계산: 피드백 키워드와 spec 텍스트 간 토큰 겹침
                                    feedback_keywords = set()
                                    for comment in feedback.values():
                                        if isinstance(comment, str) and comment and comment != 'N/A':
                                            words = [w.lower() for w in comment.split() if len(w) > 3]
                                            feedback_keywords.update(words)

                                    spec_scores = []
                                    for spec in single_spec_pool:
                                        if isinstance(spec, dict):
                                            spec_text = spec.get('text', '').lower()
                                            score = sum(1 for word in feedback_keywords if word in spec_text)
                                            spec_scores.append((score, spec))

                                    # 상위 3개 선택 (안전하게)
                                    if spec_scores:
                                        spec_scores.sort(key=lambda x: x[0], reverse=True)
                                        similar_specs = [spec for score, spec in spec_scores[:3] if score > 0]

                                except Exception as sort_error:
                                    print(f"  ⚠️ 유사도 계산 중 정렬 오류: {sort_error}")
                                    # fallback: 랜덤하게 3개 선택
                                    similar_specs = single_spec_pool[:3]

                            # 로그 저장 (간단 버전)
                            if rag_log_dir:
                                os.makedirs(rag_log_dir, exist_ok=True)
                                rag_record = {
                                    'generation': generation,
                                    'group_id': group.get('id'),
                                    'strategy': 'add_new_spec',
                                    'feedback': feedback,
                                    'similar_specs_count': len(similar_specs),
                                    'selected_specs': [
                                        {
                                            'id': s.get('id'),
                                            'score': s.get('score', 0),
                                            'elo': s.get('elo'),
                                            'text': s.get('text', '')[:200]
                                        } for s in similar_specs
                                    ]
                                }
                                fname = os.path.join(
                                    rag_log_dir, f"gen{generation:03d}_{group.get('id','group')}_rag.json"
                                )
                                with open(fname, 'w', encoding='utf-8') as rf:
                                    json.dump(rag_record, rf, ensure_ascii=False, indent=2)

                            # 프롬프트 컨텍스트에 피드백과 유사 spec들 추가
                            if similar_specs:
                                feedback_block = f"\nCURRENT GROUP FEEDBACK:\n{feedback_text}"
                                refs_lines = []
                                for idx, s in enumerate(similar_specs, 1):
                                    snippet = s.get('text', '').strip().replace('\n', ' ')
                                    if len(snippet) > 100:
                                        snippet = snippet[:100] + '...'
                                    refs_lines.append(f"  - [{idx}] (Score: {s.get('score','N/A')}) {snippet}")

                                refs_block = "\nSIMILAR EXISTING SPECS:\n" + "\n".join(refs_lines)
                                learning_context += f"\n\n{feedback_block}{refs_block}"

                        except Exception as _e:
                            print(f"  ⚠️ 피드백 컨텍스트 추가 중 오류: {_e}")

                    # 🔥 다양성 강제: 현재 그룹에서 자주 나온 주제 추출
                    forbidden_topics_str = extract_forbidden_topics(specs, top_n=8)
                    
                    new_specs = apply_variation_multi_parent(
                        parent_specs,
                        generator, constitution, domain_profile,
                        task_profile + "\n\n" + group_context + learning_context,
                        generation, generator_log_dir, domain_name, task_name,
                        forbidden_topics=forbidden_topics_str
                    )
                    if new_specs:
                        # 생성된 spec 중 최고 성능 하나만 추가
                        best_new_spec = max(new_specs, key=lambda s: float(s.get('score', 0)))
                        specs.append(best_new_spec)
                        print(f"  ➕ 그룹 변이: 새 spec 추가 (점수: {best_new_spec.get('score', 0)})")
                except Exception as e:
                    print(f"  ⚠️ 그룹 변이 중 새 spec 생성 실패: {e}")

            elif specs and (group_score > 80 or random.random() < 0.5):
                # 2. 그룹 점수가 높으면 기존 spec 개선 (그룹 품질 향상)
                try:
                    # 그룹 내 최저 성능 spec을 개선 대상으로 선택
                    target_spec = min(specs, key=lambda s: float(s.get('score', 0)))
                    other_specs = [s for s in specs if s != target_spec]

                    # 개선을 위한 부모 선택: target + 그룹 내 최고 성능 spec들
                    improvement_parents = [target_spec]
                    if other_specs:
                        best_others = sorted(other_specs, key=lambda s: float(s.get('score', 0)), reverse=True)
                        improvement_parents.extend(best_others[:2])

                    # 그룹 컨텍스트 정보 + 평가 피드백 활용
                    if hasattr(unified_judge, '_analyze_group_feedback'):
                        feedback = unified_judge._analyze_group_feedback(group.get('group_comments', {}))
                    else:
                        feedback = {'cohesion': 'Good group structure', 'coverage': 'Good coverage', 'redundancy': 'Balanced redundancy', 'practicality': 'Practical implementation'}
                    improvement_context = f"Group Context: Improving a spec in a {len(specs)}-spec group. " \
                                        f"Average group score: {group_score:.1f}/100. " \
                                        f"Cohesion: {feedback['cohesion']}. " \
                                        f"Coverage: {feedback['coverage']}. " \
                                        f"Target Spec: {target_spec.get('text', '')[:100]}... " \
                                        f"Focus on enhancing this spec while maintaining group cohesion."

                    # 평가 피드백 기반 examples 추가 + RAG 보강
                    learning_context = f"\n\nLEARNING FROM PAST GROUPS:\n{good_examples_text}\n{bad_examples_text}"

                    if getattr(cfg, 'use_rag', False) and single_spec_pool:
                        try:
                            # 피드백을 직접 컨텍스트에 추가
                            feedback_text = "\n".join([
                                f"• Cohesion: {feedback.get('cohesion', 'N/A')}",
                                f"• Coverage: {feedback.get('coverage', 'N/A')}",
                                f"• Redundancy: {feedback.get('redundancy', 'N/A')}",
                                f"• Practicality: {feedback.get('practicality', 'N/A')}"
                            ])

                            # 개선 대상 spec과 유사한 기존 spec들 찾아서 컨텍스트 추가
                            similar_specs = []
                            if single_spec_pool:
                                try:
                                    target_text = target_spec.get('text', '').lower()
                                    # target spec의 키워드 추출
                                    target_words = [w.lower() for w in target_text.split() if len(w) > 3]

                                    # 피드백 키워드 추출
                                    feedback_keywords = set()
                                    for comment in feedback.values():
                                        if isinstance(comment, str) and comment and comment != 'N/A':
                                            words = [w.lower() for w in comment.split() if len(w) > 3]
                                            feedback_keywords.update(words)

                                    spec_scores = []
                                    for spec in single_spec_pool:
                                        if isinstance(spec, dict):
                                            spec_text = spec.get('text', '').lower()
                                            # target spec과의 유사도 + 피드백 키워드 유사도
                                            target_score = sum(1 for word in target_words if word in spec_text)
                                            feedback_score = sum(1 for word in feedback_keywords if word in spec_text)
                                            total_score = target_score + feedback_score
                                            spec_scores.append((total_score, spec))

                                    # 상위 3개 선택 (안전하게)
                                    if spec_scores:
                                        spec_scores.sort(key=lambda x: x[0], reverse=True)
                                        similar_specs = [spec for score, spec in spec_scores[:3] if score > 0]

                                except Exception as sort_error:
                                    print(f"  ⚠️ 유사도 계산 중 정렬 오류: {sort_error}")
                                    # fallback: 랜덤하게 3개 선택
                                    similar_specs = single_spec_pool[:3]

                            # 로그 저장 (간단 버전)
                            if rag_log_dir:
                                os.makedirs(rag_log_dir, exist_ok=True)
                                rag_record = {
                                    'generation': generation,
                                    'group_id': group.get('id'),
                                    'strategy': 'improve_spec',
                                    'target_spec_id': target_spec.get('id'),
                                    'feedback': feedback,
                                    'similar_specs_count': len(similar_specs),
                                    'selected_specs': [
                                        {
                                            'id': s.get('id'),
                                            'score': s.get('score', 0),
                                            'elo': s.get('elo'),
                                            'text': s.get('text', '')[:200]
                                        } for s in similar_specs
                                    ]
                                }
                                fname = os.path.join(
                                    rag_log_dir, f"gen{generation:03d}_{group.get('id','group')}_improve_rag.json"
                                )
                                with open(fname, 'w', encoding='utf-8') as rf:
                                    json.dump(rag_record, rf, ensure_ascii=False, indent=2)

                            # 프롬프트 컨텍스트에 피드백과 유사 spec들 추가
                            if similar_specs:
                                feedback_block = f"\nCURRENT GROUP FEEDBACK:\n{feedback_text}"
                                refs_lines = []
                                for idx, s in enumerate(similar_specs, 1):
                                    snippet = s.get('text', '').strip().replace('\n', ' ')
                                    if len(snippet) > 100:
                                        snippet = snippet[:100] + '...'
                                    refs_lines.append(f"  - [{idx}] (Score: {s.get('score','N/A')}) {snippet}")

                                refs_block = "\nSIMILAR EXISTING SPECS:\n" + "\n".join(refs_lines)
                                learning_context += f"\n\n{feedback_block}{refs_block}"

                        except Exception as _e:
                            print(f"  ⚠️ 피드백 컨텍스트 추가 중 오류: {_e}")

                    # 🔥 다양성 강제: 현재 그룹에서 자주 나온 주제 추출
                    forbidden_topics_str = extract_forbidden_topics(specs, top_n=8)
                    
                    improved_specs = apply_variation_multi_parent(
                        improvement_parents,
                        generator, constitution, domain_profile,
                        task_profile + "\n\n" + improvement_context + learning_context,
                        generation, generator_log_dir, domain_name, task_name,
                        forbidden_topics=forbidden_topics_str
                    )

                    if improved_specs:
                        # 개선된 spec 중 최고 성능으로 교체
                        best_improved = max(improved_specs, key=lambda s: float(s.get('score', 0)))
                        specs = [s if s != target_spec else best_improved for s in specs]
                        print(f"  🔄 그룹 변이: spec 개선 ({target_spec.get('score', 0)} → {best_improved.get('score', 0)})")
                except Exception as e:
                    print(f"  ⚠️ 그룹 변이 중 spec 개선 실패: {e}")

            # 크기 조정
            specs = specs[:cfg.max_group_size]

            mutated_groups.append({
                'specs': specs,
                'meta': {'origin': 'mutation', 'strategy': 'adaptive_improvement',
                        'parent': group.get('id'), 'parent_score': group_score}
            })
        else:
            mutated_groups.append(group)

    return mutated_groups


def run_group_evolution_from_archive(archive: Archive,
                                   task_name: str,
                                   constitution: str,
                                   domain_profile: str,
                                   task_profile: str,
                                   cfg: GroupEvolverConfig,
                                   unified_judge: UnifiedGroupJudge,
                                   base_output_dir: str = None,
                                   domain_name: str = None) -> GroupArchive:
    """기존 개별 spec archive로부터 집단 기반 정책 셋 생성 및 진화

    목표:
    1. 개별 평가를 통해 선정된 spec들을 조합하여 방대한 risk 범위를 커버하는 정책 셋 생성
    2. 의미적으로 coherent한 그룹 형성
    3. 그룹 단위 평가 및 진화로 최적의 정책 조합 탐색
    """

    # 기존 archive에서 상위 spec들 추출
    all_elite_specs = archive.all_elites()
    print(f"🚀 기존 archive에서 집단 기반 정책 셋 생성: {task_name}")
    print(f"  📊 기존 archive 크기: {len(all_elite_specs)}개 spec")
    print(f"  🎯 목표 그룹 크기: {cfg.min_group_size}-{cfg.max_group_size} (target: {cfg.target_group_size})")
    print(f"  📈 최고 점수: {all_elite_specs[0].get('score', 0)}")

    # 상위 spec들을 기반으로 초기 그룹 생성
    top_specs = all_elite_specs[:min(100, len(all_elite_specs))]  # 상위 100개 사용
    
    # 모델 클라이언트 설정 (Generator는 Claude Sonnet 사용)
    if not cfg.anthropic_api_key:
        raise ValueError("❌ Generator용 ANTHROPIC_API_KEY가 필요합니다.")
    
    client_gen = AnthropicClientWrapper(api_key=cfg.anthropic_api_key, model="claude-sonnet-4-20250514")
    
    # unified_judge는 파라미터로 전달받음 (중복 생성 방지)

    # 최고/최저 성능 그룹들을 추적하여 good/bad examples 생성용
    best_groups_history = []
    worst_groups_history = []
    
    # 출력 디렉터리 설정
    if base_output_dir is None:
        base_output_dir = cfg.output_dir
    
    if cfg.use_timestamp_suffix:
        timestamp_suffix = time.strftime("_%Y%m%d_%H%M%S")
        base_output_dir = base_output_dir + timestamp_suffix
    
    if domain_name:
        safe_domain = domain_name.replace(' & ', '_and_').replace(' ', '_')
        safe_task = task_name.replace(' & ', '_and_').replace(' ', '_')
        out_dir = os.path.join(base_output_dir, safe_domain, safe_task)
    else:
        out_dir = os.path.join(base_output_dir, task_name.replace(' ', '_'))
    
    ensure_dir(out_dir)
    best_dir = os.path.join(out_dir, 'best_groups')
    generator_dir = os.path.join(out_dir, 'generator')
    judges_dir = os.path.join(out_dir, 'group_judges')
    rag_dir = os.path.join(out_dir, 'rag')
    ensure_dir(best_dir); ensure_dir(generator_dir); ensure_dir(judges_dir); ensure_dir(rag_dir)
    
    # 집단 Archive 초기화
    group_archive = GroupArchive(max_capacity=50)
    
    # 초기 집단 생성 (기존 archive의 상위 spec들로부터)
    print(f"\n🧬 기존 archive로부터 초기 집단 생성 중...")
    initial_groups = create_spec_groups_from_clustering(top_specs, cfg)
    
    # 초기 집단 평가
    print(f"  ⚖️ 초기 집단 평가 중... ({len(initial_groups)}개 그룹)")
    for i, group_specs in enumerate(initial_groups):
        print(f"    그룹 {i+1}/{len(initial_groups)}: {len(group_specs)}개 spec 평가 중...")
        evaluated_group = evaluate_spec_group(
            group_specs, unified_judge,
            domain_profile, task_profile, -1, judges_dir
        )
        group_archive.add_group(evaluated_group)
    
    print(f"  📊 초기 그룹: {len(group_archive.all_groups())}개")

    # 초기 그룹들로 히스토리 초기화
    if group_archive.all_groups():
        sorted_groups = sorted(group_archive.all_groups(), key=lambda g: g.get('group_score', 0), reverse=True)
        best_groups_history.extend(sorted_groups[:2])  # 상위 2개
        worst_groups_history.extend(sorted_groups[-2:])  # 하위 2개

    # 진화 루프
    history_path = os.path.join(out_dir, 'group_history.jsonl')
    
    with open(history_path, 'a', encoding='utf-8') as hf:
        for gen in range(cfg.generations):
            print(f"\n🚀 Generation {gen + 1}/{cfg.generations}")
            print("=" * 60)
            
            # 부모 그룹 선택
            all_groups = group_archive.all_groups()
            parent_groups = group_archive.sample_parent_groups(cfg.population_per_gen)
            print(f"  🧬 선택된 부모 그룹: {len(parent_groups)}개")
            
            # 교배 및 변이
            print(f"  🔄 교배 및 변이 중...")
            offspring_groups = group_crossover(parent_groups, cfg)
            mutated_groups = group_mutation(
                offspring_groups, client_gen, unified_judge, best_groups_history, worst_groups_history,
                constitution, domain_profile, task_profile, cfg, gen, generator_dir, domain_name, task_name,
                single_spec_pool=top_specs, rag_log_dir=rag_dir
            )
            
            print(f"  📊 새로운 후보 그룹: {len(mutated_groups)}개")
            
            # 후보 그룹 평가
            print(f"  ⚖️ 후보 그룹 평가 중...")
            evaluated_groups = []
            for i, group in enumerate(mutated_groups):
                print(f"    그룹 {i+1}/{len(mutated_groups)}: {len(group['specs'])}개 spec 평가 중...")
                # evaluate_spec_group의 결과를 직접 사용
                evaluated_group = evaluate_spec_group(
                    group['specs'], unified_judge,
                    domain_profile, task_profile, gen, judges_dir
                )

                # 메타데이터 보존
                if 'meta' in group:
                    evaluated_group['meta'] = group['meta']
                evaluated_groups.append(evaluated_group)
            
            # 점수 정규화 및 그룹 Elo 경쟁
            if cfg.use_score_normalization:
                all_for_norm = group_archive.all_groups() + evaluated_groups
                # 그룹 점수 정규화
                if all_for_norm:
                    scores = [g.get('group_score', 0) for g in all_for_norm]
                    if len(set(scores)) > 1:
                        import statistics as st
                        mean_score = st.mean(scores)
                        std_score = st.pstdev(scores) or 1.0
                        for g in all_for_norm:
                            g['group_score_norm'] = (g.get('group_score', 0) - mean_score) / std_score

            # Pointwise 평가만 사용 (Elo 경쟁 제거)
            
            # 최고/최저 성능 그룹 기록 (good/bad examples용)
            if evaluated_groups:
                sorted_by_score = sorted(evaluated_groups, key=lambda g: g.get('group_score', 0), reverse=True)
                best_groups_history.extend(sorted_by_score[:2])  # 상위 2개
                worst_groups_history.extend(sorted_by_score[-2:])  # 하위 2개

                # 히스토리 크기 제한
                best_groups_history = best_groups_history[-5:]  # 최근 5개 유지
                worst_groups_history = worst_groups_history[-5:]  # 최근 5개 유지

            # Archive에 추가
            for group in evaluated_groups:
                group_archive.add_group(group)
            
            # 통계 및 로그
            all_groups = group_archive.all_groups()
            best_group = all_groups[0] if all_groups else None
            best_score = best_group.get('group_score', 0) if best_group else 0
            
            record = {
                'generation': gen,
                'best_group_score': best_score,
                'archive_size': len(all_groups),
                'avg_group_size': sum(len(g.get('specs', [])) for g in all_groups) / len(all_groups) if all_groups else 0,
                'timestamp': time.time(),
            }
            hf.write(json.dumps(record, ensure_ascii=False) + "\n")
            hf.flush()
            
            # 상위 그룹들 저장
            fname = os.path.join(best_dir, f"gen{gen:03d}_top_groups.md")
            with open(fname, 'w', encoding='utf-8') as wf:
                wf.write(f"# Generation {gen} - Top Groups\n\n")
                wf.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                wf.write(f"Archive size: {len(all_groups)}\n")
                wf.write(f"Best Group Score: {best_score:.1f}\n\n")
                
                for i, group in enumerate(all_groups[:10]):
                    wf.write(f"## Group #{i+1}\n\n")
                    wf.write(f"**ID:** {group['id']}\n")

                    wf.write(f"**Group Score:** {group.get('group_score', 0)}/100\n")
                    wf.write(f"**Scores:** Cohesion: {group.get('group_scores', {}).get('cohesion', 0)}/30, ")
                    wf.write(f"Coverage: {group.get('group_scores', {}).get('coverage', 0)}/25, ")
                    wf.write(f"Redundancy: {group.get('group_scores', {}).get('redundancy', 0)}/25, ")
                    wf.write(f"Practicality: {group.get('group_scores', {}).get('practicality', 0)}/20\n")
                    wf.write(f"**Group Size:** {len(group.get('specs', []))}\n\n")
                    
                    wf.write(f"**Specifications:**\n")
                    for j, spec in enumerate(group.get('specs', []), 1):
                        wf.write(f"{j}. {spec.get('text', '')}\n")
                    wf.write("\n" + "-"*60 + "\n\n")
            
            print(f"  📊 Gen {gen}: best_score={best_score:.1f}, archive_size={len(all_groups)}")
    
    # ========== 최종 Spec 추출 및 저장 (ver3용) ==========
    print(f"\n🏆 최고 점수 그룹 추출 및 중복 제거 중...")
    
    all_final_groups = group_archive.all_groups()
    if all_final_groups:
        # 최고 점수 그룹 추출
        best_group = all_final_groups[0]
        best_specs = best_group.get('specs', [])
        
        print(f"  📊 최고 그룹: {len(best_specs)}개 spec (점수: {best_group.get('group_score', 0)}/100)")
        
        # LLM으로 중복 제거
        deduplicated_specs = deduplicate_specs_with_llm(
            best_specs, 
            client_gen,  # Generator client 사용
            domain_profile, 
            task_profile
        )
        
        # JSON 저장
        final_spec_path = os.path.join(out_dir, 'final_spec.json')
        final_spec_data = {
            'domain': domain_name,
            'task': task_name,
            'generation': cfg.generations,
            'original_group_score': best_group.get('group_score', 0),
            'original_count': len(best_specs),
            'deduplicated_count': len(deduplicated_specs),
            'specifications': [
                {
                    'id': spec.get('id', ''),
                    'text': spec.get('text', ''),
                    'score': spec.get('score', 0),
                    'elo': spec.get('elo', 0)
                }
                for spec in deduplicated_specs
            ]
        }
        
        with open(final_spec_path, 'w', encoding='utf-8') as f:
            json.dump(final_spec_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 최종 Spec 저장: {final_spec_path}")
        print(f"  📝 최종 Spec 개수: {len(deduplicated_specs)}개")
    
    print(f"\n✅ 집단 진화 완료! 결과: {out_dir}")
    return group_archive


def run_domain_group_evolution(target_domain: str = "Legal_and_Regulatory", output_dir: str = "final_spec_ver1", generations: int = 5):
    """특정 도메인의 모든 태스크에 대해 그룹 진화를 실행"""
    import os
    import sys
    from alpha_elo import Archive

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # Client 생성 (UnifiedGroupJudge용 - GPT-4o 필수)
    if not openai_api_key:
        raise ValueError("❌ Judge용 OPENAI_API_KEY가 필요합니다.")
    
    from alpha_elo import OpenAIClientWrapper
    client = OpenAIClientWrapper(api_key=openai_api_key, model="gpt-4o")

    config = GroupEvolverConfig(
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        generations=generations,  
        population_per_gen=6,  # 그룹 수 증가
        output_dir=output_dir,
        use_timestamp_suffix=False
    )

    # 도메인과 태스크를 기반으로 가장 최근 archive 자동 찾기
    def find_latest_archive(domain_name, task_name):
        """도메인/태스크 조합의 가장 최근 archive를 찾는 함수"""
        import os
        import re
        from datetime import datetime

        base_dir = "/Users/vashazza/Desktop/PJ/Fellowship/AlphaEvolve"

        # single_spec_result_ver1 폴더 찾기
        results_dir = "single_spec_result_ver1"
        
        # 도메인/태스크 조합의 archive 찾기
        domain_task_path = os.path.join(base_dir, results_dir, domain_name, task_name)
        archive_path = os.path.join(domain_task_path, "top100_archive.json")

        if os.path.exists(archive_path):
            return archive_path

        return None

    # 도메인 설정
    domain_name = target_domain
    print(f"🚀 도메인 '{domain_name}'의 그룹 진화 시작")

    # 해당 도메인의 few_shot 폴더에서 모든 태스크 찾기
    few_shot_folder = "few_shot_examples"
    safe_domain = target_domain
    domain_folder = os.path.join(few_shot_folder, safe_domain)
    ensure_dir(domain_folder)

    if not os.path.exists(domain_folder):
        print(f"❌ 도메인 폴더를 찾을 수 없습니다: {domain_folder}")
        return

    task_files = [f for f in os.listdir(domain_folder) if f.endswith('.txt')]
    if not task_files:
        print(f"⚠️ {domain_folder}에 태스크 파일이 없습니다.")
        return

    print(f"📋 발견된 태스크들: {task_files}")

    # UnifiedGroupJudge 생성 (한 번만)
    unified_judge = UnifiedGroupJudge(client)

    # 각 태스크에 대해 그룹 진화 실행
    for task_file in task_files:
        task_name = task_file.replace('.txt', '').replace('_and_', ' & ').replace('_', ' ')
        # 실제 저장된 폴더명 (언더스코어 유지)
        task_folder_name = task_file.replace('.txt', '')
        print(f"\n🎯 태스크 '{task_name}' 그룹 진화 시작")

        # 가장 최근 archive 찾기 (폴더명으로 검색)
        archive_path = find_latest_archive(domain_name, task_folder_name)

        if not archive_path:
            print(f"  ❌ {domain_name}/{task_folder_name} 조합의 archive를 찾을 수 없습니다. 건너뜁니다.")
            continue

        try:
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_data = json.load(f)

            # 상위 80개 spec만 사용
            test_specs = []
            for item in archive_data[:80]:
                test_specs.append({
                    'text': item['text'],
                    'id': item['id'],
                    'score': item['score'],
                    'elo': item['elo']
                })

        except Exception as e:
            print(f"  ❌ Archive 로드 실패: {e}. 건너뜁니다.")
            continue

        # Archive 생성
        test_archive = Archive()
        for spec in test_specs:
            test_archive.add(spec)

        # 도메인/태스크 프로필 설정
        constitution = "Build secure, reliable, and user-friendly AI systems."
        domain_profile = f"### Domain: {domain_name.replace('_and_', ' & ').replace('_', ' ')}\n- Description: Domain-specific requirements"
        task_profile = f"### Task: {task_name}\n- Description: Task-specific requirements"

        try:
            group_archive = run_group_evolution_from_archive(
                test_archive, task_name, constitution, domain_profile, task_profile,
                config, unified_judge, domain_name=target_domain
            )
            print(f"  🏆 {task_name} 진화 완료! 총 {len(group_archive.all_groups())}개 그룹 생성")

            # 최종 결과 출력 (상위 그룹 1개만)
            best_groups = group_archive.all_groups()[:1]
            if best_groups:
                group = best_groups[0]
                print(f"     그룹 점수: {group.get('group_score', 0)}/100")
                print(f"     그룹 크기: {len(group.get('specs', []))}개 spec")

        except Exception as e:
            print(f"  ❌ {task_name} 그룹 진화 중 오류: {e}")
            continue

    print(f"\n✅ 도메인 '{domain_name}'의 모든 태스크 그룹 진화 완료!")


def run_single_task_group_evolution(target_domain: str, target_task: str, output_dir: str = "final_spec_ver1", generations: int = 5):
    """특정 도메인의 특정 태스크에 대해서만 그룹 진화를 실행"""
    import os
    import sys
    from alpha_elo import Archive

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # Client 생성 (UnifiedGroupJudge용 - GPT-4o 필수)
    if not openai_api_key:
        raise ValueError("❌ Judge용 OPENAI_API_KEY가 필요합니다.")
    
    from alpha_elo import OpenAIClientWrapper
    client = OpenAIClientWrapper(api_key=openai_api_key, model="gpt-4o")

    config = GroupEvolverConfig(
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        generations=generations,
        population_per_gen=6,
        output_dir=output_dir,
        use_timestamp_suffix=False
    )

    # 도메인과 태스크를 기반으로 가장 최근 archive 자동 찾기
    def find_latest_archive(domain_name, task_name):
        """도메인/태스크 조합의 가장 최근 archive를 찾는 함수"""
        import os
        import re
        from datetime import datetime

        base_dir = "/Users/vashazza/Desktop/PJ/Fellowship/AlphaEvolve"

        # single_spec_result_ver1 폴더 찾기
        results_dir = "single_spec_result_ver1"
        
        # 도메인/태스크 조합의 archive 찾기
        domain_task_path = os.path.join(base_dir, results_dir, domain_name, task_name)
        archive_path = os.path.join(domain_task_path, "top100_archive.json")

        if os.path.exists(archive_path):
            return archive_path

        return None

    # 도메인과 태스크 설정
    domain_name = target_domain
    task_name = target_task
    # 실제 저장된 폴더명 (언더스코어 유지)
    task_folder_name = target_task.replace(' & ', '_and_').replace(' ', '_')
    print(f"🚀 단일 태스크 그룹 진화: {domain_name} / {task_name}")

    # 가장 최근 archive 찾기 (폴더명으로 검색)
    archive_path = find_latest_archive(domain_name, task_folder_name)

    if not archive_path:
        print(f"❌ {domain_name}/{task_name} 조합의 archive를 찾을 수 없습니다.")
        return

    try:
        with open(archive_path, 'r', encoding='utf-8') as f:
            archive_data = json.load(f)

        # 상위 80개 spec만 사용
        test_specs = []
        for item in archive_data[:80]:
            test_specs.append({
                'text': item['text'],
                'id': item['id'],
                'score': item['score'],
                'elo': item['elo']
            })

    except Exception as e:
        print(f"❌ Archive 로드 실패: {e}")
        return

    # Archive 생성
    test_archive = Archive()
    for spec in test_specs:
        test_archive.add(spec)

    # 도메인/태스크 프로필 설정
    constitution = "Build secure, reliable, and user-friendly AI systems."
    domain_profile = f"### Domain: {domain_name.replace('_and_', ' & ').replace('_', ' ')}\n- Description: Domain-specific requirements"
    task_profile = f"### Task: {task_name}\n- Description: Task-specific requirements"

    # UnifiedGroupJudge 생성
    unified_judge = UnifiedGroupJudge(client)

    try:
        group_archive = run_group_evolution_from_archive(
            test_archive, task_name, constitution, domain_profile, task_profile,
            config, unified_judge, domain_name=target_domain
        )
        print(f"🏆 {task_name} 진화 완료! 총 {len(group_archive.all_groups())}개 그룹 생성")

        # 최종 결과 출력 (상위 그룹 1개만)
        best_groups = group_archive.all_groups()[:1]
        if best_groups:
            group = best_groups[0]
            print(f"   그룹 점수: {group.get('group_score', 0)}/100")
            print(f"   그룹 크기: {len(group.get('specs', []))}개 spec")

    except Exception as e:
        print(f"❌ {task_name} 그룹 진화 중 오류: {e}")


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='그룹 진화 실행')
    parser.add_argument('domain', type=str, nargs='?', default="Legal_and_Regulatory",
                        help='타겟 도메인 (예: General, Healthcare_and_Medicine)')
    parser.add_argument('task', type=str, nargs='?', default=None,
                        help='타겟 태스크 (선택사항, 지정하지 않으면 모든 태스크)')
    parser.add_argument('--output-dir', type=str, default="final_spec_ver1",
                        help='출력 디렉터리 (기본값: final_spec_ver1)')
    parser.add_argument('--generations', type=int, default=5,
                        help='진화 세대 수 (기본값: 5)')
    
    args = parser.parse_args()

    if args.task:
        # 도메인과 태스크 둘 다 지정된 경우
        print(f"🎯 Target: {args.domain} / {args.task}")
        print(f"📁 Output: {args.output_dir}")
        print(f"🔄 Generations: {args.generations}")
        run_single_task_group_evolution(args.domain, args.task, args.output_dir, args.generations)
    else:
        # 도메인만 지정된 경우
        print(f"🎯 Target Domain: {args.domain} (모든 태스크)")
        print(f"📁 Output: {args.output_dir}")
        print(f"🔄 Generations: {args.generations}")
        run_domain_group_evolution(args.domain, args.output_dir, args.generations)
