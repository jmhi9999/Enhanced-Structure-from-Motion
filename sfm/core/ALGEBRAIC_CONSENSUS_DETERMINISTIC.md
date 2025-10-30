# Deterministic Algebraic Consensus — Smart Mode Design

**Goal**: Keep the *“mathematically simple but beautiful”* philosophy while delivering a deterministic, explainable alternative to RANSAC for affine verification.

---

## 1. Motivation
- CVPR 제안서에서 강조한 핵심은 **결정론적**이고 **증명이 가능한** 합의 절차.
- 현재 파이프라인은 빠른 실행을 위해 RANSAC을 기본값으로 사용하지만, 기본값을 **알gebraic smart mode**로 전환해도 충분히 효율적인 설계가 가능함.
- 아이디어: Orientation polytope, 미리 계산된 elimination 템플릿, 증명 가능한 Gröbner certificate를 결합해서 “빠르고 deterministic”한 경로를 만든다.

---

## 2. High-Level Pipeline
```
Input  →  Orientation Polytope Reduction  →  Polytope Vertex Enumeration
          ↓                                    ↓
        Deterministic Candidate Generation  →  Gröbner Certificate Refinement
          ↓                                    ↓
    Best Candidate Selection             Certificate / Transformation Output
```

모든 단계는 고정된 순서와 유한한 후보 집합을 기반으로 하므로 완전히 결정론적이다.

---

## 3. Orientation Polytope Acceleration

### 3.1 Polytope Construction
- 각 correspondence \( i \)에 대해 OrientationFilter가 반환하는 허용 회전 범위 \([ \theta_i^- , \theta_i^+ ]\) 사용.
- \( (a_{11}, a_{21}) \) 평면에서 각 한계를 **선형 부등식**으로 표현:
  \[ L_i^-(a_{11}, a_{21}) \le 0,\quad L_i^+(a_{11}, a_{21}) \le 0 \]
- 모든 제약을 모아 convex polytope \( P(\tau) \) 구성.

### 3.2 Vertex Enumeration (Deterministic Candidate Set)
- Half-space 교차점은 최대 \( O(n^2) \)개 vertex.
- 두 개의 orientation facet을 선택하고, 필요한 경우 scale/translation 제약을 추가해 vertex 후보를 구한다.
- Vertex마다 대응하는 affine 회전 성분을 고정하고, Translation은 최소제곱으로 해결.
- 결과: **결정론적이고 작게 bounded**된 후보 변환 집합 \( \{T_k\} \).

---

## 4. Precomputed Elimination Template

### 4.1 Offline 준비
1. 3 correspondence에 대한 다항식 시스템을 symbolic tool(Sage/Maple)로 한 번만 풀어 *elimination template*을 추출.
2. Buchberger 단계에서 생기는 고정된 모노미얼 순서를 기록하고, Macaulay 행렬의 SVD/QR factorization 형태로 저장.
3. 코드를 생성(예: NumPy, 혹은 Numba/C++) 해서 런타임에는 숫자 coefficient만 대입.

### 4.2 Runtime 사용
```text
for each deterministic minimal set (from vertex or ordered combinations):
    plug coefficients into precomputed template
    solve small linear system → candidate transformation
```
- Buchberger를 매번 돌리지 않아도 되므로 SymPy 대비 수십 배 빠름.
- 모든 minimal set을 순회하므로 랜덤성이 없음.

---

## 5. Incremental Gröbner Certificates

### 5.1 Deterministic Ordering
- correspondence를 orientation alignment, descriptor score, 혹은 공간적 거리 기준으로 정렬.
- 순서가 고정되면 iteration마다 같은 certificate progression을 보장.

### 5.2 Incremental Update
1. 빈 ideal \( I_0 \)에서 시작.
2. 각 correspondence \( c_j \)를 추가하면서 \( I_j = I_{j-1} \cup \{g_j\} \).
3. 미리 계산된 템플릿을 활용해 빠르게 Gröbner basis 업데이트(F5 스타일).
4. \( 1 \in I_j \)가 되는 순간, \( c_j \)가 모순을 유발한 outlier임을 명시적으로 보고.

이 과정은 후보 변환 검증과 병렬로 수행해도 좋으며, **증명 가능한 outlier log**를 생성한다.

---

## 6. Candidate Scoring & Selection

- Vertex 기반 후보와 minimal-set 기반 후보를 모두 합쳐서 유한한 deterministic 세트 확보.
- 각 후보 \( T_k \)에 대해 전체 correspondence residual을 평가하고 최적값을 선택.
- 동시에 incremental Gröbner certificate가 보여주는 outlier 정보를 기록.

결과:
- `method='deterministic_groebner'`
- `certificate`: 최종 basis 요약이나 outlier 목록.

---

## 7. Complexity & Practical Notes
- Orientation 필터 이후 typical match 수: 20–40.
- Vertex 수 \( O(n^2) \)와 minimal set 수 \( \binom{n}{3} \) 모두 결정론적으로 bounded.
- Precomputed template 덕에 각 minimal set 해석은 **linear algebra 수준**의 비용.
- Incremental certificate 업데이트는 F5 variant를 사용하면 \( O(n) \) update로 매끄럽게 유지.

---

## 8. Implementation Checklist
- [x] OrientationFilter가 vertex/angle 후보를 제공하도록 개선.
- [x] Elimination template 생성 스크립트 작성 (SymPy 기반).
- [x] 템플릿을 사용한 빠른 numeric solver 구현 (NumPy).
- [ ] Deterministic ordering & incremental Gröbner 업데이트 유틸 추가 *(ordering 완료, certificates TODO)*.
- [x] `AlgebraicConsensus`에 `mode='deterministic'` 옵션 추가 후 smart mode를 기본값으로 전환.
- [x] README 및 문서에서 RANSAC은 fallback임을 명확히 기술.

---

## 9. Roadmap
1. **Prototype**: Orientation vertex enumeration + 템플릿 solver + deterministic loop.
2. **Benchmark**: SymPy 기반 Gröbner vs template solver 속도 비교, RANSAC 대비 품질 검증.
3. **Certification UX**: Incremental certificate를 어떻게 노출할지(텍스트 로그, 구조화된 outlier 레코드 등) 결정.
4. **Fallbacks**: extreme outlier 비율에서 여전히 RANSAC 옵션이 유용하므로, 모듈 옵션으로 유지.

---

**Outcome**: 파이프라인 전부가 “orientation polytope → finite algebraic candidates → Gröbner certificate”라는 깔끔한 수학적 이야기로 정리되며, CVPR 제안서와 README의 서사가 일치한다.
