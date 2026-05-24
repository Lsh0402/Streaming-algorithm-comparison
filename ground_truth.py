# 정답(Ground Truth)을 정확히 계산하고 스트리밍 근사 결과와 비교하는 모듈
from collections import Counter
from typing import Any

from data_prep import stream_ratings
from bloom_filter import BloomFilter
from count_min_sketch import CountMinSketch


class GroundTruth:
    """전체 데이터를 정확히 보관해 근사 알고리즘의 비교 기준을 제공한다.

    - Bloom Filter 대응 : 실제 set 기반 포함 여부
    - Count-Min Sketch 대응 : dictionary(Counter) 기반 정확한 빈도
    근사 구조와 달리 메모리를 그대로 다 쓰므로, 정확성 검증 용도로만 사용한다.
    """

    def __init__(self):
        self.members: set = set()        # 지금까지 등장한 모든 키 (정확한 집합)
        self.freq: Counter = Counter()   # 항목별 정확한 빈도

    def observe_membership(self, key: Any) -> bool:
        """키를 보기 전 정확한 포함 여부를 반환하고, 그 뒤 집합에 추가한다.

        반환값은 add 이전 상태 기준이라 '실제 중복 여부'를 뜻한다.
        """
        was_present = key in self.members
        self.members.add(key)
        return was_present

    def observe_frequency(self, item: Any, count: int = 1):
        self.freq[item] += count

    def contains(self, key: Any) -> bool:
        return key in self.members

    def frequency(self, item: Any) -> int:
        return self.freq[item]

    def unique_count(self) -> int:
        return len(self.members)


def compare():
    """스트림을 한 번 흘리며 정답과 근사 구조를 동시에 갱신하고 정확도 지표를 출력한다."""
    bloom = BloomFilter(expected_items=1_000_000, fp_rate=0.01)
    cms = CountMinSketch(epsilon=0.0001, delta=0.001)
    gt = GroundTruth()

    # Bloom 판정을 정답과 대조한 혼동행렬 (tp/fp/tn/fn)
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for rec in stream_ratings():
        key = f"{rec.user_id}:{rec.movie_id}"

        # Bloom: add 이전 정답 기준으로 판정 결과를 분류
        approx_dup = key in bloom
        actual_dup = gt.observe_membership(key)  # 정답 갱신 + 실제 중복 여부
        bloom.add(key)

        if actual_dup and approx_dup:
            cm["tp"] += 1
        elif not actual_dup and approx_dup:
            cm["fp"] += 1   # 거짓 양성: 정답엔 신규인데 봤다고 함
        elif not actual_dup and not approx_dup:
            cm["tn"] += 1
        else:
            cm["fn"] += 1   # 거짓 음성: 발생하면 안 됨

        # CMS: 빈도 누적
        cms.add(rec.movie_id)
        gt.observe_frequency(rec.movie_id)

    print("=" * 64)
    print("알고리즘 비교: Ground Truth vs 스트리밍 근사")
    print("=" * 64)

    # ---- Bloom Filter ----
    print("\n[Bloom Filter]  Ground Truth = 실제 set 기반 포함 여부")
    print(f"  메모리   : 근사 {bloom.m/8/1024:.0f} KB  vs  정답 set {len(gt.members):,} keys")
    print(f"  TP(맞은 중복)     : {cm['tp']:,}")
    print(f"  FN(놓친 중복)     : {cm['fn']:,}   <- 0이어야 정상 (거짓 음성 없음)")
    print(f"  FP(거짓 양성)     : {cm['fp']:,}")
    print(f"  TN(맞은 신규)     : {cm['tn']:,}")
    denom = cm["fp"] + cm["tn"]
    fp_rate = cm["fp"] / denom if denom else 0
    print(f"  경험적 FP율       : {fp_rate:.4%}  (이론 상한 {bloom.current_fp_rate():.4%})")

    # ---- Count-Min Sketch ----
    print("\n[Count-Min Sketch]  Ground Truth = dict 기반 정확한 빈도")
    print(f"  메모리   : 근사 {cms.w*cms.d:,} counters  vs  정답 dict {len(gt.freq):,} keys")
    errors = [cms.estimate(m) - c for m, c in gt.freq.items()]
    print(f"  최대 과대추정     : +{max(errors)}")
    print(f"  평균 과대추정     : +{sum(errors)/len(errors):.4f}")
    exact_hits = sum(1 for e in errors if e == 0)
    print(f"  정확 일치 항목    : {exact_hits:,} / {len(errors):,} ({exact_hits/len(errors):.2%})")
    print("\n  Top 5 빈도 (추정 vs 정답):")
    print(f"    {'movie':>6} {'추정':>8} {'정답':>8} {'오차':>6}")
    for movie_id, actual in gt.freq.most_common(5):
        est = cms.estimate(movie_id)
        print(f"    {movie_id:>6} {est:>8,} {actual:>8,} {est-actual:>+6}")


if __name__ == "__main__":
    compare()
