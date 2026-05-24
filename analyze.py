# 정확도/메모리/시간 3개 항목으로 스트리밍 알고리즘과 정답을 측정·비교하는 성능 분석 스크립트
import time
import tracemalloc

from data_prep import stream_ratings
from bloom_filter import BloomFilter
from count_min_sketch import CountMinSketch
from ground_truth import GroundTruth


def measure(build_fn):
    """build_fn을 실행하며 (소요시간 초, 피크 메모리 바이트, 반환값)을 측정한다."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = build_fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, result


def build_ground_truth():
    gt = GroundTruth()
    for rec in stream_ratings():
        gt.observe_membership(f"{rec.user_id}:{rec.movie_id}")
        gt.observe_frequency(rec.movie_id)
    return gt


def build_bloom():
    bf = BloomFilter(expected_items=1_000_000, fp_rate=0.01)
    for rec in stream_ratings():
        bf.add(f"{rec.user_id}:{rec.movie_id}")
    return bf


def build_cms():
    cms = CountMinSketch(epsilon=0.0001, delta=0.001)
    for rec in stream_ratings():
        cms.add(rec.movie_id)
    return cms


def fmt_bytes(n):
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def run():
    # 정답 먼저 구축 (정확도 비교 기준 + 자체 시간/메모리도 측정)
    gt_time, gt_mem, gt = measure(build_ground_truth)
    n = gt.unique_count()

    bf_time, bf_mem, bf = measure(build_bloom)
    cms_time, cms_mem, cms = measure(build_cms)

    # ---- 정확도 ----
    # Bloom: 모든 키가 add됐으므로 contains는 전부 True여야 함(거짓 음성 0).
    #        존재하지 않는 키를 던져 거짓양성률(FPR)을 실측한다.
    fp = sum(1 for i in range(n) if f"missing-{i}" in bf)
    fpr = fp / n

    # CMS: 상대오차 = |추정-실제| / 실제, 전체 항목 평균
    rel_errors = [abs(cms.estimate(m) - c) / c for m, c in gt.freq.items()]
    mean_rel_err = sum(rel_errors) / len(rel_errors)
    max_abs_err = max(cms.estimate(m) - c for m, c in gt.freq.items())

    print("=" * 70)
    print(f"성능 분석  (총 {sum(gt.freq.values()):,} 레코드, 고유 키 {n:,})")
    print("=" * 70)

    print("\n[시간]  전체 처리 시간 / 초당 처리량")
    total = sum(gt.freq.values())
    for name, t in [("Ground Truth", gt_time), ("Bloom Filter", bf_time), ("Count-Min Sketch", cms_time)]:
        print(f"  {name:<16}: {t:6.2f} s   ({total/t:>12,.0f} rec/s)")

    print("\n[메모리]  실제 피크 할당량(tracemalloc) / 자료구조 크기")
    print(f"  Ground Truth    : {fmt_bytes(gt_mem):>10}   (set+dict, 전체 보관)")
    print(f"  Bloom Filter    : {fmt_bytes(bf_mem):>10}   (비트배열 {fmt_bytes(bf.m/8)}, m={bf.m:,} k={bf.k})")
    print(f"  Count-Min Sketch: {fmt_bytes(cms_mem):>10}   (카운터 {cms.w*cms.d:,}개, w={cms.w:,} d={cms.d})")
    print(f"  -> Bloom은 정답 대비 {gt_mem/bf_mem:.1f}배, CMS는 {gt_mem/cms_mem:.1f}배 절약")

    print("\n[정확도]")
    print(f"  Bloom Filter    : 거짓음성 0 (보장), 거짓양성률(FPR) {fpr:.4%} (이론 상한 {bf.current_fp_rate():.4%})")
    print(f"  Count-Min Sketch: 평균 상대오차 {mean_rel_err:.4%}, 최대 절대오차 +{max_abs_err}")


if __name__ == "__main__":
    run()
