# Bloom Filter와 Count-Min Sketch의 파라미터를 바꿔가며 정확도/메모리를 비교하는 실험 스크립트
import itertools
from collections import Counter

from data_prep import stream_ratings
from bloom_filter import BloomFilter
from count_min_sketch import CountMinSketch

SAMPLE = 100_000  # 실험 반복을 위해 앞쪽 N개 레코드만 메모리에 적재해 재사용


def load_sample():
    """앞쪽 SAMPLE개 레코드를 한 번만 읽어 키/영화ID 리스트와 정답을 만든다."""
    keys = []
    movies = []
    for rec in itertools.islice(stream_ratings(), SAMPLE):
        keys.append(f"{rec.user_id}:{rec.movie_id}")
        movies.append(rec.movie_id)
    freq = Counter(movies)         # 정답 빈도
    n_unique = len(set(keys))      # 고유 키 수
    return keys, movies, freq, n_unique


def bloom_fpr(keys, m, k):
    """주어진 m, k로 Bloom을 채운 뒤, 존재하지 않는 키로 거짓양성률을 실측한다."""
    bf = BloomFilter(m=m, k=k)
    for key in keys:
        bf.add(key)
    probes = 50_000
    fp = sum(1 for i in range(probes) if f"absent-{i}" in bf)
    return fp / probes, bf.m / 8  # FPR, 비트배열 바이트


def cms_relerr(movies, freq, w, d):
    """주어진 w, d로 CMS를 채운 뒤 평균 상대오차를 측정한다."""
    cms = CountMinSketch(w=w, d=d)
    for m in movies:
        cms.add(m)
    rel = [abs(cms.estimate(mv) - c) / c for mv, c in freq.items()]
    return sum(rel) / len(rel)  # 평균 상대오차


def run():
    keys, movies, freq, n_unique = load_sample()
    print(f"샘플: {len(keys):,} 레코드, 고유 키 {n_unique:,}, 고유 영화 {len(freq):,}\n")

    # ===== Bloom Filter: 비트 배열 크기(m) 변화 (해시 k=6 고정) =====
    print("=" * 60)
    print("[Bloom] 비트 배열 크기 m 변화 (k=6 고정)")
    print(f"  {'m(bits)':>12} {'크기':>10} {'FPR(실측)':>12}")
    for m in [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
        fpr, size = bloom_fpr(keys, m=m, k=6)
        print(f"  {m:>12,} {size/1024:>8.0f}KB {fpr:>12.4%}")

    # ===== Bloom Filter: 해시 함수 개수(k) 변화 (m 고정) =====
    print("\n[Bloom] 해시 함수 개수 k 변화 (m=1,000,000 고정)")
    print("  (이론 최적 k = m/n * ln2 ~= %.1f)" % (1_000_000 / n_unique * 0.693))
    print(f"  {'k':>4} {'FPR(실측)':>12}")
    for k in [1, 2, 4, 6, 8, 12, 16]:
        fpr, _ = bloom_fpr(keys, m=1_000_000, k=k)
        print(f"  {k:>4} {fpr:>12.4%}")

    # ===== Count-Min Sketch: width(w) 변화 (depth d=5 고정) =====
    print("\n" + "=" * 60)
    print("[CMS] width w 변화 (d=5 고정)")
    print(f"  {'w':>8} {'카운터수':>10} {'평균상대오차':>14}")
    for w in [50, 100, 500, 2_000, 10_000]:
        err = cms_relerr(movies, freq, w=w, d=5)
        print(f"  {w:>8,} {w*5:>10,} {err:>14.4%}")

    # ===== Count-Min Sketch: depth(d) 변화 (width w=200 고정) =====
    print("\n[CMS] depth d 변화 (w=200 고정)")
    print(f"  {'d':>4} {'카운터수':>10} {'평균상대오차':>14}")
    for d in [1, 2, 3, 5, 8]:
        err = cms_relerr(movies, freq, w=200, d=d)
        print(f"  {d:>4} {200*d:>10,} {err:>14.4%}")


if __name__ == "__main__":
    run()
