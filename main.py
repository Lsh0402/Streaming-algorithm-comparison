# 평점 스트림을 Bloom Filter와 Count-Min Sketch에 흘려 정답과 비교 검증하는 실행 스크립트
from collections import Counter

from data_prep import stream_ratings
from bloom_filter import BloomFilter
from count_min_sketch import CountMinSketch


def run():
    # 근사 자료구조 (한 줄씩 스트림 처리, 전체를 dict/set으로 들고 있지 않아도 동작)
    bloom = BloomFilter(expected_items=1_000_000, fp_rate=0.01)
    cms = CountMinSketch(epsilon=0.0001, delta=0.001)

    # 검증용 정답 (실제값과 비교하기 위한 용도로만 보관)
    seen_exact = set()          # 이미 본 (user, movie) 키 정답
    movie_exact = Counter()     # movie_id별 실제 등장 횟수 정답

    dup_bloom_hits = 0          # bloom이 "이미 봤다"고 판정한 횟수
    dup_actual = 0              # 실제 중복 횟수
    false_positive = 0          # 처음 보는데 bloom이 봤다고 한 횟수 (거짓 양성)

    for rec in stream_ratings():
        key = f"{rec.user_id}:{rec.movie_id}"

        # --- Bloom Filter: 중복(이미 본 평점) 감지 ---
        in_bloom = key in bloom
        in_exact = key in seen_exact
        if in_bloom:
            dup_bloom_hits += 1
        if in_exact:
            dup_actual += 1
        elif in_bloom:
            false_positive += 1  # 정답엔 없는데 bloom이 있다고 함
        bloom.add(key)
        seen_exact.add(key)

        # --- Count-Min Sketch: movie_id별 빈도 ---
        cms.add(rec.movie_id)
        movie_exact[rec.movie_id] += 1

    print("=" * 60)
    print("[Bloom Filter] 중복 (user,movie) 키 감지")
    print(f"  구조: m={bloom.m:,} bits ({bloom.m/8/1024:.0f} KB), k={bloom.k} hashes")
    print(f"  고유 키 수      : {len(seen_exact):,}")
    print(f"  실제 중복 수    : {dup_actual:,}")
    print(f"  Bloom 중복 판정 : {dup_bloom_hits:,}")
    print(f"  거짓 양성 수    : {false_positive:,}")
    print(f"  이론적 FP율     : {bloom.current_fp_rate():.4%}")

    print("=" * 60)
    print("[Count-Min Sketch] movie_id별 빈도 추정")
    print(f"  구조: w={cms.w:,} x d={cms.d} = {cms.w*cms.d:,} counters")
    print(f"  Top 5 인기 영화 (추정값 vs 실제값):")
    for movie_id, actual in movie_exact.most_common(5):
        est = cms.estimate(movie_id)
        err = est - actual
        print(f"    movie {movie_id:>5}: 추정 {est:>6,} | 실제 {actual:>6,} | 오차 +{err}")

    # 전체 항목 평균 절대오차
    total_err = sum(cms.estimate(m) - c for m, c in movie_exact.items())
    print(f"  전체 {len(movie_exact):,}개 영화 평균 과대추정: {total_err/len(movie_exact):.2f}")


if __name__ == "__main__":
    run()
