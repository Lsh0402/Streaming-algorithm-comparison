# STEP 01. 데이터 준비 — MovieLens 1M 평점 스트림 로드 모듈
from collections import namedtuple
from pathlib import Path
from typing import Iterator

# 스트림에서 흘러나오는 단일 평점 이벤트 레코드 타입
Rating = namedtuple("Rating", ["user_id", "movie_id", "rating", "timestamp"])

SEP = "::"

# ── [1] 데이터셋 로드 ──────────────────────────────────────────────────────────
# MovieLens 1M 데이터셋 경로를 지정한다.
# 파일 형식: userId::movieId::rating::timestamp  (인코딩: latin-1)
# 예) 1::1193::5::978300760
DEFAULT_PATH = Path(__file__).parent / "ml-1m" / "ratings.dat"


def stream_ratings(path: Path = DEFAULT_PATH) -> Iterator[Rating]:
    # ── [3] 전체 데이터를 한 번에 메모리에 올리지 않는 방식 ─────────────────────
    # 이 함수는 yield를 사용하는 제너레이터 함수다.
    # 호출해도 즉시 실행되지 않고, 호출부가 next()를 요청할 때마다
    # 한 줄씩 읽어 Rating 하나를 반환한 뒤 일시 정지한다.
    # 파일 전체를 list/dict로 적재하지 않으므로 메모리 사용량이
    # 파일 크기(23.5 MB)와 무관하게 수십 KB 수준을 유지한다.
    with open(path, "r", encoding="latin-1") as f:

        # ── [2] 스트림 형태로 한 줄씩 처리 ────────────────────────────────────
        # `for line in f` 는 파이썬 파일 이터레이터를 사용한다.
        # f.read() / f.readlines() 처럼 전체를 올리는 것이 아니라
        # OS 버퍼에서 한 줄씩 가져오므로 I/O와 처리가 동시에 진행된다.
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEP)
            if len(parts) != 4:      # 형식이 깨진 줄 건너뜀
                continue
            user_id, movie_id, rating, timestamp = parts
            yield Rating(int(user_id), int(movie_id), int(rating), int(timestamp))
            # yield 지점에서 함수가 일시 정지 → 다음 레코드는 호출부가 요청할 때 읽힘


if __name__ == "__main__":
    # 동작 검증: 전체 레코드 수 + 처음 3개 출력
    count = 0
    first = []
    for rec in stream_ratings():
        if count < 3:
            first.append(rec)
        count += 1

    print(f"총 레코드 수: {count:,}")
    print("처음 3개:")
    for rec in first:
        print(f"  {rec}")
