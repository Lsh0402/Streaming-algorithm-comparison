# 항목별 빈도를 근사 추정하는 Count-Min Sketch 직접 구현 (표준 라이브러리만 사용)
#
# 핵심 로직 설명:
#   - d개 행 x w개 열로 이루어진 정수 카운터 테이블로 빈도를 근사한다.
#   - add(x)      : 행마다 해시한 칸을 한 개씩 골라 +1 (행별로 서로 다른 해시 사용)
#   - estimate(x) : 행별 해당 칸 값 중 최솟값(min)을 반환 → 충돌로 부풀려진 값을 최대한 제거
#   - 해시 충돌은 카운터를 키우기만 하므로 과소추정은 없고 추정값은 항상 실제값 이상이다.
import hashlib
import math
from typing import Any


class CountMinSketch:
    """d개의 해시 행 x w개의 카운터 열로 항목 빈도를 근사 추정한다.

    - add(x, c): d개 행에서 각각 한 칸씩 해시해 카운터를 c만큼 증가
    - estimate(x): d개 행의 해당 칸 값 중 최솟값을 반환
    해시 충돌 때문에 실제값보다 작아질 수는 없고(과소추정 없음), 항상 실제값 이상으로 추정된다.
    """

    def __init__(self, epsilon: float = 0.001, delta: float = 0.001,
                 w: int = None, d: int = None):
        # 파라미터 실험용으로 w, d를 직접 지정할 수 있다.
        # 둘 다 None이면 오차폭 epsilon, 실패확률 delta로 폭 w와 깊이 d를 계산한다.
        #   w = ceil(e / epsilon)   -> 추정 오차는 epsilon * (전체 카운트 합) 이내
        #   d = ceil(ln(1/delta))   -> 위 오차 보장이 깨질 확률이 delta 이하
        if w is not None and d is not None:
            self.w = max(1, w)
            self.d = max(1, d)
        else:
            self.w = max(1, int(math.ceil(math.e / epsilon)))
            self.d = max(1, int(math.ceil(math.log(1 / delta))))
        self.table = [[0] * self.w for _ in range(self.d)]
        self.total = 0  # 누적된 전체 카운트 합

    def _columns(self, item: Any):
        """d개 행 각각에 대한 열 인덱스를 생성한다.

        SHA-256 한 번으로 h1, h2를 뽑고 g_i(x) = (h1 + i*h2) mod w로
        행마다 서로 다른 열을 결정한다 (행별 독립 해시 역할).
        """
        data = str(item).encode("utf-8")
        digest = hashlib.sha256(data).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")
        for i in range(self.d):
            yield i, (h1 + i * h2) % self.w

    def add(self, item: Any, count: int = 1):
        for row, col in self._columns(item):
            self.table[row][col] += count
        self.total += count

    def estimate(self, item: Any) -> int:
        # 각 행의 해당 칸 값 중 최솟값 -> 충돌로 부풀려진 값을 최대한 걷어낸다
        return min(self.table[row][col] for row, col in self._columns(item))

    def __getitem__(self, item: Any) -> int:
        return self.estimate(item)
