# 원소 포함 여부를 근사 판정하는 Bloom Filter 직접 구현 (표준 라이브러리만 사용)
#
# 핵심 로직 설명:
#   - m비트짜리 비트배열(bytearray)과 k개의 해시 함수로 집합을 표현한다.
#   - add(x)   : x를 k개 위치에 해시해 해당 비트를 모두 1로 세팅
#   - contains : k개 위치가 모두 1이면 "있을 수 있음"(True), 하나라도 0이면 "확실히 없음"(False)
#   - 해시 합성: SHA-256 한 번으로 h1,h2를 뽑고 g_i(x)=(h1+i*h2) mod m 으로 k개 해시를 만든다(double hashing).
#   - 거짓 음성(있는데 없다고 함)은 절대 없고, 거짓 양성(없는데 있다고 함)만 발생한다.
import hashlib
import math
from typing import Any


class BloomFilter:
    """비트 배열 + k개의 해시 함수로 집합 포함 여부를 근사 판정한다.

    - add(x):   x를 k개 위치에 해시해 해당 비트를 1로 세팅
    - contains(x): k개 위치가 모두 1이면 "있을 수 있음"(True), 하나라도 0이면 "확실히 없음"(False)
    거짓 양성(없는데 있다고 함)은 발생하지만, 거짓 음성(있는데 없다고 함)은 절대 없다.
    """

    def __init__(self, expected_items: int = None, fp_rate: float = 0.01,
                 m: int = None, k: int = None):
        # 파라미터 실험용으로 m, k를 직접 지정할 수 있다.
        # 둘 다 None이면 목표 원소 수 n과 목표 거짓양성률 p로 최적값을 계산한다.
        #   m = -n * ln(p) / (ln2)^2
        #   k = (m / n) * ln2
        self.n = expected_items
        self.p = fp_rate
        if m is not None and k is not None:
            self.m = max(1, m)
            self.k = max(1, k)
        else:
            self.m = max(1, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
            self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)  # m비트를 바이트 배열로
        self.count = 0  # add 호출 횟수

    def _hashes(self, item: Any):
        """double hashing으로 k개의 비트 위치를 생성한다.

        SHA-256 한 번으로 128비트씩 두 정수 h1, h2를 뽑고,
        g_i(x) = (h1 + i*h2) mod m  으로 k개 위치를 만든다.
        해시 함수를 k번 따로 돌리지 않아 빠르면서도 분포가 충분히 독립적이다.
        """
        data = str(item).encode("utf-8")
        digest = hashlib.sha256(data).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def _set_bit(self, pos: int):
        self.bits[pos >> 3] |= (1 << (pos & 7))

    def _get_bit(self, pos: int) -> bool:
        return bool(self.bits[pos >> 3] & (1 << (pos & 7)))

    def add(self, item: Any):
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.count += 1

    def contains(self, item: Any) -> bool:
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def current_fp_rate(self) -> float:
        """현재까지 add된 원소 수 기준 이론적 거짓양성률 추정.

        (1 - e^(-k*n/m))^k
        """
        if self.count == 0:
            return 0.0
        return (1 - math.exp(-self.k * self.count / self.m)) ** self.k
