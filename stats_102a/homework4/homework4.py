# pqnumber.py

from typing import List

class PQNumber:
    def __init__(self, sign: int, p: int, q: int, nums: List[int]):
        if sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative")
        if len(nums) != p + q + 1:
            raise ValueError("nums must have length p + q + 1")
        self.sign = sign
        self.p = p
        self.q = q
        self.nums = nums

    def __repr__(self):
        return f"PQNumber(sign={self.sign}, p={self.p}, q={self.q}, nums={self.nums})"

    def decimal_value(self) -> float:
        integer_part = self.nums[: self.p + 1]
        frac_part = self.nums[self.p + 1 :]
        int_val = int("".join(map(str, integer_part)))
        frac_val = int("".join(map(str, frac_part))) / (10 ** len(frac_part)) if frac_part else 0
        return self.sign * (int_val + frac_val)

    def print(self, DEC: bool = False):
        if DEC:
            print(self.decimal_value())
        else:
            print(self)


def is_pqnumber(x) -> bool:
    return isinstance(x, PQNumber)


def as_pqnumber(x: float, p: int, q: int) -> PQNumber:
    sign = 1 if x >= 0 else -1
    x = abs(x)
    fmt = f"{{:.{q}f}}".format(x)
    int_str, _, frac_str = fmt.partition(".")
    nums = list(map(int, list(int_str + frac_str)))
    return PQNumber(sign, len(int_str) - 1, q, nums)


def as_numeric_pqnumber(x: PQNumber) -> float:
    return x.decimal_value()


def align_pqnumber(x: PQNumber, p: int, q: int) -> PQNumber:
    nums = x.nums.copy()
    while len(nums) < p + q + 1:
        nums.insert(0, 0)
    while len(nums) > p + q + 1:
        nums.pop(0)
    return PQNumber(x.sign, p, q, nums)


def shave_off_zeros(nums: List[int], p: int, q: int):
    while len(nums) > 1 and nums[0] == 0 and p > 0:
        nums.pop(0)
        p -= 1
    while q > 0 and nums[-1] == 0:
        nums.pop()
        q -= 1
    return {"nums": nums, "p": p, "q": q}


def add(x: PQNumber, y: PQNumber) -> PQNumber:
    val = x.decimal_value() + y.decimal_value()
    q = max(x.q, y.q)
    p = max(x.p, y.p)
    return as_pqnumber(val, p, q)


def subtract(x: PQNumber, y: PQNumber) -> PQNumber:
    val = x.decimal_value() - y.decimal_value()
    q = max(x.q, y.q)
    p = max(x.p, y.p)
    return as_pqnumber(val, p, q)


def multiply(x: PQNumber, y: PQNumber) -> PQNumber:
    val = x.decimal_value() * y.decimal_value()
    q = x.q + y.q
    p = x.p + y.p
    return as_pqnumber(val, p, q)