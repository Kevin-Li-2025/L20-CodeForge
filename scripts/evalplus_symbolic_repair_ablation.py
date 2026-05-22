from __future__ import annotations

import argparse
import json
from pathlib import Path


SYMBOLIC_REPAIRS = [
    {
        "task_id": "HumanEval/32",
        "solution": """import math

def poly(xs: list, x: float):
    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))

def find_zero(xs: list):
    def f(x):
        return poly(xs, x)

    if f(0.0) == 0:
        return 0.0

    bound = 1.0
    left, right = -bound, bound
    fl, fr = f(left), f(right)
    while fl * fr > 0:
        bound *= 2.0
        left, right = -bound, bound
        fl, fr = f(left), f(right)

    for _ in range(100):
        mid = (left + right) / 2.0
        fm = f(mid)
        if fm == 0:
            return mid
        if fl * fm <= 0:
            right = mid
            fr = fm
        else:
            left = mid
            fl = fm
    return (left + right) / 2.0
""",
    },
    {
        "task_id": "HumanEval/132",
        "solution": """def is_nested(string):
    target = '[[]]'
    index = 0
    for char in string:
        if char == target[index]:
            index += 1
            if index == len(target):
                return True
    return False
""",
    },
    {
        "task_id": "HumanEval/145",
        "solution": """def order_by_points(nums):
    def digit_sum(num):
        digits = str(abs(num))
        total = sum(int(digit) for digit in digits)
        if num < 0:
            total -= 2 * int(digits[0])
        return total

    return sorted(nums, key=digit_sum)
""",
    },
]


def write_symbolic_repairs(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in SYMBOLIC_REPAIRS),
        encoding="utf-8",
    )
    return len(SYMBOLIC_REPAIRS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write transparent task-specific EvalPlus symbolic repair candidates. "
            "This is an ablation artifact, not a general model-improvement method."
        )
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = write_symbolic_repairs(args.output)
    print(json.dumps({"output": str(args.output), "records": count}, indent=2))


if __name__ == "__main__":
    main()
