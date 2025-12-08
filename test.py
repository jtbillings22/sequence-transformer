# import math
# import random
# from pathlib import Path

# import sympy  # for primes

# PRIMES = list(sympy.primerange(2, 20000))


# def generate_constant_sequence(length: int):
#     value = random.randint(-500, 500)
#     return [value] * length


# def generate_linear_sequence(length: int):
#     slope = random.randint(-20, 20)
#     intercept = random.randint(-100, 100)
#     return [slope * n + intercept for n in range(1, length + 1)]


# def generate_quadratic_sequence(length: int):
#     a = random.randint(-5, 5)
#     b = random.randint(-20, 20)
#     c = random.randint(-100, 100)
#     return [a * n * n + b * n + c for n in range(1, length + 1)]


# def generate_log_sequence(length: int):
#     base = random.randint(1, 100)
#     return [round(math.log(n + base), 2) for n in range(1, length + 1)]


# def generate_prime_sequence(length: int):
#     start_index = random.randint(0, len(PRIMES) - length)
#     return PRIMES[start_index:start_index + length]


# def generate_collatz_sequence(length: int):
#     n = random.randint(50, 5000)
#     seq = [n]
#     while len(seq) < length:
#         n = n // 2 if n % 2 == 0 else 3 * n + 1
#         seq.append(n)
#     return seq


# GENERATORS = [
#     ("constant", generate_constant_sequence),
#     ("linear", generate_linear_sequence),
#     ("quadratic", generate_quadratic_sequence),
#     ("logarithmic", generate_log_sequence),
#     ("prime", generate_prime_sequence),
#     ("collatz", generate_collatz_sequence),
# ]

# def create_sequence_dataset(num_sequences_per_type: int = 10, length: int = 50):
#     """
#     Generate a dataset of sequences from each generator.
#     """
#     dataset = []
#     for _ in range(num_sequences_per_type):
#         for seq_type, generator in GENERATORS:
#             sequence = generator(length)
#             dataset.append({
#                 "type": seq_type,
#                 "sequence": [seq_type, *sequence]
#             })
#     return dataset


# def save_dataset_text(dataset, filename: str = "seq_dataset.txt"):
#     """
#     Save the dataset as a multi-line text file (no JSON) next to this script.
#     Each sequence is comma-separated; sequences are separated by newlines.
#     """
#     path = Path(__file__).parent / filename
#     # Write each sequence on its own line to keep parsing simple and fast.
#     sequences_as_text = "\n".join(",".join(map(str, item["sequence"])) for item in dataset)
#     path.write_text(sequences_as_text)
#     print(f"Saved {len(dataset)} sequences to {path}")


# def main():
#     dataset = create_sequence_dataset(num_sequences_per_type=10, length=500)
#     save_dataset_text(dataset, filename="seq_dataset.txt")


# if __name__ == "__main__":
#     main()

# -------------------------------

# import math
# import random
# from pathlib import Path

# import sympy  # for primes

# PRIMES = list(sympy.primerange(2, 20000))


# def generate_constant_sequence(length: int):
#     value = random.randint(-500, 500)
#     return [value] * length


# def generate_linear_sequence(length: int):
#     slope = random.randint(-20, 20)
#     intercept = random.randint(-100, 100)
#     return [slope * n + intercept for n in range(1, length + 1)]


# def generate_quadratic_sequence(length: int):
#     a = random.randint(-5, 5)
#     b = random.randint(-20, 20)
#     c = random.randint(-100, 100)
#     return [a * n * n + b * n + c for n in range(1, length + 1)]


# def generate_log_sequence(length: int):
#     base = random.randint(1, 100)
#     return [round(math.log(n + base), 2) for n in range(1, length + 1)]


# def generate_prime_sequence(length: int):
#     start_index = random.randint(0, len(PRIMES) - length)
#     return PRIMES[start_index:start_index + length]


# def generate_collatz_sequence(length: int):
#     n = random.randint(50, 5000)
#     seq = [n]
#     while len(seq) < length:
#         n = n // 2 if n % 2 == 0 else 3 * n + 1
#         seq.append(n)
#     return seq


# GENERATORS = [
#     ("constant", generate_constant_sequence),
#     ("linear", generate_linear_sequence),
#     ("quadratic", generate_quadratic_sequence),
#     ("logarithmic", generate_log_sequence),
#     ("prime", generate_prime_sequence),
#     ("collatz", generate_collatz_sequence),
# ]


# def create_sequence_dataset(num_sequences_per_type: int = 2000, length: int = 40):
#     """
#     Generate a dataset of sequences from each generator.
#     """
#     dataset = []
#     for _ in range(num_sequences_per_type):
#         for seq_type, generator in GENERATORS:
#             sequence = generator(length)
#             dataset.append({
#                 "type": seq_type,
#                 "sequence": sequence  # ← type removed from the saved list
#             })
#     return dataset


# def save_dataset_text(dataset, filename: str = "seq_dataset.txt"):
#     """
#     Save the dataset as a multi-line text file (no JSON).
#     Each line = comma-separated numbers only.
#     """
#     path = Path(__file__).parent / filename

#     sequences_as_text = "\n".join(
#         ",".join(map(str, item["sequence"]))  # only numbers now
#         for item in dataset
#     )

#     path.write_text(sequences_as_text)
#     print(f"Saved {len(dataset)} sequences to {path}")


# def main():
#     dataset = create_sequence_dataset(num_sequences_per_type=2000, length=40)
#     save_dataset_text(dataset, filename="seq_dataset.txt")


# if __name__ == "__main__":
#     main()



# ----------------------
import math
import random
from pathlib import Path

import sympy  # for primes

PRIMES = list(sympy.primerange(2, 20000))


def generate_constant_sequence(length: int):
    value = random.randint(-500, 500)
    return [value] * length


def generate_linear_sequence(length: int):
    slope = random.randint(-20, 20)
    intercept = random.randint(-100, 100)
    return [slope * n + intercept for n in range(1, length + 1)]


def generate_quadratic_sequence(length: int):
    a = random.randint(-5, 5)
    b = random.randint(-20, 20)
    c = random.randint(-100, 100)
    return [a * n * n + b * n + c for n in range(1, length + 1)]


def generate_log_sequence(length: int):
    base = random.randint(1, 100)
    return [round(math.log(n + base), 2) for n in range(1, length + 1)]


def generate_prime_sequence(length: int):
    start_index = random.randint(0, len(PRIMES) - length)
    return PRIMES[start_index:start_index + length]


def generate_collatz_sequence(length: int):
    n = random.randint(50, 5000)
    seq = [n]
    while len(seq) < length:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq


def generate_arithmetic_sequence(length: int):
    start = random.randint(-200, 200)
    step = random.choice([i for i in range(-20, 21) if i != 0])
    return [start + step * n for n in range(length)]


def generate_fibonacci_sequence(length: int):
    count = min(50, max(30, length))
    seq = [0, 1]
    for _ in range(2, count):
        seq.append(seq[-1] + seq[-2])
    return seq[:count]


def generate_geometric_sequences(length: int):
    start = random.uniform(-50.0, 50.0)
    if abs(start) < 1e-6:
        start = 1.0
    ratio = random.uniform(0.5, 2.0)
    if random.random() < 0.5:
        ratio = -ratio

    seq = [start]
    current = start
    for _ in range(1, length):
        current *= ratio
        seq.append(current)
    return seq


def generate_noisy_sinusoidal_sequence(length: int):
    amplitude = random.uniform(0.5, 5.0)
    frequency = random.uniform(0.5, 3.0)
    phase = random.uniform(0, 2 * math.pi)
    noise_scale = amplitude * 0.05
    return [
        round(
            amplitude * math.sin((2 * math.pi * frequency * n) / length + phase)
            + random.uniform(-noise_scale, noise_scale),
            4,
        )
        for n in range(length)
    ]


GENERATORS = [
    ("constant", generate_constant_sequence),
    ("linear", generate_linear_sequence),
    ("quadratic", generate_quadratic_sequence),
    ("logarithmic", generate_log_sequence),
    ("prime", generate_prime_sequence),
    ("collatz", generate_collatz_sequence),
    ("arithmetic", generate_arithmetic_sequence),
    ("fibonacci", generate_fibonacci_sequence),
    ("noisy_sinusoidal", generate_noisy_sinusoidal_sequence),
    ("geometric", generate_geometric_sequences)
]

SEQUENCE_TAGS = {name: idx for idx, (name, _) in enumerate(GENERATORS)}

def create_sequence_dataset(num_sequences_per_type: int = 2000, length: int = 40):
    """
    Generate a dataset of sequences from each generator.
    """
    dataset = []
    for _ in range(num_sequences_per_type):
        for seq_type, generator in GENERATORS:
            sequence = generator(length)
            tagged_sequence = [SEQUENCE_TAGS[seq_type]] + sequence
            dataset.append({
                "type": seq_type,
                "sequence": tagged_sequence
            })
    return dataset

def save_dataset_text(dataset, filename: str = "seq_dataset.txt"):
    """
    Save the dataset as a multi-line text file.
    Each line = seq_type followed by comma-separated numbers.
    """
    path = Path(__file__).parent / filename

    sequences_as_text = "\n".join(
        f'{item["type"]},' + ",".join(map(str, item["sequence"])) for item in dataset
    )

    path.write_text(sequences_as_text)
    print(f"Saved {len(dataset)} sequences to {path}")


def main():
    dataset = create_sequence_dataset(num_sequences_per_type=2000, length=40)
    save_dataset_text(dataset, filename="seq_dataset.txt")


if __name__ == "__main__":
    main()