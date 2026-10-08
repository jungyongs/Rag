from pathlib import Path

OUT = Path("data/raw/dummy_1gb.txt")
TARGET_BYTES = 1024 * 1024 * 100  # 100 MiB

LINE = (
    "The patient presented for follow-up. Clinical history, medications, laboratory findings, "
    "treatment response, diagnoses, symptoms, and possible adverse events were reviewed. "
    "The care team documented the assessment and plan for continued management.\n"
).encode("utf-8")

OUT.parent.mkdir(parents=True, exist_ok=True)

# Write in ~8 MiB blocks for speed.
block = LINE * max(1, (8 * 1024 * 1024) // len(LINE))

written = 0
with OUT.open("wb") as f:
    while written + len(block) <= TARGET_BYTES:
        f.write(block)
        written += len(block)

    remaining = TARGET_BYTES - written
    if remaining > 0:
        full_repeats, tail = divmod(remaining, len(LINE))

        if full_repeats:
            f.write(LINE * full_repeats)

        if tail:
            f.write(LINE[:tail])

size = OUT.stat().st_size

print(f"Created: {OUT.resolve()}")
print(f"Size: {size:,} bytes")
print(f"GiB: {size / (1024 ** 3):.3f}")
