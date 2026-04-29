#### understand major numerical formats

ref - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

| Format | Total Bits | Sign Bit (S) | Exponent (E) | Mantissa / Fraction (M) | Numerical Range (Approx.) | Memory Usage (Relative to FP32) | Primary Uses and Characteristics |
|-------|------------|--------------|--------------|--------------------------|----------------------------|----------------------------------|---------------------------------|
| FP32  | 32 bits    | 1 bit        | 8 bits       | 23 bits                  | ±3.4 × 10³⁸               | 1x (baseline)                   | high precision, baseline for training and inference |
| FP16  | 16 bits    | 1 bit        | 5 bits       | 10 bits                  | ±6.5 × 10⁴                | 50%                             | faster training and inference, limited dynamic range, usually needs loss scaling |
| BF16  | 16 bits    | 1 bit        | 8 bits       | 7 bits                   | ±3.4 × 10³⁸               | 50%                             | used in bf16/fp32 mixed training, wide dynamic range |
| FP8 (E4M3 / E5M2) | 8 bits | 1 bit | 4–5 bits* | 2–3 bits* | ~10²–10³* (varies by format) | 25% | aggressive low-precision training and inference, hardware dependent, needs careful scaling |
| MXFP8 | 8 bits + block scale | 1 bit (per FP8 value) | 4–5 bits (FP8 encoding) | 2–3 bits (FP8 encoding) | ~10²–10³ per value, but extended dynamically via block scaling | ~25% + small overhead for scale storage | microscaled FP8, values share a block scale factor,  higher accuracy than raw FP8 in llm training/inference |



for the tensara problem, we need to convert mxfp8 to fp32

1. we have q of shape `M x K` of mxfp8
2. scale of shape `M x (K / 32)`, this means every block of 32 elements share a scale.
3. K is divisible by 32.


so you could have q = (1000, 64) and k = (1000, 2)