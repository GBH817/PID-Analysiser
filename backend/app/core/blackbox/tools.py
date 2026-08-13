"""Small bit-manipulation helpers ported from betaflight/blackbox-tools (tools.c)."""


def int32(value: int) -> int:
    """Wrap a value into a signed 32-bit integer (sign-extends the low 32 bits)."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def uint32(value: int) -> int:
    """Wrap a value into an unsigned 32-bit integer."""
    return value & 0xFFFFFFFF


def zigzag_decode(value: int) -> int:
    """Decode a ZigZag-encoded signed integer."""
    return (value >> 1) ^ -(value & 1)


def zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def sign_extend_2bit(byte: int) -> int:
    # If the sign bit is set, fill the top bits with 1s to sign-extend
    return int32(byte | 0xFFFFFFFC) if byte & 0x02 else byte


def sign_extend_4bit(nibble: int) -> int:
    return int32(nibble | 0xFFFFFFF0) if nibble & 0x08 else nibble


def sign_extend_6bit(byte: int) -> int:
    return int32(byte | 0xFFFFFFC0) if byte & 0x20 else byte


def sign_extend_14bit(word: int) -> int:
    return int32(word | 0xFFFFC000) if word & 0x2000 else word


def sign_extend_24bit(u: int) -> int:
    return int32(u | 0xFF000000) if u & 0x800000 else u


def div_trunc(num: int, den: int) -> int:
    """C-style integer division that truncates toward zero."""
    q = abs(num) // abs(den)
    return q if (num < 0) == (den < 0) else -q


def starts_with(string: str, prefix: str) -> bool:
    return string.startswith(prefix)


def ends_with(string: str, suffix: str) -> bool:
    return string.endswith(suffix)
