"""Field encodings, ported from betaflight/blackbox-tools (decoders.c)."""

import struct

from .bitstream import BitStream, EOF
from .tools import (
    sign_extend_2bit, sign_extend_4bit, sign_extend_6bit, sign_extend_24bit,
    zigzag_decode,
)


def read_tag2_3s32(stream: BitStream):
    """TAG2_3S32: 3 signed 32-bit values packed with a 2-bit tag per value."""
    values = [0, 0, 0]
    lead_byte = stream.read_byte()

    selector = lead_byte >> 6
    if selector == 0:
        # 2-bit fields
        values[0] = sign_extend_2bit((lead_byte >> 4) & 0x03)
        values[1] = sign_extend_2bit((lead_byte >> 2) & 0x03)
        values[2] = sign_extend_2bit(lead_byte & 0x03)
    elif selector == 1:
        # 4-bit fields
        values[0] = sign_extend_4bit(lead_byte & 0x0F)
        lead_byte = stream.read_byte()
        values[1] = sign_extend_4bit(lead_byte >> 4)
        values[2] = sign_extend_4bit(lead_byte & 0x0F)
    elif selector == 2:
        # 6-bit fields
        values[0] = sign_extend_6bit(lead_byte & 0x3F)
        lead_byte = stream.read_byte()
        values[1] = sign_extend_6bit(lead_byte & 0x3F)
        lead_byte = stream.read_byte()
        values[2] = sign_extend_6bit(lead_byte & 0x3F)
    else:
        # 8/16/24/32-bit fields, selector decides per-value width
        for i in range(3):
            width = lead_byte & 0x03
            if width == 0:
                byte1 = stream.read_byte()
                values[i] = byte1 - 0x100 if byte1 & 0x80 else byte1  # (int8_t)
            elif width == 1:
                byte1 = stream.read_byte()
                byte2 = stream.read_byte()
                raw = byte1 | (byte2 << 8)
                values[i] = raw - 0x10000 if raw & 0x8000 else raw  # (int16_t)
            elif width == 2:
                byte1 = stream.read_byte()
                byte2 = stream.read_byte()
                byte3 = stream.read_byte()
                values[i] = sign_extend_24bit(byte1 | (byte2 << 8) | (byte3 << 16))
            else:
                byte1 = stream.read_byte()
                byte2 = stream.read_byte()
                byte3 = stream.read_byte()
                byte4 = stream.read_byte()
                raw = byte1 | (byte2 << 8) | (byte3 << 16) | (byte4 << 24)
                values[i] = raw - 0x100000000 if raw & 0x80000000 else raw  # (int32_t)
            lead_byte >>= 2

    return values


def read_tag8_4s16_v1(stream: BitStream):
    """TAG8_4S16 (data version < 2): 4 signed 16-bit values, 2-bit tag each."""
    values = [0, 0, 0, 0]
    selector = stream.read_byte()

    i = 0
    while i < 4:
        tag = selector & 0x03
        if tag == 0:
            values[i] = 0
        elif tag == 1:
            combined = stream.read_byte()
            values[i] = sign_extend_4bit(combined & 0x0F)
            i += 1
            selector >>= 2
            values[i] = sign_extend_4bit(combined >> 4)
        elif tag == 2:
            byte = stream.read_byte()
            values[i] = byte - 0x100 if byte & 0x80 else byte  # (int8_t)
        else:
            char1 = stream.read_byte()
            char2 = stream.read_byte()
            raw = char1 | (char2 << 8)
            values[i] = raw - 0x10000 if raw & 0x8000 else raw  # (int16_t)
        selector >>= 2
        i += 1

    return values


def read_tag8_4s16_v2(stream: BitStream):
    """TAG8_4S16 (data version >= 2): 4 signed 16-bit values, nibble-packed."""
    values = [0, 0, 0, 0]
    selector = stream.read_byte()
    buffer = 0
    nibble_index = 0

    for i in range(4):
        tag = selector & 0x03
        if tag == 0:
            values[i] = 0
        elif tag == 1:
            if nibble_index == 0:
                buffer = stream.read_byte()
                values[i] = sign_extend_4bit(buffer >> 4)
                nibble_index = 1
            else:
                values[i] = sign_extend_4bit(buffer & 0x0F)
                nibble_index = 0
        elif tag == 2:
            if nibble_index == 0:
                byte = stream.read_byte()
                values[i] = byte - 0x100 if byte & 0x80 else byte  # (int8_t)
            else:
                char1 = (buffer << 4) & 0xFF  # C: uint8_t char1 = buffer << 4
                buffer = stream.read_byte()
                char1 |= buffer >> 4
                values[i] = char1 - 0x100 if char1 & 0x80 else char1  # (int8_t)
        else:
            if nibble_index == 0:
                char1 = stream.read_byte()
                char2 = stream.read_byte()
                raw = (char1 << 8) | char2
                values[i] = raw - 0x10000 if raw & 0x8000 else raw  # (int16_t)
            else:
                char1 = stream.read_byte()
                char2 = stream.read_byte()
                raw = ((buffer << 12) | (char1 << 4) | (char2 >> 4)) & 0xFFFF  # C: (uint16_t) cast
                values[i] = raw - 0x10000 if raw & 0x8000 else raw  # (int16_t)
                buffer = char2
        selector >>= 2

    return values


def read_tag8_8svb(stream: BitStream, value_count: int):
    """TAG8_8SVB: 8 values, each either zero or a signed VB integer."""
    values = [0] * 8
    if value_count == 1:
        values[0] = stream.read_signed_vb()
    else:
        header = stream.read_byte()
        for i in range(8):
            if header & 0x01:
                values[i] = stream.read_signed_vb()
            header >>= 1
    return values


def read_raw_float(stream: BitStream) -> float:
    """Read a little-endian IEEE-754 single-precision float."""
    raw = bytes([stream.read_byte(), stream.read_byte(), stream.read_byte(), stream.read_byte()])
    return struct.unpack("<f", raw)[0]


# ---------------------------------------------------------------------------
# Elias codes
# ---------------------------------------------------------------------------

def read_elias_delta_u32(stream: BitStream) -> int:
    """Elias-Delta encoded unsigned 32-bit integer."""
    max_bit_read_size = 32

    length_val_bits = 0
    while length_val_bits <= max_bit_read_size and stream.read_bit() == 0:
        length_val_bits += 1

    if stream.eof or length_val_bits > max_bit_read_size:
        return 0

    length_low_bits = stream.read_bits(length_val_bits)
    if stream.eof:
        return 0

    length = ((1 << length_val_bits) | length_low_bits) - 1
    if length > max_bit_read_size:
        return 0

    result_low_bits = stream.read_bits(length)
    if stream.eof:
        return 0

    result = (1 << length) | result_low_bits

    # The highest value is an escape code meaning MAXINT-1 or MAXINT
    if result == 0xFFFFFFFF:
        escape_val = stream.read_bit()
        if escape_val == 0:
            return 0xFFFFFFFF - 1
        elif escape_val == 1:
            return 0xFFFFFFFF
        return 0

    return result - 1


def read_elias_delta_s32(stream: BitStream) -> int:
    return zigzag_decode(read_elias_delta_u32(stream))


def read_elias_gamma_u32(stream: BitStream) -> int:
    """Elias-Gamma encoded unsigned 32-bit integer."""
    max_bit_read_size = 32

    val_bits = 0
    while val_bits <= max_bit_read_size and stream.read_bit() == 0:
        val_bits += 1

    if stream.eof or val_bits > max_bit_read_size:
        return 0

    value_low_bits = stream.read_bits(val_bits - 1)
    if stream.eof:
        return 0

    result = (1 << (val_bits - 1)) | value_low_bits

    if result == 0xFFFFFFFF:
        escape_val = stream.read_bit()
        if escape_val == 0:
            return 0xFFFFFFFF - 1
        elif escape_val == 1:
            return 0xFFFFFFFF
        return 0

    return result - 1


def read_elias_gamma_s32(stream: BitStream) -> int:
    return zigzag_decode(read_elias_gamma_u32(stream))
