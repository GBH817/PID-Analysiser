"""Bit-level stream reader, ported from betaflight/blackbox-tools (stream.c).

The `bit_pos` semantics match the C implementation exactly:
- bit_pos == 7  -> the stream is byte-aligned
- a bit is read from the current byte at index bit_pos (7 down to 0)
- reading a bit at index 0 advances to the next byte and resets bit_pos to 7
"""

from typing import Optional

EOF = -1


class BitStream:
    __slots__ = ("data", "size", "start", "pos", "end", "bit_pos", "eof")

    def __init__(self, data: bytes):
        self.data = data
        self.size = len(data)
        self.start = 0
        self.pos = 0
        self.end = len(data)
        self.bit_pos = 7  # CHAR_BIT - 1 => byte aligned
        self.eof = False

    # ------------------------------------------------------------------ chars
    def peek_char(self) -> int:
        if self.pos < self.end:
            return self.data[self.pos]
        self.eof = True
        return EOF

    def read_byte(self) -> int:
        if self.pos < self.end:
            result = self.data[self.pos]
            self.pos += 1
            return result
        self.eof = True
        return EOF

    def read_char(self) -> int:
        return self.read_byte()

    def unread_char(self) -> None:
        self.pos -= 1

    def read(self, length: int) -> bytes:
        if length > self.end - self.pos:
            length = self.end - self.pos
            self.eof = True
        result = self.data[self.pos:self.pos + length]
        self.pos += length
        return result

    # ------------------------------------------------------------------ bits
    def read_bits(self, num_bits: int) -> int:
        assert num_bits <= 32
        num_bytes = (num_bits + 7) // 8

        if self.pos + num_bytes <= self.end:
            result = 0
            while num_bits > 0:
                result |= ((self.data[self.pos] >> self.bit_pos) & 0x01) << (num_bits - 1)
                if self.bit_pos == 0:
                    self.pos += 1
                    self.bit_pos = 7
                else:
                    self.bit_pos -= 1
                num_bits -= 1
            return result

        self.pos = self.end
        self.eof = True
        self.bit_pos = 7
        return EOF

    def read_bit(self) -> int:
        return self.read_bits(1)

    def byte_align(self) -> None:
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.pos += 1

    # ------------------------------------------------------------- variable-byte
    def read_unsigned_vb(self) -> int:
        """Read a little-endian 7-bit-grouped variable-byte integer (max 5 bytes)."""
        data = self.data
        pos = self.pos
        end = self.end
        result = 0
        shift = 0
        for _ in range(5):
            if pos < end:
                c = data[pos]
                pos += 1
            else:
                self.pos = end
                self.eof = True
                return 0
            result |= (c & 0x7F) << shift
            if c < 128:
                self.pos = pos
                return result
            shift += 7
        self.pos = pos
        # This VB-encoded int is too long
        return 0

    def read_signed_vb(self) -> int:
        """Read a ZigZag-encoded signed variable-byte integer (max 5 bytes)."""
        data = self.data
        pos = self.pos
        end = self.end
        result = 0
        shift = 0
        for _ in range(5):
            if pos < end:
                c = data[pos]
                pos += 1
            else:
                self.pos = end
                self.eof = True
                return 0
            result |= (c & 0x7F) << shift
            if c < 128:
                self.pos = pos
                # zigzag_decode(result)
                return (result >> 1) ^ -(result & 1)
            shift += 7
        self.pos = pos
        # This VB-encoded int is too long
        return 0
