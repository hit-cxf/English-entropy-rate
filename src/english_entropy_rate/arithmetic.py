from __future__ import annotations

from bisect import bisect_right


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3
STATE_MASK = FULL_RANGE - 1


class BitOutput:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0
        self._bit_count = 0

    def write(self, bit: int) -> None:
        self._current_byte = (self._current_byte << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_count = 0

    def finish(self) -> bytes:
        if self._bit_count:
            self._buffer.append(self._current_byte << (8 - self._bit_count))
            self._current_byte = 0
            self._bit_count = 0
        return bytes(self._buffer)


class BitInput:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._byte_index = 0
        self._bit_index = 0

    def read(self) -> int:
        if self._byte_index >= len(self._payload):
            return 0
        bit = (self._payload[self._byte_index] >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._byte_index += 1
            self._bit_index = 0
        return bit


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = STATE_MASK
        self._pending_bits = 0
        self._output = BitOutput()

    def write(self, cumulative: list[int], symbol: int) -> None:
        total = cumulative[-1]
        low_count = cumulative[symbol]
        high_count = cumulative[symbol + 1]
        if not 0 <= low_count < high_count <= total:
            raise ValueError("invalid cumulative distribution or symbol")

        current_range = self.high - self.low + 1
        self.high = self.low + (current_range * high_count // total) - 1
        self.low = self.low + (current_range * low_count // total)

        while True:
            if self.high < HALF_RANGE:
                self._write_bit_plus_pending(0)
            elif self.low >= HALF_RANGE:
                self._write_bit_plus_pending(1)
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self._pending_bits += 1
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) & STATE_MASK) | 1

    def finish(self) -> bytes:
        self._pending_bits += 1
        if self.low < QUARTER_RANGE:
            self._write_bit_plus_pending(0)
        else:
            self._write_bit_plus_pending(1)
        return self._output.finish()

    def _write_bit_plus_pending(self, bit: int) -> None:
        self._output.write(bit)
        opposite = bit ^ 1
        for _ in range(self._pending_bits):
            self._output.write(opposite)
        self._pending_bits = 0


class ArithmeticDecoder:
    def __init__(self, payload: bytes) -> None:
        self.low = 0
        self.high = STATE_MASK
        self._input = BitInput(payload)
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self._input.read()

    def read(self, cumulative: list[int]) -> int:
        total = cumulative[-1]
        current_range = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // current_range
        symbol = bisect_right(cumulative, value) - 1

        low_count = cumulative[symbol]
        high_count = cumulative[symbol + 1]
        self.high = self.low + (current_range * high_count // total) - 1
        self.low = self.low + (current_range * low_count // total)

        while True:
            if self.high < HALF_RANGE:
                pass
            elif self.low >= HALF_RANGE:
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
                self.code -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
                self.code -= QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) & STATE_MASK) | 1
            self.code = ((self.code << 1) & STATE_MASK) | self._input.read()

        return symbol
