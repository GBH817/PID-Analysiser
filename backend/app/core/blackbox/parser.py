"""Betaflight Blackbox (.bbl) flight log parser.

A faithful Python port of the official decoder
(betaflight/blackbox-tools: parser.c / stream.c / decoders.c / tools.c).
"""

import math
import re
import struct
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .constants import *
from .bitstream import BitStream, EOF
from .tools import (
    int32, uint32, div_trunc, starts_with, ends_with,
    sign_extend_14bit,
)
from .encodings import (
    read_tag2_3s32, read_tag8_4s16_v1, read_tag8_4s16_v2, read_tag8_8svb,
    read_elias_delta_u32, read_elias_delta_s32,
    read_elias_gamma_u32, read_elias_gamma_s32, read_raw_float,
)


class BlackboxError(Exception):
    """Raised when a log cannot be parsed at all (bad file, missing defs...)."""


# ---------------------------------------------------------------------------
# Frame / field definitions
# ---------------------------------------------------------------------------

class FrameDef:
    __slots__ = ("frame_type", "field_count", "field_names", "field_signed",
                 "field_width", "predictor", "encoding")

    def __init__(self, frame_type: int):
        self.frame_type = frame_type
        self.field_count = 0
        self.field_names: List[str] = []
        self.field_signed = [0] * FLIGHT_LOG_MAX_FIELDS
        self.field_width = [4] * FLIGHT_LOG_MAX_FIELDS
        self.predictor = [0] * FLIGHT_LOG_MAX_FIELDS
        self.encoding = [0] * FLIGHT_LOG_MAX_FIELDS


class MainFieldIndexes:
    def __init__(self):
        self.loop_iteration = -1
        self.time = -1
        self.pid = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]  # [P, I, D][axis]
        self.rc_command = [-1] * 4
        self.vbat_latest = -1
        self.amperage_latest = -1
        self.mag_adc = [-1] * 3
        self.baro_alt = -1
        self.sonar_raw = -1
        self.rssi = -1
        self.gyro_adc = [-1] * 3
        self.acc_smooth = [-1] * 3
        self.motor = [-1] * FLIGHT_LOG_MAX_MOTORS
        self.servo = [-1] * FLIGHT_LOG_MAX_SERVOS


class GpsGFieldIndexes:
    def __init__(self):
        self.time = -1
        self.gps_num_sat = -1
        self.gps_coord = [-1, -1]
        self.gps_altitude = -1
        self.gps_speed = -1
        self.gps_ground_course = -1


class GpsHFieldIndexes:
    def __init__(self):
        self.gps_home = [-1, -1]


class SlowFieldIndexes:
    def __init__(self):
        self.flight_mode_flags = -1
        self.state_flags = -1
        self.failsafe_phase = -1


class SysConfig:
    def __init__(self):
        self.minthrottle = 1150
        self.maxthrottle = 1850
        self.motor_output_low = 1150
        self.motor_output_high = 1850
        self.rc_rate = 90
        self.yaw_rate = 0
        self.acc_1g = 1
        self.gyro_scale = 1.0
        self.vbatscale = 110
        self.vbatref = 4095
        self.vbatmincellvoltage = 33
        self.vbatmaxcellvoltage = 43
        self.vbatwarningcellvoltage = 35
        self.current_meter_offset = 0
        self.current_meter_scale = 400
        self.firmware_type = FIRMWARE_TYPE_UNKNOWN
        # PID / filter settings (Betaflight 4.x logs these in the header)
        self.pid_roll = [0, 0, 0]   # [P, I, D]
        self.pid_pitch = [0, 0, 0]
        self.pid_yaw = [0, 0, 0]
        self.d_min = [0, 0, 0]      # roll, pitch, yaw
        self.ff_weight = [0, 0, 0]  # roll, pitch, yaw
        self.dterm_lpf1_static_hz = 0
        self.dterm_lpf2_static_hz = 0
        self.gyro_lpf1_static_hz = 0
        self.gyro_lpf2_static_hz = 0
        self.yaw_lowpass_hz = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class FlightLogParser:
    """Parses one contiguous log section (header + data) from raw bytes."""

    def __init__(self, data: bytes, raw: bool = False, log_start: int = 0, log_end: Optional[int] = None):
        self.data = data
        self.raw = raw
        self.stream = BitStream(data)
        if log_start:
            self.stream.start = log_start
            self.stream.pos = log_start
        if log_end is not None:
            self.stream.end = log_end

        self.frame_defs: Dict[int, FrameDef] = {c: FrameDef(c) for c in range(256)}
        self.sys_config = SysConfig()
        self.main_field_indexes = MainFieldIndexes()
        self.gps_field_indexes = GpsGFieldIndexes()
        self.gps_home_field_indexes = GpsHFieldIndexes()
        self.slow_field_indexes = SlowFieldIndexes()

        self.frame_interval_i = 32
        self.frame_interval_p_num = 1
        self.frame_interval_p_denom = 1
        self.data_version = 0
        self.fc_version = ""
        self.date_time: Optional[datetime] = None

        # ---- private parse state (mirrors flightLogPrivate_t) ----
        self._time_rollover_accumulator = 0
        self._last_main_frame_iteration = uint32(-1)
        self._last_main_frame_time = -1
        self._last_skipped_frames = 0
        self.main_stream_is_valid = False
        self.gps_home_is_valid = False

        self._cur = [0] * FLIGHT_LOG_MAX_FIELDS
        self._prev: Optional[List[int]] = None
        self._prev2: Optional[List[int]] = None
        self._gps_home_cur = [0] * FLIGHT_LOG_MAX_FIELDS
        self._gps_home_prev: Optional[List[int]] = None
        self._last_gps = [0] * FLIGHT_LOG_MAX_FIELDS
        self._last_slow = [0] * FLIGHT_LOG_MAX_FIELDS

        self._last_event_type = -1
        self._last_event_data: dict = {}
        self._parser_state = PARSER_STATE_HEADER

        # ---- statistics ----
        self._field_stats = []           # [min, max] per field, [] until first frame
        self._have_field_stats = False
        self._frame_stats: Dict[int, dict] = {}
        for ft in FRAME_TYPES:
            self._frame_stats[ft] = {"bytes": 0, "valid_count": 0,
                                     "corrupt_count": 0, "desync_count": 0}
        self._total_corrupt_frames = 0
        self._intentionally_absent_iterations = 0

        # ---- collected output ----
        self.field_names: List[str] = []
        self.main_frames: Optional[np.ndarray] = None   # (N, field_count) int64
        self._n_frames = 0
        self.slow_frames: List[dict] = []
        self.gps_frames: List[dict] = []
        self.events: List[dict] = []

    # ------------------------------------------------------------ header lines
    def _parse_header_line(self):
        if self.stream.read_byte() != ord('H'):
            return 0
        if self.stream.read_byte() != ord(' '):
            return 1

        line_start = self.stream.pos
        value_buffer = bytearray()
        separator_pos = None
        i = 0
        while i < FLIGHT_LOG_MAX_FRAME_HEADER_LENGTH:
            c = self.stream.read_char()
            if c == ord(':') and separator_pos is None:
                separator_pos = self.stream.pos - 1
            if c == ord('\n'):
                i += 1  # size includes the newline
                break
            if c == EOF or c == 0:
                return i  # line ended before newline / contains binary junk
            value_buffer.append(c)
            i += 1
        frame_size = i + 2  # we have read two bytes ('H', ' ') previously

        if separator_pos is None:
            return frame_size

        line_end = self.stream.pos

        # Generic end-of-header detection: if the next char is not another 'H',
        # the ASCII header block is finished -> request a state transition.
        next_char = self.stream.peek_char()
        if next_char != ord('H'):
            self._parser_state = PARSER_STATE_TRANSITION

        line = bytes(value_buffer)
        sep_offset = separator_pos - line_start
        field_name = line[:sep_offset].decode("utf-8", "replace")
        field_value = line[sep_offset + 1: line_end - line_start - 1]

        self._handle_header_field(field_name, field_value)
        return frame_size

    def _handle_header_field(self, field_name: str, field_value: bytes):
        if starts_with(field_name, "Field "):
            frame_type = ord(field_name[len("Field ")])
            frame_def = self.frame_defs[frame_type]

            if ends_with(field_name, " name"):
                names = [n.decode("utf-8", "replace") for n in field_value.split(b",")]
                frame_def.field_names = names
                frame_def.field_count = len(names)
                self._identify_fields(frame_type, frame_def)
                if frame_type == FRAME_MAIN:
                    # P frames are derived from I frames, copy common data over
                    p = self.frame_defs[FRAME_INTER]
                    p.field_names = list(names)
                    p.field_count = len(names)
            elif ends_with(field_name, " signed"):
                self._parse_int_list(field_value, frame_def.field_signed)
                if frame_type == FRAME_MAIN:
                    self.frame_defs[FRAME_INTER].field_signed = list(frame_def.field_signed)
            elif ends_with(field_name, " predictor"):
                self._parse_int_list(field_value, frame_def.predictor)
            elif ends_with(field_name, " encoding"):
                self._parse_int_list(field_value, frame_def.encoding)
        elif field_name == "I interval":
            v = self._atoi(field_value)
            self.frame_interval_i = v if v >= 1 else 1
        elif field_name == "P interval":
            value_str = field_value.decode("utf-8", "replace")
            if "/" in value_str:
                num_str, denom_str = value_str.split("/", 1)
                self.frame_interval_p_num = self._atoi(num_str.encode())
                self.frame_interval_p_denom = self._atoi(denom_str.encode())
        elif field_name == "Data version":
            self.data_version = self._atoi(field_value)
        elif field_name == "Firmware type":
            value_str = field_value.decode("utf-8", "replace")
            if value_str == "Cleanflight":
                self.sys_config.firmware_type = FIRMWARE_TYPE_CLEANFLIGHT
            else:
                self.sys_config.firmware_type = FIRMWARE_TYPE_BASEFLIGHT
        elif field_name == "Firmware revision":
            value_str = field_value.decode("utf-8", "replace")
            parts = value_str.split(" ")
            if parts and parts[0] == "Betaflight":
                self.fc_version = parts[1] if len(parts) > 1 else ""
        elif field_name == "minthrottle":
            self.sys_config.minthrottle = self._atoi(field_value)
            self.sys_config.motor_output_low = self.sys_config.minthrottle
        elif field_name == "maxthrottle":
            self.sys_config.maxthrottle = self._atoi(field_value)
            self.sys_config.motor_output_high = self.sys_config.maxthrottle
        elif field_name == "rcRate":
            self.sys_config.rc_rate = self._atoi(field_value)
        elif field_name == "vbatscale":
            self.sys_config.vbatscale = self._atoi(field_value)
        elif field_name == "vbatref":
            self.sys_config.vbatref = self._atoi(field_value)
        elif field_name == "vbatcellvoltage":
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 3:
                self.sys_config.vbatmincellvoltage = vals[0]
                self.sys_config.vbatwarningcellvoltage = vals[1]
                self.sys_config.vbatmaxcellvoltage = vals[2]
        elif field_name == "currentMeter":
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 2:
                self.sys_config.current_meter_offset = vals[0]
                self.sys_config.current_meter_scale = vals[1]
        elif field_name in ("gyro.scale", "gyro_scale"):
            try:
                self.sys_config.gyro_scale = struct.unpack("<f", struct.pack("<I", int(field_value, 16)))[0]
            except (ValueError, struct.error):
                self.sys_config.gyro_scale = 1.0
            if self.sys_config.firmware_type != FIRMWARE_TYPE_BASEFLIGHT:
                self.sys_config.gyro_scale = self.sys_config.gyro_scale * (math.pi / 180.0) * 0.000001
        elif field_name == "acc_1G":
            self.sys_config.acc_1g = self._atoi(field_value)
        elif field_name == "motorOutput":
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 2:
                self.sys_config.motor_output_low = vals[0]
                self.sys_config.motor_output_high = vals[1]
        elif field_name in ("rollPID", "pitchPID", "yawPID"):
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 3:
                target = {"rollPID": self.sys_config.pid_roll,
                          "pitchPID": self.sys_config.pid_pitch,
                          "yawPID": self.sys_config.pid_yaw}[field_name]
                target[:] = vals[:3]
        elif field_name == "d_min":
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 3:
                self.sys_config.d_min[:] = vals[:3]
        elif field_name == "ff_weight":
            vals = [self._atoi(x) for x in field_value.split(b",") if x]
            if len(vals) >= 3:
                self.sys_config.ff_weight[:] = vals[:3]
        elif field_name == "dterm_lpf1_static_hz":
            self.sys_config.dterm_lpf1_static_hz = self._atoi(field_value)
        elif field_name == "dterm_lpf2_static_hz":
            self.sys_config.dterm_lpf2_static_hz = self._atoi(field_value)
        elif field_name == "gyro_lpf1_static_hz":
            self.sys_config.gyro_lpf1_static_hz = self._atoi(field_value)
        elif field_name == "gyro_lpf2_static_hz":
            self.sys_config.gyro_lpf2_static_hz = self._atoi(field_value)
        elif field_name == "yaw_lowpass_hz":
            self.sys_config.yaw_lowpass_hz = self._atoi(field_value)
        elif starts_with(field_name, "Log start datetime"):
            self.date_time = self._parse_datetime(field_value.decode("utf-8", "replace"))

    @staticmethod
    def _atoi(b: bytes) -> int:
        s = b.decode("utf-8", "replace").strip()
        try:
            return int(float(s)) if re.search(r"[.eE]", s) else int(s, 10)
        except ValueError:
            return 0

    @staticmethod
    def _parse_int_list(b: bytes, target: list):
        vals = [FlightLogParser._atoi(x) for x in b.split(b",") if x]
        n = min(len(vals), FLIGHT_LOG_MAX_FIELDS)
        target[:n] = vals[:n]
        for i in range(n, FLIGHT_LOG_MAX_FIELDS):
            target[i] = 0

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)", value)
        if m:
            try:
                return datetime(*[int(x) for x in m.groups()], tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    # ---------------------------------------------------------- field idents
    def _identify_fields(self, frame_type: int, frame_def: FrameDef):
        if frame_type == FRAME_MAIN:
            self._identify_main_fields(frame_def)
        elif frame_type == FRAME_GPS:
            self._identify_gps_fields(frame_def)
        elif frame_type == FRAME_GPS_HOME:
            self._identify_gps_home_fields(frame_def)
        elif frame_type == FRAME_SLOW:
            self._identify_slow_fields(frame_def)

    def _identify_main_fields(self, frame_def: FrameDef):
        idx = self.main_field_indexes
        for i, name in enumerate(frame_def.field_names):
            if starts_with(name, "motor["):
                m = self._bracket_index(name, "motor[")
                if 0 <= m < FLIGHT_LOG_MAX_MOTORS:
                    idx.motor[m] = i
            elif starts_with(name, "rcCommand["):
                m = self._bracket_index(name, "rcCommand[")
                if 0 <= m < 4:
                    idx.rc_command[m] = i
            elif starts_with(name, "axis"):
                kind = name[4] if len(name) > 4 else ""
                row = {"P": 0, "I": 1, "D": 2}.get(kind)
                if row is not None:
                    axis = self._bracket_index(name, "axis" + kind + "[")
                    if 0 <= axis < 3:
                        idx.pid[row][axis] = i
            elif starts_with(name, "gyroData["):
                axis = self._bracket_index(name, "gyroData[")
                if 0 <= axis < 3:
                    idx.gyro_adc[axis] = i
            elif starts_with(name, "gyroADC["):
                axis = self._bracket_index(name, "gyroADC[")
                if 0 <= axis < 3:
                    idx.gyro_adc[axis] = i
            elif starts_with(name, "magADC["):
                axis = self._bracket_index(name, "magADC[")
                if 0 <= axis < 3:
                    idx.mag_adc[axis] = i
            elif starts_with(name, "accSmooth["):
                axis = self._bracket_index(name, "accSmooth[")
                if 0 <= axis < 3:
                    idx.acc_smooth[axis] = i
            elif starts_with(name, "servo["):
                s = self._bracket_index(name, "servo[")
                if 0 <= s < FLIGHT_LOG_MAX_SERVOS:
                    idx.servo[s] = i
            elif name == "vbatLatest":
                idx.vbat_latest = i
            elif name == "amperageLatest":
                idx.amperage_latest = i
            elif name == "BaroAlt":
                idx.baro_alt = i
            elif name == "sonarRaw":
                idx.sonar_raw = i
            elif name == "rssi":
                idx.rssi = i
            elif name == "loopIteration":
                idx.loop_iteration = i
            elif name == "time":
                idx.time = i

    @staticmethod
    def _bracket_index(name: str, prefix: str) -> int:
        inner = name[len(prefix):]
        inner = inner[:-1] if inner.endswith("]") else inner
        try:
            return int(inner)
        except ValueError:
            return -1

    def _identify_gps_fields(self, frame_def: FrameDef):
        idx = self.gps_field_indexes
        for i, name in enumerate(frame_def.field_names):
            if name == "time":
                idx.time = i
            elif name == "GPS_numSat":
                idx.gps_num_sat = i
            elif name == "GPS_altitude":
                idx.gps_altitude = i
            elif name == "GPS_speed":
                idx.gps_speed = i
            elif name == "GPS_ground_course":
                idx.gps_ground_course = i
            elif starts_with(name, "GPS_coord["):
                c = self._bracket_index(name, "GPS_coord[")
                if 0 <= c < 2:
                    idx.gps_coord[c] = i

    def _identify_gps_home_fields(self, frame_def: FrameDef):
        idx = self.gps_home_field_indexes
        for i, name in enumerate(frame_def.field_names):
            if name == "GPS_home[0]":
                idx.gps_home[0] = i
            elif name == "GPS_home[1]":
                idx.gps_home[1] = i

    def _identify_slow_fields(self, frame_def: FrameDef):
        idx = self.slow_field_indexes
        for i, name in enumerate(frame_def.field_names):
            if name == "flightModeFlags":
                idx.flight_mode_flags = i
            elif name == "stateFlags":
                idx.state_flags = i
            elif name == "failsafePhase":
                idx.failsafe_phase = i

    # ------------------------------------------------------------ sampling
    def _should_have_frame(self, frame_index: int) -> bool:
        return ((frame_index % self.frame_interval_i + self.frame_interval_p_num - 1)
                % self.frame_interval_p_denom < self.frame_interval_p_num)

    def _count_intentionally_skipped_frames(self) -> int:
        if self._last_main_frame_iteration == uint32(-1):
            return 0
        count = 0
        frame_index = self._last_main_frame_iteration + 1
        while not self._should_have_frame(frame_index):
            count += 1
            frame_index += 1
        return count

    def _count_intentionally_skipped_frames_to(self, target_iteration: int) -> int:
        if self._last_main_frame_iteration == uint32(-1):
            return 0
        count = 0
        for frame_index in range(self._last_main_frame_iteration + 1, target_iteration):
            if not self._should_have_frame(frame_index):
                count += 1
        return count

    # ----------------------------------------------------------- prediction
    def _apply_prediction(self, field_index: int, predictor: int, value: int,
                          current: List[int], previous: Optional[List[int]],
                          previous2: Optional[List[int]]) -> int:
        if predictor == PREDICTOR_0:
            pass
        elif predictor == PREDICTOR_MINTHROTTLE:
            value += self.sys_config.minthrottle
        elif predictor == PREDICTOR_1500:
            value += 1500
        elif predictor == PREDICTOR_MOTOR_0:
            if self.main_field_indexes.motor[0] < 0:
                raise BlackboxError("Attempted to base prediction on motor[0] without that field being defined")
            value += current[self.main_field_indexes.motor[0]]
        elif predictor == PREDICTOR_VBATREF:
            value += self.sys_config.vbatref
        elif predictor == PREDICTOR_PREVIOUS:
            if previous is not None:
                value += previous[field_index]
        elif predictor == PREDICTOR_STRAIGHT_LINE:
            if previous is not None and previous2 is not None:
                value += 2 * previous[field_index] - previous2[field_index]
        elif predictor == PREDICTOR_AVERAGE_2:
            if previous is not None and previous2 is not None:
                value += div_trunc(previous[field_index] + previous2[field_index], 2)
        elif predictor == PREDICTOR_HOME_COORD:
            if self.gps_home_field_indexes.gps_home[0] < 0:
                raise BlackboxError("Attempted to base prediction on GPS home position without GPS home frame definition")
            if self._gps_home_prev is not None:
                value += self._gps_home_prev[self.gps_home_field_indexes.gps_home[0]]
        elif predictor == PREDICTOR_HOME_COORD_1:
            if self.gps_home_field_indexes.gps_home[1] < 1:
                raise BlackboxError("Attempted to base prediction on GPS home position without GPS home frame definition")
            if self._gps_home_prev is not None:
                value += self._gps_home_prev[self.gps_home_field_indexes.gps_home[1]]
        elif predictor == PREDICTOR_LAST_MAIN_FRAME_TIME:
            if self._prev is not None:
                value += self._prev[FLIGHT_LOG_FIELD_INDEX_TIME]
        elif predictor == PREDICTOR_MINMOTOR:
            value += self.sys_config.motor_output_low
        else:
            raise BlackboxError(f"Unsupported field predictor {predictor}")
        return value

    # -------------------------------------------------------------- frames
    def _parse_frame(self, frame_type: int, frame: List[int],
                     previous: Optional[List[int]], previous2: Optional[List[int]],
                     skipped_frames: int):
        frame_def = self.frame_defs[frame_type]
        predictor = frame_def.predictor
        encoding = frame_def.encoding
        field_signed = frame_def.field_signed
        field_width = frame_def.field_width

        i = 0
        while i < frame_def.field_count:
            if predictor[i] == PREDICTOR_INC:
                frame[i] = skipped_frames + 1
                if previous is not None:
                    frame[i] += previous[i]
                i += 1
                continue

            enc = encoding[i]
            if enc == ENCODING_SIGNED_VB:
                self.stream.byte_align()
                value = self.stream.read_signed_vb()
            elif enc == ENCODING_UNSIGNED_VB:
                self.stream.byte_align()
                value = self.stream.read_unsigned_vb()
            elif enc == ENCODING_NEG_14BIT:
                self.stream.byte_align()
                value = -sign_extend_14bit(self.stream.read_unsigned_vb())
            elif enc == ENCODING_TAG8_4S16:
                self.stream.byte_align()
                if self.data_version < 2:
                    values = read_tag8_4s16_v1(self.stream)
                else:
                    values = read_tag8_4s16_v2(self.stream)
                for j in range(4):
                    frame[i] = self._apply_prediction(
                        i, PREDICTOR_0 if self.raw else predictor[i], values[j],
                        frame, previous, previous2)
                    i += 1
                continue
            elif enc == ENCODING_TAG2_3S32:
                self.stream.byte_align()
                values = read_tag2_3s32(self.stream)
                for j in range(3):
                    frame[i] = self._apply_prediction(
                        i, PREDICTOR_0 if self.raw else predictor[i], values[j],
                        frame, previous, previous2)
                    i += 1
                continue
            elif enc == ENCODING_TAG8_8SVB:
                self.stream.byte_align()
                j = i + 1
                while j < i + 8 and j < frame_def.field_count:
                    if encoding[j] != ENCODING_TAG8_8SVB:
                        break
                    j += 1
                group_count = j - i
                values = read_tag8_8svb(self.stream, group_count)
                for j in range(group_count):
                    frame[i] = self._apply_prediction(
                        i, PREDICTOR_0 if self.raw else predictor[i], values[j],
                        frame, previous, previous2)
                    i += 1
                continue
            elif enc == ENCODING_ELIAS_DELTA_U32:
                value = read_elias_delta_u32(self.stream)
            elif enc == ENCODING_ELIAS_DELTA_S32:
                value = read_elias_delta_s32(self.stream)
            elif enc == ENCODING_ELIAS_GAMMA_U32:
                value = read_elias_gamma_u32(self.stream)
            elif enc == ENCODING_ELIAS_GAMMA_S32:
                value = read_elias_gamma_s32(self.stream)
            elif enc == ENCODING_NULL:
                value = 0
            else:
                raise BlackboxError(f"Unsupported field encoding {enc}")

            value = self._apply_prediction(
                i, PREDICTOR_0 if self.raw else predictor[i], value,
                frame, previous, previous2)

            if field_width[i] != 8:
                # Assume 32-bit...
                if field_signed[i]:
                    value = int32(value)
                else:
                    value = uint32(value)

            frame[i] = value
            i += 1

        self.stream.byte_align()

    # ------------------------------------------------------------- rollover
    def _apply_main_frame_time_rollover(self):
        self._cur[FLIGHT_LOG_FIELD_INDEX_TIME] = self._detect_and_apply_timestamp_rollover(
            self._cur[FLIGHT_LOG_FIELD_INDEX_TIME])

    def _apply_gps_frame_time_rollover(self):
        time_index = self.gps_field_indexes.time
        if time_index != -1:
            self._last_gps[time_index] = self._detect_and_apply_timestamp_rollover(
                self._last_gps[time_index])

    def _detect_and_apply_timestamp_rollover(self, timestamp: int) -> int:
        if self._last_main_frame_time != -1:
            if (uint32(timestamp) < uint32(self._last_main_frame_time)
                    and uint32(uint32(timestamp) - uint32(self._last_main_frame_time))
                    < MAXIMUM_TIME_JUMP_BETWEEN_FRAMES):
                self._time_rollover_accumulator += 0x100000000
        return uint32(timestamp) + self._time_rollover_accumulator

    # ------------------------------------------------------------ validation
    def _validate_main_frame_values(self) -> bool:
        return (
            uint32(self._cur[FLIGHT_LOG_FIELD_INDEX_ITERATION]) >= self._last_main_frame_iteration
            and uint32(self._cur[FLIGHT_LOG_FIELD_INDEX_ITERATION])
            < self._last_main_frame_iteration + MAXIMUM_ITERATION_JUMP_BETWEEN_FRAMES
            and self._cur[FLIGHT_LOG_FIELD_INDEX_TIME] >= self._last_main_frame_time
            and self._cur[FLIGHT_LOG_FIELD_INDEX_TIME] < self._last_main_frame_time + MAXIMUM_TIME_JUMP_BETWEEN_FRAMES
        )

    def _invalidate_stream(self):
        self.main_stream_is_valid = False
        self._prev = None
        self._prev2 = None

    def _update_main_field_statistics(self, fields: List[int]):
        field_count = self.frame_defs[FRAME_MAIN].field_count
        if not self._have_field_stats:
            self._field_stats = [[fields[i], fields[i]] for i in range(field_count)]
            self._have_field_stats = True
        else:
            for i in range(field_count):
                if fields[i] > self._field_stats[i][1]:
                    self._field_stats[i][1] = fields[i]
                if fields[i] < self._field_stats[i][0]:
                    self._field_stats[i][0] = fields[i]

    # ------------------------------------------------------- complete frames
    def _complete_intraframe(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        self._apply_main_frame_time_rollover()

        if (not self.raw and self._last_main_frame_iteration != uint32(-1)
                and not self._validate_main_frame_values()):
            self._invalidate_stream()
        else:
            self.main_stream_is_valid = True

        if self.main_stream_is_valid:
            self._intentionally_absent_iterations += self._count_intentionally_skipped_frames_to(
                uint32(self._cur[FLIGHT_LOG_FIELD_INDEX_ITERATION]))
            self._last_main_frame_iteration = uint32(self._cur[FLIGHT_LOG_FIELD_INDEX_ITERATION])
            self._last_main_frame_time = self._cur[FLIGHT_LOG_FIELD_INDEX_TIME]
            self._update_main_field_statistics(self._cur)

        self._collect_main_frame()

        if self.main_stream_is_valid:
            # Both previous states become the I-frame (can't look further back)
            self._prev = list(self._cur[:FLIGHT_LOG_MAX_FIELDS])
            self._prev2 = list(self._prev)
            self._cur = [0] * FLIGHT_LOG_MAX_FIELDS

        return self.main_stream_is_valid

    def _complete_interframe(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        self._apply_main_frame_time_rollover()

        if self.main_stream_is_valid and not self.raw and not self._validate_main_frame_values():
            self._invalidate_stream()

        if self.main_stream_is_valid:
            self._last_main_frame_iteration = uint32(self._cur[FLIGHT_LOG_FIELD_INDEX_ITERATION])
            self._last_main_frame_time = self._cur[FLIGHT_LOG_FIELD_INDEX_TIME]
            self._intentionally_absent_iterations += self._last_skipped_frames
            self._update_main_field_statistics(self._cur)

        # Receiving a P frame can't resynchronise the stream
        self._collect_main_frame()

        if self.main_stream_is_valid:
            self._prev2 = self._prev
            self._prev = list(self._cur[:FLIGHT_LOG_MAX_FIELDS])
            self._cur = [0] * FLIGHT_LOG_MAX_FIELDS

        return self.main_stream_is_valid

    def _complete_event_frame(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        if self._last_event_type != -1:
            if self._last_event_type == EVENT_LOGGING_RESUME:
                self._last_main_frame_iteration = self._last_event_data["log_iteration"]
                self._last_main_frame_time = self._last_event_data["current_time"]
            self._collect_event()
            return True
        return False

    def _complete_gps_home_frame(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        self._gps_home_prev = list(self._gps_home_cur[:FLIGHT_LOG_MAX_FIELDS])
        self.gps_home_is_valid = True
        self._collect_frame(FRAME_GPS_HOME, True)
        return True

    def _complete_gps_frame(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        self._apply_gps_frame_time_rollover()
        self._collect_frame(FRAME_GPS, self.gps_home_is_valid)
        return True

    def _complete_slow_frame(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        self._collect_frame(FRAME_SLOW, True)
        return True

    # --------------------------------------------------------------- events
    def _parse_event_frame(self):
        self._last_event_data = {}
        event_type = self.stream.read_byte()
        self._last_event_type = event_type

        if event_type == EVENT_SYNC_BEEP:
            self._last_event_data["time"] = (self.stream.read_unsigned_vb()
                                             + self._time_rollover_accumulator)
        elif event_type == EVENT_INFLIGHT_ADJUSTMENT:
            function = self.stream.read_byte()
            if function > 127:
                self._last_event_data["new_float_value"] = read_raw_float(self.stream)
            else:
                self._last_event_data["new_value"] = self.stream.read_signed_vb()
            self._last_event_data["adjustment_function"] = function
        elif event_type == EVENT_LOGGING_RESUME:
            self._last_event_data["log_iteration"] = self.stream.read_unsigned_vb()
            self._last_event_data["current_time"] = (self.stream.read_unsigned_vb()
                                                     + self._time_rollover_accumulator)
        elif event_type == EVENT_LOG_END:
            end_message = self.stream.read(EVENT_END_OF_LOG_MESSAGE_LEN)
            if end_message == EVENT_END_OF_LOG_MESSAGE:
                # Adjust the end of the stream so we stop reading
                self.stream.end = self.stream.pos
            else:
                self._last_event_type = -1
        else:
            self._last_event_type = -1

    # ------------------------------------------------------------ collection
    def _collect_main_frame(self):
        field_count = self.frame_defs[FRAME_MAIN].field_count
        if not self.main_stream_is_valid or field_count == 0:
            return
        if self.main_frames is None:
            self.main_frames = np.empty((4096, field_count), dtype=np.int64)
            self._n_frames = 0
        if self._n_frames >= self.main_frames.shape[0]:
            new_shape = (self.main_frames.shape[0] * 2, field_count)
            new = np.empty(new_shape, dtype=np.int64)
            new[:self._n_frames] = self.main_frames[:self._n_frames]
            self.main_frames = new
        self.main_frames[self._n_frames] = self._cur[:field_count]
        self._n_frames += 1

    def _collect_frame(self, frame_type: int, frame_valid: bool):
        frame_def = self.frame_defs[frame_type]
        if not frame_valid or frame_def.field_count == 0:
            return
        if frame_type == FRAME_GPS_HOME:
            values = self._gps_home_cur
        elif frame_type == FRAME_GPS:
            values = self._last_gps
        else:  # slow
            values = self._last_slow
        record = {name: values[i] for i, name in enumerate(frame_def.field_names)}
        self.gps_frames.append(record) if frame_type in (FRAME_GPS, FRAME_GPS_HOME) \
            else self.slow_frames.append(record)

    def _collect_event(self):
        t = self._last_event_type
        if t == EVENT_SYNC_BEEP:
            self.events.append({"type": t, "name": "Sync beep", "time": self._last_event_data.get("time", 0)})
        elif t == EVENT_INFLIGHT_ADJUSTMENT:
            fn = self._last_event_data.get("adjustment_function", 0)
            fn_name = INFLIGHT_ADJUSTMENT_FUNCTIONS[fn & 127] if (fn & 127) < len(INFLIGHT_ADJUSTMENT_FUNCTIONS) else str(fn & 127)
            if "new_float_value" in self._last_event_data:
                value = self._last_event_data["new_float_value"]
            else:
                value = self._last_event_data.get("new_value", 0)
            self.events.append({"type": t, "name": "Inflight adjustment",
                                "time": self._last_main_frame_time,
                                "data": {"adjustment_function": fn_name, "value": value}})
        elif t == EVENT_LOGGING_RESUME:
            self.events.append({"type": t, "name": "Logging resume",
                                "time": self._last_event_data.get("current_time", 0),
                                "data": {"log_iteration": self._last_event_data.get("log_iteration", 0)}})
        elif t == EVENT_LOG_END:
            self.events.append({"type": t, "name": "Log clean end", "time": self._last_main_frame_time})

    # -------------------------------------------------------------- parsing
    def parse(self) -> bool:
        """Run the parser over this log section. Returns True on success."""
        self._parser_state = PARSER_STATE_HEADER

        while True:
            command = self.stream.peek_char()

            if command == ord('H') and self._parser_state == PARSER_STATE_HEADER:
                self._parse_header_line()
            elif command == EOF:
                break

            if self._parser_state == PARSER_STATE_TRANSITION:
                frame_type = command if command in FRAME_TYPES else None
                if frame_type is not None:
                    if self.frame_defs[FRAME_MAIN].field_count == 0:
                        raise BlackboxError("Data file is missing field name definitions")

                    # Rewrite the second of each HOME_COORD predictor pair in G frames
                    g = self.frame_defs[FRAME_GPS]
                    for i in range(1, g.field_count):
                        if (g.predictor[i - 1] == PREDICTOR_HOME_COORD
                                and g.predictor[i] == PREDICTOR_HOME_COORD):
                            g.predictor[i] = PREDICTOR_HOME_COORD_1

                    self._parser_state = PARSER_STATE_DATA
                else:
                    # Skip garbage which apparently precedes the first data frame
                    self.stream.read_byte()
            elif self._parser_state == PARSER_STATE_DATA:
                if command == EOF:
                    break

                frame_type = command if command in FRAME_TYPES else None
                self.stream.read_byte()  # skip over the initial frame letter
                frame_size = 0

                if frame_type is not None:
                    frame_start = self.stream.pos
                    self._dispatch_parse(frame_type)
                    frame_size = self.stream.pos - frame_start
                else:
                    self.main_stream_is_valid = False

                premature_eof = self.stream.eof

                if frame_type is not None:
                    # Like the C decoder, a frame is considered complete when it
                    # decoded to a sane length (command != EOF is guaranteed here)
                    if frame_size <= FLIGHT_LOG_MAX_FRAME_LENGTH:
                        frame_accepted = self._dispatch_complete(
                            frame_type, self.stream.pos - frame_size, self.stream.pos)
                        if frame_accepted:
                            st = self._frame_stats[frame_type]
                            st["bytes"] += frame_size
                            st["valid_count"] += 1
                        else:
                            self._frame_stats[frame_type]["desync_count"] += 1
                    else:
                        # The previous frame was corrupt; resync after its first byte
                        self.main_stream_is_valid = False
                        self._frame_stats[frame_type]["corrupt_count"] += 1
                        self._total_corrupt_frames += 1
                        self.stream.read_byte()
                        continue
            elif self._parser_state == PARSER_STATE_HEADER:
                # Still in HEADER state after parsing a line: if the next byte
                # is not another 'H' header, we reached the end of the header
                # block (mirrors the C parser's generic end-of-header rule).
                # With command == 'H' we just loop back and parse the next line.
                if command != ord('H') and command != EOF:
                    self._parser_state = PARSER_STATE_TRANSITION

        self.field_names = list(self.frame_defs[FRAME_MAIN].field_names)
        if self.main_frames is not None:
            self.main_frames = self.main_frames[:self._n_frames]
        return True

    def _dispatch_parse(self, frame_type: int):
        if frame_type == FRAME_MAIN:
            self._parse_frame(FRAME_MAIN, self._cur, self._prev, None, 0)
        elif frame_type == FRAME_INTER:
            self._last_skipped_frames = self._count_intentionally_skipped_frames()
            self._parse_frame(FRAME_INTER, self._cur, self._prev, self._prev2,
                              self._last_skipped_frames)
        elif frame_type == FRAME_GPS:
            self._parse_frame(FRAME_GPS, self._last_gps, None, None, 0)
        elif frame_type == FRAME_GPS_HOME:
            self._parse_frame(FRAME_GPS_HOME, self._gps_home_cur, None, None, 0)
        elif frame_type == FRAME_SLOW:
            self._parse_frame(FRAME_SLOW, self._last_slow, None, None, 0)
        elif frame_type == FRAME_EVENT:
            self._parse_event_frame()

    def _dispatch_complete(self, frame_type: int, frame_start: int, frame_end: int) -> bool:
        if frame_type == FRAME_MAIN:
            return self._complete_intraframe(frame_type, frame_start, frame_end)
        if frame_type == FRAME_INTER:
            return self._complete_interframe(frame_type, frame_start, frame_end)
        if frame_type == FRAME_EVENT:
            return self._complete_event_frame(frame_type, frame_start, frame_end)
        if frame_type == FRAME_GPS_HOME:
            return self._complete_gps_home_frame(frame_type, frame_start, frame_end)
        if frame_type == FRAME_GPS:
            return self._complete_gps_frame(frame_type, frame_start, frame_end)
        if frame_type == FRAME_SLOW:
            return self._complete_slow_frame(frame_type, frame_start, frame_end)
        return False

    # ------------------------------------------------------------- conversions
    def gyro_raw_to_rad_per_s(self, raw: int) -> float:
        return self.sys_config.gyro_scale * 1000000 * raw

    def gyro_raw_to_deg_per_s(self, raw: int) -> float:
        return self.gyro_raw_to_rad_per_s(raw) * (180.0 / math.pi)

    def vbat_raw_to_volts(self, raw: int) -> float:
        """vbat scaling changed in firmware 4.3.0."""
        if self._fc_version_gte("4.3.0"):
            return raw / 100.0
        return raw / 10.0

    def amperage_raw_to_amps(self, raw: int) -> float:
        return raw / 100.0

    def _fc_version_gte(self, version: str) -> bool:
        if not self.fc_version:
            return False
        def parse(v):
            parts = re.split(r"[^0-9]", v)
            return [int(x) for x in parts[:3] if x != ""]
        try:
            return parse(self.fc_version) >= parse(version)
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Multi-log file handling
# ---------------------------------------------------------------------------

def find_log_sections(data: bytes) -> List[tuple]:
    """Return (start, end) offsets of every log in the file (mirrors flightLogCreate)."""
    sections = []
    search_start = 0
    while search_start < len(data):
        pos = data.find(LOG_START_MARKER, search_start)
        if pos == -1:
            break
        sections.append(pos)
        search_start = pos + len(LOG_START_MARKER)
    if not sections:
        return []
    bounds = []
    for i, start in enumerate(sections):
        end = sections[i + 1] if i + 1 < len(sections) else len(data)
        bounds.append((start, end))
    return bounds


def parse_bbl(data: bytes, raw: bool = False) -> List[FlightLogParser]:
    """Parse every flight log in the file. Returns a list of parsed parsers."""
    bounds = find_log_sections(data)
    if not bounds:
        raise BlackboxError("Couldn't find the header of a flight log in the file, is this the right kind of file?")
    results = []
    for start, end in bounds:
        parser = FlightLogParser(data, raw=raw, log_start=start, log_end=end)
        parser.parse()
        results.append(parser)
    return results


def build_summary(parser: FlightLogParser) -> dict:
    """Build a metadata/statistics summary for the API."""
    defs = parser.frame_defs[FRAME_MAIN]
    idx = parser.main_field_indexes

    field_min_max = []
    if parser._have_field_stats:
        for name, (mn, mx) in zip(defs.field_names, parser._field_stats):
            field_min_max.append({"name": name, "min": mn, "max": mx})

    frame_counts = {}
    for ft in FRAME_TYPES:
        st = parser._frame_stats[ft]
        frame_counts[chr(ft)] = {
            "bytes": st["bytes"], "valid": st["valid_count"],
            "corrupt": st["corrupt_count"], "desync": st["desync_count"],
        }

    summary = {
        "field_names": list(defs.field_names),
        "field_count": defs.field_count,
        "data_version": parser.data_version,
        "fc_version": parser.fc_version,
        "firmware_type": ["unknown", "Baseflight", "Cleanflight", "Betaflight"][parser.sys_config.firmware_type],
        "date_time": parser.date_time.isoformat() if parser.date_time else None,
        "frame_intervals": {
            "i": parser.frame_interval_i,
            "p_num": parser.frame_interval_p_num,
            "p_denom": parser.frame_interval_p_denom,
        },
        "sys_config": {
            "minthrottle": parser.sys_config.minthrottle,
            "maxthrottle": parser.sys_config.maxthrottle,
            "motor_output_low": parser.sys_config.motor_output_low,
            "motor_output_high": parser.sys_config.motor_output_high,
            "vbatref": parser.sys_config.vbatref,
            "vbatscale": parser.sys_config.vbatscale,
            "vbatmincellvoltage": parser.sys_config.vbatmincellvoltage,
            "vbatwarningcellvoltage": parser.sys_config.vbatwarningcellvoltage,
            "vbatmaxcellvoltage": parser.sys_config.vbatmaxcellvoltage,
            "current_meter_offset": parser.sys_config.current_meter_offset,
            "current_meter_scale": parser.sys_config.current_meter_scale,
            "acc_1g": parser.sys_config.acc_1g,
            "gyro_scale": parser.sys_config.gyro_scale,
            "rc_rate": parser.sys_config.rc_rate,
            "pid_roll": list(parser.sys_config.pid_roll),
            "pid_pitch": list(parser.sys_config.pid_pitch),
            "pid_yaw": list(parser.sys_config.pid_yaw),
            "d_min": list(parser.sys_config.d_min),
            "ff_weight": list(parser.sys_config.ff_weight),
            "dterm_lpf1_static_hz": parser.sys_config.dterm_lpf1_static_hz,
            "dterm_lpf2_static_hz": parser.sys_config.dterm_lpf2_static_hz,
            "gyro_lpf1_static_hz": parser.sys_config.gyro_lpf1_static_hz,
            "gyro_lpf2_static_hz": parser.sys_config.gyro_lpf2_static_hz,
            "yaw_lowpass_hz": parser.sys_config.yaw_lowpass_hz,
        },
        "field_indexes": {
            "loop_iteration": idx.loop_iteration,
            "time": idx.time,
            "pid": idx.pid,
            "rc_command": idx.rc_command,
            "gyro_adc": idx.gyro_adc,
            "acc_smooth": idx.acc_smooth,
            "mag_adc": idx.mag_adc,
            "motor": idx.motor,
            "servo": idx.servo,
            "vbat_latest": idx.vbat_latest,
            "amperage_latest": idx.amperage_latest,
            "baro_alt": idx.baro_alt,
            "rssi": idx.rssi,
        },
        "fields_min_max": field_min_max,
        "frame_counts": frame_counts,
        "total_corrupt_frames": parser._total_corrupt_frames,
        "intentionally_absent_iterations": parser._intentionally_absent_iterations,
        "n_main_frames": parser._n_frames,
        "n_events": len(parser.events),
        "n_gps_frames": len(parser.gps_frames),
        "n_slow_frames": len(parser.slow_frames),
    }

    # Duration from time field statistics
    if parser._have_field_stats and idx.time != -1 and idx.time < len(parser._field_stats):
        tmin = parser._field_stats[idx.time][0]
        tmax = parser._field_stats[idx.time][1]
        summary["duration_us"] = tmax - tmin
        summary["start_time_us"] = tmin
        summary["end_time_us"] = tmax
    return summary
