"""Blackbox (.bbl) format constants.

Ported from betaflight/blackbox-tools (blackbox_fielddefs_tools.h, parser.h).
"""

# ---------------------------------------------------------------------------
# Log limits
# ---------------------------------------------------------------------------
FLIGHT_LOG_MAX_LOGS_IN_FILE = 1000
FLIGHT_LOG_MAX_FIELDS = 128
FLIGHT_LOG_MAX_FRAME_LENGTH = 256
FLIGHT_LOG_MAX_FRAME_HEADER_LENGTH = 1024
FLIGHT_LOG_MAX_MOTORS = 8
FLIGHT_LOG_MAX_SERVOS = 8

# Well-known field indexes inside the main stream frame
FLIGHT_LOG_FIELD_INDEX_ITERATION = 0
FLIGHT_LOG_FIELD_INDEX_TIME = 1

# Frames are rejected if the iteration/time jumps more than this much
MAXIMUM_TIME_JUMP_BETWEEN_FRAMES = 10 * 1000000
MAXIMUM_ITERATION_JUMP_BETWEEN_FRAMES = 500 * 10

LOG_START_MARKER = b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"

# ---------------------------------------------------------------------------
# Field predictors
# ---------------------------------------------------------------------------
PREDICTOR_0 = 0
PREDICTOR_PREVIOUS = 1
PREDICTOR_STRAIGHT_LINE = 2
PREDICTOR_AVERAGE_2 = 3
PREDICTOR_MINTHROTTLE = 4
PREDICTOR_MOTOR_0 = 5
PREDICTOR_INC = 6
PREDICTOR_HOME_COORD = 7
PREDICTOR_1500 = 8
PREDICTOR_VBATREF = 9
PREDICTOR_LAST_MAIN_FRAME_TIME = 10
PREDICTOR_MINMOTOR = 11
# Home coord predictors appear in pairs; the second one of a pair is
# rewritten to this value to make parsing easier
PREDICTOR_HOME_COORD_1 = 256

# ---------------------------------------------------------------------------
# Field encodings
# ---------------------------------------------------------------------------
ENCODING_SIGNED_VB = 0
ENCODING_UNSIGNED_VB = 1
ENCODING_NEG_14BIT = 3
ENCODING_ELIAS_DELTA_U32 = 4
ENCODING_ELIAS_DELTA_S32 = 5
ENCODING_TAG8_8SVB = 6
ENCODING_TAG2_3S32 = 7
ENCODING_TAG8_4S16 = 8
ENCODING_NULL = 9
ENCODING_ELIAS_GAMMA_U32 = 10
ENCODING_ELIAS_GAMMA_S32 = 11

# ---------------------------------------------------------------------------
# Frame types (marker characters)
# ---------------------------------------------------------------------------
FRAME_MAIN = ord('I')
FRAME_INTER = ord('P')
FRAME_GPS = ord('G')
FRAME_GPS_HOME = ord('H')
FRAME_EVENT = ord('E')
FRAME_SLOW = ord('S')

FRAME_TYPES = (FRAME_MAIN, FRAME_INTER, FRAME_GPS, FRAME_GPS_HOME, FRAME_EVENT, FRAME_SLOW)

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
EVENT_SYNC_BEEP = 0
EVENT_INFLIGHT_ADJUSTMENT = 13
EVENT_LOGGING_RESUME = 14
EVENT_FLIGHTMODE = 30
EVENT_LOG_END = 255

EVENT_END_OF_LOG_MESSAGE = b"End of log\x00"
EVENT_END_OF_LOG_MESSAGE_LEN = len(EVENT_END_OF_LOG_MESSAGE)

# ---------------------------------------------------------------------------
# Parser states
# ---------------------------------------------------------------------------
PARSER_STATE_HEADER = 0
PARSER_STATE_TRANSITION = 1
PARSER_STATE_DATA = 2

# ---------------------------------------------------------------------------
# Firmware types
# ---------------------------------------------------------------------------
FIRMWARE_TYPE_UNKNOWN = 0
FIRMWARE_TYPE_BASEFLIGHT = 1
FIRMWARE_TYPE_CLEANFLIGHT = 2
FIRMWARE_TYPE_BETAFLIGHT = 3

# Inflight adjustment function names (for event reporting)
INFLIGHT_ADJUSTMENT_FUNCTIONS = [
    "NONE", "RC_RATE", "RC_EXPO", "THROTTLE_EXPO",
    "PITCH_ROLL_RATE", "YAW_RATE", "PITCH_ROLL_P", "PITCH_ROLL_I",
    "PITCH_ROLL_D", "YAW_P", "YAW_I", "YAW_D", "RATE_PROFILE",
    "PITCH_RATE", "ROLL_RATE", "PITCH_P", "PITCH_I", "PITCH_D",
    "ROLL_P", "ROLL_I", "ROLL_D",
]

FLIGHT_MODE_NAMES = [
    "ANGLE_MODE", "HORIZON_MODE", "MAG_MODE", "BARO_MODE",
    "GPS_HOME_MODE", "GPS_HOLD_MODE", "HEADFREE_MODE", "UNUSED_MODE",
    "PASSTHRU_MODE", "RANGEFINDER_MODE", "FAILSAFE_MODE",
]

FLIGHT_STATE_NAMES = [
    "GPS_FIX_HOME", "GPS_FIX", "CALIBRATE_MAG", "SMALL_ANGLE", "FIXED_WING",
]

FAILSAFE_PHASE_NAMES = [
    "IDLE", "RX_LOSS_DETECTED", "LANDING", "LANDED",
    "RX_LOSS_MONITORING", "RX_LOSS_RECOVERED",
]

ADCVREF = 33
