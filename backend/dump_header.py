"""Dump the raw header lines of the BBL to inspect rcCommand/setpoint field defs."""
with open(r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL", "rb") as f:
    data = f.read()

# print header lines that mention rcCommand or setpoint
pos = data.find(b"H Product:")
end = data.find(b"I ", pos)
header = data[pos:end].decode("latin-1")
for line in header.splitlines():
    if any(k in line for k in ("rcCommand", "setpoint", "Field I")):
        print(line)
