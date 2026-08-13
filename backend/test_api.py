"""Quick smoke test of the decode/channels API against the sample log."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api"
BBL = r"..\samples\BTFL_KWONGKAN_10inch_0326_00_Filter.BBL"


def post_decode(path):
    boundary = "----pidtest"
    with open(path, "rb") as f:
        data = f.read()
    body = bytearray()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
             "Content-Type: application/octet-stream\r\n\r\n"
             % (boundary, path.replace("\\", "/").split("/")[-1])).encode()
    body += data + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(BASE + "/decode", data=bytes(body),
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


t0 = time.time()
r = post_decode(BBL)
print("decode took %.1fs" % (time.time() - t0))
print("logs:", len(r["logs"]))
s = r["logs"][0]
sid = s["id"]
print("id:", sid, "| name:", s["name"])
print("fc_version:", s["fc_version"], "| data_version:", s["data_version"])
print("fields:", s["field_count"], "| main_frames:", s["n_main_frames"],
      "| duration_s:", round(s.get("duration_us", 0) / 1e6, 2))
print("corrupt:", s["total_corrupt_frames"])

names = ["gyroADC[0]", "axisP[1]", "rcCommand[1]", "setpoint[3]", "motor[0]", "vbatLatest", "amperageLatest"]
t1 = time.time()
ch = get(f"{BASE}/decode/{sid}/channels?names=" + ",".join(names) + "&max_points=2000")
print("channels took %.2fs" % (time.time() - t1))
for name, info in ch["channels"].items():
    pts = info["data"]
    print(f"  {name:16s} unit={info['unit']:6s} points={len(pts)} "
          f"first=({pts[0][0]:.3f},{pts[0][1]:.1f}) last=({pts[-1][0]:.3f},{pts[-1][1]:.1f})")

fld = get(f"{BASE}/decode/{sid}/fields")
print("fields endpoint:", len(fld["fields"]), "fields")
