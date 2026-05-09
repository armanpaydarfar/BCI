from pupil_labs.realtime_api.simple import discover_devices

print("searching 8s for Neon devices on LAN...", flush=True)
try:
    devs = discover_devices(timeout=8.0)
except TypeError:
    devs = discover_devices(8.0)
if not devs:
    print("NO DEVICES FOUND")
else:
    for d in devs:
        print(d)
