"""
FIT Gap Repair — Flask backend
Endpoints:
  POST /analyze   → upload FIT, returns gap list + defaults
  POST /repair    → upload FIT + gap params, returns repaired FIT bytes
"""

import io
import json
import math
import traceback

import fitparse
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import (
    Activity,
    Event,
    EventType,
    FileType,
    Manufacturer,
    Sport,
)

app = Flask(__name__)
CORS(app)

FIT_EPOCH_OFFSET = 631065600  # seconds between 1989-12-31 and 1970-01-01


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def get_field(msg, name, default=None):
    try:
        v = msg.get_value(name)
        return v if v is not None else default
    except Exception:
        return default


def semicircles_to_deg(v):
    if v is None:
        return None
    return v * (180.0 / (2 ** 31))


# ──────────────────────────────────────────────────────────────
# Parse
# ──────────────────────────────────────────────────────────────

def parse_fit(data: bytes):
    fit = fitparse.FitFile(io.BytesIO(data))
    records, laps, session, activity, file_id = [], [], None, None, None

    for msg in fit.get_messages():
        n = msg.name
        if n == "record":
            ts = get_field(msg, "timestamp")
            if ts is None:
                continue
            lat_raw = get_field(msg, "position_lat")
            lon_raw = get_field(msg, "position_long")
            # fitparse returns degrees already for lat/long when it can;
            # fall back to semicircle conversion for raw int values
            lat = semicircles_to_deg(lat_raw) if isinstance(lat_raw, (int, float)) else lat_raw
            lon = semicircles_to_deg(lon_raw) if isinstance(lon_raw, (int, float)) else lon_raw
            records.append({
                "timestamp_unix_ms": int(ts.timestamp() * 1000),
                "lat": lat,
                "lon": lon,
                "distance": get_field(msg, "distance"),
                "speed": get_field(msg, "speed"),
                "heart_rate": get_field(msg, "heart_rate"),
                "power": get_field(msg, "power"),
                "cadence": get_field(msg, "cadence"),
                "altitude": get_field(msg, "altitude") or get_field(msg, "enhanced_altitude"),
                "temperature": get_field(msg, "temperature"),
            })
        elif n == "lap":
            laps.append(msg)
        elif n == "session" and session is None:
            session = msg
        elif n == "activity" and activity is None:
            activity = msg
        elif n == "file_id" and file_id is None:
            file_id = msg

    return records, laps, session, activity, file_id


# ──────────────────────────────────────────────────────────────
# Gap detection
# ──────────────────────────────────────────────────────────────

def detect_gaps(records, min_time_gap=30, min_gps_gap=50):
    gaps = []
    for i in range(1, len(records)):
        r0, r1 = records[i - 1], records[i]
        dt = (r1["timestamp_unix_ms"] - r0["timestamp_unix_ms"]) / 1000.0
        if dt <= min_time_gap:
            continue
        lat0, lon0 = r0.get("lat"), r0.get("lon")
        lat1, lon1 = r1.get("lat"), r1.get("lon")
        if None in (lat0, lon0, lat1, lon1):
            continue
        gps_gap = haversine_m(lat0, lon0, lat1, lon1)
        if gps_gap < min_gps_gap:
            continue
        gaps.append({
            "idx_before": i - 1,
            "idx_after": i,
            "time_gap_s": dt,
            "gps_gap_m": gps_gap,
            "dist_before_m": r0.get("distance") or 0,
        })
    return gaps


# ──────────────────────────────────────────────────────────────
# Surrounding averages
# ──────────────────────────────────────────────────────────────

def surrounding_avg(records, idx_before, idx_after, n=50):
    lo = max(0, idx_before - n + 1)
    hi = min(len(records) - 1, idx_after + n - 1)
    window = records[lo: idx_before + 1] + records[idx_after: hi + 1]

    def avg(key):
        vals = [r[key] for r in window if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "heart_rate": avg("heart_rate"),
        "power": avg("power"),
        "cadence": avg("cadence"),
        "altitude": avg("altitude"),
        "temperature": avg("temperature"),
        "speed": avg("speed"),
    }


# ──────────────────────────────────────────────────────────────
# Synthetic record builder
# ──────────────────────────────────────────────────────────────

def build_synthetic_records(records, gap, stop_time_s, ride_speed_ms):
    r0 = records[gap["idx_before"]]
    r1 = records[gap["idx_after"]]
    avgs = surrounding_avg(records, gap["idx_before"], gap["idx_after"])

    total_gap_s = gap["time_gap_s"]
    ride_time_s = max(0, total_gap_s - stop_time_s)
    added_distance_m = ride_time_s * ride_speed_ms

    lat0, lon0 = r0["lat"], r0["lon"]
    lat1, lon1 = r1["lat"], r1["lon"]
    dist_start = r0.get("distance") or 0

    n = int(total_gap_s) - 1
    if n <= 0:
        return [], added_distance_m

    synth = []
    t0_ms = r0["timestamp_unix_ms"]

    for i in range(1, n + 1):
        frac = i / (n + 1)
        t_ms = int(t0_ms + i * 1000)
        lat = lat0 + frac * (lat1 - lat0)
        lon = lon0 + frac * (lon1 - lon0)
        ride_frac = max(0, (i - stop_time_s) / ride_time_s) if ride_time_s > 0 else 0
        ride_frac = min(ride_frac, 1.0)
        dist = dist_start + ride_frac * added_distance_m
        speed = ride_speed_ms if i > stop_time_s else 0.0
        synth.append({
            "timestamp_unix_ms": t_ms,
            "lat": lat,
            "lon": lon,
            "distance": dist,
            "speed": speed,
            "heart_rate": avgs.get("heart_rate"),
            "power": avgs.get("power"),
            "cadence": avgs.get("cadence"),
            "altitude": avgs.get("altitude"),
            "temperature": avgs.get("temperature"),
            "_synthetic": True,
        })

    return synth, added_distance_m


# ──────────────────────────────────────────────────────────────
# FIT writer
# ──────────────────────────────────────────────────────────────

def write_fit(all_records, laps_raw, session_raw, activity_raw, file_id_raw, added_distances):
    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    # file_id
    fid = FileIdMessage()
    fid.type = FileType.ACTIVITY
    if file_id_raw:
        mfr = get_field(file_id_raw, "manufacturer")
        prod = get_field(file_id_raw, "product")
        sn = get_field(file_id_raw, "serial_number")
        tc = get_field(file_id_raw, "time_created")
        if mfr is not None:
            try:
                fid.manufacturer = Manufacturer(int(mfr))
            except Exception:
                pass
        if prod is not None:
            try:
                fid.product = int(prod)
            except Exception:
                pass
        if sn is not None:
            fid.serial_number = int(sn)
        if tc is not None:
            fid.time_created = int(tc.timestamp() * 1000)
    builder.add(fid)

    # start event
    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[0]["timestamp_unix_ms"]
        ev.event = Event.TIMER
        ev.event_type = EventType.START
        builder.add(ev)

    # records
    for r in all_records:
        rm = RecordMessage()
        rm.timestamp = r["timestamp_unix_ms"]
        if r.get("lat") is not None:
            rm.position_lat = r["lat"]
        if r.get("lon") is not None:
            rm.position_long = r["lon"]
        if r.get("distance") is not None:
            rm.distance = float(r["distance"])
        if r.get("speed") is not None:
            rm.speed = max(0.0, min(50.0, float(r["speed"])))
        if r.get("heart_rate") is not None:
            rm.heart_rate = int(round(r["heart_rate"]))
        if r.get("power") is not None:
            rm.power = int(round(r["power"]))
        if r.get("cadence") is not None:
            rm.cadence = int(round(r["cadence"]))
        if r.get("altitude") is not None:
            rm.enhanced_altitude = float(r["altitude"])
        if r.get("temperature") is not None:
            rm.temperature = int(round(r["temperature"]))
        builder.add(rm)

    # stop event
    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[-1]["timestamp_unix_ms"]
        ev.event = Event.TIMER
        ev.event_type = EventType.STOP_ALL
        builder.add(ev)

    # laps
    total_added = sum(added_distances.values())
    for lap_raw in laps_raw:
        lm = LapMessage()
        ts = get_field(lap_raw, "timestamp")
        ts_unix_ms = None
        if ts is not None:
            ts_unix_ms = int(ts.timestamp() * 1000)
            lm.timestamp = ts_unix_ms
        start_t = get_field(lap_raw, "start_time")
        if start_t is not None:
            lm.start_time = int(start_t.timestamp() * 1000)
        td = get_field(lap_raw, "total_distance")
        if td is not None:
            lap_added = sum(
                v for gap_ts_ms, v in added_distances.items()
                if ts_unix_ms is not None and ts_unix_ms >= gap_ts_ms
            )
            lm.total_distance = float(td) + lap_added
        tt = get_field(lap_raw, "total_timer_time")
        if tt is not None:
            lm.total_timer_time = float(tt)
        te = get_field(lap_raw, "total_elapsed_time")
        if te is not None:
            lm.total_elapsed_time = float(te)
        lm.event = Event.LAP
        lm.event_type = EventType.STOP
        builder.add(lm)

    # session
    sm = SessionMessage()
    sm.event = Event.SESSION
    sm.event_type = EventType.STOP
    if all_records:
        sm.start_time = all_records[0]["timestamp_unix_ms"]
        sm.timestamp = all_records[-1]["timestamp_unix_ms"]
        elapsed = (all_records[-1]["timestamp_unix_ms"] - all_records[0]["timestamp_unix_ms"]) / 1000.0
        sm.total_elapsed_time = elapsed
    if session_raw:
        tt = get_field(session_raw, "total_timer_time")
        if tt is not None:
            sm.total_timer_time = float(tt)
        td = get_field(session_raw, "total_distance")
        if td is not None:
            sm.total_distance = float(td) + total_added
        for attr, field in [("avg_speed", float), ("max_speed", float),
                             ("avg_heart_rate", int), ("avg_power", int), ("total_calories", int)]:
            v = get_field(session_raw, attr)
            if v is not None:
                setattr(sm, attr, field(v))
        sport_val = get_field(session_raw, "sport")
        try:
            sm.sport = Sport(int(sport_val)) if sport_val is not None else Sport.CYCLING
        except Exception:
            sm.sport = Sport.CYCLING
    else:
        sm.sport = Sport.CYCLING
    builder.add(sm)

    # activity
    am = ActivityMessage()
    if all_records:
        am.timestamp = all_records[-1]["timestamp_unix_ms"]
        am.total_timer_time = (all_records[-1]["timestamp_unix_ms"] - all_records[0]["timestamp_unix_ms"]) / 1000.0
    am.num_sessions = 1
    am.type = Activity.MANUAL
    am.event = Event.ACTIVITY
    am.event_type = EventType.STOP
    builder.add(am)

    return builder.build().to_bytes()


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    data = request.files["file"].read()
    try:
        records, laps_raw, session_raw, _, _ = parse_fit(data)
    except Exception as e:
        return jsonify({"error": f"Parse failed: {e}"}), 400

    if not records:
        return jsonify({"error": "No record messages found"}), 400

    gaps = detect_gaps(records)
    original_dist_m = records[-1].get("distance") or 0

    gap_list = []
    for g in gaps:
        avgs = surrounding_avg(records, g["idx_before"], g["idx_after"])
        default_speed_kmh = round((avgs["speed"] or 0) * 3.6, 1) if avgs.get("speed") else 25.0
        gap_list.append({
            "idx_before": g["idx_before"],
            "idx_after": g["idx_after"],
            "time_gap_s": g["time_gap_s"],
            "gps_gap_m": round(g["gps_gap_m"], 1),
            "dist_before_km": round(g["dist_before_m"] / 1000, 2),
            "default_speed_kmh": default_speed_kmh,
        })

    return jsonify({
        "num_records": len(records),
        "num_laps": len(laps_raw),
        "original_dist_km": round(original_dist_m / 1000, 2),
        "gaps": gap_list,
    })


@app.route("/repair", methods=["POST"])
def repair():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    if "params" not in request.form:
        return jsonify({"error": "No params supplied"}), 400

    data = request.files["file"].read()
    filename = request.files["file"].filename or "activity.fit"
    params = json.loads(request.form["params"])  # list of {stop_s, ride_kmh} per gap

    try:
        records, laps_raw, session_raw, activity_raw, file_id_raw = parse_fit(data)
    except Exception as e:
        return jsonify({"error": f"Parse failed: {e}"}), 400

    gaps = detect_gaps(records)
    if len(gaps) != len(params):
        return jsonify({"error": f"Gap count mismatch: found {len(gaps)}, got {len(params)} param sets"}), 400

    merged = list(records)
    added_distances = {}

    for gi in reversed(range(len(gaps))):
        gap = gaps[gi]
        stop_s = float(params[gi]["stop_s"])
        ride_ms = float(params[gi]["ride_kmh"]) / 3.6
        synth, added_m = build_synthetic_records(merged, gap, stop_s, ride_ms)

        for j in range(gap["idx_after"], len(merged)):
            if merged[j].get("distance") is not None:
                merged[j] = dict(merged[j])
                merged[j]["distance"] += added_m

        merged = merged[: gap["idx_after"]] + synth + merged[gap["idx_after"] :]
        gap_ts_after_ms = records[gap["idx_after"]]["timestamp_unix_ms"]
        added_distances[gap_ts_after_ms] = added_m

    original_dist_m = records[-1].get("distance") or 0
    total_added_m = sum(added_distances.values())

    try:
        fit_bytes = write_fit(merged, laps_raw, session_raw, activity_raw, file_id_raw, added_distances)
    except Exception as e:
        return jsonify({"error": f"FIT write failed: {e}\n{traceback.format_exc()}"}), 500

    out_name = filename.rsplit(".", 1)[0] + "_repaired.fit"
    return send_file(
        io.BytesIO(fit_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=out_name,
        headers={
            "X-Original-Dist-Km": str(round(original_dist_m / 1000, 2)),
            "X-Added-Dist-Km": str(round(total_added_m / 1000, 2)),
            "X-New-Dist-Km": str(round((original_dist_m + total_added_m) / 1000, 2)),
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
