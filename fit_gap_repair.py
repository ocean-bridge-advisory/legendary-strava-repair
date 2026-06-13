"""
FIT File Gap Repair Tool
Detects and fills timestamp/GPS gaps in Garmin cycling FIT files.
"""

import io
import math
import struct
import streamlit as st
import fitparse
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.profile_type import (
    FileType, Sport, SubSport, Manufacturer,
    Event, EventType, Activity
)

# FIT epoch: 1989-12-31 00:00:00 UTC → Unix timestamp
FIT_EPOCH_OFFSET = 631065600
SEMICIRCLE_TO_DEG = 180.0 / (2**31)
DEG_TO_SEMICIRCLE = (2**31) / 180.0


# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def get_field(msg, name, default=None):
    """Safely read a fitparse message field."""
    try:
        v = msg.get_value(name)
        return v if v is not None else default
    except Exception:
        return default


def semicircles_to_deg(v):
    if v is None:
        return None
    return v * SEMICIRCLE_TO_DEG


def fit_ts_to_unix_ms(fit_ts):
    """FIT epoch seconds → Unix epoch milliseconds."""
    return (fit_ts + FIT_EPOCH_OFFSET) * 1000


# ──────────────────────────────────────────────────────────────
# Step 1: Parse FIT file
# ──────────────────────────────────────────────────────────────

def parse_fit(data: bytes):
    """
    Returns:
      records  – list of dicts with all fields
      laps     – list of raw fitparse lap messages
      session  – raw fitparse session message (first one found)
      activity – raw fitparse activity message
      file_id  – raw fitparse file_id message
    """
    fit = fitparse.FitFile(io.BytesIO(data))

    records = []
    laps = []
    session = None
    activity = None
    file_id = None

    for msg in fit.get_messages():
        name = msg.name

        if name == "record":
            ts = get_field(msg, "timestamp")
            if ts is None:
                continue
            lat_raw = get_field(msg, "position_lat")
            lon_raw = get_field(msg, "position_long")
            rec = {
                "timestamp_fit": int(ts.timestamp()) - FIT_EPOCH_OFFSET,  # FIT seconds
                "timestamp_unix_ms": int(ts.timestamp() * 1000),
                "lat": semicircles_to_deg(lat_raw) if isinstance(lat_raw, (int, float)) else (lat_raw if lat_raw else None),
                "lon": semicircles_to_deg(lon_raw) if isinstance(lon_raw, (int, float)) else (lon_raw if lon_raw else None),
                "distance": get_field(msg, "distance"),     # metres
                "speed": get_field(msg, "speed"),           # m/s
                "heart_rate": get_field(msg, "heart_rate"),
                "power": get_field(msg, "power"),
                "cadence": get_field(msg, "cadence"),
                "altitude": get_field(msg, "altitude") or get_field(msg, "enhanced_altitude"),
                "temperature": get_field(msg, "temperature"),
            }
            records.append(rec)

        elif name == "lap":
            laps.append(msg)
        elif name == "session" and session is None:
            session = msg
        elif name == "activity" and activity is None:
            activity = msg
        elif name == "file_id" and file_id is None:
            file_id = msg

    return records, laps, session, activity, file_id


# ──────────────────────────────────────────────────────────────
# Step 2: Detect gaps
# ──────────────────────────────────────────────────────────────

def detect_gaps(records, min_time_gap=30, min_gps_gap=50):
    """
    Returns list of gap dicts:
      idx_before, idx_after, time_gap_s, gps_gap_m, dist_before_m
    """
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
# Step 3: Compute surrounding averages
# ──────────────────────────────────────────────────────────────

def surrounding_avg(records, idx_before, idx_after, n=50):
    """Average HR, power, cadence, altitude, temp over n records either side."""
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
# Step 4: Build synthetic records for one gap
# ──────────────────────────────────────────────────────────────

def build_synthetic_records(records, gap, stop_time_s, ride_speed_ms):
    """
    Returns list of synthetic record dicts to insert between
    gap['idx_before'] and gap['idx_after'].
    Also returns added_distance_m (distance filled in).
    """
    r0 = records[gap["idx_before"]]
    r1 = records[gap["idx_after"]]
    avgs = surrounding_avg(records, gap["idx_before"], gap["idx_after"])

    total_gap_s = gap["time_gap_s"]
    ride_time_s = max(0, total_gap_s - stop_time_s)
    added_distance_m = ride_time_s * ride_speed_ms

    lat0, lon0 = r0["lat"], r0["lon"]
    lat1, lon1 = r1["lat"], r1["lon"]
    dist_start = r0.get("distance") or 0

    # Number of synthetic seconds
    n = int(total_gap_s) - 1  # exclude boundary records
    if n <= 0:
        return [], added_distance_m

    synth = []
    t0_ms = r0["timestamp_unix_ms"]

    for i in range(1, n + 1):
        frac = i / (n + 1)
        t_ms = int(t0_ms + i * 1000)

        # Interpolate GPS linearly
        lat = lat0 + frac * (lat1 - lat0)
        lon = lon0 + frac * (lon1 - lon0)

        # Distance: only accumulate during ride portion (after stop)
        ride_frac = max(0, (i - stop_time_s) / ride_time_s) if ride_time_s > 0 else 0
        ride_frac = min(ride_frac, 1.0)
        dist = dist_start + ride_frac * added_distance_m

        speed = ride_speed_ms if i > stop_time_s else 0.0

        rec = {
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
        }
        synth.append(rec)

    return synth, added_distance_m


# ──────────────────────────────────────────────────────────────
# Step 5: Write output FIT file
# ──────────────────────────────────────────────────────────────

def write_fit(all_records, laps_raw, session_raw, activity_raw, file_id_raw, added_distances):
    """
    Build a FIT file from the merged record list.
    added_distances: dict mapping lap fit-timestamps-after-gap to added metres.
    """
    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    # ── file_id ──────────────────────────────────────────────
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

    # ── start event ──────────────────────────────────────────
    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[0]["timestamp_unix_ms"]
        ev.event = Event.TIMER
        ev.event_type = EventType.START
        builder.add(ev)

    # ── records ──────────────────────────────────────────────
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
            spd = float(r["speed"])
            rm.speed = max(0.0, min(50.0, spd))
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

    # ── stop event ───────────────────────────────────────────
    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[-1]["timestamp_unix_ms"]
        ev.event = Event.TIMER
        ev.event_type = EventType.STOP_ALL
        builder.add(ev)

    # ── laps ─────────────────────────────────────────────────
    total_added = sum(added_distances.values())
    for lap_raw in laps_raw:
        lm = LapMessage()
        ts = get_field(lap_raw, "timestamp")
        if ts is not None:
            ts_unix_ms = int(ts.timestamp() * 1000)
            lm.timestamp = ts_unix_ms
        st = get_field(lap_raw, "start_time")
        if st is not None:
            lm.start_time = int(st.timestamp() * 1000)
        td = get_field(lap_raw, "total_distance")
        if td is not None:
            # Add missing distance to laps after any gap
            lap_added = sum(
                v for gap_ts_ms, v in added_distances.items()
                if ts_unix_ms >= gap_ts_ms
            ) if ts is not None else 0
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

    # ── session ──────────────────────────────────────────────
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
        avgs_spd = get_field(session_raw, "avg_speed")
        if avgs_spd is not None:
            sm.avg_speed = float(avgs_spd)
        max_spd = get_field(session_raw, "max_speed")
        if max_spd is not None:
            sm.max_speed = float(max_spd)
        avg_hr = get_field(session_raw, "avg_heart_rate")
        if avg_hr is not None:
            sm.avg_heart_rate = int(avg_hr)
        avg_pwr = get_field(session_raw, "avg_power")
        if avg_pwr is not None:
            sm.avg_power = int(avg_pwr)
        cal = get_field(session_raw, "total_calories")
        if cal is not None:
            sm.total_calories = int(cal)
        sport_val = get_field(session_raw, "sport")
        if sport_val is not None:
            try:
                sm.sport = Sport(int(sport_val))
            except Exception:
                sm.sport = Sport.CYCLING
        else:
            sm.sport = Sport.CYCLING
    else:
        sm.sport = Sport.CYCLING
    builder.add(sm)

    # ── activity ─────────────────────────────────────────────
    am = ActivityMessage()
    if all_records:
        am.timestamp = all_records[-1]["timestamp_unix_ms"]
        elapsed = (all_records[-1]["timestamp_unix_ms"] - all_records[0]["timestamp_unix_ms"]) / 1000.0
        am.total_timer_time = elapsed
    am.num_sessions = 1
    am.type = Activity.MANUAL
    am.event = Event.ACTIVITY
    am.event_type = EventType.STOP
    builder.add(am)

    fit_file = builder.build()
    return fit_file.to_bytes()


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="FIT Gap Repair", page_icon="🚴", layout="centered")

st.title("🚴 FIT File Gap Repair")
st.caption(
    "Detects and fills timestamp + GPS gaps in Garmin cycling FIT files. "
    "Outputs a corrected FIT ready for Strava / Garmin Connect."
)

uploaded = st.file_uploader("Upload your FIT file", type=["fit"])

if not uploaded:
    st.stop()

raw_bytes = uploaded.read()

with st.spinner("Parsing FIT file…"):
    try:
        records, laps_raw, session_raw, activity_raw, file_id_raw = parse_fit(raw_bytes)
    except Exception as e:
        st.error(f"Could not parse FIT file: {e}")
        st.stop()

if not records:
    st.error("No record messages found in this FIT file.")
    st.stop()

st.success(f"Loaded **{len(records):,}** records, **{len(laps_raw)}** laps")

# ── Gap detection ─────────────────────────────────────────────
gaps = detect_gaps(records)

if not gaps:
    st.info("No gaps detected (no timestamp jumps > 30 s with GPS displacement > 50 m).")
    st.stop()

st.subheader(f"{'Gap' if len(gaps) == 1 else f'{len(gaps)} Gaps'} Detected")

# ── Per-gap user inputs ───────────────────────────────────────
gap_params = []

for gi, gap in enumerate(gaps):
    avgs = surrounding_avg(records, gap["idx_before"], gap["idx_after"])
    default_speed_kmh = (avgs["speed"] * 3.6) if avgs.get("speed") else 25.0
    dist_km = (gap["dist_before_m"] or 0) / 1000.0

    with st.expander(
        f"Gap {gi + 1}  —  at {dist_km:.2f} km  |  "
        f"Δt {gap['time_gap_s']:.0f} s  |  "
        f"GPS jump {gap['gps_gap_m']:.0f} m",
        expanded=True,
    ):
        col1, col2, col3 = st.columns(3)
        col1.metric("Time jump", f"{gap['time_gap_s'] / 60:.1f} min")
        col2.metric("GPS jump (straight line)", f"{gap['gps_gap_m']:.0f} m")
        col3.metric("Position in ride", f"{dist_km:.2f} km")

        c1, c2 = st.columns(2)
        stop_min = c1.number_input(
            "Stopped time (minutes)",
            min_value=0.0,
            max_value=gap["time_gap_s"] / 60,
            value=0.0,
            step=0.5,
            key=f"stop_{gi}",
        )
        ride_kmh = c2.number_input(
            "Riding speed during gap (km/h)",
            min_value=1.0,
            max_value=80.0,
            value=round(default_speed_kmh, 1),
            step=0.5,
            key=f"speed_{gi}",
            help=f"Default = average of surrounding records ({default_speed_kmh:.1f} km/h)",
        )

        stop_s = stop_min * 60
        ride_s = max(0, gap["time_gap_s"] - stop_s)
        ride_ms = ride_kmh / 3.6
        est_dist_km = ride_s * ride_ms / 1000.0

        st.caption(
            f"Riding time in gap: **{ride_s / 60:.1f} min** at **{ride_kmh:.1f} km/h** "
            f"→ estimated missing distance: **{est_dist_km:.2f} km**"
        )

        gap_params.append({"stop_s": stop_s, "ride_ms": ride_ms})

# ── Build button ──────────────────────────────────────────────
st.divider()
if st.button("🔧 Repair FIT File", type="primary", use_container_width=True):
    with st.spinner("Building corrected FIT file…"):
        # Process gaps in reverse order so indices stay valid
        merged = list(records)
        added_distances = {}  # gap_ts_after_ms → added_m

        for gi in reversed(range(len(gaps))):
            gap = gaps[gi]
            params = gap_params[gi]
            synth, added_m = build_synthetic_records(
                merged, gap, params["stop_s"], params["ride_ms"]
            )

            # Shift all records after this gap by added_m
            for j in range(gap["idx_after"], len(merged)):
                if merged[j].get("distance") is not None:
                    merged[j] = dict(merged[j])
                    merged[j]["distance"] += added_m

            # Insert synthetic records
            insert_at = gap["idx_after"]
            merged = merged[:insert_at] + synth + merged[insert_at:]

            # Record gap timestamp for lap adjustment
            gap_ts_after_ms = records[gap["idx_after"]]["timestamp_unix_ms"]
            added_distances[gap_ts_after_ms] = added_m

        # Summary
        original_dist_m = records[-1].get("distance") or 0
        total_added_m = sum(added_distances.values())
        new_dist_m = original_dist_m + total_added_m

        try:
            fit_bytes = write_fit(merged, laps_raw, session_raw, activity_raw, file_id_raw, added_distances)
        except Exception as e:
            st.error(f"FIT write failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    st.success("✅ FIT file repaired successfully")

    col1, col2, col3 = st.columns(3)
    col1.metric("Original distance", f"{original_dist_m / 1000:.2f} km")
    col2.metric("Distance added", f"{total_added_m / 1000:.2f} km")
    col3.metric("Corrected total", f"{new_dist_m / 1000:.2f} km")

    out_name = uploaded.name.replace(".fit", "_repaired.fit").replace(".FIT", "_repaired.FIT")
    st.download_button(
        label="⬇️ Download Repaired FIT",
        data=fit_bytes,
        file_name=out_name,
        mime="application/octet-stream",
        use_container_width=True,
    )
