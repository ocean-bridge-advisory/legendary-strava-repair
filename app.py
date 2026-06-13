"""
FIT Gap Repair — Flask backend with async job processing
Workaround for Render free tier 30s request timeout.
"""

import io
import json
import math
import os
import tempfile
import threading
import time
import traceback
import uuid

import fitdecode
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
    Activity, Event, EventType, FileType, Manufacturer, Sport,
)

app = Flask(__name__)
CORS(app)

FIT_EPOCH_OFFSET = 631065600
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)

# In-memory job store: job_id -> job dict
# Jobs expire after 30 minutes
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 1800  # seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def gv(frame, name, default=None):
    try:
        return frame.get_value(name) if frame.has_field(name) else default
    except Exception:
        return default


def ts_to_unix_ms(dt):
    if dt is None:
        return None
    import calendar
    return int(calendar.timegm(dt.timetuple()) * 1000)


def semicircles_to_deg(v):
    if v is None or not isinstance(v, (int, float)):
        return None
    return v * SEMICIRCLE_TO_DEG


def cleanup_old_jobs():
    """Remove jobs older than JOB_TTL seconds."""
    now = time.time()
    with JOBS_LOCK:
        expired = [jid for jid, j in JOBS.items() if now - j['created_at'] > JOB_TTL]
        for jid in expired:
            # Clean up temp file if present
            tmp = JOBS[jid].get('tmp_path')
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            del JOBS[jid]


# ── FIT parsing ───────────────────────────────────────────────────────────────

def parse_fit(data: bytes):
    records, laps, session, activity, file_id = [], [], None, None, None

    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            name = frame.name

            if name == 'record':
                ts = gv(frame, 'timestamp')
                if ts is None:
                    continue
                lat_raw = gv(frame, 'position_lat')
                lon_raw = gv(frame, 'position_long')
                records.append({
                    'timestamp_unix_ms': ts_to_unix_ms(ts),
                    'lat': semicircles_to_deg(lat_raw),
                    'lon': semicircles_to_deg(lon_raw),
                    'distance': gv(frame, 'distance'),
                    'speed': gv(frame, 'enhanced_speed') or gv(frame, 'speed'),
                    'heart_rate': gv(frame, 'heart_rate'),
                    'power': gv(frame, 'power'),
                    'cadence': gv(frame, 'cadence'),
                    'altitude': gv(frame, 'enhanced_altitude') or gv(frame, 'altitude'),
                    'temperature': gv(frame, 'temperature'),
                })
            elif name == 'lap':
                laps.append(frame)
            elif name == 'session' and session is None:
                session = frame
            elif name == 'activity' and activity is None:
                activity = frame
            elif name == 'file_id' and file_id is None:
                file_id = frame

    return records, laps, session, activity, file_id


# ── Gap detection ─────────────────────────────────────────────────────────────

def detect_gaps(records, min_time_gap=30, min_gps_gap=50):
    gaps = []
    for i in range(1, len(records)):
        r0, r1 = records[i - 1], records[i]
        dt = (r1['timestamp_unix_ms'] - r0['timestamp_unix_ms']) / 1000.0
        if dt <= min_time_gap:
            continue
        lat0, lon0 = r0.get('lat'), r0.get('lon')
        lat1, lon1 = r1.get('lat'), r1.get('lon')
        if None in (lat0, lon0, lat1, lon1):
            continue
        gps_gap = haversine_m(lat0, lon0, lat1, lon1)
        if gps_gap < min_gps_gap:
            continue
        gaps.append({
            'idx_before': i - 1,
            'idx_after': i,
            'time_gap_s': dt,
            'gps_gap_m': gps_gap,
            'dist_before_m': r0.get('distance') or 0,
        })
    return gaps


def surrounding_avg(records, idx_before, idx_after, n=50):
    lo = max(0, idx_before - n + 1)
    hi = min(len(records) - 1, idx_after + n - 1)
    window = records[lo: idx_before + 1] + records[idx_after: hi + 1]

    def avg(key):
        vals = [r[key] for r in window if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {k: avg(k) for k in ('heart_rate', 'power', 'cadence', 'altitude', 'temperature', 'speed')}


def build_synthetic_records(records, gap, stop_time_s, ride_speed_ms):
    r0 = records[gap['idx_before']]
    r1 = records[gap['idx_after']]
    avgs = surrounding_avg(records, gap['idx_before'], gap['idx_after'])

    total_gap_s = gap['time_gap_s']
    ride_time_s = max(0, total_gap_s - stop_time_s)
    added_distance_m = ride_time_s * ride_speed_ms

    lat0, lon0 = r0['lat'], r0['lon']
    lat1, lon1 = r1['lat'], r1['lon']
    dist_start = r0.get('distance') or 0

    n = int(total_gap_s) - 1
    if n <= 0:
        return [], added_distance_m

    synth = []
    t0_ms = r0['timestamp_unix_ms']
    for i in range(1, n + 1):
        frac = i / (n + 1)
        ride_frac = max(0, (i - stop_time_s) / ride_time_s) if ride_time_s > 0 else 0
        ride_frac = min(ride_frac, 1.0)
        synth.append({
            'timestamp_unix_ms': int(t0_ms + i * 1000),
            'lat': lat0 + frac * (lat1 - lat0),
            'lon': lon0 + frac * (lon1 - lon0),
            'distance': dist_start + ride_frac * added_distance_m,
            'speed': ride_speed_ms if i > stop_time_s else 0.0,
            'heart_rate': avgs.get('heart_rate'),
            'power': avgs.get('power'),
            'cadence': avgs.get('cadence'),
            'altitude': avgs.get('altitude'),
            'temperature': avgs.get('temperature'),
        })
    return synth, added_distance_m


# ── FIT writer ────────────────────────────────────────────────────────────────

def write_fit(all_records, laps_raw, session_raw, activity_raw, file_id_raw, added_distances):
    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    fid = FileIdMessage()
    fid.type = FileType.ACTIVITY
    if file_id_raw:
        for attr, cast, setter in [
            ('manufacturer', lambda v: Manufacturer(int(v)), 'manufacturer'),
            ('product', int, 'product'),
            ('serial_number', int, 'serial_number'),
        ]:
            v = gv(file_id_raw, attr)
            if v is not None:
                try:
                    setattr(fid, setter, cast(v))
                except Exception:
                    pass
        tc = gv(file_id_raw, 'time_created')
        if tc is not None:
            fid.time_created = ts_to_unix_ms(tc)
    builder.add(fid)

    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[0]['timestamp_unix_ms']
        ev.event = Event.TIMER
        ev.event_type = EventType.START
        builder.add(ev)

    for r in all_records:
        rm = RecordMessage()
        rm.timestamp = r['timestamp_unix_ms']
        if r.get('lat') is not None: rm.position_lat = r['lat']
        if r.get('lon') is not None: rm.position_long = r['lon']
        if r.get('distance') is not None: rm.distance = float(r['distance'])
        if r.get('speed') is not None: rm.speed = max(0.0, min(50.0, float(r['speed'])))
        if r.get('heart_rate') is not None: rm.heart_rate = int(round(r['heart_rate']))
        if r.get('power') is not None: rm.power = int(round(r['power']))
        if r.get('cadence') is not None: rm.cadence = int(round(r['cadence']))
        if r.get('altitude') is not None: rm.enhanced_altitude = float(r['altitude'])
        if r.get('temperature') is not None: rm.temperature = int(round(r['temperature']))
        builder.add(rm)

    if all_records:
        ev = EventMessage()
        ev.timestamp = all_records[-1]['timestamp_unix_ms']
        ev.event = Event.TIMER
        ev.event_type = EventType.STOP_ALL
        builder.add(ev)

    total_added = sum(added_distances.values())
    for lap_raw in laps_raw:
        lm = LapMessage()
        ts = gv(lap_raw, 'timestamp')
        ts_unix_ms = None
        if ts is not None:
            ts_unix_ms = ts_to_unix_ms(ts)
            lm.timestamp = ts_unix_ms
        st = gv(lap_raw, 'start_time')
        if st is not None: lm.start_time = ts_to_unix_ms(st)
        td = gv(lap_raw, 'total_distance')
        if td is not None:
            lap_added = sum(v for gap_ts_ms, v in added_distances.items()
                           if ts_unix_ms is not None and ts_unix_ms >= gap_ts_ms)
            lm.total_distance = float(td) + lap_added
        tt = gv(lap_raw, 'total_timer_time')
        if tt is not None: lm.total_timer_time = float(tt)
        te = gv(lap_raw, 'total_elapsed_time')
        if te is not None: lm.total_elapsed_time = float(te)
        lm.event = Event.LAP
        lm.event_type = EventType.STOP
        builder.add(lm)

    sm = SessionMessage()
    sm.event = Event.SESSION
    sm.event_type = EventType.STOP
    if all_records:
        sm.start_time = all_records[0]['timestamp_unix_ms']
        sm.timestamp = all_records[-1]['timestamp_unix_ms']
        sm.total_elapsed_time = (all_records[-1]['timestamp_unix_ms'] - all_records[0]['timestamp_unix_ms']) / 1000.0
    if session_raw:
        tt = gv(session_raw, 'total_timer_time')
        if tt is not None: sm.total_timer_time = float(tt)
        td = gv(session_raw, 'total_distance')
        if td is not None: sm.total_distance = float(td) + total_added
        for attr, cast in [('avg_speed', float), ('max_speed', float),
                           ('avg_heart_rate', int), ('avg_power', int), ('total_calories', int)]:
            v = gv(session_raw, attr)
            if v is not None:
                try: setattr(sm, attr, cast(v))
                except Exception: pass
        sport_val = gv(session_raw, 'sport')
        try: sm.sport = Sport(int(sport_val)) if sport_val is not None else Sport.CYCLING
        except Exception: sm.sport = Sport.CYCLING
    else:
        sm.sport = Sport.CYCLING
    builder.add(sm)

    am = ActivityMessage()
    if all_records:
        am.timestamp = all_records[-1]['timestamp_unix_ms']
        am.total_timer_time = (all_records[-1]['timestamp_unix_ms'] - all_records[0]['timestamp_unix_ms']) / 1000.0
    am.num_sessions = 1
    am.type = Activity.MANUAL
    am.event = Event.ACTIVITY
    am.event_type = EventType.STOP
    builder.add(am)

    return builder.build().to_bytes()


# ── Background workers ────────────────────────────────────────────────────────

def run_analyze(job_id, data):
    """Parse FIT and detect gaps. Runs in background thread."""
    try:
        records, laps_raw, session_raw, _, _ = parse_fit(data)
        if not records:
            with JOBS_LOCK:
                JOBS[job_id]['status'] = 'error'
                JOBS[job_id]['error'] = 'No record messages found'
            return

        gaps = detect_gaps(records)
        original_dist_m = records[-1].get('distance') or 0

        gap_list = []
        for g in gaps:
            avgs = surrounding_avg(records, g['idx_before'], g['idx_after'])
            default_speed_kmh = round((avgs['speed'] or 0) * 3.6, 1) if avgs.get('speed') else 25.0
            gap_list.append({
                'idx_before': g['idx_before'],
                'idx_after': g['idx_after'],
                'time_gap_s': g['time_gap_s'],
                'gps_gap_m': round(g['gps_gap_m'], 1),
                'dist_before_km': round(g['dist_before_m'] / 1000, 2),
                'default_speed_kmh': default_speed_kmh,
            })

        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'done'
            JOBS[job_id]['result'] = {
                'num_records': len(records),
                'num_laps': len(laps_raw),
                'original_dist_km': round(original_dist_m / 1000, 2),
                'gaps': gap_list,
            }
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = str(e)


def run_repair(job_id, data, filename, params):
    """Repair FIT file. Runs in background thread."""
    try:
        records, laps_raw, session_raw, activity_raw, file_id_raw = parse_fit(data)
        gaps = detect_gaps(records)

        if len(gaps) != len(params):
            with JOBS_LOCK:
                JOBS[job_id]['status'] = 'error'
                JOBS[job_id]['error'] = f'Gap count mismatch: found {len(gaps)}, got {len(params)} param sets'
            return

        merged = list(records)
        added_distances = {}

        for gi in reversed(range(len(gaps))):
            gap = gaps[gi]
            stop_s = float(params[gi]['stop_s'])
            ride_ms = float(params[gi]['ride_kmh']) / 3.6
            synth, added_m = build_synthetic_records(merged, gap, stop_s, ride_ms)

            for j in range(gap['idx_after'], len(merged)):
                if merged[j].get('distance') is not None:
                    merged[j] = dict(merged[j])
                    merged[j]['distance'] += added_m

            merged = merged[:gap['idx_after']] + synth + merged[gap['idx_after']:]
            gap_ts_after_ms = records[gap['idx_after']]['timestamp_unix_ms']
            added_distances[gap_ts_after_ms] = added_m

        original_dist_m = records[-1].get('distance') or 0
        total_added_m = sum(added_distances.values())

        fit_bytes = write_fit(merged, laps_raw, session_raw, activity_raw, file_id_raw, added_distances)

        # Write to temp file for download
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.fit')
        tmp.write(fit_bytes)
        tmp.close()

        out_name = filename.rsplit('.', 1)[0] + '_repaired.fit'

        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'done'
            JOBS[job_id]['tmp_path'] = tmp.name
            JOBS[job_id]['out_name'] = out_name
            JOBS[job_id]['original_dist_km'] = round(original_dist_m / 1000, 2)
            JOBS[job_id]['added_dist_km'] = round(total_added_m / 1000, 2)
            JOBS[job_id]['new_dist_km'] = round((original_dist_m + total_added_m) / 1000, 2)

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = f'{e}\n{traceback.format_exc()}'


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    """Start an analyze job, return job_id immediately."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    data = request.files['file'].read()
    job_id = str(uuid.uuid4())

    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'running', 'created_at': time.time(), 'type': 'analyze'}

    t = threading.Thread(target=run_analyze, args=(job_id, data), daemon=True)
    t.start()

    cleanup_old_jobs()
    return jsonify({'job_id': job_id})


@app.route('/repair', methods=['POST'])
def repair():
    """Start a repair job, return job_id immediately."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    if 'params' not in request.form:
        return jsonify({'error': 'No params supplied'}), 400

    data = request.files['file'].read()
    filename = request.files['file'].filename or 'activity.fit'
    params = json.loads(request.form['params'])
    job_id = str(uuid.uuid4())

    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'running', 'created_at': time.time(), 'type': 'repair'}

    t = threading.Thread(target=run_repair, args=(job_id, data, filename, params), daemon=True)
    t.start()

    cleanup_old_jobs()
    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    """Poll for job status."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        return jsonify({'status': 'not_found'}), 404

    if job['status'] == 'running':
        return jsonify({'status': 'running'})

    if job['status'] == 'error':
        return jsonify({'status': 'error', 'error': job.get('error', 'Unknown error')}), 500

    # Done
    if job.get('type') == 'analyze':
        return jsonify({'status': 'done', **job['result']})
    else:
        # repair — return metadata; file downloaded separately
        return jsonify({
            'status': 'done',
            'original_dist_km': job['original_dist_km'],
            'added_dist_km': job['added_dist_km'],
            'new_dist_km': job['new_dist_km'],
            'out_name': job['out_name'],
        })


@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    """Download the repaired FIT file."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None or job.get('status') != 'done' or job.get('type') != 'repair':
        return jsonify({'error': 'Job not found or not ready'}), 404

    tmp_path = job.get('tmp_path')
    out_name = job.get('out_name', 'repaired.fit')

    if not tmp_path or not os.path.exists(tmp_path):
        return jsonify({'error': 'Output file not found'}), 404

    return send_file(
        tmp_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=out_name,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
