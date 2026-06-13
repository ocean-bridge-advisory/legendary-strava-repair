"""
FIT Gap Repair v3 — Flask backend
- Async job processing (workaround for Render 30s limit)
- Gap classification: < 20m GPS = rest stop, >= 20m = recording gap
- Speed default: avg over 30 min before gap
- Pre-gap averages (50 records before) for HR/power/cadence/alt/temp
"""

import io, json, math, os, tempfile, threading, time, traceback, uuid
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
from fit_tool.profile.profile_type import Activity, Event, EventType, FileType, Manufacturer, Sport

app = Flask(__name__)
CORS(app)

SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 1800

REST_STOP_THRESHOLD_M = 20.0   # GPS jump below this = rest stop
SPEED_WINDOW_S = 30 * 60       # 30 minutes in seconds
PRE_GAP_AVG_N = 50             # records before gap for HR/power/etc averages


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def gv(frame, name, default=None):
    try: return frame.get_value(name) if frame.has_field(name) else default
    except: return default

def ts_to_unix_ms(dt):
    if dt is None: return None
    import calendar; return int(calendar.timegm(dt.timetuple()) * 1000)

def semicircles_to_deg(v):
    if v is None or not isinstance(v, (int, float)): return None
    return v * SEMICIRCLE_TO_DEG

def cleanup_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired = [jid for jid, j in JOBS.items() if now - j['created_at'] > JOB_TTL]
        for jid in expired:
            tmp = JOBS[jid].get('tmp_path')
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except: pass
            del JOBS[jid]


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_fit(data: bytes):
    records, laps, session, activity, file_id = [], [], None, None, None
    with fitdecode.FitReader(io.BytesIO(data)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage): continue
            name = frame.name
            if name == 'record':
                ts = gv(frame, 'timestamp')
                if ts is None: continue
                records.append({
                    'timestamp_unix_ms': ts_to_unix_ms(ts),
                    'lat': semicircles_to_deg(gv(frame, 'position_lat')),
                    'lon': semicircles_to_deg(gv(frame, 'position_long')),
                    'distance': gv(frame, 'distance'),
                    'speed': gv(frame, 'enhanced_speed') or gv(frame, 'speed'),
                    'heart_rate': gv(frame, 'heart_rate'),
                    'power': gv(frame, 'power'),
                    'cadence': gv(frame, 'cadence'),
                    'altitude': gv(frame, 'enhanced_altitude') or gv(frame, 'altitude'),
                    'temperature': gv(frame, 'temperature'),
                })
            elif name == 'lap': laps.append(frame)
            elif name == 'session' and session is None: session = frame
            elif name == 'activity' and activity is None: activity = frame
            elif name == 'file_id' and file_id is None: file_id = frame
    return records, laps, session, activity, file_id


# ── Gap detection & analysis ──────────────────────────────────────────────────

def detect_gaps(records, min_time_gap=30, min_gps_gap=5):
    gaps = []
    for i in range(1, len(records)):
        r0, r1 = records[i-1], records[i]
        dt = (r1['timestamp_unix_ms'] - r0['timestamp_unix_ms']) / 1000.0
        if dt <= min_time_gap: continue
        lat0, lon0 = r0.get('lat'), r0.get('lon')
        lat1, lon1 = r1.get('lat'), r1.get('lon')
        if None in (lat0, lon0, lat1, lon1): continue
        gps_gap = haversine_m(lat0, lon0, lat1, lon1)
        gaps.append({
            'idx_before': i-1,
            'idx_after': i,
            'time_gap_s': dt,
            'gps_gap_m': gps_gap,
            'dist_before_m': r0.get('distance') or 0,
            'is_rest_stop': gps_gap < REST_STOP_THRESHOLD_M,
        })
    return gaps


def gap_analysis(records, gap):
    """Compute speed default and pre-gap averages for one gap."""
    idx = gap['idx_before']
    t_cutoff = records[idx]['timestamp_unix_ms'] - SPEED_WINDOW_S * 1000

    # Speed: avg over 30 min before gap (moving records only, speed > 0.5 m/s)
    speed_window = [r for r in records[:idx+1]
                    if r['timestamp_unix_ms'] >= t_cutoff
                    and r.get('speed') is not None and r['speed'] > 0.5]
    avg_speed_ms = sum(r['speed'] for r in speed_window) / len(speed_window) if speed_window else 5.0

    # Pre-gap averages: 50 records before gap
    pre_window = records[max(0, idx - PRE_GAP_AVG_N + 1): idx + 1]

    def avg(key):
        vals = [r[key] for r in pre_window if r.get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        'avg_speed_kmh': round(avg_speed_ms * 3.6, 1),
        'avg_speed_ms': avg_speed_ms,
        'heart_rate': avg('heart_rate'),
        'power': avg('power'),
        'cadence': avg('cadence'),
        'altitude': avg('altitude'),
        'temperature': avg('temperature'),
    }


# ── Synthetic records ─────────────────────────────────────────────────────────

def build_synthetic_records(records, gap, stop_time_s, road_distance_m, avg_speed_ms, pre_avgs, route_coords=None):
    r0 = records[gap['idx_before']]
    r1 = records[gap['idx_after']]

    total_gap_s = gap['time_gap_s']
    ride_time_s = max(0, total_gap_s - stop_time_s)
    # Use user-supplied road distance; speed is road_distance / ride_time
    if ride_time_s > 0:
        effective_speed_ms = road_distance_m / ride_time_s
    else:
        effective_speed_ms = 0.0

    lat0, lon0 = r0['lat'], r0['lon']
    lat1, lon1 = r1['lat'], r1['lon']
    dist_start = r0.get('distance') or 0

    n = int(total_gap_s) - 1
    if n <= 0:
        return [], road_distance_m

    # Build route interpolator
    if route_coords and len(route_coords) >= 2:
        cum = [0.0]
        for j in range(1, len(route_coords)):
            p0, p1 = route_coords[j-1], route_coords[j]
            cum.append(cum[-1] + haversine_m(p0[0],p0[1],p1[0],p1[1]))
        total_route_m = cum[-1]
        def get_pos(ride_frac, linear_frac):
            target = ride_frac * total_route_m
            for j in range(1, len(cum)):
                if cum[j] >= target or j == len(cum)-1:
                    sf = (target - cum[j-1]) / max(1e-9, cum[j] - cum[j-1])
                    sf = max(0.0, min(1.0, sf))
                    p0, p1 = route_coords[j-1], route_coords[j]
                    return p0[0]+sf*(p1[0]-p0[0]), p0[1]+sf*(p1[1]-p0[1])
            return route_coords[-1][0], route_coords[-1][1]
    else:
        def get_pos(ride_frac, linear_frac):
            f = linear_frac
            return lat0+f*(lat1-lat0), lon0+f*(lon1-lon0)

    synth = []
    t0_ms = r0['timestamp_unix_ms']
    for i in range(1, n + 1):
        frac = i / (n + 1)
        ride_frac = max(0, (i - stop_time_s) / ride_time_s) if ride_time_s > 0 else 0
        ride_frac = min(ride_frac, 1.0)
        lat, lon = get_pos(ride_frac, frac)
        synth.append({
            'timestamp_unix_ms': int(t0_ms + i * 1000),
            'lat': lat, 'lon': lon,
            'distance': dist_start + ride_frac * road_distance_m,
            'speed': effective_speed_ms if i > stop_time_s else 0.0,
            'heart_rate': pre_avgs.get('heart_rate'),
            'power': pre_avgs.get('power'),
            'cadence': pre_avgs.get('cadence'),
            'altitude': pre_avgs.get('altitude'),
            'temperature': pre_avgs.get('temperature'),
        })
    return synth, road_distance_m


# ── FIT writer ────────────────────────────────────────────────────────────────

def write_fit(all_records, laps_raw, session_raw, activity_raw, file_id_raw, added_distances):
    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    fid = FileIdMessage(); fid.type = FileType.ACTIVITY
    if file_id_raw:
        for attr, cast, setter in [
            ('manufacturer', lambda v: Manufacturer(int(v)), 'manufacturer'),
            ('product', int, 'product'),
            ('serial_number', int, 'serial_number'),
        ]:
            v = gv(file_id_raw, attr)
            if v is not None:
                try: setattr(fid, setter, cast(v))
                except: pass
        tc = gv(file_id_raw, 'time_created')
        if tc is not None: fid.time_created = ts_to_unix_ms(tc)
    builder.add(fid)

    if all_records:
        ev = EventMessage(); ev.timestamp = all_records[0]['timestamp_unix_ms']
        ev.event = Event.TIMER; ev.event_type = EventType.START; builder.add(ev)

    for r in all_records:
        rm = RecordMessage(); rm.timestamp = r['timestamp_unix_ms']
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
        ev = EventMessage(); ev.timestamp = all_records[-1]['timestamp_unix_ms']
        ev.event = Event.TIMER; ev.event_type = EventType.STOP_ALL; builder.add(ev)

    total_added = sum(added_distances.values())
    for lap_raw in laps_raw:
        lm = LapMessage()
        ts = gv(lap_raw, 'timestamp'); ts_unix_ms = None
        if ts is not None: ts_unix_ms = ts_to_unix_ms(ts); lm.timestamp = ts_unix_ms
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
        lm.event = Event.LAP; lm.event_type = EventType.STOP; builder.add(lm)

    sm = SessionMessage(); sm.event = Event.SESSION; sm.event_type = EventType.STOP
    if all_records:
        sm.start_time = all_records[0]['timestamp_unix_ms']
        sm.timestamp = all_records[-1]['timestamp_unix_ms']
        sm.total_elapsed_time = (all_records[-1]['timestamp_unix_ms'] - all_records[0]['timestamp_unix_ms']) / 1000.0
    if session_raw:
        tt = gv(session_raw, 'total_timer_time')
        if tt is not None: sm.total_timer_time = float(tt)
        td = gv(session_raw, 'total_distance')
        if td is not None: sm.total_distance = float(td) + total_added
        for attr, cast in [('avg_speed',float),('max_speed',float),('avg_heart_rate',int),('avg_power',int),('total_calories',int)]:
            v = gv(session_raw, attr)
            if v is not None:
                try: setattr(sm, attr, cast(v))
                except: pass
        sport_val = gv(session_raw, 'sport')
        try: sm.sport = Sport(int(sport_val)) if sport_val is not None else Sport.CYCLING
        except: sm.sport = Sport.CYCLING
    else:
        sm.sport = Sport.CYCLING
    builder.add(sm)

    am = ActivityMessage()
    if all_records:
        am.timestamp = all_records[-1]['timestamp_unix_ms']
        am.total_timer_time = (all_records[-1]['timestamp_unix_ms'] - all_records[0]['timestamp_unix_ms']) / 1000.0
    am.num_sessions = 1; am.type = Activity.MANUAL
    am.event = Event.ACTIVITY; am.event_type = EventType.STOP; builder.add(am)

    return builder.build().to_bytes()


# ── Background workers ────────────────────────────────────────────────────────

def run_analyze(job_id, data):
    try:
        records, laps_raw, session_raw, _, _ = parse_fit(data)
        if not records:
            with JOBS_LOCK: JOBS[job_id].update({'status':'error','error':'No record messages found'}); return

        gaps = detect_gaps(records)
        original_dist_m = records[-1].get('distance') or 0

        gap_list = []
        for g in gaps:
            analysis = gap_analysis(records, g)
            r0b = records[g['idx_before']]
            r1b = records[g['idx_after']]
            track_before = [[r['lat'],r['lon']] for r in records[max(0,g['idx_before']-99):g['idx_before']+1] if r.get('lat') is not None]
            track_after  = [[r['lat'],r['lon']] for r in records[g['idx_after']:g['idx_after']+100] if r.get('lat') is not None]
            gap_list.append({
                'idx_before': g['idx_before'],
                'idx_after': g['idx_after'],
                'time_gap_s': g['time_gap_s'],
                'gps_gap_m': round(g['gps_gap_m'], 1),
                'dist_before_km': round(g['dist_before_m'] / 1000, 2),
                'is_rest_stop': g['is_rest_stop'],
                'lat_before': r0b.get('lat'), 'lon_before': r0b.get('lon'),
                'lat_after':  r1b.get('lat'), 'lon_after':  r1b.get('lon'),
                'track_before': track_before,
                'track_after':  track_after,
                'avg_speed_kmh': analysis['avg_speed_kmh'],
                'avg_speed_ms': analysis['avg_speed_ms'],
                'heart_rate': analysis['heart_rate'],
                'power': analysis['power'],
                'cadence': analysis['cadence'],
                'altitude': analysis['altitude'],
                'temperature': analysis['temperature'],
            })

        with JOBS_LOCK:
            JOBS[job_id].update({
                'status': 'done',
                'result': {
                    'num_records': len(records),
                    'num_laps': len(laps_raw),
                    'original_dist_km': round(original_dist_m / 1000, 2),
                    'gaps': gap_list,
                }
            })
    except Exception as e:
        with JOBS_LOCK: JOBS[job_id].update({'status':'error','error':str(e)})


def run_repair(job_id, data, filename, params):
    """params: list of {include, stop_s, road_distance_m} per gap"""
    try:
        records, laps_raw, session_raw, activity_raw, file_id_raw = parse_fit(data)
        gaps = detect_gaps(records)

        if len(gaps) != len(params):
            with JOBS_LOCK: JOBS[job_id].update({'status':'error','error':f'Gap count mismatch'}); return

        merged = list(records)
        added_distances = {}

        for gi in reversed(range(len(gaps))):
            if not params[gi].get('include', False):
                continue
            gap = gaps[gi]
            stop_s = float(params[gi]['stop_s'])
            road_distance_m = float(params[gi]['road_distance_m'])
            analysis = gap_analysis(records, gap)
            avg_speed_ms = analysis['avg_speed_ms']
            pre_avgs = {k: analysis[k] for k in ('heart_rate','power','cadence','altitude','temperature')}

            route_coords = params[gi].get('route_coords')
            synth, added_m = build_synthetic_records(
                merged, gap, stop_s, road_distance_m, avg_speed_ms, pre_avgs, route_coords)

            for j in range(gap['idx_after'], len(merged)):
                if merged[j].get('distance') is not None:
                    merged[j] = dict(merged[j]); merged[j]['distance'] += added_m

            merged = merged[:gap['idx_after']] + synth + merged[gap['idx_after']:]
            gap_ts_after_ms = records[gap['idx_after']]['timestamp_unix_ms']
            added_distances[gap_ts_after_ms] = added_m

        original_dist_m = records[-1].get('distance') or 0
        total_added_m = sum(added_distances.values())

        fit_bytes = write_fit(merged, laps_raw, session_raw, activity_raw, file_id_raw, added_distances)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.fit')
        tmp.write(fit_bytes); tmp.close()
        out_name = filename.rsplit('.', 1)[0] + '_repaired.fit'

        with JOBS_LOCK:
            JOBS[job_id].update({
                'status': 'done', 'tmp_path': tmp.name, 'out_name': out_name,
                'original_dist_km': round(original_dist_m / 1000, 2),
                'added_dist_km': round(total_added_m / 1000, 2),
                'new_dist_km': round((original_dist_m + total_added_m) / 1000, 2),
            })
    except Exception as e:
        with JOBS_LOCK: JOBS[job_id].update({'status':'error','error':f'{e}\n{traceback.format_exc()}'})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}), 400
    data = request.files['file'].read()
    job_id = str(uuid.uuid4())
    with JOBS_LOCK: JOBS[job_id] = {'status':'running','created_at':time.time(),'type':'analyze'}
    threading.Thread(target=run_analyze, args=(job_id, data), daemon=True).start()
    cleanup_old_jobs()
    return jsonify({'job_id': job_id})


@app.route('/repair', methods=['POST'])
def repair():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}), 400
    if 'params' not in request.form: return jsonify({'error':'No params supplied'}), 400
    data = request.files['file'].read()
    filename = request.files['file'].filename or 'activity.fit'
    params = json.loads(request.form['params'])
    job_id = str(uuid.uuid4())
    with JOBS_LOCK: JOBS[job_id] = {'status':'running','created_at':time.time(),'type':'repair'}
    threading.Thread(target=run_repair, args=(job_id, data, filename, params), daemon=True).start()
    cleanup_old_jobs()
    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    with JOBS_LOCK: job = JOBS.get(job_id)
    if job is None: return jsonify({'status':'not_found'}), 404
    if job['status'] == 'running': return jsonify({'status':'running'})
    if job['status'] == 'error': return jsonify({'status':'error','error':job.get('error','Unknown error')}), 500
    if job.get('type') == 'analyze': return jsonify({'status':'done', **job['result']})
    return jsonify({
        'status':'done', 'original_dist_km':job['original_dist_km'],
        'added_dist_km':job['added_dist_km'], 'new_dist_km':job['new_dist_km'],
        'out_name':job['out_name'],
    })


@app.route('/download/<job_id>', methods=['GET'])
def download(job_id):
    with JOBS_LOCK: job = JOBS.get(job_id)
    if job is None or job.get('status') != 'done' or job.get('type') != 'repair':
        return jsonify({'error':'Job not found or not ready'}), 404
    tmp_path = job.get('tmp_path')
    if not tmp_path or not os.path.exists(tmp_path):
        return jsonify({'error':'Output file not found'}), 404
    return send_file(tmp_path, mimetype='application/octet-stream',
                     as_attachment=True, download_name=job.get('out_name','repaired.fit'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
