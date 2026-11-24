const $ = id => document.getElementById(id);
const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(body || {})}).then(r => r.json());

// ---------- live status ----------
async function refresh() {
  let s;
  try { s = await fetch('/api/status').then(r => r.json()); }
  catch { $('error').textContent = 'Lost connection to the Alerta server.'; $('error').classList.remove('hidden'); return; }

  $('error').textContent = s.error || '';
  $('error').classList.toggle('hidden', !s.error);
  const lv = s.level || 0;
  $('pill').textContent = s.level_name; $('pill').className = 'pill level' + lv;
  $('levelName').textContent = s.level_name;
  document.querySelectorAll('.levels span').forEach(e => e.classList.toggle('on', +e.dataset.l <= lv && lv > 0));
  $('videoWrap').className = 'video-wrap' + (lv ? ' l' + lv : '');
  $('banner').textContent = s.distracted ? 'LOOK AT THE ROAD' : 'WAKE UP!';
  $('banner').classList.toggle('hidden', !(s.distracted || lv === 3));
  if (s.fps !== undefined) $('fps').textContent = s.fps + ' fps';
  $('session').textContent = `Session ${s.session_minutes ?? 0} min · ${s.alarms_total ?? 0} alarms`;
  $('pauseBtn').textContent = s.paused ? 'Resume camera' : 'Pause camera';

  // eyes + openness vs personal threshold (feature 8)
  $('eyeState').textContent = s.eye_state ? s.eye_state[0].toUpperCase() + s.eye_state.slice(1) : '–';
  const p = s.p_open ?? 0;
  $('openBar').style.width = (p * 100) + '%';
  $('thrMark').style.left = ((s.threshold ?? .5) * 100) + '%';
  $('openText').textContent = s.p_open == null ? 'openness –'
    : `openness ${p.toFixed(2)} · closed below ${(+s.threshold).toFixed(2)}`;
  $('closedFor').textContent = (s.closed_for ?? 0).toFixed(1) + ' s';

  // blink rate (feature 4)
  $('bpm').textContent = s.blinks_per_min == null ? 'measuring…' : s.blinks_per_min + ' /min';
  $('blinkLen').textContent = s.avg_blink_ms == null ? 'avg blink –' : `avg blink ${s.avg_blink_ms} ms`
    + (s.avg_blink_ms > 400 ? ' · slow blinks' : '');

  // distraction (feature 6)
  $('attention').textContent = s.distracted ? 'Distracted' : (s.face ? 'On road' : 'Face lost');
  $('noFace').textContent = s.face ? 'face in view' : `no face for ${s.no_face_for} s`;

  // night mode (feature 9)
  $('nightInfo').textContent = `Brightness ${s.brightness ?? '–'} · enhancement ${s.night_active ? 'ON' : 'off'}`;

  // calibration progress
  $('calibProg').classList.toggle('hidden', !s.calibrating);
  if (s.calibrating) $('calibProg').firstElementChild.style.width = (s.calib_progress * 100) + '%';
  $('calibMsg').textContent = s.calib_message || (s.baseline_open != null
    ? `Calibrated: your openness ${s.baseline_open}, closed below ${s.threshold}` : 'Not calibrated (default threshold 0.5)');
  $('calibBtn').disabled = s.calibrating;

  // emergency alert status (feature 11)
  const t = s.telegram || {};
  if (t.sending) $('tgMsg').textContent = 'Sending…';
  else if (t.last_error) $('tgMsg').textContent = 'Last send failed: ' + t.last_error;
  else if (t.last_sent) $('tgMsg').textContent = 'Last message sent at ' + new Date(t.last_sent * 1000).toLocaleTimeString();
  else $('tgMsg').textContent = t.enabled ? (t.configured ? `Armed · ${s.critical_recent} critical alarm(s) in last 10 min`
    : 'Enabled, but token / chat id missing') : 'Disabled';

  $('log').innerHTML = (s.events || []).map(e =>
    `<tr><td>${e.time}</td><td>${e.event}</td><td>${e.detail}</td></tr>`).join('')
    || '<tr><td colspan="3" class="muted">No events yet</td></tr>';
}
setInterval(refresh, 250);
refresh();

// ---------- controls ----------
function showSettings(c) {
  document.querySelectorAll('#night button').forEach(b => b.classList.toggle('on', b.dataset.v === c.night_mode));
  $('volume').value = c.volume; $('muted').checked = c.muted; $('distract').value = c.distraction_seconds;
}
fetch('/api/settings').then(r => r.json()).then(showSettings);
document.querySelectorAll('#night button').forEach(b =>
  b.onclick = () => post('/api/settings', {night_mode: b.dataset.v}).then(showSettings));
$('volume').oninput = () => post('/api/settings', {volume: +$('volume').value});
$('muted').onchange = () => post('/api/settings', {muted: $('muted').checked});
$('distract').onchange = () => post('/api/settings', {distraction_seconds: +$('distract').value}).then(showSettings);
$('calibBtn').onclick = () => post('/api/calibrate');
$('calibReset').onclick = () => post('/api/settings', {reset_calibration: true});
$('pauseBtn').onclick = () => post('/api/pause', {paused: $('pauseBtn').textContent.startsWith('Pause')});

function showTelegram(c) {
  $('tgEnabled').checked = c.enabled; $('tgChat').value = c.chat_id; $('tgAfter').value = c.emergency_after;
  $('tgToken').value = ''; $('tgToken').placeholder = c.token_saved ? 'saved (leave blank to keep)' : 'from @BotFather';
}
fetch('/api/telegram').then(r => r.json()).then(showTelegram);
$('tgSave').onclick = () => post('/api/telegram', {enabled: $('tgEnabled').checked, bot_token: $('tgToken').value,
  chat_id: $('tgChat').value, emergency_after: +$('tgAfter').value}).then(showTelegram);
$('tgTest').onclick = () => post('/api/telegram/test').then(r => { if (!r.ok) $('tgMsg').textContent = r.error; });
