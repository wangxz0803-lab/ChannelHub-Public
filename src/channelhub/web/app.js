'use strict';
const $ = id => document.getElementById(id);
let defaults, capabilities, currentPage = 'workspace';
const n = id => Number($(id).value);
const labels = {queued:'排队中', running:'生成中', completed:'已完成', failed:'生成失败', cancelled:'已取消', interrupted:'已中断'};
const clone = object => JSON.parse(JSON.stringify(object));
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map(e => `${e.loc.join('.')}: ${e.msg}`).join('\n') : data.detail || '请求失败');
  return data;
}
function notice(message, error = false) {
  const box = $('notice'); box.textContent = message; box.className = error ? 'error' : ''; box.hidden = false;
  $(currentPage).prepend(box);
}
function page(name) {
  currentPage = name;
  for (const p of document.querySelectorAll('.page')) p.hidden = p.id !== name;
  for (const nav of document.querySelectorAll('.nav')) nav.classList.toggle('active', nav.dataset.page === name);
  $('page-label').textContent = {workspace:'配置与生成',runs:'实验记录',guide:'数据与接入'}[name];
  if (name === 'runs') refreshJobs();
  window.scrollTo({top:0, behavior:'smooth'});
}
for (const button of document.querySelectorAll('[data-page]')) button.addEventListener('click', () => page(button.dataset.page));
for (const family of ['CDL','TDL']) for (const letter of 'ABCDE') {
  const option = new Option(`${family}-${letter}`, `${family}-${letter}`);
  $('profile').add(option);
}
$('profile').value = 'CDL-C';
function vector(id) {
  const result = $(id).value.split(/[,，\s]+/).filter(Boolean).map(Number);
  if (result.length !== 3 || !result.every(Number.isFinite)) throw new Error('位置和速度需要三个数字，例如 50, 0, 1.5');
  return result;
}
function formConfig() {
  const cfg = clone(defaults);
  Object.assign(cfg, {engine:$('engine').value,profile:$('profile').value,carrier_ghz:n('carrier'),scs_khz:n('scs'),num_rb:n('rb'),num_slots:n('slots'),delay_spread_ns:n('delay'),seed:n('seed'),noise_psd_dbm_hz:n('noise'),noise_figure_db:n('nf'),noise_enabled:$('noise-on').checked});
  Object.assign(cfg.tx, {position_m:vector('tx-position'),power_dbm:n('power')});
  Object.assign(cfg.tx.array, {rows:n('tx-rows'),cols:n('tx-cols'),pattern:$('pattern').value});
  Object.assign(cfg.rx, {position_m:vector('rx-position'),velocity_m_s:vector('velocity')});
  Object.assign(cfg.rx.array, {rows:n('rx-rows'),cols:n('rx-cols'),pattern:$('pattern').value});
  Object.assign(cfg.tx.waveform, {kind:$('waveform').value,identity:n('identity'),beam_index:n('beam')});
  Object.assign(cfg.tx.waveform.srs, {c_srs:n('srs-c'),comb:n('comb'),ports:n('srs-ports'),period_slots:n('period')});
  Object.assign(cfg.rt, {scene:$('scene').value,max_depth:n('depth'),samples_per_source:n('rays')});
  cfg.interferers = [];
  if ($('interference-on').checked) for (let i = 0; i < n('interferer-count'); i++) {
    const tx = clone(cfg.tx), angle = (i+0.35)*2*Math.PI/n('interferer-count');
    Object.assign(tx, {position_m:[cfg.rx.position_m[0]+80*Math.cos(angle),cfg.rx.position_m[1]+80*Math.sin(angle),10],power_dbm:n('interferer-power'),activity:n('activity')});
    Object.assign(tx.waveform, {kind:$('interferer-wave').value,identity:(i+1)*13,beam_index:0});
    cfg.interferers.push(tx);
  }
  return cfg;
}
function config() { return $('advanced-on').checked ? JSON.parse($('config-json').value) : formConfig(); }
function update() {
  const rt = $('engine').value === 'sionna_rt', wave = $('waveform').value;
  $('rt-fields').hidden = !rt; $('profile-label').hidden = rt; $('delay-label').hidden = rt;
  $('srs-fields').hidden = wave !== 'srs'; $('custom-fields').hidden = wave !== 'custom';
  $('beam-label').hidden = !['ssb','csi_rs'].includes(wave);
  $('power').disabled = wave === 'custom';
  $('wave-note').textContent = {srs:'SRS 默认在每个时隙最后一个 OFDM 符号发送；高级配置可设置跳频、周期与资源偏移。',csi_rs:'CSI-RS 采用 row 2：单端口、密度 1、无 CDM，默认位于符号 4。',ssb:'SSB 仅生成 PSS、SSS 和 PBCH-DMRS；不包含 PBCH 编码数据。至少需要 20 RB。',qpsk:'每个发射端口在整个资源网格独立发送 QPSK 符号。',custom:'自定义 X 直接作为物理发射网格使用，保持用户指定幅度。'}[wave];
  for (const id of ['interferer-count','interferer-power','activity','interferer-wave']) $(id).disabled = !$('interference-on').checked;
  if (!defaults) return;
  try {
    const cfg = config(), ports = a => a.rows*a.cols*(a.polarization === 'VH' ? 2 : 1);
    $('summary-engine').textContent = cfg.engine === 'statistical' ? cfg.profile : 'Sionna RT';
    $('summary-grid').textContent = `${cfg.num_slots*14} × ${cfg.num_rb*12}`;
    $('summary-ports').textContent = `${ports(cfg.tx.array)} → ${ports(cfg.rx.array)}`;
    $('summary-bw').textContent = `${(cfg.num_rb*12*cfg.scs_khz/1000).toFixed(2)} MHz`;
    const key = cfg.engine === 'statistical' ? 'sionna' : 'sionna-rt';
    $('engine-status').textContent = capabilities?.installed[key] ? `${key} ${capabilities.installed[key]} 已安装` : `${key} 未安装，请按 README 配置引擎`;
    topology(cfg);
  } catch { $('summary-engine').textContent = '配置待完善'; }
}
function topology(cfg) {
  const points = [{p:cfg.tx.position_m,color:'#245ed0',label:'TX'},{p:cfg.rx.position_m,color:'#087f75',label:'RX'},...(cfg.interferers||[]).map((t,i)=>({p:t.position_m,color:'#d97a23',label:`I${i+1}`}))];
  const xs = points.map(x=>x.p[0]), ys = points.map(x=>x.p[1]);
  const x0 = Math.min(...xs),y0 = Math.min(...ys),dx = Math.max(...xs)-x0,dy = Math.max(...ys)-y0;
  const scale = Math.min(220/Math.max(dx,1),110/Math.max(dy,1));
  const coords = points.map(p=>({x:150+(p.p[0]-x0-dx/2)*scale,y:90-(p.p[1]-y0-dy/2)*scale,...p}));
  const svg = $('topology'); svg.replaceChildren();
  function element(tag,attrs) { const e = document.createElementNS('http://www.w3.org/2000/svg',tag);for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);svg.append(e);return e; }
  for (let x=20;x<300;x+=20) for(let y=10;y<180;y+=20) element('circle',{cx:x,cy:y,r:.7,fill:'#d7e1ea'});
  for (let i=0;i<coords.length;i++) {
    const c=coords[i];if(i!==1) element('line',{x1:c.x,y1:c.y,x2:coords[1].x,y2:coords[1].y,stroke:i?'#eac8aa':'#a6bfdc','stroke-dasharray':'4 4'});
  }
  for(const c of coords){element('circle',{cx:c.x,cy:c.y,r:10,fill:'white',stroke:c.color,'stroke-width':1.5});element('circle',{cx:c.x,cy:c.y,r:3,fill:c.color});element('text',{x:c.x,y:c.y+25,fill:c.color,'text-anchor':'middle','font-size':10,'font-family':'sans-serif'}).textContent=c.label;}
}
$('experiment-form').addEventListener('input', update);
$('experiment-form').addEventListener('change', update);
$('sync-json').onclick = () => {try{$('config-json').value = JSON.stringify(formConfig(),null,2);update();}catch(e){notice(e.message,true);}};
$('reset').onclick = () => {$('experiment-form').reset();$('profile').value='CDL-C';$('config-json').value=JSON.stringify(defaults,null,2);update();};
function saveFile(name,text){const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export-config').onclick=()=>{try{saveFile('channelhub-config.json',JSON.stringify(config(),null,2));}catch(e){notice(e.message,true);}};
$('import-config').onchange=async()=>{try{const f=$('import-config').files[0];if(!f)return;if(f.size>512000)throw new Error('配置文件过大');const c=JSON.parse(await f.text());$('config-json').value=JSON.stringify(c,null,2);$('advanced-on').checked=true;update();notice('配置已导入，生成实验前会进行完整校验。');}catch(e){notice(e.message,true);}};
$('experiment-form').onsubmit=async event=>{
  event.preventDefault();$('run').disabled=true;$('notice').hidden=true;
  try{
    const cfg=config();let custom_x_id=null;
    if(cfg.tx.waveform.kind==='custom'){
      const file=$('custom-x').files[0];if(!file)throw new Error('请先上传自定义 X 的 .npy 文件');
      const data=await api('/api/uploads/x',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});custom_x_id=data.id;
    }
    await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:cfg,custom_x_id})});
    page('runs');
  }catch(e){notice(e.message,true);}finally{$('run').disabled=false;}
};
function el(tag,className,text){const element=document.createElement(tag);if(className)element.className=className;if(text!==undefined)element.textContent=text;return element;}
async function refreshJobs(){
  try{
    const jobs=await api('/api/jobs');$('job-count').textContent=jobs.length;
    const list=$('job-list');list.replaceChildren();
    if(!jobs.length){const empty=el('div','card empty');empty.append(el('strong','','开始你的第一个实验'),el('p','','配置一个场景，生成可复现的 H、X 和 Y。'));list.append(empty);return;}
    for(const job of jobs){
      const card=el('article','card job'),info=el('div'),state=el('div'),actions=el('div','job-actions');
      info.append(el('h3','',`${job.engine==='statistical'?job.profile:'Sionna RT'} · ${job.waveform.toUpperCase()}`),el('span','job-id',`${job.id.slice(0,12)} · ${new Date(job.created_at*1000).toLocaleString()}`),el('p','help',job.message));
      state.append(el('div',`job-state ${job.status}`,`${labels[job.status]||job.status} · ${job.percent}%`));const progress=el('div','progress'),bar=el('i');bar.style.width=`${job.percent}%`;progress.append(bar);state.append(progress);
      if(job.status==='completed'){
        const preview=el('button','secondary','查看结果');preview.onclick=()=>showResult(job.id);actions.append(preview);
        const download=el('a','primary','下载 ↓');download.href=`/api/jobs/${job.id}/download/result.zip`;actions.append(download);
      }else if(['queued','running'].includes(job.status)){const cancel=el('button','secondary','取消');cancel.onclick=async()=>{await api(`/api/jobs/${job.id}/cancel`,{method:'POST'});refreshJobs();};actions.append(cancel);}
      else{const log=el('a','secondary','查看日志');log.href=`/api/jobs/${job.id}/download/worker.log`;actions.append(log);}
      card.append(info,state,actions);list.append(card);
    }
  }catch(e){if(currentPage==='runs')notice(e.message,true);}
}
function plot(canvas,x,series){
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
  const values=series.flatMap(s=>s.y).filter(y=>Number.isFinite(y)&&y>-300);let ymin=Math.floor(Math.min(...values)/10)*10-5,ymax=Math.ceil(Math.max(...values)/10)*10+5;
  if(!values.length){ymin=-150;ymax=0;} const xmin=x[0]||0,xmax=x.at(-1)||1;
  const px=v=>54+(v-xmin)/(xmax-xmin||1)*(w-75),py=v=>h-32-(Math.max(ymin,v)-ymin)/(ymax-ymin)*(h-52);
  ctx.font='11px Segoe UI';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;ctx.strokeStyle='#e4ebf1';ctx.beginPath();ctx.moveTo(52,py(y));ctx.lineTo(w-20,py(y));ctx.stroke();ctx.fillStyle='#8495a3';ctx.fillText(y.toFixed(0),9,py(y)+4);const xv=xmin+(xmax-xmin)*i/4;ctx.fillText(xv.toFixed(2),px(xv)-10,h-10);}
  for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();s.y.forEach((y,i)=>{if(i===0)ctx.moveTo(px(x[i]),py(y));else ctx.lineTo(px(x[i]),py(y));});ctx.stroke();}
}
async function showResult(id){try{const r=await api(`/api/jobs/${id}/preview`);$('result').hidden=false;$('result-description').textContent=`${id.slice(0,12)} · ${r.metadata.schema}`;$('download-result').href=`/api/jobs/${id}/download/result.zip`;$('array-info').textContent=JSON.stringify(r.metadata.arrays,null,2);plot($('channel-plot'),r.frequency_mhz,[{y:r.channel_db,color:'#245ed0'}]);plot($('power-plot'),r.time_ms,[{y:r.powers_dbm.signal,color:'#2667ca'},{y:r.powers_dbm.interference,color:'#d97a23'},{y:r.powers_dbm.noise,color:'#83909c'},{y:r.powers_dbm.y,color:'#087f75'}]);$('result').scrollIntoView({behavior:'smooth'});}catch(e){notice(e.message,true);}}
Promise.all([api('/api/config'),api('/api/capabilities')]).then(([data,caps])=>{defaults=data.defaults;capabilities=caps;$('config-json').value=JSON.stringify(defaults,null,2);update();refreshJobs();}).catch(e=>notice(e.message,true));
setInterval(()=>{if(currentPage==='runs')refreshJobs();},2000);
