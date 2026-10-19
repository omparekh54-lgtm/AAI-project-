'use strict';
const $ = id => document.getElementById(id);
const METHODS = {
  dqn: {name:'Parameter-shared DQN', description:'Each cab chooses the highest-scoring valid action from the trained DQN, using its own zone and adjacent-zone observations. Network weights are shared across cabs.'},
  nearest: {name:'Nearest Cab', description:'The central dispatcher gives each waiting rider the closest available cab within 3 zones. Unassigned cabs wait where they are.'},
  zone: {name:'Zone Balancing', description:'The central dispatcher matches nearby cabs to riders, then sends spare cabs toward fixed historical demand zones, including the airport.'}
};
const ACTIONS = ['Wait here','Accept a ride','Move north','Move south','Move west','Move east'];
const STATUS = ['Available','Driving to pickup','Passenger on board'];
let replay=null, frameIndex=420, selected=0, timer=null, running=false, started=false, loading=false, loadVersion=0;
let research=null, curve=null;
const cache=new Map();
const time=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`;
const zone=(x,y)=>`${String.fromCharCode(65+x)}${y+1}`;
const pct=n=>`${(100*Number(n||0)).toFixed(1)}%`;
const num=(n,d=1)=>Number(n||0).toFixed(d);
const mean=(r,k)=>Number(r[`${k}_mean`]??r[k]??0);
const distance=(a,b)=>Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]);
function stop(){clearTimeout(timer);timer=null;running=false;updateControls();}
function updateControls(){
  $('pause').textContent=loading?'Loading policy…':running?'Pause':started?'Resume':'Start';
  for(const id of ['pause','step','reset','timeline']) $(id).disabled=loading||!replay;
  if(replay&&!loading){$('playState').textContent=frameIndex===1440?'DAY COMPLETE':running?'PLAYING POLICY REPLAY':started?'PAUSED · INSPECT A DECISION':'READY · PRESS START TO BEGIN';}
}
function schedule(){clearTimeout(timer);if(running)timer=setTimeout(()=>{if(!running)return;if(frameIndex<1440){frameIndex++;render();schedule();}else stop();},Number($('speed').value));}
function start(){if(!replay||loading)return;if(frameIndex===1440)frameIndex=420;started=true;running=true;render();schedule();}
function showError(message,retry){
  const box=$('error');box.textContent=message;box.hidden=false;
  if(retry){const button=document.createElement('button');button.textContent='Retry loading';button.onclick=retry;box.append(button);}
}
async function json(path){const response=await fetch(path);if(!response.ok)throw new Error(`${path}: HTTP ${response.status}`);return response.json();}
async function trace(path){
  const response=await fetch(path);if(!response.ok)throw new Error(`${path}: HTTP ${response.status}`);
  let bytes=new Uint8Array(await response.arrayBuffer());
  if(bytes[0]===31&&bytes[1]===139){
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes=new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}
async function loadReplay(autoplay=running){
  const version=++loadVersion;stop();loading=true;updateControls();$('error').hidden=true;
  const key=`${$('algorithm').value}-${$('fleet').value}`;
  $('methodDescription').textContent=METHODS[$('algorithm').value].description;
  $('playState').textContent='LOADING POLICY EXECUTION';
  try{
    const data=cache.get(key)||await trace(`data/traces/${key}.json.gz`);
    if(version!==loadVersion)return;
    if(data.frames?.length!==1441||!data.frames[420]?.c.length)throw new Error('Incomplete recorded trajectory');
    cache.set(key,data);if(cache.size>3)cache.delete(cache.keys().next().value);
    replay=data;selected=Math.min(selected,data.fleet-1);loading=false;render();renderEvidence();if(autoplay)start();
  }catch(error){if(version!==loadVersion)return;loading=false;replay=null;updateControls();$('playState').textContent='SIMULATION UNAVAILABLE';showError('The recorded policy could not load. Check your connection and retry. Research results can still be viewed below.',()=>loadReplay(autoplay));console.error(error);}
}
function validActions(frame,cab){
  if(cab[3]!==0||frame.t<360||frame.t>=1320)return [0];
  const [id,x,y]=cab;const actions=[0];
  if(frame.r.some(r=>r[6]===null&&distance([x,y],[r[1],r[2]])<=1))actions.push(1);
  if(x>0)actions.push(2);if(x<4)actions.push(3);if(y>0)actions.push(4);if(y<4)actions.push(5);
  return actions;
}
function renderMap(frame){
  const selectedCab=frame.c[selected]||frame.c[0];
  const grid=document.createDocumentFragment();
  for(let x=0;x<5;x++)for(let y=0;y<5;y++){
    const z=document.createElement('div');z.className='zone';
    if(selectedCab&&distance([x,y],[selectedCab[1],selectedCab[2]])<=1)z.classList.add('sensed');
    if(selectedCab&&x===selectedCab[1]&&y===selectedCab[2])z.classList.add('selected-zone');
    if(x===0&&y===4)z.classList.add('airport');
    const label=document.createElement('div');label.className='zone-label';label.textContent=zone(x,y);
    if(x===0&&y===4){const a=document.createElement('span');a.className='airport-name';a.textContent='AIRPORT';label.append(a);}z.append(label);
    const pile=document.createElement('div');pile.className='cab-pile';
    const zoneCabs=frame.c.filter(c=>c[1]===x&&c[2]===y);
    const visibleCabs=zoneCabs.slice(0,2);
    const focused=zoneCabs.find(c=>c[0]===selected);
    if(focused&&!visibleCabs.includes(focused))visibleCabs[visibleCabs.length-1]=focused;
    for(const c of visibleCabs){
      const b=document.createElement('button');b.className=`cab ${['idle','pickup','occupied'][c[3]]}${c[0]===selected?' selected':''}`;
      b.textContent=String(c[0]+1).padStart(2,'0');b.setAttribute('aria-label',`Inspect Cab ${c[0]+1}, ${STATUS[c[3]]}`);b.setAttribute('aria-pressed',String(c[0]===selected));b.title=`Cab ${c[0]+1}: ${STATUS[c[3]]}`;b.onclick=()=>{selected=c[0];render();};pile.append(b);
    }z.append(pile);
    if(zoneCabs.length>2){const count=document.createElement('span');count.className='pile-count';count.textContent=`+${zoneCabs.length-2}`;count.title=`${zoneCabs.length} cabs here. Select a cab in the fleet list below.`;z.append(count);}
    const riders=document.createElement('div');riders.className='riders';
    frame.r.filter(r=>r[1]===x&&r[2]===y&&r[7]===null).forEach(r=>{
      const dot=document.createElement('span');dot.className='request'+(r[6]!==null?' assigned':'');dot.title=`Rider ${r[0]}: ${zone(r[1],r[2])} to ${zone(r[3],r[4])}, waiting ${frame.t-r[5]} min${r[6]!==null?`, Cab ${r[6]+1} assigned`:''}`;riders.append(dot);
    });z.append(riders);grid.append(z);
  }$('city').replaceChildren(grid);
  $('fleetList').replaceChildren(...frame.c.map(c=>{const b=document.createElement('button');b.textContent=`Cab ${String(c[0]+1).padStart(2,'0')}`;b.className=c[0]===selected?'selected':'';b.setAttribute('aria-pressed',String(c[0]===selected));b.onclick=()=>{selected=c[0];render();};return b;}));
}
function renderInspector(frame){
  const c=frame.c[selected];if(!c)return;
  const r=frame.r.find(r=>r[0]===c[7]);
  const local=frame.r.filter(r=>r[6]===null&&distance([c[1],c[2]],[r[1],r[2]])<=1);
  const neighbours=frame.c.filter(n=>n[0]!==c[0]&&n[3]===0&&distance([c[1],c[2]],[n[1],n[2]])<=1).length;
  const active=frame.t>=360&&frame.t<1320;
  $('cabTitle').textContent=`Cab ${String(c[0]+1).padStart(2,'0')}`;
  $('cabStatus').textContent=!active&&c[3]===0?'Off shift':STATUS[c[3]];
  $('cabSummary').textContent=r?`Rider ${r[0]}: ${zone(r[1],r[2])} → ${zone(r[3],r[4])}. ${c[3]===1?'Collecting the rider':'Travelling with the passenger'}; ${c[8]} minute${c[8]===1?'':'s'} to ${c[3]===1?'pickup':'drop-off'}.`:`${active?'Ready for a dispatch decision':'Shift is 06:00–22:00'}. Currently in ${zone(c[1],c[2])}${c[1]===0&&c[2]===4?' (Airport)':''}.`;
  $('observations').innerHTML=`<dt>Current zone</dt><dd>${zone(c[1],c[2])}</dd><dt>Local open riders</dt><dd>${local.length}</dd><dt>Nearby idle peers</dt><dd>${neighbours}</dd><dt>Earned so far</dt><dd>${c[6]} fare units</dd><dt>Previous step reward</dt><dd>${num(c[5],2)}</dd>`;
  const busy=c[3]!==0;
  $('decision').textContent=busy?(c[3]===1?'Continue to pickup':'Continue to drop-off'):!active?'Wait · off shift':ACTIONS[c[4]];
  $('reason').textContent=busy?'An assigned ride follows its route one zone at a time. The cab makes a new dispatch choice when it becomes available.':!active?'New rides and repositioning are disabled outside the driver shift.':replay.method==='dqn'?`The DQN selected ${ACTIONS[c[4]].toLowerCase()} from its valid actions using the highlighted local view.`:replay.method==='nearest'?(c[4]===1?'The dispatcher reserved this cab for a nearby waiting rider.':'No rider was assigned to this cab this minute.'):(c[4]===1?'A waiting rider was matched to this cab.':c[4]===0?'The cab is waiting in its target coverage zone.':'No ride was assigned, so the cab moves toward its static coverage target.');
  $('qPanel').hidden=replay.method!=='dqn';
  if(frame.q){
    $('qPanel').querySelector('h4').textContent=busy?'Dispatch scores while this cab is assigned':!active?'Dispatch scores outside the shift':'How the DQN chose this action';
    $('qPanel').querySelector('p').textContent=busy?'Only wait is valid for dispatch while the simulator advances the existing trip. The scores below are the network outputs at this state.':!active?'The shift constraint forces wait. These scores do not override the driver shift.':'The network scores each valid action. The highest score wins. These are model outputs, not probabilities.';
    const q=frame.q[selected],valid=validActions(frame,c),lo=Math.min(...q),hi=Math.max(...q);
    $('qValues').replaceChildren(...q.map((value,a)=>{
      const row=document.createElement('div');const eligible=valid.includes(a);row.className='qrow'+(a===c[4]&&eligible?' chosen':'');
      const label=document.createElement('span');label.textContent=ACTIONS[a]+(!eligible?' ×':'');
      const bar=document.createElement('div');bar.className='qbar';const fill=document.createElement('i');fill.style.width=`${eligible?10+90*(value-lo)/Math.max(.001,hi-lo):0}%`;bar.append(fill);
      const v=document.createElement('b');v.textContent=eligible?num(value,2):'invalid';row.append(label,bar,v);return row;
    }));
  }
}
function eventsAt(index){
  if(index===0)return ['The city begins a new simulated day.'];
  const prev=replay.frames[index-1],now=replay.frames[index],out=[];
  for(const c of now.c){const p=prev.c[c[0]];const name=`Cab ${String(c[0]+1).padStart(2,'0')}`;
    if(c[6]>p[6])out.push(`${name} completed a trip and earned ${c[6]-p[6]} fare units.`);
    else if(p[3]===1&&c[3]===2)out.push(`${name} picked up its passenger.`);
    else if(p[3]===0&&c[3]!==0)out.push(`${name} accepted rider ${c[7]}${c[3]===2?' and picked them up in the same zone':''}.`);
    else if(p[1]!==c[1]||p[2]!==c[2])out.push(`${name} moved ${zone(p[1],p[2])} → ${zone(c[1],c[2])}${p[3]===0?' to reposition':p[3]===1?' toward pickup':' with a passenger'}.`);
  }
  const newRiders=now.m.requests-prev.m.requests;if(newRiders>0)out.push(`${newRiders} new rider${newRiders>1?'s':''} requested a trip.`);
  const cancels=now.m.cancelled-prev.m.cancelled;if(cancels>0)out.push(`${cancels} unmatched rider${cancels>1?'s':''} cancelled after 8 minutes.`);
  return out;
}
function renderEvents(){
  const entries=[];for(let i=frameIndex;i>=Math.max(0,frameIndex-8)&&entries.length<7;i--){for(const msg of eventsAt(i)){entries.push(`${time(i)} · ${msg}`);if(entries.length>=7)break;}}
  if(!entries.length)entries.push(`${time(frameIndex)} · Cabs are waiting for demand${frameIndex<360||frameIndex>=1320?' outside the driver shift':''}.`);
  $('events').replaceChildren(...entries.map(msg=>{const li=document.createElement('li');li.textContent=msg;return li;}));
}
function render(){
  if(!replay)return;
  const frame=replay.frames[frameIndex];$('clock').textContent=time(frame.t);$('timelineTime').textContent=time(frame.t);$('timeline').value=frame.t;
  $('period').textContent=frame.t<360||frame.t>=1320?'Outside driver shift':(frame.t>=420&&frame.t<600)||(frame.t>=1020&&frame.t<1260)?'Rush hour':'Quieter hours';
  const waiting=frame.r.filter(r=>r[7]===null).length;
  const stats=[['Waiting for pickup',waiting,'open + assigned riders'],['Trips completed',frame.m.completed,'since midnight'],['Average rider wait',`${num(frame.m.avg_wait)} min`,'among picked-up riders'],['Available cabs',`${frame.c.filter(c=>c[3]===0).length} / ${replay.fleet}`,'idle, including off shift'],['Cancellations',frame.m.cancelled,'unmatched after 8 minutes']];
  $('liveStats').innerHTML=stats.map(([label,v,note])=>`<div class="live-stat"><span>${label}</span><strong>${v}</strong><small>${note}</small></div>`).join('');
  renderMap(frame);renderInspector(frame);renderEvents();updateControls();
}
function renderEvidence(){
  if(!research)return;
  const fleet=Number($('fleet').value),chosen=METHODS[$('algorithm').value].name;
  const rows=research.results.filter(r=>Number(r.fleet_size)===fleet);
  $('benchmarkTable').replaceChildren(...rows.map(r=>{
    const tr=document.createElement('tr');if(r.method===chosen)tr.className='selected';
    [r.method,`${num(mean(r,'avg_wait'),2)} min`,`${num(mean(r,'avg_pickup_distance'),2)} zones`,pct(mean(r,'service_rate')),pct(mean(r,'cancellation_rate')),pct(mean(r,'empty_driving_ratio')),pct(mean(r,'utilisation')),num(mean(r,'earnings'),0),num(mean(r,'earnings_fairness'),2)].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});return tr;
  }));
  const meta=research.metadata||{};
  $('evidenceMeta').textContent=`${meta.evaluation_episodes} evaluation days per method · ${meta.day_minutes} minutes per day · ${fleet} cabs · ${meta.training_episodes} DQN training episodes`;
  const selectedRow=rows.find(r=>r.method===chosen);
  $('benchmarkMetrics').innerHTML=selectedRow?`<div><b>${num(mean(selectedRow,'avg_wait'),2)} ± ${num(selectedRow.avg_wait_std,2)} min</b><small>${chosen}: mean rider wait ± standard deviation</small></div><div><b>${pct(mean(selectedRow,'service_rate'))}</b><small>Full-day trips completed / all requests</small></div>`:'No benchmark recorded for this fleet.';
  const periods=(research.period_results||[]).filter(r=>r.method===chosen);
  $('periodTable').innerHTML='<div class="period-row"><span>Period</span><span>Rider wait</span><span>Service rate</span></div>'+periods.map(r=>`<div class="period-row"><span>${r.period==='peak'?'Rush hour':'Quieter hours'}</span><span>${num(mean(r,'avg_wait'),2)} min</span><span>${pct(mean(r,'service_rate'))}</span></div>`).join('');
}
function drawChart(){
  if(!curve)return;
  const canvas=$('rewardChart'),ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=210,dpr=window.devicePixelRatio||1;
  canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
  const vals=curve.rewards||[];if(!vals.length)return;
  const left=48,right=12,top=18,bottom=30,lo=Math.min(...vals),hi=Math.max(...vals),range=Math.max(1,hi-lo);
  ctx.font='10px system-ui';
  for(let i=0;i<4;i++){let y=top+i*(h-top-bottom)/3;ctx.strokeStyle='#304937';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();ctx.fillStyle='#acbdb0';ctx.fillText(String(Math.round(hi-i*range/3)),0,y+3);}
  ctx.strokeStyle='#c4f572';ctx.lineWidth=1.7;ctx.beginPath();vals.forEach((v,i)=>{const x=left+i/Math.max(1,vals.length-1)*(w-left-right),y=top+(hi-v)/range*(h-top-bottom);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.fillStyle='#acbdb0';ctx.fillText('Episode 1',left,h-5);ctx.fillText(`Episode ${vals.length}`,Math.max(left,w-85),h-5);
}
async function loadEvidence(){
  const results=await Promise.allSettled([json('data/experiment_results.json'),json('data/training_curve.json')]);
  if(results[0].status==='fulfilled'){research=results[0].value;renderEvidence();}else{$('evidenceMeta').textContent='Benchmark data could not load. Reload the page to retry.';console.error(results[0].reason);}
  if(results[1].status==='fulfilled'){curve=results[1].value;$('trainingMeta').textContent=`${curve.episodes} full-day training episodes × ${curve.day_minutes} simulated minutes. Shared DQN checkpoint.`;drawChart();}else{$('trainingMeta').textContent='Training curve unavailable. Reload to retry.';console.error(results[1].reason);}
}
$('pause').onclick=()=>running?stop():start();
$('step').onclick=()=>{stop();started=true;frameIndex=Math.min(1440,frameIndex+1);render();};
$('reset').onclick=()=>{stop();started=false;frameIndex=420;render();};
$('speed').onchange=()=>{if(running)schedule();};
$('algorithm').onchange=()=>loadReplay(running);
$('fleet').onchange=()=>loadReplay(running);
$('timeline').oninput=()=>{stop();started=true;frameIndex=Number($('timeline').value);render();};
window.addEventListener('resize',drawChart);
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stop();});
// Render structure immediately, even if network data is slow or unavailable.
renderMap({c:[],r:[]});
loadReplay(false);
loadEvidence();
