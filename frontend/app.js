const city = document.getElementById('city');
const fleetSlider = document.getElementById('fleet');
const fleetValue = document.getElementById('fleetValue');
const runBtn = document.getElementById('runBtn');
const algorithm = document.getElementById('algorithm');
const demand = document.getElementById('demand');
const speed = document.getElementById('speed');
let timer = null, selectedCab = 3, minute = 0, cabs = [];
let research = {results: [], period_results: []}, curve = {rewards: []};
const actions = ['WAIT','ACCEPT RIDE','REPOSITION NORTH','REPOSITION SOUTH','REPOSITION WEST','REPOSITION EAST'];
const valueOf = (row, key) => Number(row?.[`${key}_mean`] ?? row?.[key] ?? 0);

async function loadResearchData(){
  try {
    const [r,c] = await Promise.all([
      fetch('/results/experiment_results.json').then(x => x.json()),
      fetch('/results/training_curve.json').then(x => x.json())
    ]);
    research = Array.isArray(r) ? {results:r} : r;
    curve = c;
    const n = research.metadata?.training_episodes || curve.episodes || 360;
    document.getElementById('episodeValue').textContent = n.toLocaleString();
    document.querySelector('.chart-note').textContent = `${n} full-day training episodes`;
    renderResearchMetrics(); drawChart(); drawBars();
  } catch(e) { console.error('Research data unavailable', e); }
}
function resultFor(method, fleet=8){const rows=research.results||[];return rows.find(r=>r.method===method&&Number(r.fleet_size)===fleet)||rows.find(r=>r.method===method)||null;}
function seedCabs(n){cabs=Array.from({length:n},(_,i)=>({id:i,x:Math.floor(i/5),y:i%5,status:'IDLE',action:'WAIT'}));}
function requests(){const count=demand.value==='Peak'?10:demand.value==='Airport-heavy'?8:5;return Array.from({length:count},(_,i)=>({x:(i*3+minute)%5,y:(i*2+1)%5,hot:demand.value==='Airport-heavy'||i%3===0}));}
function render(){city.innerHTML='';const reqs=requests();for(let x=0;x<5;x++)for(let y=0;y<5;y++){const z=document.createElement('div');z.className='zone';if((x+y)%4===0)z.classList.add('hotspot');if(x===0&&y===4)z.classList.add('airport');const r=reqs.find(q=>q.x===x&&q.y===y);if(r){const d=document.createElement('i');d.className='request';z.appendChild(d)}cabs.filter(c=>c.x===x&&c.y===y).forEach(c=>{const el=document.createElement('div');el.className='cab'+(c.id===selectedCab?' selected':'');el.textContent=String(c.id+1).padStart(2,'0');el.onclick=()=>{selectedCab=c.id;updateInspector()};z.appendChild(el)});city.appendChild(z)}const h=Math.floor((minute%1440)/60),m=minute%60;document.getElementById('clock').textContent=`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;updateInspector();}
function moveCab(c){if(c.status!=='IDLE')return;const req=requests().sort((a,b)=>(Math.abs(c.x-a.x)+Math.abs(c.y-a.y))-(Math.abs(c.x-b.x)+Math.abs(c.y-b.y)))[0];if(algorithm.value==='Nearest Cab'&&req&&Math.abs(c.x-req.x)+Math.abs(c.y-req.y)<=3){c.status='ON TRIP';c.action='ACCEPT RIDE';return}if(algorithm.value==='Zone Balancing'&&req){if(Math.abs(c.x-req.x)>Math.abs(c.y-req.y))c.x+=Math.sign(req.x-c.x);else c.y+=Math.sign(req.y-c.y);c.action='REPOSITION';return}if(req&&Math.random()<.52){c.status='ON TRIP';c.action='ACCEPT RIDE';return}const dirs=[[1,0,'SOUTH'],[-1,0,'NORTH'],[0,1,'EAST'],[0,-1,'WEST']],d=dirs[Math.floor(Math.random()*dirs.length)];c.x=Math.max(0,Math.min(4,c.x+d[0]));c.y=Math.max(0,Math.min(4,c.y+d[1]));c.action='REPOSITION '+d[2];}
function tick(){minute=(minute+1)%1440;cabs.forEach(c=>{if(c.status==='ON TRIP'){if(Math.random()<.38)c.status='IDLE';else c.action='IN TRIP'}else moveCab(c)});render();}
function updateInspector(){const c=cabs[selectedCab]||cabs[0];if(!c)return;document.getElementById('agentTitle').textContent=`Cab ${String(c.id+1).padStart(2,'0')}`;document.getElementById('agentStatus').textContent=c.status;const zone=c.x*5+c.y;document.getElementById('agentZone').textContent=`${zone} · ${zoneName(c.x,c.y)}`;document.getElementById('decision').textContent=c.action;const nearby=requests().filter(r=>Math.abs(r.x-c.x)+Math.abs(r.y-c.y)<=2).length;document.getElementById('obsDemand').textContent=`${nearby} requests`;const grid=document.getElementById('obsGrid');grid.innerHTML='';['X '+c.x,'Y '+c.y,'T '+String(minute).padStart(4,'0'),'IDLE '+(c.status==='IDLE'?1:0),`REQ ${nearby}`,`IDLE FLEET ${cabs.filter(x=>x.status==='IDLE').length}`,`DEMAND ${demand.value}`,'NEIGHBOUR VIEW'].forEach(t=>{const s=document.createElement('span');s.textContent=t;grid.appendChild(s)});const q=document.getElementById('qValues');q.innerHTML='';actions.forEach((a,i)=>{const v=i===1&&nearby?.86:.35+((i*17+minute*3)%50)/100;const row=document.createElement('div');row.className='qrow';row.innerHTML=`<span>${a}</span><div class="qbar"><i style="width:${Math.round(Math.min(v,1)*100)}%"></i></div>`;q.appendChild(row)});}
function zoneName(x,y){if(x===0&&y===4)return'Airport';if(x===2&&y===2)return'Central';if(x<2)return'North';if(x>2)return'South';return'Urban'}
function selectedResult(){const map={DQN:'Parameter-shared DQN','Nearest Cab':'Nearest Cab','Zone Balancing':'Zone Balancing'};return resultFor(map[algorithm.value],8)}
function updateMetric(id,value){document.getElementById(id).textContent=value}
function renderResearchMetrics(){const r=selectedResult();if(!r)return;const sr=valueOf(r,'service_rate');updateMetric('waitMetric',`${valueOf(r,'avg_wait').toFixed(2)} min`);updateMetric('serviceMetric',`${(sr*100).toFixed(1)}%`);updateMetric('cancelMetric',`${(100-sr*100).toFixed(1)}%`);updateMetric('emptyMetric',`${(valueOf(r,'empty_driving_ratio')*100).toFixed(1)}%`);updateMetric('utilMetric',`${(valueOf(r,'utilisation')*100).toFixed(1)}%`);updateMetric('earnMetric',`₹${valueOf(r,'earnings').toFixed(0)}`)}
function drawChart(){const c=document.getElementById('rewardChart');if(!c)return;const ctx=c.getContext('2d'),dpr=devicePixelRatio||1,w=c.clientWidth||500,h=c.clientHeight||240;c.width=w*dpr;c.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#20312b';ctx.lineWidth=1;for(let i=1;i<5;i++){const y=i*h/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}const rewards=curve.rewards||[];if(!rewards.length)return;const lo=Math.min(...rewards),hi=Math.max(...rewards),range=Math.max(1,hi-lo);ctx.beginPath();rewards.forEach((v,i)=>{const x=(i/(rewards.length-1))*w,y=h-((v-lo)/range*.78+.1)*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle='#a8ff72';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#8ea29c';ctx.font='10px DM Mono';ctx.fillText('reward',8,14);ctx.fillText('ep 1',8,h-4);ctx.fillText(`ep ${rewards.length}`,w-65,h-4)}
function drawBars(){const box=document.getElementById('barChart');if(!box)return;box.innerHTML='';const rows=[['Avg wait',r=>valueOf(r,'avg_wait'),' min'],['Service rate',r=>valueOf(r,'service_rate')*100,'%'],['Empty driving',r=>valueOf(r,'empty_driving_ratio')*100,'%'],['Earnings',r=>valueOf(r,'earnings'),'']];rows.forEach(([label,get,suffix])=>{const vals=['Nearest Cab','Zone Balancing','Parameter-shared DQN'].map(m=>{const r=resultFor(m,8);return r?get(r):0});const max=Math.max(...vals)||1;const r=resultFor('Parameter-shared DQN',8),value=r?get(r):0;const block=document.createElement('div');block.className='barline';block.innerHTML=`<span>${label}</span><div class="bar"><i style="width:${Math.max(4,Math.round((value/max)*100))}%"></i></div><b>${value.toFixed(label==='Earnings'?0:1)}${suffix}</b>`;box.appendChild(block)})}
fleetSlider.oninput=()=>{fleetValue.textContent=fleetSlider.value;seedCabs(+fleetSlider.value);render()};runBtn.onclick=()=>{clearInterval(timer);minute=0;seedCabs(+fleetSlider.value);render();timer=setInterval(tick,+speed.value)};speed.onchange=()=>{if(timer){clearInterval(timer);timer=setInterval(tick,+speed.value)}};algorithm.onchange=()=>{renderResearchMetrics();drawBars();render()};demand.onchange=render;window.onresize=drawChart;seedCabs(8);render();loadResearchData();
