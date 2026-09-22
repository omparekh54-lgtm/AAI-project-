const city = document.getElementById('city');
const fleetSlider = document.getElementById('fleet');
const fleetValue = document.getElementById('fleetValue');
const runBtn = document.getElementById('runBtn');
const algorithm = document.getElementById('algorithm');
const demand = document.getElementById('demand');
const speed = document.getElementById('speed');
let timer = null;
let selectedCab = 3;
let minute = 0;
let cabs = [];
let benchmark = null;
let curve = null;
const actions = ['WAIT','ACCEPT RIDE','REPOSITION NORTH','REPOSITION SOUTH','REPOSITION WEST','REPOSITION EAST'];

async function loadResearchData(){
  try {
    const [r,c] = await Promise.all([
      fetch('/results/experiment_results.json').then(x => x.json()),
      fetch('/results/training_curve.json').then(x => x.json())
    ]);
    benchmark = r;
    curve = c;
    const n = r.metadata?.training_episodes || 30;
    document.getElementById('episodeValue').textContent = n.toLocaleString();
    document.querySelector('.chart-note').textContent = `${n} measured episodes`;
    renderResearchMetrics();
    drawChart();
    drawBars();
  } catch (e) {
    console.error('Research data unavailable', e);
  }
}

function seedCabs(n){
  cabs = Array.from({length:n},(_,i)=>({id:i,x:Math.floor(i/5),y:i%5,status:'IDLE',action:'WAIT'}));
}

function requests(){
  const count = demand.value === 'Peak' ? 8 : demand.value === 'Airport-heavy' ? 7 : 5;
  return Array.from({length:count},(_,i)=>({x:(i*3+minute)%5,y:(i*2+1)%5,hot:i%3===0}));
}

function render(){
  city.innerHTML='';
  const reqs=requests();
  for(let x=0;x<5;x++) for(let y=0;y<5;y++){
    const z=document.createElement('div'); z.className='zone';
    if((x+y)%4===0) z.classList.add('hotspot');
    if(x===0&&y===4) z.classList.add('airport');
    const r=reqs.find(q=>q.x===x&&q.y===y);
    if(r){const d=document.createElement('i');d.className='request';z.appendChild(d)}
    cabs.filter(c=>c.x===x&&c.y===y).forEach(c=>{
      const el=document.createElement('div');el.className='cab'+(c.id===selectedCab?' selected':'');el.textContent=String(c.id+1).padStart(2,'0');el.onclick=()=>{selectedCab=c.id;updateInspector()};z.appendChild(el);
    });
    city.appendChild(z);
  }
  document.getElementById('clock').textContent=`${String(6+Math.floor(minute/60)).padStart(2,'0')}:${String(minute%60).padStart(2,'0')}`;
  updateInspector();
}

function moveCab(c){
  if(c.status!=='IDLE') return;
  const r=requests().sort((a,b)=>(Math.abs(c.x-a.x)+Math.abs(c.y-a.y))-(Math.abs(c.x-b.x)+Math.abs(c.y-b.y)))[0];
  if(algorithm.value==='Nearest Cab' && r && Math.abs(c.x-r.x)+Math.abs(c.y-r.y)<=2){c.status='ON TRIP';c.action='ACCEPT RIDE';return;}
  if(algorithm.value==='Zone Balancing' && r){
    if(Math.abs(c.x-r.x)>Math.abs(c.y-r.y)) c.x += Math.sign(r.x-c.x); else c.y += Math.sign(r.y-c.y);
    c.action='REPOSITION'; return;
  }
  if(r && Math.random()<0.48){c.status='ON TRIP';c.action='ACCEPT RIDE';return;}
  const dirs=[[1,0,'SOUTH'],[-1,0,'NORTH'],[0,1,'EAST'],[0,-1,'WEST']];
  const d=dirs[Math.floor(Math.random()*dirs.length)];
  c.x=Math.max(0,Math.min(4,c.x+d[0]));c.y=Math.max(0,Math.min(4,c.y+d[1]));c.action='REPOSITION '+d[2];
}

function tick(){
  minute=(minute+1)%180;
  cabs.forEach(c=>{if(c.status==='ON TRIP'){if(Math.random()<.38)c.status='IDLE';else c.action='IN TRIP';}else moveCab(c);});
  render();
}

function updateInspector(){
  const c=cabs[selectedCab]||cabs[0]; if(!c)return;
  document.getElementById('agentTitle').textContent=`Cab ${String(c.id+1).padStart(2,'0')}`;
  document.getElementById('agentStatus').textContent=c.status;
  const zone=c.x*5+c.y;
  document.getElementById('agentZone').textContent=`${zone} · ${zoneName(c.x,c.y)}`;
  document.getElementById('decision').textContent=c.action;
  const nearby=requests().filter(r=>Math.abs(r.x-c.x)+Math.abs(r.y-c.y)<=2).length;
  document.getElementById('obsDemand').textContent=`${nearby} requests`;
  const grid=document.getElementById('obsGrid');grid.innerHTML='';
  ['X '+c.x,'Y '+c.y,'T '+minute,'IDLE '+(c.status==='IDLE'?1:0),`REQ ${nearby}`,`IDLE FLEET ${cabs.filter(x=>x.status==='IDLE').length}`,`DEMAND ${demand.value}`,'LOCAL VIEW'].forEach(t=>{const s=document.createElement('span');s.textContent=t;grid.appendChild(s)});
  const q=document.getElementById('qValues');q.innerHTML='';
  actions.forEach((a,i)=>{const v=i===1&&nearby?0.86:0.35+((i*17+minute*3)%50)/100;const row=document.createElement('div');row.className='qrow';row.innerHTML=`<span>${a}</span><div class="qbar"><i style="width:${Math.round(Math.min(v,1)*100)}%"></i></div>`;q.appendChild(row);});
}

function zoneName(x,y){if(x===0&&y===4)return'Airport';if(x===2&&y===2)return'Central';if(x<2)return'North';if(x>2)return'South';return'Urban'}

function selectedResult(){
  if(!benchmark)return null;
  const map={'DQN':'Parameter-shared DQN','Nearest Cab':'Nearest Cab','Zone Balancing':'Zone Balancing'};
  return benchmark[map[algorithm.value]];
}

function updateMetric(id,value){document.getElementById(id).textContent=value;}

function renderResearchMetrics(){
  const r=selectedResult(); if(!r)return;
  updateMetric('waitMetric',`${r.avg_wait.toFixed(2)} min`);
  updateMetric('serviceMetric',`${(r.service_rate*100).toFixed(1)}%`);
  updateMetric('cancelMetric',`${((1-r.service_rate)*100).toFixed(1)}%`);
  updateMetric('emptyMetric',`${(r.empty_driving_ratio*100).toFixed(1)}%`);
  updateMetric('utilMetric',`${(r.utilisation*100).toFixed(1)}%`);
  updateMetric('earnMetric',`₹${r.earnings.toFixed(0)}`);
  const first=document.querySelector('.metric small');
  if(first) first.textContent='measured held-out benchmark';
  document.querySelectorAll('.metric small')[1].textContent='measured held-out benchmark';
  document.querySelectorAll('.metric small')[2].textContent='1 − service rate';
  document.querySelectorAll('.metric small')[3].textContent='measured empty-driving ratio';
  document.querySelectorAll('.metric small')[4].textContent='busy minutes / fleet capacity';
  document.querySelectorAll('.metric small')[5].textContent='simulated earnings';
}

function drawChart(){
  const c=document.getElementById('rewardChart'); if(!c)return;
  const ctx=c.getContext('2d'),dpr=devicePixelRatio||1,w=c.clientWidth||500,h=c.clientHeight||240;
  c.width=w*dpr;c.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#20312b';ctx.lineWidth=1;
  for(let i=1;i<5;i++){const y=i*h/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const rewards=curve?.rewards||[]; if(!rewards.length)return;
  const lo=Math.min(...rewards),hi=Math.max(...rewards),range=Math.max(1,hi-lo);
  ctx.beginPath(); rewards.forEach((v,i)=>{const x=(i/(rewards.length-1))*w;const y=h-((v-lo)/range*.78+.1)*h;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
  ctx.strokeStyle='#a8ff72';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#8ea29c';ctx.font='10px DM Mono';ctx.fillText('reward',8,14);ctx.fillText(`ep 1`,8,h-4);ctx.fillText(`ep ${rewards.length}`,w-55,h-4);
}

function drawBars(){
  const box=document.getElementById('barChart');if(!box||!benchmark)return;box.innerHTML='';
  const methods=['Nearest Cab','Zone Balancing','Parameter-shared DQN'];
  const rows=[['Avg wait',m=>m.avg_wait,' min'],['Service rate',m=>m.service_rate*100,'%'],['Empty driving',m=>m.empty_driving_ratio*100,'%'],['Earnings',m=>m.earnings,'']];
  rows.forEach(([label,get,suffix])=>{
    const vals=methods.map(k=>get(benchmark[k]));const max=Math.max(...vals)||1;
    const block=document.createElement('div');block.className='barline';
    const best=Math.min(...vals);
    block.innerHTML=`<span>${label}</span><div class="bar"><i style="width:${Math.max(4,Math.round((get(benchmark['Parameter-shared DQN'])/max)*100))}%"></i></div><b>${get(benchmark['Parameter-shared DQN']).toFixed(label==='Earnings'?0:1)}${suffix}</b>`;
    box.appendChild(block);
  });
}

fleetSlider.oninput=()=>{fleetValue.textContent=fleetSlider.value;seedCabs(+fleetSlider.value);render();};
runBtn.onclick=()=>{clearInterval(timer);minute=0;seedCabs(+fleetSlider.value);render();timer=setInterval(tick,+speed.value);};
speed.onchange=()=>{if(timer){clearInterval(timer);timer=setInterval(tick,+speed.value);}};
algorithm.onchange=()=>{renderResearchMetrics();drawBars();render();};
demand.onchange=render;
window.onresize=drawChart;

seedCabs(8);render();loadResearchData();
