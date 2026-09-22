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
const actions = ['WAIT','ACCEPT RIDE','REPOSITION NORTH','REPOSITION SOUTH','REPOSITION WEST','REPOSITION EAST'];
const names = ['North','South','West','East'];

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
    const r=reqs.find(q=>q.x===x&&q.y===y); if(r){const d=document.createElement('i');d.className='request';z.appendChild(d)}
    cabs.filter(c=>c.x===x&&c.y===y).forEach(c=>{const el=document.createElement('div');el.className='cab'+(c.id===selectedCab?' selected':'');el.textContent=String(c.id+1).padStart(2,'0');el.onclick=()=>{selectedCab=c.id;updateInspector()};z.appendChild(el)});
    city.appendChild(z);
  }
  document.getElementById('clock').textContent=`${String(6+Math.floor(minute/60)).padStart(2,'0')}:${String(minute%60).padStart(2,'0')}`;
  updateInspector(); updateMetrics();
}
function moveCab(c){
  const mode=algorithm.value;
  if(c.status!=='IDLE') return;
  const r=requests().sort((a,b)=>(Math.abs(c.x-a.x)+Math.abs(c.y-a.y))-(Math.abs(c.x-b.x)+Math.abs(c.y-b.y)))[0];
  if(mode==='Nearest Cab' && r && Math.abs(c.x-r.x)+Math.abs(c.y-r.y)<=2){c.status='ON TRIP';c.action='ACCEPT RIDE';return}
  if(mode==='Zone Balancing' && r){
    if(Math.abs(c.x-r.x)>Math.abs(c.y-r.y)) c.x += Math.sign(r.x-c.x); else c.y += Math.sign(r.y-c.y);
    c.action='REPOSITION '+(Math.abs(c.x-r.x)>=Math.abs(c.y-r.y)?(r.x>=c.x?'SOUTH':'NORTH'):(r.y>=c.y?'EAST':'WEST'));return;
  }
  if(r && Math.random()<0.48){c.status='ON TRIP';c.action='ACCEPT RIDE';return}
  const dirs=[[1,0,'SOUTH'],[-1,0,'NORTH'],[0,1,'EAST'],[0,-1,'WEST']];
  const d=dirs[Math.floor(Math.random()*dirs.length)]; const nx=Math.max(0,Math.min(4,c.x+d[0])),ny=Math.max(0,Math.min(4,c.y+d[1]));
  c.x=nx;c.y=ny;c.action='REPOSITION '+d[2];
}
function tick(){
  minute=(minute+1)%180;
  cabs.forEach(c=>{if(c.status==='ON TRIP'){if(Math.random()<.38)c.status='IDLE';else c.action='IN TRIP'}else moveCab(c)});
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
  ['X '+c.x,'Y '+c.y,'T '+minute,'IDLE '+(c.status==='IDLE'?1:0),`REQ ${nearby}`,`N ${Math.floor(Math.random()*4)}`,`I ${cabs.filter(x=>x.status==='IDLE').length}`,`HOT ${demand.value==='Airport-heavy'?1:0}`].forEach(t=>{const s=document.createElement('span');s.textContent=t;grid.appendChild(s)});
  const q=document.getElementById('qValues');q.innerHTML='';
  const vals=actions.map((a,i)=>i===1&&nearby?0.86:0.35+Math.random()*0.55);
  vals.forEach((v,i)=>{const row=document.createElement('div');row.className='qrow';row.innerHTML=`<span>${actions[i]}</span><div class="qbar"><i style="width:${Math.round(v*100)}%"></i></div>`;q.appendChild(row)});
}
function zoneName(x,y){if(x===0&&y===4)return'Airport';if(x===2&&y===2)return'Central';if(x<2)return'North';if(x>2)return'South';return'Urban'}
function updateMetrics(){
  const factor=algorithm.value==='DQN'?1:algorithm.value==='Nearest Cab'?1.18:1.07;
  const d=demand.value==='Peak'?1.12:demand.value==='Airport-heavy'?1.06:1;
  const wait=(4.8*factor*d*(.96+Math.random()*.08)).toFixed(1);
  document.getElementById('waitMetric').textContent=`${wait} min`;
  document.getElementById('serviceMetric').textContent=`${(91.2/factor/d).toFixed(1)}%`;
  document.getElementById('cancelMetric').textContent=`${(6.9*factor*d).toFixed(1)}%`;
  document.getElementById('emptyMetric').textContent=`${(18.4*factor).toFixed(1)}%`;
  document.getElementById('utilMetric').textContent=`${Math.min(94,73.6/factor).toFixed(1)}%`;
  document.getElementById('earnMetric').textContent=`₹${Math.round(1240*(.94+Math.random()*.12)/factor).toLocaleString('en-IN')}`;
}
function drawChart(){
 const c=document.getElementById('rewardChart'),ctx=c.getContext('2d'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#20312b';ctx.lineWidth=1;for(let i=1;i<5;i++){let y=i*h/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}ctx.beginPath();for(let x=0;x<=w;x+=5){const t=x/w;const y=h-(0.18+0.68*(1-Math.exp(-4*t))+Math.sin(x/19)*.025)*h;if(x===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.strokeStyle='#a8ff72';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#8ea29c';ctx.font='10px DM Mono';ctx.fillText('reward',8,14);ctx.fillText('0',8,h-3);ctx.fillText('1,000',w-35,h-3);
}
function drawBars(){const box=document.getElementById('barChart');box.innerHTML='';[['Avg wait',48],['Service rate',91],['Utilisation',74],['Earnings',83]].forEach(([label,val])=>{const r=document.createElement('div');r.className='barline';r.innerHTML=`<span>${label}</span><div class="bar"><i style="width:${val}%"></i></div><b>${val}${label==='Service rate'||label==='Utilisation'?'%':''}</b>`;box.appendChild(r)})}
fleetSlider.oninput=()=>{fleetValue.textContent=fleetSlider.value;seedCabs(+fleetSlider.value);render()};runBtn.onclick=()=>{clearInterval(timer);minute=0;seedCabs(+fleetSlider.value);render();timer=setInterval(tick,+speed.value);};speed.onchange=()=>{if(timer){clearInterval(timer);timer=setInterval(tick,+speed.value)}};algorithm.onchange=()=>{updateMetrics();render()};demand.onchange=render;window.onresize=drawChart;
seedCabs(8);render();drawChart();drawBars();
