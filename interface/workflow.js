// Presentation helpers: no changes to model recommendations or scoring.
function horizontalChart(titleText, entries, format=num, note=''){
 const max=Math.max(...entries.map(e=>Math.abs(e.value)),1);
 return `<div class="card section visual"><h3>${esc(titleText)}</h3>${entries.map(e=>`<div class="plot-row"><span>${esc(e.label)}</span><div class="plot-track"><div style="width:${Math.abs(e.value)/max*100}%"></div></div><strong>${esc(format(e.value))}</strong></div>`).join('')}<p class="notice">${esc(note)}</p></div>`;
}
function addVisuals(page){
 if(page==='overview'){
  const intro=$('#content .intro');
  intro.insertAdjacentHTML('afterend',`<div class="journey" aria-label="Suggested workflow"><a href="#forecast"><small>01</small> Plan demand <span>What to prepare</span></a><a href="#combinations"><small>02</small> Explore promotion <span>What to pair</span></a><a href="#pricing"><small>03</small> Compare prices <span>What could change</span></a><a href="#decisions"><small>04</small> Review decisions <span>What to test</span></a></div>`);
 }
 if(page==='pricing'){
  const rows=data.pricing.filter(r=>branch==='Overall'||r.Branch===branch);
  const entries=['low','mid','high'].map((key,i)=>({label:['Less sensitive','Middle scenario','More sensitive'][i],value:rows.reduce((s,r)=>s+r['profit_change_per_day_'+key],0)}));
  $('#content .table-wrap').insertAdjacentHTML('beforebegin',horizontalChart('Estimated daily profit change',entries,money,'Assumed price responses. Not measured gains.'));
 }
 if(page==='branches'){
  const entries=branchRows().filter(r=>r.prediction_complete).sort((a,b)=>Math.abs(b.cost_residual_pct)-Math.abs(a.cost_residual_pct)).slice(0,5).map(r=>({label:`${r.Product} · ${r.Branch}`,value:r.cost_residual_pct}));
  $('#content .table-wrap').insertAdjacentHTML('beforebegin',horizontalChart('Largest cost differences',entries,v=>(v>0?'+':'')+num(v)+'%','Bar length shows the size of the difference. The sign shows above or below expected cost.'));
 }
}
function enhanceTable(page){
 // Marketing already has search, tier filtering, detail views, and pagination.
 if(!['pricing','forecast','branches'].includes(page))return;
 const table=$('#content table');if(!table)return;
 const all=[...table.querySelectorAll('tbody tr')], wrap=table.closest('.table-wrap');
 const controls=document.createElement('div');controls.className='controls table-tools';
 controls.innerHTML=`<label class="sr-only" for="result-search">Search results</label><input id="result-search" type="search" placeholder="Find a product…"><label>Sort <select id="result-sort"><option value="original">Recommended order</option><option value="name">Product A–Z</option>${page==='pricing'?'<option value="gain">Estimated gain: high to low</option>':''}</select></label>${page==='pricing'?'<label>Action <select id="result-action"><option value="all">All actions</option><option value="raise">Raise</option><option value="cut">Cut</option><option value="keep">Keep</option></select></label>':''}<button id="result-reset">Reset</button><button id="result-export">Export visible results</button>`;
 wrap.before(controls);
 const pager=document.createElement('div');pager.className='row controls';pager.innerHTML='<span class="muted" role="status" id="result-count"></span><div><button id="result-prev">Previous</button> <button id="result-next">Next 10</button></div>';wrap.after(pager);
 let offset=0,filteredRows=all;
 const original=new Map(all.map((r,i)=>[r,i]));
 function refresh(){const query=$('#result-search').value.toLowerCase(),action=$('#result-action')?.value||'all';
 filteredRows=all.filter(r=>r.textContent.toLowerCase().includes(query)&&(action==='all'||r.cells[3].textContent.trim().startsWith(action)));
 const sort=$('#result-sort').value;
 const gain=row=>Number(row.cells[4].textContent.replaceAll('$','').match(/-?[\d,.]+/)?.[0].replaceAll(',',''));
 filteredRows.sort((a,b)=>sort==='name'?a.cells[0].textContent.localeCompare(b.cells[0].textContent):sort==='gain'?gain(b)-gain(a):original.get(a)-original.get(b));
 all.forEach(r=>r.hidden=true);filteredRows.forEach(r=>table.tBodies[0].append(r));filteredRows.slice(offset,offset+10).forEach(r=>r.hidden=false);
 $('#result-count').textContent=filteredRows.length?`${offset+1}–${Math.min(offset+10,filteredRows.length)} of ${filteredRows.length} results`:'No matching results. Try another search or reset.';
 $('#result-prev').disabled=offset===0;$('#result-next').disabled=offset+10>=filteredRows.length;
 }
 controls.querySelectorAll('input,select').forEach(e=>e.oninput=()=>{offset=0;refresh()});
 $('#result-prev').onclick=()=>{offset=Math.max(0,offset-10);refresh()};$('#result-next').onclick=()=>{offset+=10;refresh()};
 $('#result-reset').onclick=()=>{$('#result-search').value='';$('#result-sort').value='original';if($('#result-action'))$('#result-action').value='all';offset=0;refresh()};
 $('#result-export').onclick=()=>{const csvCell=v=>'"'+(/^[=+@-]/.test(v)?'\u0027':'')+v.replaceAll('"','""')+'"';const lines=[[...table.querySelectorAll('thead th')],...filteredRows.slice(offset,offset+10).map(r=>[...r.cells])].map(cells=>cells.map(c=>csvCell(c.innerText.replaceAll('\n',' '))).join(',')).join('\r\n');const url=URL.createObjectURL(new Blob([lines],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`stories-${page}-${branch}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
 refresh();
}
function enhanceDecisions(page){
 if(page!=='decisions')return;
 const input=$('#decision-search');
 if(input)input.oninput=event=>{decisionSearch=event.target.value;decisionOffset=0;render();requestAnimationFrame(()=>{const next=$('#decision-search');next?.focus();next?.setSelectionRange(decisionSearch.length,decisionSearch.length)})};
 $('#decision-prev')?.addEventListener('click',()=>{decisionOffset=Math.max(0,decisionOffset-decisionPageSize);render()});
 $('#decision-next')?.addEventListener('click',()=>{decisionOffset+=decisionPageSize;render()});
 document.querySelectorAll('[data-decision-detail]').forEach(button=>button.addEventListener('click',()=>showDecisionDetail(button.dataset.decisionDetail)));
}
const workflowRender=render;
render=function(){workflowRender();const page=location.hash.slice(1)||'overview';addVisuals(page);enhanceTable(page);enhanceDecisions(page);document.querySelectorAll('nav a').forEach(a=>{if(a.classList.contains('active'))a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
 const next={forecast:['Explore product pairings','#combinations'],combinations:['Compare price options','#pricing'],branches:['Review decision progress','#decisions'],pricing:['Review decision progress','#decisions']}[page];
 if(next)$('#content').insertAdjacentHTML('beforeend',`<div class="next-step"><span class="muted">Continue exploring</span><a href="${next[1]}">${next[0]} ↗</a></div>`);
};
window.onhashchange=()=>{render();window.scrollTo({top:0,behavior:'instant'})};
render();
