  const initialSessionId=new URLSearchParams(location.search).get("session");
  const state={activePane:"task",snapshot:null,taskId:null,assetId:null,dataAsset:"all",selectedEpisode:null,logType:initialSessionId?"session":null,logId:initialSessionId,toastTimer:null,timeSort:"desc",assetPickerMode:"session",annotationIndex:null,annotationIndexTask:null,annotationSaveChain:Promise.resolve(),reviewPane:null,reviewCamera:"cam_high",reviewRate:1,markStart:null,annotation:null,lang:localStorage.getItem("clothdojo-lang")||"zh",theme:localStorage.getItem("clothdojo-theme")||"dark"};
  const I18N={
    zh:{
      title:"数据控制台",
      instances:"采集实例",task:"任务",pick_asset:"选择衣服",col_instance:"实例",col_status:"状态",col_task:"任务",col_actions:"操作",
      log:"日志",log_empty:"启动实例后显示日志",log_select:"选择工作实例后显示日志",
      task_data:"采集数据",asset_filter:"资产筛选",sort_time:"采集时间",sort_flatten:"展平时间",
      stat_count:"条数据",stat_frames:"总帧数",stat_duration:"总时长",stat_annotations:"标注",
      review:"审核",pane_empty:"点击左侧轨迹条目<br>在此查看第一人称画面",
      frames_unit:"帧",sec_unit:"秒",min_unit:"分",
      available:"空闲",starting:"启动中",waiting_for_vr:"等待 VR",collecting:"采集中",stopping:"关闭中",exporting:"生成数据",failed:"故障",offline:"未就绪",port_conflict:"端口占用",
      st_complete:"完成",st_collecting:"采集中",st_unavailable:"不可续采",st_pending:"待采集",st_max:"达到上限",st_aborted:"异常",st_recording:"未完成",
      all_assets:"全部资产",count_unit:"条",pending_unit:"待采",
      no_task:"没有配置采集任务",no_episodes:"还没有采集数据",no_continuation:"还没有可采集的展平轨迹",
      pick_asset_title:"选择衣服资产",default_no_preview:"默认资产没有预览图",front:"正面",back:"背面",
      pane_unusable:"该轨迹不可用",pane_no_data:"该轨迹还没有生成数据",pane_generating:"审核数据生成中…",pane_not_generated:"审核数据还没有生成",
      marking:"标记中…点击结束",marked_count:"已标注",segment:"段",end_mark:"结束标记",mark_bad:"标记 bad 片段",good_btn:"Good",good_title:"整条轨迹审核通过",good_done:"已标记 Good",good_cleared:"已取消 Good",
      mark_start_toast:"已标记起点，再次点击结束标记",saved:"已保存标记",save_fail:"保存失败：",
      start_session:"启动",vr_connect:"VR 连接",qr:"二维码",close:"关闭",stop_session:"关闭这个遥操作实例？",stopping_session:"正在关闭实例",
      review_export:"审核导出",
      service_offline:"采集服务未启动",batch_done:"本批已完成 · 关闭后重新启动下一批",recording:"录制中",batch_remaining:"当前批次剩余",
      no_pending:"当前衣服没有待采展平终态",asset_prefix:"衣服",
      qr_title:"Quest 连接",qr_alt:"Quest 连接二维码",qr_help:"首次显示 Certificate Accepted 时返回并再次扫码，然后点击 CONNECT",
      delete:"删除",delete_confirm_perm:"永久删除",delete_confirm_fold:"删除",
      log_waiting:"进程已启动，等待日志…",session_starting:"正在启动",
      data_deleted:"数据已删除",fold_deleted:"折叠轨迹已删除，已恢复待采集",undo:"撤销",
      hil_title:"HIL 介入",assets_title:"资产筛选",hil_points:"介入点",hil_stop:"停止",hil_start:"启动 HIL",hil_interventions:"介入",no_rollouts:"没有可用 rollout",
      hil_pane_empty:"选择左侧 rollout 轨迹<br>查看第一人称画面",hil_empty_hint:"点击“标记介入点”在当前帧打点",hil_add_point:"标记介入点",hil_frame:"帧",
      hil_starting:"正在启动…",hil_ended:"已结束",hil_failed:"启动失败",hil_active_sessions:"个 HIL 会话运行中",
    },
    en:{
      title:"Data Console",
      instances:"Teleop Instances",task:"Task",pick_asset:"Pick asset",col_instance:"Instance",col_status:"Status",col_task:"Task",col_actions:"Actions",
      log:"Log",log_empty:"Start an instance to see logs",log_select:"Select a worker to see logs",
      task_data:"Collected Data",asset_filter:"Asset filter",sort_time:"Collect time",sort_flatten:"Flatten time",
      stat_count:"episodes",stat_frames:"total frames",stat_duration:"total duration",stat_annotations:"annotations",
      review:"Review",pane_empty:"Click a trajectory on the left<br>to preview its high view",
      frames_unit:"frames",sec_unit:"s",min_unit:"m",
      available:"Idle",starting:"Starting",waiting_for_vr:"Waiting VR",collecting:"Collecting",stopping:"Stopping",exporting:"Exporting",failed:"Failed",offline:"Offline",port_conflict:"Port conflict",
      st_complete:"Done",st_collecting:"Collecting",st_unavailable:"Unavailable",st_pending:"Pending",st_max:"Max actions",st_aborted:"Aborted",st_recording:"Incomplete",
      all_assets:"All assets",count_unit:"items",pending_unit:"pending",
      no_task:"No collection task configured",no_episodes:"No data collected yet",no_continuation:"No flatten states available",
      pick_asset_title:"Pick asset",default_no_preview:"Default asset has no preview",front:"Front",back:"Back",
      pane_unusable:"This trajectory is unusable",pane_no_data:"No data generated for this trajectory",pane_generating:"Generating review data…",pane_not_generated:"Review data not generated",
      marking:"Marking… click to end",marked_count:"Marked",segment:"segments",end_mark:"End mark",mark_bad:"Mark bad segment",good_btn:"Good",good_title:"Whole trajectory reviewed good",good_done:"Marked good",good_cleared:"Good mark cleared",
      mark_start_toast:"Start marked, click again to end",saved:"Mark saved",save_fail:"Save failed: ",
      start_session:"Start",vr_connect:"VR connect",qr:"QR",close:"Close",stop_session:"Stop this teleop instance?",stopping_session:"Stopping instance",
      review_export:"Review export",
      service_offline:"Collection service offline",batch_done:"Batch done · close and restart for next batch",recording:"Recording",batch_remaining:"Batch remaining",
      no_pending:"No pending flatten states for this asset",asset_prefix:"Asset",
      qr_title:"Quest connect",qr_alt:"Quest connect QR",qr_help:"When Certificate Accepted shows, go back and scan again, then CONNECT",
      delete:"Delete",delete_confirm_perm:"Permanently delete",delete_confirm_fold:"Delete",
      log_waiting:"Process started, waiting for logs…",session_starting:"Starting",
      data_deleted:"Data deleted",fold_deleted:"Fold deleted, restored to pending",undo:"Undo",
      hil_title:"HIL Interventions",assets_title:"Asset Curation",hil_points:"Breakpoints",hil_stop:"Stop",hil_start:"Start HIL",hil_interventions:"interventions",no_rollouts:"No rollouts available",
      hil_pane_empty:"Select a rollout episode<br>to preview its primary view",hil_empty_hint:"Click “Mark intervention” to add a point at the current frame",hil_add_point:"Mark intervention",hil_frame:"frame",
      hil_starting:"Starting…",hil_ended:"Ended",hil_failed:"Failed to start",hil_active_sessions:"HIL session(s) running",
    }
  };
  const t=key=>(I18N[state.lang]||I18N.zh)[key]??(I18N.zh[key]??key);
  function applyTheme(){document.documentElement.dataset.theme=state.theme;$("theme-btn").textContent=state.theme==="dark"?"☀":"🌙"}
  function toggleTheme(){state.theme=state.theme==="dark"?"light":"dark";localStorage.setItem("clothdojo-theme",state.theme);applyTheme()}
  function applyI18n(){document.querySelectorAll("[data-i18n]").forEach(el=>{el.textContent=t(el.getAttribute("data-i18n"))});$("lang-btn").textContent=state.lang==="zh"?"EN":"中";$("theme-btn").textContent=state.theme==="dark"?"☀":"🌙"}
  function toggleLang(){state.lang=state.lang==="zh"?"en":"zh";localStorage.setItem("clothdojo-lang",state.lang);applyI18n();if(state.snapshot)render(state.snapshot)}
  const $=id=>document.getElementById(id);
  const esc=value=>{const node=document.createElement("span");node.textContent=value??"—";return node.innerHTML};
  function errorMessage(detail,fallback){if(typeof detail==="string")return detail;if(Array.isArray(detail))return detail.map(item=>item.msg||JSON.stringify(item)).join("；");if(detail&&typeof detail==="object")return JSON.stringify(detail);return fallback}
  async function request(url,options={}){const response=await fetch(url,{...options,headers:{"Content-Type":"application/json",...(options.headers||{})}});if(!response.ok){let message=response.statusText;try{message=errorMessage((await response.json()).detail,message)}catch{}throw new Error(message)}return response.json()}
  function toast(message,error=false){clearTimeout(state.toastTimer);$("toast").textContent=message;$("toast").className="toast show"+(error?" error":"");state.toastTimer=setTimeout(()=>$("toast").className="toast",2600)}
  function duration(seconds){if(seconds==null)return "—";const value=Math.max(0,Math.round(seconds));if(value<60)return value+" "+t("sec_unit");const minutes=Math.floor(value/60),rest=value%60;return `${minutes} ${t("min_unit")} ${String(rest).padStart(2,"0")} ${t("sec_unit")}`}
  function setHtml(el,html){if(el.__html!==html){el.__html=html;el.innerHTML=html}}
  function episodeTime(id){const match=String(id).match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);if(!match)return id;const date=new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}Z`);return new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(date)}
  function selectedTask(){return state.snapshot?.tasks.find(task=>task.id===state.taskId)||state.snapshot?.tasks[0]}
  function pendingCount(task,assetId){return (task?.episodes||[]).filter(episode=>episode.asset_id===assetId&&episode.status==="pending").length}
  function sortedAssets(task){return [...(task?.assets||[])].sort((left,right)=>(right.default?1:0)-(left.default?1:0)||(task?.continuation?pendingCount(task,right.id)-pendingCount(task,left.id)||right.episode_count-left.episode_count||left.id.localeCompare(right.id):right.episode_count-left.episode_count||left.id.localeCompare(right.id)))}
  function render(snapshot){
    state.snapshot=snapshot;
    if(!state.taskId||!snapshot.tasks.some(task=>task.id===state.taskId))state.taskId=snapshot.tasks[0]?.id||null;
    const task=selectedTask();
    const assets=sortedAssets(task);
    if(!state.assetId||!assets.some(asset=>asset.id===state.assetId))state.assetId=assets[0]?.id||null;
    setHtml($("task-select"),snapshot.tasks.map(task=>`<option value="${esc(task.id)}" ${task.id===state.taskId?"selected":""}>${esc(task.display_name)}</option>`).join(""));
    if(state.dataAsset!=='all'&&!assets.some(asset=>asset.id===state.dataAsset))state.dataAsset='all';
    $("data-asset-pick").textContent=state.dataAsset==='all'?`${t("all_assets")}（${assets.length}）`:state.dataAsset;
    renderAssetVisual();
    renderAssetPick();
    const targets=snapshot.slots.flatMap(slot=>slot.session?[["session",slot.session.id]]:slot.export?[["export",slot.export.id]]:[]);
    if(state.logId&&!targets.some(([type,id])=>type===state.logType&&id===state.logId)){state.logType=null;state.logId=null}
    if(!state.logId&&targets.length)[state.logType,state.logId]=targets[0];
    setHtml($("slots"),snapshot.slots.map(slot=>renderSlot(slot)).join(""));
    if(state.annotationIndexTask!==state.taskId){state.annotationIndexTask=state.taskId;state.annotationIndex=null;loadAnnotationIndex(state.taskId)}
    renderData(selectedTask());
    updateLog();
  }
  function renderSlot(slot){
    const session=slot.session,job=slot.export,selected=(session&&state.logType==="session"&&session.id===state.logId)||(job&&state.logType==="export"&&job.id===state.logId);
    const task=selectedTask(),hasPending=!task?.continuation||pendingCount(task,state.assetId)>0;
    const statusClass=slot.status==="available"?"available":slot.status==="failed"||slot.status==="offline"||slot.status==="port_conflict"?"failed":slot.status==="stopping"?"stopping":"active";
    const detail=session
      ? `<div class="slot-detail"><strong>${esc(taskName(session.profile_id))} · ${esc(session.asset_id)}</strong>${session.continuation_phase==="exhausted"?'<span style="color:#ffb3ad">${t("batch_done")}</span>':session.recording?`<span>${t("recording")} · ${(session.recording_frames||0).toLocaleString()} ${t("frames_unit")}</span>`:session.continuation_phase==="active"?`<span>${t("batch_remaining")} ${session.continuation_remaining??0} ${t("count_unit")}</span>`:""}</div>`
      : job
      ? `<div class="slot-detail"><strong>${t("review_export")} · ${esc(episodeTime(job.episode_id))}</strong></div>`
      : `<div class="slot-detail">${slot.status==="offline"?`<strong>${t("service_offline")}</strong>`:slot.error?`<strong>${esc(slot.error)}</strong>`:""}</div>`;
    const actions=session
      ? `<button class="primary" onclick="openConnect(event,'${esc(slot.id)}')">${t("vr_connect")}</button><button onclick="showQr(event,'${esc(session.id)}','${esc(slot.id)}','${esc(session.endpoint)}')">${t("qr")}</button><button class="danger" onclick="stopSession(event,'${esc(session.id)}')">${t("close")}</button>`
      : job
      ? ``
      : `<button class="primary" ${slot.status!=="available"||!hasPending?"disabled":""} title="${!hasPending?t("no_pending"):""}" onclick="startSession(event,'${esc(slot.id)}')">${t("start_session")}</button>`;
    const target=session?`selectLog('session','${esc(session.id)}')`:job?`selectLog('export','${esc(job.id)}')`:"";
    return `<div class="slot ${selected?"selected":""}" onclick="${target}"><span class="gpu">${esc(slot.worker_id?`${slot.worker_id} · GPU ${slot.gpu_id}`:`GPU ${slot.gpu_id}`)}</span><span class="chip ${statusClass}">${t(slot.status)||esc(slot.status)}</span>${detail}<div class="slot-actions">${actions}</div></div>`;
  }
  function episodeKey(task,episode){return task.continuation?episode.result_episode_id:episode.id}
  function filteredRows(task){return orderedEpisodeRows(task).filter(episode=>state.dataAsset==="all"||episode.asset_id===state.dataAsset)}
  function selectEpisode(key){state.selectedEpisode=key||null;state.activePane="task";renderData(selectedTask())}
  function renderData(task){
    if(!task){setHtml($("episodes"),`<div class="empty-row">${t("no_task")}</div>`);renderReviewPane(null,null);return}
    $("time-sort").textContent=`${task.continuation?t("sort_flatten"):t("sort_time")} ${state.timeSort==='desc'?'↓':'↑'}`;
    const rows=filteredRows(task);
    const usable=rows.filter(episode=>episode.usable);
    $("episode-count").textContent=usable.length;
    $("frame-count").textContent=usable.reduce((total,episode)=>total+(episode.action_count||0),0).toLocaleString();
    $("duration").textContent=duration(usable.reduce((total,episode)=>total+(episode.duration_seconds||0),0));
    $("annotation-count").textContent=rows.filter(episode=>{
      const annotation=(state.annotationIndex||{})[episodeKey(task,episode)];
      return !!annotation&&((annotation.segments||[]).length>0||annotation.review_status==="complete");
    }).length;
    const emptyMessage=task.continuation?t("no_continuation"):t("no_episodes");
    setHtml($("episodes"),rows.length?rows.map(episode=>renderEpisodeItem(task,episode)).join(''):`<div class="empty-row">${emptyMessage}</div>`);
    if(state.selectedEpisode&&!rows.some(episode=>episodeKey(task,episode)===state.selectedEpisode))state.selectedEpisode=null;
    renderReviewPane(task,rows);
  }
  function renderReviewPane(task,rows){
    const body=$("pane-body"),actions=$("pane-actions");
    const reset=()=>{body.innerHTML=`<div class="pane-empty">${t("pane_empty")}</div>`;actions.innerHTML="";$("pane-segments").innerHTML="";state.reviewPane=null;state.markStart=null;state.annotation=null};
    if(!task){reset();return}
    const episode=(rows||filteredRows(task)).find(item=>episodeKey(task,item)===state.selectedEpisode);
    if(!episode){reset();return}
    const key=episodeKey(task,episode);
    const review=episode.review;
    if(!key||!episode.reviewable){
      body.innerHTML=`<div class="pane-empty">${episode.reviewable?t("pane_no_data"):t("pane_unusable")}</div>`;
      actions.innerHTML="";
      state.reviewPane=null;state.markStart=null;
      return;
    }
    if(!state.reviewPane||state.reviewPane.key!==key)state.markStart=null;
    let info=null;
    if(review?.status==="ready"){
      const cameras=review.cameras||[];
      if(!cameras.includes(state.reviewCamera))state.reviewCamera=cameras[0]||"cam_high";
      info={key,taskId:task.id,cameras,camera:state.reviewCamera,src:`/api/tasks/${encodeURIComponent(task.id)}/episodes/${encodeURIComponent(key)}/videos/${encodeURIComponent(state.reviewCamera)}`,fps:review.fps||50,from:0,to:(review.frames||0)/(review.fps||50),length:review.frames||0};
    }else if(review?.status==="starting"||review?.status==="running"){
      body.innerHTML=`<div class="pane-empty">${t("pane_generating")}</div>`;
      actions.innerHTML="";
      state.reviewPane=null;
      return;
    }else{
      body.innerHTML=`<div class="pane-empty">${t("pane_not_generated")}</div>`;
      actions.innerHTML="";
      state.reviewPane=null;
      return;
    }
    ensurePaneVideo(body,info);
    state.reviewPane=info;
    if(!state.annotation||state.annotation.taskId!==task.id||state.annotation.key!==key){
      state.annotation=null;
      loadAnnotation(task.id,key);
    }
    renderPaneActions();
  }
  function ensurePaneVideo(body,info){
    const stage=body.querySelector(".video-stage");
    if(stage&&stage.dataset.src===info.src)return;
    body.innerHTML=`
      <div class="video-stage" data-src="${info.src}">
        <video preload="auto" playsinline></video>
        <div class="bad-badge"></div>
      </div>
      <div class="video-bar"><div class="bar-marks"></div><div class="bar-playhead"></div></div>`;
    const video=body.querySelector("video");
    video.src=info.src;
    video.playbackRate=state.reviewRate;
    video.addEventListener("loadedmetadata",()=>{video.currentTime=info.resumeFrame==null?(info.from||0):info.resumeFrame/(info.fps||50);syncPaneTimeline()});
    video.addEventListener("timeupdate",syncPaneTimeline);
    video.addEventListener("seeked",syncPaneTimeline);
    video.addEventListener("click",()=>{state.activePane="task";togglePanePlay()});
    body.querySelector(".video-bar").addEventListener("click",e=>{state.activePane="task";seekPaneFromBar(e)});
  }
  function paneVideo(){return $("pane-body").querySelector("video")}
  function paneInfo(){return state.reviewPane||{fps:50,from:0,length:0}}
  function paneFrameAt(){const v=paneVideo();if(!v)return 0;const info=paneInfo(),total=paneTotalFrames(),frame=Math.floor((v.currentTime-(info.from||0))*(info.fps||50));return Math.max(0,Math.min(Math.max(0,total-1),frame))}
  function paneTotalFrames(){const info=paneInfo();if(info.length)return info.length;const v=paneVideo();return v&&v.duration?Math.round(v.duration*(info.fps||50)):0}
  function syncPaneTimeline(){
    const body=$("pane-body"),v=paneVideo();
    if(!v)return;
    const marks=body.querySelector(".bar-marks"),playhead=body.querySelector(".bar-playhead"),badge=body.querySelector(".bad-badge");
    const info=paneInfo(),total=paneTotalFrames(),frame=paneFrameAt(),dur=v.duration||0;
    if(dur>0)playhead.style.left=`${(v.currentTime/dur)*100}%`;
    const segments=(state.annotation?.segments)||[];
    marks.innerHTML=total?segments.map(s=>{const left=(s.start_frame/total)*100;const width=Math.max(0.2,(s.end_frame-s.start_frame+1)/total*100);return `<div class="bad" style="left:${left}%;width:${width}%"></div>`}).join(""):"";
    badge.style.display=segments.some(s=>frame>=s.start_frame&&frame<=s.end_frame)?"block":"none";
  }
  function togglePanePlay(){const v=paneVideo();if(!v)return;if(v.paused)v.play();else v.pause()}
  const reviewRates=[0.5,0.75,1,1.25,1.5,2,2.5,3];
  function reviewRateOptions(){return reviewRates.map(rate=>`<option value="${rate}" ${rate===state.reviewRate?"selected":""}>${rate}×</option>`).join("")}
  function setPaneRate(value){state.reviewRate=Number(value);const v=paneVideo();if(v)v.playbackRate=state.reviewRate}
  function seekPaneFromBar(e){const v=paneVideo();if(!v||!v.duration)return;const bar=$("pane-body").querySelector(".video-bar");const r=bar.getBoundingClientRect();v.currentTime=Math.max(0,Math.min(v.duration,(e.clientX-r.left)/r.width*v.duration))}
  function paneStep(delta){const v=paneVideo();if(!v)return;const fps=paneInfo().fps||50;v.currentTime=Math.max(0,Math.min(v.duration||0,v.currentTime+delta/fps))}
  function renderPaneActions(){
    const actions=$("pane-actions"),seg=$("pane-segments");
    const ann=state.annotation||{segments:[],review_status:"unreviewed"};
    const ready=!!state.annotation&&state.annotation.taskId===selectedTask()?.id&&state.annotation.key===state.reviewPane?.key;
    const marking=state.markStart!=null;
    const segments=ann.segments||[];
    const count=segments.length;
    seg.innerHTML=count
      ? segments.map((s,i)=>`<div class="seg-row"><span class="seg-idx">#${i+1}</span><span class="seg-range">${s.start_frame}-${s.end_frame} ${t("frames_unit")}</span><button class="seg-del" onclick="removeSegment(${i})">×</button></div>`).join("")
      : (marking?`<div class="seg-empty">${t("marking")}</div>`:"");
    const good=ann.review_status==="complete"&&count===0;
    const cameras=state.reviewPane?.cameras||[];
    const cameraOptions=cameras.map(camera=>`<option value="${esc(camera)}" ${camera===state.reviewCamera?"selected":""}>${esc(camera)}</option>`).join("");
    actions.innerHTML=`<select class="speed-select" aria-label="审核视角" title="审核视角" onchange="setReviewCamera(this.value)">${cameraOptions}</select><select class="speed-select" aria-label="播放倍速" title="播放倍速" onchange="setPaneRate(this.value)">${reviewRateOptions()}</select><button class="${marking?"danger":""}" ${ready?"":"disabled"} onclick="toggleMark()">${marking?t("end_mark"):t("mark_bad")}</button><button ${count===0?"disabled":""} onclick="undoSegment()">${t("undo")}</button><button class="${good?"good-btn on":"good-btn"}" ${count||!ready?"disabled":""} title="${t("good_title")}" onclick="toggleGood()">${good?"✓ ":""}${t("good_btn")}</button>`;
  }
  function setReviewCamera(camera){
    const info=state.reviewPane;if(!info||!info.cameras.includes(camera)||camera===info.camera)return;
    const next={...info,camera,src:`/api/tasks/${encodeURIComponent(info.taskId)}/episodes/${encodeURIComponent(info.key)}/videos/${encodeURIComponent(camera)}`,resumeFrame:paneFrameAt()};
    state.reviewCamera=camera;state.reviewPane=next;ensurePaneVideo($("pane-body"),next);renderPaneActions();
  }
  function undoSegment(){
    const ann=state.annotation;
    if(!ann||!(ann.segments||[]).length)return;
    ann.segments.pop();
    renderPaneActions();
    saveAnnotation();
  }
  function removeSegment(i){
    const ann=state.annotation;
    if(!ann||!(ann.segments||[]).length)return;
    ann.segments.splice(i,1);
    renderPaneActions();
    saveAnnotation();
  }
  function toggleMark(){
    const info=state.reviewPane;
    if(!info||state.annotation?.taskId!==selectedTask()?.id||state.annotation?.key!==info.key)return;
    const video=paneVideo();
    if(!video)return;
    if(state.markStart==null){
      state.markStart=paneFrameAt();
      toast(t("mark_start_toast"));
      renderPaneActions();
    }else{
      const start=state.markStart,end=paneFrameAt();
      state.markStart=null;
      const s=Math.min(start,end),e=Math.max(start,end);
      const ann=state.annotation||{taskId:selectedTask()?.id,key:state.reviewPane.key,segments:[],review_status:"unreviewed"};
      ann.segments.push({start_frame:s,end_frame:e,label:"bad"});
      state.annotation=ann;
      renderPaneActions();
      saveAnnotation();
    }
  }
  async function loadAnnotation(taskId,key){
    try{
      const a=await request(`/api/tasks/${encodeURIComponent(taskId)}/episodes/${encodeURIComponent(key)}/annotation`);
      if(selectedTask()?.id!==taskId||state.selectedEpisode!==key)return;
      state.annotation={taskId,key,segments:(a.segments||[]).map(s=>({...s})),review_status:a.review_status||"unreviewed"};
    }catch(err){
      if(selectedTask()?.id!==taskId||state.selectedEpisode!==key)return;
      state.annotation=null;
      toast(err.message,true);
    }
    renderPaneActions();
  }
  async function loadAnnotationIndex(taskId){
    try{
      const index=await request(`/api/tasks/${encodeURIComponent(taskId)}/annotations`);
      if(state.annotationIndexTask!==taskId)return;
      state.annotationIndex=index||{};
    }catch(err){if(state.annotationIndexTask===taskId)state.annotationIndex={}}
    renderData(selectedTask());
  }
  function persistAnnotation(taskId,key,reviewStatus,segments){
    const payload={review_status:reviewStatus,segments:segments.map(segment=>({...segment}))};
    const operation=state.annotationSaveChain.then(()=>request(`/api/tasks/${encodeURIComponent(taskId)}/episodes/${encodeURIComponent(key)}/annotation`,{method:"PUT",body:JSON.stringify({...payload,expected_revision:state.annotation?.revision??null})})).then(saved=>{if(state.annotation?.taskId===taskId&&state.annotation?.key===key)state.annotation.revision=saved.revision;return saved});
    state.annotationSaveChain=operation.catch(()=>{});
    return operation;
  }
  async function saveAnnotation(){
    const ann=state.annotation;const key=ann?.key;const task=selectedTask();
    if(!ann||!key||!task)return;
    const taskId=task.id,segments=ann.segments.map(segment=>({...segment}));
    try{
      await persistAnnotation(taskId,key,"in_progress",segments);
      toast(t("saved"));
      if(state.annotationIndex&&state.annotationIndexTask===taskId){state.annotationIndex[key]={segments,review_status:"in_progress"};renderData(selectedTask())}
    }catch(err){toast(t("save_fail")+err.message,true)}
  }
  async function toggleGood(){
    const ann=state.annotation;const key=ann?.key;const task=selectedTask();
    if(!ann||!key||!task)return;
    if((ann.segments||[]).length)return;
    const target=ann.review_status==="complete"?"in_progress":"complete";
    const taskId=task.id;
    try{
      await persistAnnotation(taskId,key,target,[]);
      if(selectedTask()?.id===taskId&&state.selectedEpisode===key)state.annotation={...ann,review_status:target,segments:[]};
      if(state.annotationIndex&&state.annotationIndexTask===taskId)state.annotationIndex[key]={segments:[],review_status:target};
      toast(target==="complete"?t("good_done"):t("good_cleared"));
      renderData(selectedTask());
    }catch(err){toast(err.message,true)}
  }

  function orderedEpisodeRows(task){
    const direction=state.timeSort==="desc"?-1:1;
    return [...(task?.episodes||[])].sort((left,right)=>direction*String(left.id).localeCompare(String(right.id)));
  }
  function renderEpisodeItem(task,episode){
    const key=episodeKey(task,episode);
    const limited=episode.stop_reason==='max_actions_reached';
    const label=task.continuation
      ? episode.status==='complete'?t("st_complete"):episode.status==='collecting'?t("st_collecting"):episode.status==='unavailable'?t("st_unavailable"):t("st_pending")
      : limited?t("st_max"):episode.status==='complete'?t("st_complete"):episode.status==='aborted'?t("st_aborted"):episode.status==='recording'?t("st_recording"):episode.status;
    const statusClass=episode.status==='complete'&&episode.usable?'available':episode.status==='collecting'?'active':episode.status==='unavailable'?'aborted':task.continuation?'':'aborted';
    const frames=episode.action_count==null?'—':episode.action_count.toLocaleString();
    const ann=(state.annotationIndex||{})[key];
    const flagged=!!ann&&(ann.segments||[]).length>0;
    const good=!!ann&&ann.review_status==="complete"&&!flagged;
    const flagMark=flagged?`<span class="flag-dot" title="${t("marked_count")} ${ann.segments.length} ${t("segment")}"></span>`:(good?`<span class="flag-dot good" title="${t("good_title")}"></span>`:'');
    const delArgs=`'${esc(task.id)}','${esc(key)}',${task.continuation?`'${esc(episode.id)}'`:'null'}`
    const deleteAction=key&&episode.deletable!==false?`<button class="delete" title="${t("delete")}" onclick="event.stopPropagation();deleteEpisode(${delArgs})">×</button>`:'';
    return `<div class="episode-item ${key===state.selectedEpisode?'selected':''} " onclick="selectEpisode('${esc(key||'')}')"><span class="status-cell"><span class="chip ${statusClass}">${esc(label)}</span>${flagMark}</span><span class="episode-id">${esc(episodeTime(episode.id))}</span><span class="asset">${esc(episode.asset_id)}</span><span class="meta">${frames} ${t("frames_unit")} · ${duration(episode.duration_seconds)}</span>${deleteAction}</div>`;
  }
  function taskName(id){return state.snapshot?.tasks.find(task=>task.id===id)?.display_name||id}
  function renderAssetPick(){
    $("asset-pick").textContent=state.assetId?`${t("asset_prefix")} · ${state.assetId}`:t("pick_asset");
  }
  function renderAssetVisual(){
    const el=$("asset-visual");
    const task=selectedTask();
    if(!task){el.innerHTML="";return}
    if(state.dataAsset==='all'){el.innerHTML=`<div class="preview-empty">${t("asset_filter")}</div>`;return}
    const base=`/api/tasks/${encodeURIComponent(task.id)}/assets/${encodeURIComponent(state.dataAsset)}/preview`;
    setHtml(el,`<img loading="lazy" decoding="async" src="${base}/front" alt="${t("front")}"><img loading="lazy" decoding="async" src="${base}/back" alt="${t("back")}">`);
  }
  function openAssetPicker(mode){
    state.assetPickerMode=mode==="filter"?"filter":mode==="hil"?"hil":"session";
    let assets,current,title,task=null;
    if(state.assetPickerMode==="hil"){
      const profile=hilProfile();if(!profile)return;
      const counts=new Map();
      (profile.runs||[]).forEach(r=>{const c=counts.get(r.asset_id)||{runs:0,episodes:0};c.runs++;c.episodes+=r.episodes.length;counts.set(r.asset_id,c)});
      assets=[...counts.entries()].map(([id,c])=>({id,episode_count:c.episodes,hilRuns:c.runs})).sort((l,r)=>r.hilRuns-l.hilRuns||l.id.localeCompare(r.id));
      current=hilState.asset;
      title=`${t("asset_filter")} · ${profile.display_name}`;
    }else{
      task=selectedTask();if(!task)return;
      assets=sortedAssets(task);
      current=state.assetPickerMode==="filter"?state.dataAsset:state.assetId;
      title=`${state.assetPickerMode==="filter"?t("asset_filter"):t("pick_asset_title")} · ${task.display_name}`;
    }
    const previewTask=state.assetPickerMode==="hil"?hilState.profile:task.id;
    const allCard=state.assetPickerMode==="session"?"":`<button class="asset-card ${current==="all"?"selected":""}" onclick="pickAssetFromModal(\x27all\x27)"><div class="images"><div style="display:grid;place-items:center;color:var(--muted);grid-column:1/-1">${t("all_assets")}</div></div><span class="meta"><span class="id">${t("all_assets")}</span><span class="count">${assets.length}</span></span></button>`;
    $("asset-modal-title").textContent=title;
    setHtml($("asset-cards"),allCard+assets.map(asset=>{
      const base=`/api/tasks/${encodeURIComponent(previewTask)}/assets/${encodeURIComponent(asset.id)}/preview`;
      const images=asset.id==="default"?`<div class="images"><div style="display:grid;place-items:center;color:var(--muted);grid-column:1/-1">${t("default_no_preview")}</div></div>`:`<div class="images"><img loading="lazy" decoding="async" src="${base}/front" alt="${t("front")}"><img loading="lazy" decoding="async" src="${base}/back" alt="${t("back")}"></div>`;
      const count=state.assetPickerMode==="hil"?`${asset.hilRuns} rollout · ${asset.episode_count} ${t("count_unit")}`:`${task.continuation?`${pendingCount(task,asset.id)} ${t("pending_unit")} · `:""}${asset.episode_count} ${t("count_unit")}`;
      return `<button class="asset-card ${asset.id===current?"selected":""}" onclick="pickAssetFromModal(\x27${esc(asset.id)}\x27)">${images}<span class="meta"><span class="id">${esc(asset.id)}</span><span class="count">${count}</span></span></button>`;
    }).join(""));
    $("asset-modal").classList.add("open");
  }
  function pickAssetFromModal(id){
    closeAssetPicker();
    if(state.assetPickerMode==="filter"){state.dataAsset=id;renderData(selectedTask())}
    else if(state.assetPickerMode==="hil"){hilState.asset=id;if(!hilRuns().some(r=>r.id===hilState.run?.id)){hilState.run=null;hilState.episode=null;hilState.points=[]}hilRender()}
    else{state.assetId=id;render(state.snapshot)}
  }
  function closeAssetPicker(){$("asset-modal").classList.remove('open')}
  function selectLog(type,id){state.logType=type;state.logId=id;render(state.snapshot)}
  async function startSession(event,slotId){event.stopPropagation();try{const session=await request("/api/sessions",{method:"POST",body:JSON.stringify({slot_id:slotId,profile_id:state.taskId,asset_id:state.assetId})});state.logType="session";state.logId=session.id;toast(`${session.worker_id||slotId} ${t("session_starting")}`);await refresh()}catch(error){toast(error.message,true)}}
  async function stopSession(event,id){event.stopPropagation();if(!confirm(t("stop_session")))return;try{await request(`/api/sessions/${encodeURIComponent(id)}/stop`,{method:"POST"});toast(t("stopping_session"));await refresh()}catch(error){toast(error.message,true)}}
  function connectionUrl(slotId){const slot=state.snapshot.slots.find(item=>item.id===slotId),origin=new URL(`https://${slot.endpoint}`);return `${origin.origin}/client/?${new URLSearchParams({serverIP:origin.hostname,port:String(slot.proxy_port),codec:"h264",immersiveMode:"vr",deviceFrameRate:"72",maxStreamingBitrateMbps:"80",panelHiddenAtStart:"true",deviceProfile:"pico4ultra",useQuestColorWorkaround:"false",cloudxrAutoRefreshMode:"any"})}`}
  function openConnect(event,slotId){event.stopPropagation();location.href=connectionUrl(slotId)}
  async function deleteEpisode(taskId,id,sourceId=null){const message=sourceId?`删除 ${episodeTime(sourceId)} 对应的折叠轨迹？\\n原展平轨迹会恢复为“待采集”。`:`永久删除 ${episodeTime(id)} 的采集数据？\\n此操作无法撤销。`;if(!confirm(message))return;try{await request(`/api/tasks/${encodeURIComponent(taskId)}/episodes/${encodeURIComponent(id)}`,{method:"DELETE"});toast(sourceId?t("fold_deleted"):t("data_deleted"));await refresh()}catch(error){toast(error.message,true)}}
  async function updateLog(){
    const {logType:type,logId:id}=state;
    if(!id){$("log-session").textContent="";$("log").innerHTML=`<div class="log-empty">${t("log_select")}</div>`;return}
    const slot=state.snapshot?.slots.find(item=>(type==="session"?item.session:item.export)?.id===id);
    $("log-session").textContent=slot?(slot.worker_id?`${slot.worker_id} · GPU ${slot.gpu_id}`:`GPU ${slot.gpu_id}`):"";
    const url=type==="export"?`/api/exports/${encodeURIComponent(id)}/log`:`/api/sessions/${encodeURIComponent(id)}/log`;
    try{const payload=await request(url);const old=$("log-text"),atBottom=!old||old.scrollHeight-old.scrollTop-old.clientHeight<28;setHtml($("log"),`<pre class="log" id="log-text">${esc(payload.log||t("log_waiting"))}</pre>`);if(atBottom&&$("log-text"))$("log-text").scrollTop=$("log-text").scrollHeight}catch(error){setHtml($("log"),`<div class="log-empty">${esc(error.message)}</div>`)}
  }
  function showQr(event,id,workerId,endpoint){event.stopPropagation();$("qr-title").textContent=`${workerId} · ${t("qr_title")}`;$("qr-endpoint").textContent=endpoint;$("qr-image").src=`/api/sessions/${encodeURIComponent(id)}/qr`;$("qr-modal").classList.add("open")}
  function closeQr(){$("qr-modal").classList.remove("open");$("qr-image").removeAttribute("src")}
  async function refresh(){try{render(await request("/api/snapshot"))}catch(error){toast(error.message,true)}}
  $("task-select").onchange=event=>{state.taskId=event.target.value;state.assetId=null;state.dataAsset="all";render(state.snapshot)};
  $("asset-pick").onclick=()=>openAssetPicker("session");
  $("data-asset-pick").onclick=()=>openAssetPicker("filter");
  $("time-sort").onclick=()=>{state.timeSort=state.timeSort==="desc"?"asc":"desc";renderData(selectedTask())};
  const scrub={active:false,dir:0,held:[],gen:0,t0:0,guard:null};
  function scrubCtx(){const useHil=state.activePane==="hil"&&hilState.episode&&hilVideo();const v=useHil?hilVideo():paneVideo();return v?{v:useHil?hilVideo():paneVideo(),fps:useHil?(hilState.episode.fps||50):(paneInfo().fps||50)}:null}
  function scrubStop(){scrub.active=false;scrub.dir=0;scrub.held=[];clearTimeout(scrub.guard);scrub.gen++}
  function scrubStart(dir){if(!scrub.held.includes(dir))scrub.held.push(dir);scrub.dir=dir;if(scrub.active)return;const ctx=scrubCtx();if(!ctx)return;if(!ctx.v.paused)ctx.v.pause();scrub.active=true;scrubStep()}
  function scrubStep(){if(!scrub.active||!scrub.dir)return scrubStop();const ctx=scrubCtx();if(!ctx)return scrubStop();const v=ctx.v,next=Math.max(0,Math.min(v.duration||0,v.currentTime+scrub.dir/ctx.fps));if(Math.abs(next-v.currentTime)<1e-6)return scrubStop();scrub.gen++;const gen=scrub.gen;scrub.t0=performance.now();v.addEventListener("seeked",()=>{if(gen===scrub.gen)scrubNext()},{once:true});clearTimeout(scrub.guard);scrub.guard=setTimeout(()=>{if(gen===scrub.gen)scrubNext()},250);v.currentTime=next}
  function scrubNext(){if(!scrub.active||!scrub.dir)return;const ctx=scrubCtx();if(!ctx)return scrubStop();const wait=Math.max(0,1000/ctx.fps-(performance.now()-scrub.t0));clearTimeout(scrub.guard);scrub.guard=setTimeout(scrubStep,wait)}
  function scrubRelease(dir){scrub.held=scrub.held.filter(d=>d!==dir);if(!scrub.held.length)scrubStop();else scrub.dir=scrub.held[scrub.held.length-1]}
  document.addEventListener("keydown",e=>{const tag=(e.target&&e.target.tagName)||"";if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT"||tag==="VIDEO")return;const useHil=state.activePane==="hil"&&hilState.episode&&hilVideo();const v=useHil?hilVideo():paneVideo();if(!v)return;if(e.code==="ArrowLeft"){e.preventDefault();if(!e.repeat)scrubStart(-1)}else if(e.code==="ArrowRight"){e.preventDefault();if(!e.repeat)scrubStart(1)}else if(e.code==="Space"){e.preventDefault();scrubStop();useHil?hilTogglePlay():togglePanePlay()}});
  document.addEventListener("keyup",e=>{if(e.code==="ArrowLeft")scrubRelease(-1);else if(e.code==="ArrowRight")scrubRelease(1)});
  window.addEventListener("blur",scrubStop);
  const asState={status:"pending",page:1,pageCount:1,pollTimer:null,searchTimer:null};
  function asStateLabel(asset){if(asset.converted)return"已转换";return{queued:"等待中",running:"转换中",failed:"转换失败",deleted:"已删除"}[asset.conversion_status]||""}
  function asActions(asset){
    const busy=["queued","running"].includes(asset.conversion_status);
    if(asState.status==="pending")return`<button class="action accept" data-action="accept">接受</button><button class="action reject" data-action="reject">拒绝</button>`;
    if(asState.status==="rejected")return`<button class="action accept" data-action="accept">接受</button>`;
    const conversion=asset.converted?`<button class="action delete" data-action="delete" ${asset.trajectory_count||asset.deletion_available===false?`disabled title="${asset.trajectory_count?`已有 ${asset.trajectory_count} 条轨迹`:"转换删除尚未迁移"}"`:""}>删除</button>`:`<button class="action" data-action="convert" ${busy||asset.conversion_available===false?`disabled title="${asset.conversion_available===false?"资产转换尚未迁移":""}"`:""}>${busy?"转换中":"转换"}</button>`;
    return`<button class="action reject" data-action="reject">拒绝</button>${conversion}`;
  }
  function asCard(asset){
    const images=asset.preview?`<img loading="lazy" src="/api/assets/${encodeURIComponent(asset.id)}/preview/front" alt="正面"><img loading="lazy" src="/api/assets/${encodeURIComponent(asset.id)}/preview/back" alt="背面">`:`<div class="no-preview">暂无预览</div>`;
    const label=asStateLabel(asset),error=asset.conversion_error?`<div class="error" title="${esc(asset.conversion_error)}">${esc(asset.conversion_error)}</div>`:"",trajectory=asState.status==="accepted"?`<span class="state trajectory">${asset.trajectory_count} 条轨迹</span>`:"";
    return`<article class="as-card" data-id="${esc(asset.id)}"><div class="images">${images}</div><div class="meta"><div class="identity"><span class="id">${esc(asset.id)}</span>${trajectory}${label?`<span class="state ${asset.conversion_status==="failed"?"failed":""}">${label}</span>`:""}</div>${error}<div class="actions">${asActions(asset)}</div></div></article>`;
  }
  async function asLoad(){
    clearTimeout(asState.pollTimer);
    const query=$("as-search").value.trim();
    try{
      const params=new URLSearchParams({status:asState.status,page:String(asState.page),page_size:"48",query});
      const data=await request(`/api/assets?${params}`);
      asState.page=data.page;asState.pageCount=data.page_count;
      $("as-pending-count").textContent=data.counts.pending.toLocaleString();
      $("as-accepted-count").textContent=data.counts.accepted.toLocaleString();
      $("as-rejected-count").textContent=data.counts.rejected.toLocaleString();
      $("as-converted").textContent=`已转换 ${data.counts.converted.toLocaleString()}`;
      $("as-summary").textContent=query?`找到 ${data.total.toLocaleString()} 件`:`共 ${data.total.toLocaleString()} 件`;
      $("as-grid").innerHTML=data.assets.length?data.assets.map(asCard).join(""):`<div class="as-empty">这里暂时没有资产</div>`;
      $("as-page-label").textContent=`${asState.page} / ${asState.pageCount}`;
      $("as-jump-page").max=String(asState.pageCount);
      $("as-jump-page").placeholder=String(asState.page);
      $("as-prev").disabled=asState.page<=1;
      $("as-next").disabled=asState.page>=asState.pageCount;
      const batch=$("as-batch-convert");
      batch.hidden=asState.status!=="accepted";
      batch.disabled=data.counts.convertible===0;
      batch.textContent=data.counts.convertible?`批量转换 ${data.counts.convertible}`:data.counts.converting?`队列中 ${data.counts.converting}`:"全部已转换";
      if(data.counts.converting>0)asState.pollTimer=setTimeout(asLoad,2000);
    }catch(error){toast(error.message,true)}
  }
  $("as-tabs").onclick=event=>{const tab=event.target.closest(".as-tab");if(!tab)return;asState.status=tab.dataset.status;asState.page=1;document.querySelectorAll("#as-tabs .as-tab").forEach(item=>item.classList.toggle("active",item===tab));asLoad()};
  $("as-search").oninput=()=>{clearTimeout(asState.searchTimer);asState.page=1;asState.searchTimer=setTimeout(asLoad,220)};
  $("as-prev").onclick=()=>{if(asState.page>1){asState.page--;asLoad()}};
  $("as-next").onclick=()=>{if(asState.page<asState.pageCount){asState.page++;asLoad()}};
  $("as-jump-form").onsubmit=event=>{event.preventDefault();const requested=Number.parseInt($("as-jump-page").value,10);if(!Number.isFinite(requested))return;asState.page=Math.max(1,Math.min(asState.pageCount,requested));$("as-jump-page").value="";asLoad()};
  $("as-batch-convert").onclick=async()=>{const button=$("as-batch-convert"),count=Number(button.textContent.match(/\d+/)?.[0]||0);if(!count||!confirm(`转换全部 ${count} 件已接受但未转换的资产？`))return;try{button.disabled=true;const result=await request("/api/assets/conversions",{method:"POST"});toast(`已加入队列：${result.queued} 件`);await asLoad()}catch(error){button.disabled=false;toast(error.message,true)}};
  $("as-grid").onclick=async event=>{const button=event.target.closest("button[data-action]");if(!button||button.disabled)return;const id=button.closest(".as-card").dataset.id,action=button.dataset.action;try{button.disabled=true;if(action==="accept"||action==="reject")await request(`/api/assets/${id}/decision`,{method:"PUT",body:JSON.stringify({decision:action==="accept"?"accepted":"rejected"})});else if(action==="convert")await request(`/api/assets/${id}/conversion`,{method:"POST"});else if(action==="delete"){if(!confirm(`删除 ${id} 的转换数据？原始数据会保留。`)){button.disabled=false;return}await request(`/api/assets/${id}/converted`,{method:"DELETE"})}toast(action==="convert"?"已加入转换队列":"已保存");await asLoad()}catch(error){button.disabled=false;toast(error.message,true)}};
  asLoad();
  const hilState={data:null,profile:null,asset:"all",run:null,episode:null,points:[],session:null,pending:false};
  function hilProfiles(){return hilState.data?.profiles||[]}
  function hilProfile(){return hilProfiles().find(x=>x.id===hilState.profile)}
  function hilRuns(){return (hilProfile()?.runs||[]).map(run=>({...run,episodes:(run.episodes||[]).filter(episode=>hilState.asset==="all"||episode.asset_id===hilState.asset)})).filter(run=>run.episodes.length)}
  function hilInterventionSlots(){return (hilState.data?.slots||[]).filter(x=>x.session?.mode==="intervention")}
  function hilVideoUrl(view){return `/api/hil/profiles/${encodeURIComponent(hilState.profile)}/runs/${encodeURIComponent(hilState.run.id)}/episodes/${encodeURIComponent(hilState.episode.id)}/videos/${view}`}
  function hilRender(){
    setHtml($("hil-profile"),hilProfiles().map(x=>`<option value="${esc(x.id)}">${esc(x.display_name)}</option>`).join(""));$("hil-profile").value=hilState.profile||"";
    const withRollouts=new Set((hilProfile()?.runs||[]).flatMap(run=>(run.episodes||[]).map(episode=>episode.asset_id)));
    $("hil-asset-pick").textContent=hilState.asset==="all"?`${t("all_assets")}（${withRollouts.size}）`:hilState.asset;
    const runs=hilRuns();
    $("hil-run-count").textContent=runs.length;
    $("hil-episode-count").textContent=runs.reduce((n,r)=>n+r.episodes.length,0);
    $("hil-points-count").textContent=hilState.points.length;
    $("hil-annotation-count").textContent=runs.reduce((total,run)=>total+run.episodes.filter(episode=>episode.good||episode.interventions.length>0).length,0);
    renderHilAssetVisual();
    setHtml($("hil-list"),runs.length?runs.map(run=>`<div class="hil-run">${esc(run.id)}</div>${run.episodes.map(ep=>{
      const sel=hilState.run?.id===run.id&&hilState.episode?.id===ep.id;
      const chip=ep.interventions.length?`<span class="chip active">${ep.interventions.length} ${t("hil_interventions")}</span>`:"";
      const mark=ep.interventions.length?`<span class="flag-dot" title="${ep.interventions.length} ${t("hil_interventions")}"></span>`:(ep.good?`<span class="flag-dot good" title="${t("good_title")}"></span>`:"");
      return `<div class="episode-item hil ${sel?"selected":""}" onclick="hilSelectEpisode(\x27${esc(run.id)}\x27,\x27${esc(ep.id)}\x27)"><span class="status-cell">${chip}${mark}</span><span class="episode-id">${esc(ep.id)}</span><span class="meta">${ep.action_count.toLocaleString()} ${t("frames_unit")} · ${ep.interventions.length} ${t("hil_interventions")}</span></div>`;
    }).join("")}`).join(""):`<div class="empty-row">${t("no_rollouts")}</div>`);
    hilRenderSession();
    hilRenderPane();
  }
  function renderHilAssetVisual(){
    const el=$("hil-asset-visual");
    if(!hilState.profile||hilState.asset==="all"){el.innerHTML=`<div class="preview-empty">${t("asset_filter")}</div>`;return}
    const base=`/api/tasks/${encodeURIComponent(hilState.profile)}/assets/${encodeURIComponent(hilState.asset)}/preview`;
    setHtml(el,`<img loading="lazy" decoding="async" src="${base}/front" alt="${t("front")}"><img loading="lazy" decoding="async" src="${base}/back" alt="${t("back")}">`);
  }
  function hilSelectEpisode(runId,episodeId){
    state.activePane="hil";
    if(hilState.run?.id!==runId)hilState.points=[];
    hilState.run=hilRuns().find(x=>x.id===runId)||null;
    hilState.episode=hilState.run?.episodes.find(x=>x.id===episodeId)||null;
    hilRender();
  }
  function hilRenderPane(){
    const body=$("hil-body"),ep=hilState.episode;
    if(!ep||!hilState.run){body.innerHTML=`<div class="pane-empty">${t("hil_pane_empty")}</div>`;$("hil-segments").innerHTML="";hilRenderActions();return}
    const src=hilVideoUrl("primary");
    const stage=body.querySelector(".video-stage");
    if(!stage||stage.dataset.src!==src){
      body.innerHTML=`<div class="video-stage" data-src="${src}"><video preload="auto" playsinline></video></div><div class="video-bar"><div class="bar-marks"></div><div class="bar-playhead"></div></div>`;
      const video=body.querySelector("video");
      video.src=src;
      video.addEventListener("loadedmetadata",()=>hilSyncTimeline());
      video.addEventListener("timeupdate",hilSyncTimeline);
      video.addEventListener("seeked",hilSyncTimeline);
      video.addEventListener("click",()=>{state.activePane="hil";hilTogglePlay()});
      body.querySelector(".video-bar").addEventListener("click",e=>{state.activePane="hil";hilSeekFromBar(e)});
    }
    hilRenderActions();
    hilSyncTimeline();
  }
  function hilVideo(){return $("hil-body").querySelector("video")}
  function hilFrameAt(){const v=hilVideo();if(!v||!hilState.episode)return 0;return Math.max(0,Math.min(hilState.episode.action_count-1,Math.round(v.currentTime*hilState.episode.fps)))}
  function hilTogglePlay(){const v=hilVideo();if(!v)return;if(v.paused)v.play();else v.pause()}
  function hilSeekFromBar(e){const v=hilVideo();if(!v||!v.duration)return;const r=e.currentTarget.getBoundingClientRect();v.currentTime=Math.max(0,Math.min(v.duration,(e.clientX-r.left)/r.width*v.duration))}
  function hilStep(delta){const v=hilVideo();if(!v||!hilState.episode)return;const fps=hilState.episode.fps||50;v.currentTime=Math.max(0,Math.min(v.duration||0,v.currentTime+delta/fps))}
  function hilSyncTimeline(){
    const v=hilVideo();if(!v||!hilState.episode)return;
    const body=$("hil-body"),marks=body.querySelector(".bar-marks"),playhead=body.querySelector(".bar-playhead");
    const total=hilState.episode.action_count;
    if(v.duration)playhead.style.left=`${(v.currentTime/v.duration)*100}%`;
    const mine=hilState.episode.interventions.map(x=>`<div class="bad" style="left:${(x.sample_index/total)*100}%;width:0.35%"></div>`).join("");
    const drafts=hilState.points.filter(x=>x.episode_id===hilState.episode.id).map(x=>`<div class="draft" style="left:${(x.sample_index/total)*100}%;width:0.7%"></div>`).join("");
    marks.innerHTML=mine+drafts;
  }
  function hilRenderActions(){
    const seg=$("hil-segments"),actions=$("hil-actions"),ep=hilState.episode;
    const drafts=ep?hilState.points.filter(x=>x.episode_id===ep.id):[];
    seg.innerHTML=ep?(drafts.length
      ?drafts.map((p,i)=>`<div class="seg-row"><span class="seg-idx">#${i+1}</span><span class="seg-range">${t("hil_frame")} ${p.sample_index}</span><button class="seg-del" onclick="hilRemovePoint(${i})">×</button></div>`).join("")
      :`<div class="seg-empty">${t("hil_empty_hint")}</div>`):"";
    const goodBtn=ep&&!ep.interventions.length&&!drafts.length?`<button class="${ep.good?"good-btn on":"good-btn"}" title="${t("good_title")}" onclick="hilToggleGood()">${ep.good?"✓ ":""}${t("good_btn")}</button>`:"";
    actions.innerHTML=`<button class="primary" ${ep?"":"disabled"} onclick="hilAdd()">${t("hil_add_point")}</button><button ${drafts.length?"":"disabled"} onclick="hilUndo()">${t("undo")}</button>${goodBtn}`;
  }
  async function hilAdd(){
    const ep=hilState.episode;if(!ep)return;
    state.activePane="hil";
    if(ep.good){
      try{await request(`/api/hil/profiles/${encodeURIComponent(hilState.profile)}/episodes/${encodeURIComponent(ep.id)}/good`,{method:"PUT",body:JSON.stringify({good:false})});ep.good=false}
      catch(e){toast(e.message,true);return}
    }
    const point={episode_id:ep.id,sample_index:hilFrameAt()};
    if(!hilState.points.some(x=>x.episode_id===point.episode_id&&x.sample_index===point.sample_index))hilState.points.push(point);
    hilRender();
  }
  function hilUndo(){hilState.points.pop();hilRender()}
  function hilRemovePoint(i){
    const ep=hilState.episode;if(!ep)return;
    const drafts=hilState.points.filter(x=>x.episode_id===ep.id);
    const target=drafts[i];if(!target)return;
    hilState.points.splice(hilState.points.indexOf(target),1);
    hilRender();
  }
  async function hilToggleGood(){
    const ep=hilState.episode;if(!ep||ep.interventions.length)return;
    const next=!ep.good;
    try{
      await request(`/api/hil/profiles/${encodeURIComponent(hilState.profile)}/episodes/${encodeURIComponent(ep.id)}/good`,{method:"PUT",body:JSON.stringify({good:next})});
      ep.good=next;
      toast(next?t("good_done"):t("good_cleared"));
      hilRender();
    }catch(e){toast(e.message,true)}
  }

  function hilShowQr(){if(!hilState.session)return;$("qr-title").textContent=`HIL · ${t("qr_title")}`;$("qr-endpoint").textContent="";$("qr-image").src=`/api/sessions/${encodeURIComponent(hilState.session.id)}/qr`;$("qr-modal").classList.add("open")}
  function hilRenderSession(){
    const active=hilInterventionSlots(),slot=active.find(x=>x.session?.id===hilState.session?.id);
    const start=$("hil-start");
    if(slot){
      hilState.session=slot.session;
      $("hil-status").textContent=(active.length>1?`${active.length} ${t("hil_active_sessions")} · `:"")+(hilState.session.phase||hilState.session.status);
      $("hil-vr").hidden=false;$("hil-vr").onclick=hilShowQr;
      $("hil-stop").hidden=false;$("hil-stop").onclick=async()=>{await request(`/api/sessions/${hilState.session.id}/stop`,{method:"POST"});await hilRefresh()};
      start.disabled=true;
    }else if(hilState.session&&!hilState.pending){
      $("hil-status").textContent=hilState.session.status==="failed"?(hilState.session.error||t("hil_failed")):t("hil_ended");
      $("hil-vr").hidden=true;$("hil-stop").hidden=true;
      hilState.session=null;hilState.points=[];
      hilRenderPane();
    }else if(active.length){
      $("hil-status").textContent=`${active.length} ${t("hil_active_sessions")}`;
      start.disabled=true;
    }else{
      $("hil-status").textContent="—";
      start.disabled=!hilState.points.length||hilState.pending;
    }
  }
  async function hilStart(){
    if(hilState.pending||hilState.session||hilInterventionSlots().length)return;
    hilState.pending=true;$("hil-start").disabled=true;$("hil-status").textContent=t("hil_starting");
    try{
      hilState.session=await request("/api/hil/interventions",{method:"POST",body:JSON.stringify({profile_id:hilState.profile,run_id:hilState.run.id,breakpoints:hilState.points})});
      hilRenderSession();await hilRefresh();
    }catch(e){$("hil-status").textContent=e.message}
    finally{hilState.pending=false;hilRenderSession();if(hilState.episode)hilRenderPane()}
  }
  async function hilRefresh(){
    const runId=hilState.run?.id,episodeId=hilState.episode?.id;
    hilState.data=await request("/api/hil/snapshot");
    const active=hilInterventionSlots().sort((a,b)=>(b.session.created_at||"").localeCompare(a.session.created_at||""));
    if(active.length&&!active.some(x=>x.session.id===hilState.session?.id))hilState.session=active[0].session;
    if(!hilState.profile)hilState.profile=hilProfiles()[0]?.id;
    if(runId){hilState.run=hilRuns().find(x=>x.id===runId)||null;hilState.episode=hilState.run?.episodes.find(x=>x.id===episodeId)||null}
    hilRender();
  }
  $("hil-profile").onchange=e=>{hilState.profile=e.target.value;hilState.asset="all";hilState.run=hilState.episode=null;hilState.points=[];hilRender()};
  $("hil-asset-pick").onclick=()=>openAssetPicker("hil");
  $("hil-start").onclick=hilStart;
  hilRefresh();setInterval(hilRefresh,2000);
  applyTheme();applyI18n();refresh();setInterval(refresh,2000);
