  const initialSessionId=new URLSearchParams(location.search).get("session");
  const VR_STREAM_BASE={codec:"h264",immersiveMode:"vr",deviceFrameRate:"72",maxStreamingBitrateMbps:"80",perEyeWidth:"1280",perEyeHeight:"1152",enablePoseSmoothing:"true",posePredictionFactor:"1",panelHiddenAtStart:"true",cloudxrAutoRefreshMode:"any"};
  const VR_DEVICES={
    pico4ultra:{label:"Pico 4 Ultra",params:{...VR_STREAM_BASE,deviceProfile:"pico4ultra",enableTexSubImage2D:"false",useQuestColorWorkaround:"false"}},
    quest3:{label:"Meta Quest 3",params:{...VR_STREAM_BASE,deviceProfile:"quest3",enableTexSubImage2D:"true",useQuestColorWorkaround:"true"}},
    quest3s:{label:"Meta Quest 3S",params:{...VR_STREAM_BASE,deviceProfile:"quest3s",enableTexSubImage2D:"true",useQuestColorWorkaround:"true"}},
    quest2:{label:"Meta Quest 2",params:{...VR_STREAM_BASE,deviceProfile:"quest2",enableTexSubImage2D:"true",useQuestColorWorkaround:"false"}},
  };
  const savedVrDevice=localStorage.getItem("clothdojo-vr-device");
  const state={activePane:"task",snapshot:null,snapshotEtag:null,hilEtag:null,taskId:null,sessionMode:"collection",assetId:null,dataAsset:"all",selectedEpisode:null,hilCollectionResult:null,logType:initialSessionId?"session":null,logId:initialSessionId,logCacheId:null,logCacheText:null,toastTimer:null,timeSort:"desc",assetPickerMode:"session",annotationIndex:null,annotationIndexTask:null,annotationSaveChain:Promise.resolve(),reviewPane:null,reviewCamera:"cam_high",reviewRate:1,markStart:null,annotation:null,vrDevice:VR_DEVICES[savedVrDevice]?savedVrDevice:"pico4ultra",lang:localStorage.getItem("clothdojo-lang")||"zh",theme:localStorage.getItem("clothdojo-theme")||"dark"};
  const I18N={
    zh:{
      title:"数据控制台",
      instances:"采集实例",vr_device:"VR 设备",task:"任务",pick_asset:"选择衣服",col_instance:"实例",col_status:"状态",col_task:"任务",col_actions:"操作",
      log:"日志",log_empty:"启动实例后显示日志",log_select:"选择工作实例后显示日志",
      task_data:"采集数据",hil_task_data:"HIL 采集数据",asset_filter:"资产筛选",sort_time:"采集时间",sort_flatten:"展平时间",sort_intervention:"介入时间",
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
      start_session:"启动",switch_asset:"切换到当前衣服",switching_asset:"正在切换衣服",asset_switching:"资产切换中",asset_switch_failed:"上次切换失败",vr_connect:"VR 连接",qr:"二维码",close:"关闭",stop_session:"关闭这个遥操作实例？",stopping_session:"正在关闭实例",
      review_export:"审核导出",review_export_all:"一键导出全部 Review",
      service_offline:"采集服务未启动",batch_done:"本批恢复点已完成",recording:"录制中",saving:"保存中",batch_remaining:"当前批次剩余",
      restore_waiting:"等待 trigger 匹配",restore_ready:"已就绪 · A 录制/保存 · X 下一场景",restore_retrying:"正在恢复当前点…",
      no_pending:"当前衣服没有待采展平终态",asset_prefix:"衣服",
      qr_title:"VR 连接",qr_alt:"VR 连接二维码",qr_help:"首次显示 Certificate Accepted 时返回并再次扫码，然后点击 CONNECT",
      delete:"删除",delete_confirm_perm:"永久删除",delete_confirm_fold:"删除",
      log_waiting:"进程已启动，等待日志…",session_starting:"正在启动",
      data_deleted:"数据已删除",fold_deleted:"折叠轨迹已删除，已恢复待采集",undo:"撤销",
      hil_title:"HIL 介入",assets_title:"资产筛选",hil_points:"介入点",hil_stop:"停止",hil_interventions:"介入",no_rollouts:"没有可用 rollout",
      hil_pane_empty:"选择左侧 rollout 轨迹<br>查看第一人称画面",hil_empty_hint:"点击“标记介入点”在当前帧打点",hil_add_point:"标记介入点",hil_frame:"帧",
      hil_ended:"已结束",hil_failed:"启动失败",hil_active_sessions:"个 HIL 会话运行中",hil_completion_title:"在当前帧标记任务成功/完成；再次点击取消",hil_completion_done:"已标记任务完成点",hil_completion_cleared:"已取消任务完成点",hil_worker_hint:"没有待采 HIL 介入点",hil_worker_ready:"个介入点已就绪，请在 GPU worker 启动",hil_migrated:"个介入点已固化到项目",
      hil_collected:"已采 HIL",hil_pending:"待采 HIL",hil_marked_points:"已标介入点",hil_result_frames:"HIL 总帧数",hil_no_collection:"还没有标记或采集 HIL 介入点",hil_source:"Rollout 来源",hil_unmarked:"标记已取消",hil_results:"HIL 结果",hil_delete_confirm:"删除 HIL 轨迹",hil_delete_restore:"来源介入点将恢复为待采集。",hil_deleted:"HIL 轨迹已删除，介入点已恢复待采集",
      hil_export_review:"一键导出 Review",hil_exporting_review:"正在导出 Review",hil_export_complete:"Review 导出完成",hil_show_intervention:"查看介入结果",hil_show_rollout:"查看原始 Rollout",hil_intervention_video:"介入结果",hil_review_missing:"介入结果尚未导出 Review",
    },
    en:{
      title:"Data Console",
      instances:"Teleop Instances",vr_device:"VR device",task:"Task",pick_asset:"Pick asset",col_instance:"Instance",col_status:"Status",col_task:"Task",col_actions:"Actions",
      log:"Log",log_empty:"Start an instance to see logs",log_select:"Select a worker to see logs",
      task_data:"Collected Data",hil_task_data:"HIL Collection",asset_filter:"Asset filter",sort_time:"Collect time",sort_flatten:"Flatten time",sort_intervention:"Intervention time",
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
      start_session:"Start",switch_asset:"Switch to selected",switching_asset:"Switching asset",asset_switching:"Switching asset",asset_switch_failed:"Last switch failed",vr_connect:"VR connect",qr:"QR",close:"Close",stop_session:"Stop this teleop instance?",stopping_session:"Stopping instance",
      review_export:"Review export",review_export_all:"Export all reviews",
      service_offline:"Collection service offline",batch_done:"Restore queue complete",recording:"Recording",saving:"Saving",batch_remaining:"Batch remaining",
      restore_waiting:"Waiting for trigger match",restore_ready:"Ready · A record/save · X next scene",restore_retrying:"Restoring current point…",
      no_pending:"No pending flatten states for this asset",asset_prefix:"Asset",
      qr_title:"VR connect",qr_alt:"VR connect QR",qr_help:"When Certificate Accepted shows, go back and scan again, then CONNECT",
      delete:"Delete",delete_confirm_perm:"Permanently delete",delete_confirm_fold:"Delete",
      log_waiting:"Process started, waiting for logs…",session_starting:"Starting",
      data_deleted:"Data deleted",fold_deleted:"Fold deleted, restored to pending",undo:"Undo",
      hil_title:"HIL Interventions",assets_title:"Asset Curation",hil_points:"Breakpoints",hil_stop:"Stop",hil_interventions:"interventions",no_rollouts:"No rollouts available",
      hil_pane_empty:"Select a rollout episode<br>to preview its primary view",hil_empty_hint:"Click “Mark intervention” to add a point at the current frame",hil_add_point:"Mark intervention",hil_frame:"frame",
      hil_ended:"Ended",hil_failed:"Failed to start",hil_active_sessions:"HIL session(s) running",hil_completion_title:"Mark task success/completion at the current frame; click again to clear",hil_completion_done:"Task completion marked",hil_completion_cleared:"Task completion cleared",hil_worker_hint:"No pending HIL intervention points",hil_worker_ready:"breakpoint(s) ready; start from a GPU worker",hil_migrated:"breakpoint(s) saved to the project",
      hil_collected:"HIL collected",hil_pending:"HIL pending",hil_marked_points:"marked points",hil_result_frames:"HIL frames",hil_no_collection:"No HIL points marked or collected",hil_source:"Rollout source",hil_unmarked:"mark removed",hil_results:"HIL results",hil_delete_confirm:"Delete HIL trajectory",hil_delete_restore:"Its source breakpoint will return to pending.",hil_deleted:"HIL trajectory deleted; breakpoint restored to pending",
      hil_export_review:"Export all reviews",hil_exporting_review:"Exporting reviews",hil_export_complete:"Review export complete",hil_show_intervention:"View intervention",hil_show_rollout:"View rollout",hil_intervention_video:"Intervention result",hil_review_missing:"Intervention review has not been exported",
    }
  };
  const t=key=>(I18N[state.lang]||I18N.zh)[key]??(I18N.zh[key]??key);
  function applyTheme(){document.documentElement.dataset.theme=state.theme;$("theme-btn").textContent=state.theme==="dark"?"☀":"🌙"}
  function toggleTheme(){state.theme=state.theme==="dark"?"light":"dark";localStorage.setItem("clothdojo-theme",state.theme);applyTheme()}
  function applyI18n(){document.querySelectorAll("[data-i18n]").forEach(el=>{el.textContent=t(el.getAttribute("data-i18n"))});$("lang-btn").textContent=state.lang==="zh"?"EN":"中";$("theme-btn").textContent=state.theme==="dark"?"☀":"🌙"}
  function toggleLang(){state.lang=state.lang==="zh"?"en":"zh";localStorage.setItem("clothdojo-lang",state.lang);applyI18n();if(state.snapshot)render(state.snapshot)}
  const $=id=>document.getElementById(id);
  const ESCAPES={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
  const esc=value=>String(value??"—").replace(/[&<>"']/g,char=>ESCAPES[char]);
  function errorMessage(detail,fallback){if(typeof detail==="string")return detail;if(Array.isArray(detail))return detail.map(item=>item.msg||JSON.stringify(item)).join("；");if(detail&&typeof detail==="object")return JSON.stringify(detail);return fallback}
  async function request(url,options={}){const response=await fetch(url,{...options,headers:{"Content-Type":"application/json",...(options.headers||{})}});if(!response.ok){let message=response.statusText;try{message=errorMessage((await response.json()).detail,message)}catch{}throw new Error(message)}return response.json()}
  async function pollRequest(url,etag){const response=await fetch(url,{headers:{Accept:"application/json",...(etag?{"If-None-Match":etag}:{})}});if(response.status===304)return{data:null,etag};if(!response.ok){let message=response.statusText;try{message=errorMessage((await response.json()).detail,message)}catch{}throw new Error(message)}return{data:await response.json(),etag:response.headers.get("ETag")}}
  function toast(message,error=false){clearTimeout(state.toastTimer);$("toast").textContent=message;$("toast").className="toast show"+(error?" error":"");state.toastTimer=setTimeout(()=>$("toast").className="toast",2600)}
  function duration(seconds){if(seconds==null)return "—";const value=Math.max(0,Math.round(seconds));if(value<60)return value+" "+t("sec_unit");const minutes=Math.floor(value/60),rest=value%60;return `${minutes} ${t("min_unit")} ${String(rest).padStart(2,"0")} ${t("sec_unit")}`}
  function setHtml(el,html){if(el.__html!==html){el.__html=html;el.innerHTML=html}}
  const episodeTimeFormatter=new Intl.DateTimeFormat("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false});
  function episodeTime(id){const match=String(id).match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);if(!match)return id;const date=new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}Z`);return episodeTimeFormatter.format(date)}
  function selectedTask(){return state.snapshot?.tasks.find(task=>task.id===state.taskId)||state.snapshot?.tasks[0]}
  function hilCollection(profileId){return hilProfiles().find(profile=>profile.id===profileId)?.collection||{items:[],marked_point_count:0,pending_count:0,completed_point_count:0,trajectory_count:0,frame_count:0}}
  function hilPointCompleted(point){return !!hilEpisodeRef(point)?.interventions?.some(result=>result.sample_index===point.sample_index)}
  function hilPendingPoints(profileId){return hilProfilePoints(profileId).filter(point=>!hilPointCompleted(point))}
  function hilPendingAssets(profileId){
    const counts=new Map();
    hilPendingPoints(profileId).forEach(point=>counts.set(point.asset_id,(counts.get(point.asset_id)||0)+1));
    return [...counts.entries()].map(([id,count])=>({id,episode_count:count,breakpoint_count:count,default:false})).sort((left,right)=>right.breakpoint_count-left.breakpoint_count||left.id.localeCompare(right.id));
  }
  function hilDataAssets(profileId){
    const counts=new Map();
    hilCollection(profileId).items.forEach(item=>{const value=counts.get(item.asset_id)||{collected:0,pending:0};value.collected+=item.result_count;value.pending+=item.marked&&item.status==="pending"?1:0;counts.set(item.asset_id,value)});
    return [...counts.entries()].map(([id,value])=>({id,episode_count:value.collected,pending_count:value.pending,default:false})).sort((left,right)=>right.pending_count-left.pending_count||right.episode_count-left.episode_count||left.id.localeCompare(right.id));
  }
  function hilSessionContext(){
    if(state.sessionMode!=="intervention"||!state.taskId)return null;
    const points=hilPendingPoints(state.taskId).sort((left,right)=>left.asset_id.localeCompare(right.asset_id)||String(left.created_at).localeCompare(String(right.created_at)));
    if(!points.length)return null;
    return{profile_id:state.taskId,asset_id:points[0].asset_id,asset_count:new Set(points.map(point=>point.asset_id)).size,breakpoints:points.map(({run_id,episode_id,sample_index})=>({run_id,episode_id,sample_index}))};
  }
  function pendingCount(task,assetId){return (task?.episodes||[]).filter(episode=>episode.asset_id===assetId&&episode.status==="pending").length}
  function sortedAssets(task){return [...(task?.assets||[])].sort((left,right)=>(right.default?1:0)-(left.default?1:0)||(task?.continuation?pendingCount(task,right.id)-pendingCount(task,left.id)||right.episode_count-left.episode_count||left.id.localeCompare(right.id):right.episode_count-left.episode_count||left.id.localeCompare(right.id)))}
  function render(snapshot){
    state.snapshot=snapshot;
    if(!state.taskId||!snapshot.tasks.some(task=>task.id===state.taskId))state.taskId=snapshot.tasks[0]?.id||null;
    if(hilState.data&&hilState.profile!==state.taskId&&hilProfiles().some(profile=>profile.id===state.taskId)){
      hilState.profile=state.taskId;hilState.asset="all";hilState.run=hilState.episode=null;hilState.reviewResult=null;
    }
    const task=selectedTask();
    const assets=sortedAssets(task);
    const dataAssets=state.sessionMode==="intervention"?hilDataAssets(task.id):assets;
    const sessionAssets=state.sessionMode==="intervention"?hilPendingAssets(task.id):assets;
    if(!state.assetId||!sessionAssets.some(asset=>asset.id===state.assetId))state.assetId=sessionAssets[0]?.id||null;
    const selectedMode=`${state.taskId}::${state.sessionMode}`;
    setHtml($("task-select"),snapshot.tasks.flatMap(task=>[
      `<option value="${esc(task.id)}::collection" ${selectedMode===`${task.id}::collection`?"selected":""}>${esc(task.display_name)}</option>`,
      `<option value="${esc(task.id)}::intervention" ${selectedMode===`${task.id}::intervention`?"selected":""}>${esc(task.display_name)} HIL</option>`,
    ]).join(""));
    if(state.dataAsset!=='all'&&!dataAssets.some(asset=>asset.id===state.dataAsset))state.dataAsset='all';
    $("data-asset-pick").textContent=state.dataAsset==='all'?`${t("all_assets")}（${dataAssets.length}）`:state.dataAsset;
    renderAssetVisual();
    renderAssetPick();
    const targets=snapshot.slots.flatMap(slot=>slot.session?[["session",slot.session.id]]:slot.export?[["export",slot.export.id]]:[]);
    if(state.logId&&!targets.some(([type,id])=>type===state.logType&&id===state.logId)){state.logType=null;state.logId=null}
    if(!state.logId&&targets.length)[state.logType,state.logId]=targets[0];
    setHtml($("slots"),snapshot.slots.map(slot=>renderSlot(slot)).join(""));
    if(state.sessionMode!=="intervention"&&state.annotationIndexTask!==state.taskId){state.annotationIndexTask=state.taskId;state.annotationIndex=null;loadAnnotationIndex(state.taskId)}
    renderData(selectedTask());
    updateReviewExportButton();
    updateLog();
  }
  function restorationDetail(session){
    if(session.asset_switch_phase==="building"||session.asset_switch_phase==="opening_stage")return `<span>${t("asset_switching")}…</span>`;
    if(session.asset_switch_phase==="failed")return `<span style="color:#ffb3ad" title="${esc(session.asset_switch_error)}">${t("asset_switch_failed")}</span>`;
    if(session.recording)return `<span>${t("recording")} · ${(session.recording_frames||0).toLocaleString()} ${t("frames_unit")}</span>`;
    if(session.recording_finalizing||session.restoration_phase==="finalizing")return `<span>${t("saving")}…</span>`;
    if(session.restoration_phase==="complete"||session.continuation_phase==="exhausted")return `<span style="color:#ffb3ad">${t("batch_done")}</span>`;
    if(session.restoration_phase==="ready")return `<span>${t("restore_ready")}</span>`;
    if(session.restoration_phase==="waiting")return `<span>${t("restore_waiting")}</span>`;
    if(session.restoration_phase==="aborting")return `<span>${t("restore_retrying")}</span>`;
    if(session.continuation_phase==="active")return `<span>${t("batch_remaining")} ${session.continuation_remaining??0} ${t("count_unit")}</span>`;
    return "";
  }
  function renderSlot(slot){
    const session=slot.session,job=slot.export,selected=(session&&state.logType==="session"&&session.id===state.logId)||(job&&state.logType==="export"&&job.id===state.logId);
    const task=selectedTask(),hilContext=hilSessionContext(),hilMode=state.sessionMode==="intervention";
    const hasPending=hilMode?!!hilContext:!task?.continuation||pendingCount(task,state.assetId)>0;
    const statusClass=slot.status==="available"?"available":slot.status==="failed"||slot.status==="offline"||slot.status==="port_conflict"?"failed":slot.status==="stopping"?"stopping":"active";
    const detail=session
      ? `<div class="slot-detail"><strong>${esc(taskName(session.profile_id))}${session.mode==="intervention"?" HIL":""} · ${esc(session.asset_id)}</strong>${restorationDetail(session)}</div>`
      : job
      ? `<div class="slot-detail"><strong>${t("review_export")} · ${esc(episodeTime(job.episode_id))}</strong></div>`
      : `<div class="slot-detail">${slot.status==="offline"?`<strong>${t("service_offline")}</strong>`:slot.error?`<strong>${esc(slot.error)}</strong>`:hilMode?(hilContext?`<strong>${esc(task.display_name)} HIL · ${hilContext.breakpoints.length} ${t("hil_points")} · ${hilContext.asset_count} ${t("asset_prefix")}</strong>`:`<strong>${t("hil_worker_hint")}</strong>`):""}</div>`;
    const switchBusy=session&&(session.asset_switch_phase==="building"||session.asset_switch_phase==="opening_stage");
    const canSwitch=session&&session.mode==="collection"&&session.standby_gpu_id!=null&&session.profile_id===state.taskId&&session.asset_id!==state.assetId&&!session.recording&&!session.recording_finalizing&&!switchBusy;
    const switchAction=canSwitch?`<button onclick="switchSessionAsset(event,'${esc(session.id)}')">${t("switch_asset")}</button>`:"";
    const actions=session
      ? `${switchAction}<button class="primary" onclick="openConnect(event,'${esc(slot.id)}')">${t("vr_connect")}</button><button onclick="showQr(event,'${esc(session.id)}','${esc(slot.id)}','${esc(session.endpoint)}')">${t("qr")}</button><button class="danger" onclick="stopSession(event,'${esc(session.id)}')">${t("close")}</button>`
      : job
      ? ``
      : `<button class="primary" ${slot.status!=="available"||!hasPending?"disabled":""} title="${!hasPending?(hilMode?t("hil_worker_hint"):t("no_pending")):""}" onclick="startSession(event,'${esc(slot.id)}')">${t("start_session")}</button>`;
    const target=session?`selectLog('session','${esc(session.id)}')`:job?`selectLog('export','${esc(job.id)}')`:"";
    const gpuLabel=`GPU ${slot.gpu_id}${slot.standby_gpu_id==null?"":` + ${slot.standby_gpu_id}`}`;
    return `<div class="slot ${selected?"selected":""}" onclick="${target}"><span class="gpu">${esc(slot.worker_id?`${slot.worker_id} · ${gpuLabel}`:gpuLabel)}</span><span class="chip ${statusClass}">${t(slot.status)||esc(slot.status)}</span>${detail}<div class="slot-actions">${actions}</div></div>`;
  }
  function episodeKey(task,episode){return task.continuation?episode.result_episode_id:episode.id}
  function filteredRows(task){return orderedEpisodeRows(task).filter(episode=>state.dataAsset==="all"||episode.asset_id===state.dataAsset)}
  function selectEpisode(key){state.selectedEpisode=key||null;state.activePane="task";renderData(selectedTask())}
  function renderData(task){
    if(!task){setHtml($("episodes"),`<div class="empty-row">${t("no_task")}</div>`);renderReviewPane(null,null);return}
    if(state.sessionMode==="intervention"){renderHilCollectionData(task);return}
    $("task-data-title").textContent=t("task_data");
    $("episode-count-label").textContent=t("stat_count");
    $("frame-count-label").textContent=t("stat_frames");
    $("duration-label").textContent=t("stat_duration");
    $("annotation-count-label").textContent=t("stat_annotations");
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
  function filteredHilCollectionRows(profileId){
    const direction=state.timeSort==="desc"?-1:1;
    return hilCollection(profileId).items.filter(item=>state.dataAsset==="all"||item.asset_id===state.dataAsset).sort((left,right)=>{if(!!left.collected_at!==!!right.collected_at)return left.collected_at?-1:1;return direction*String(left.collected_at||left.id).localeCompare(String(right.collected_at||right.id))});
  }
  function selectHilCollectionItem(id){state.selectedEpisode=id||null;state.hilCollectionResult=null;state.activePane="task";renderData(selectedTask())}
  function renderHilCollectionData(task){
    const rows=filteredHilCollectionRows(task.id);
    const collected=rows.reduce((total,item)=>total+item.result_count,0);
    const pending=rows.filter(item=>item.marked&&item.status==="pending").length;
    $("task-data-title").textContent=`${task.display_name} · ${t("hil_task_data")}`;
    $("episode-count").textContent=collected.toLocaleString();
    $("episode-count-label").textContent=t("hil_collected");
    $("frame-count").textContent=pending.toLocaleString();
    $("frame-count-label").textContent=t("hil_pending");
    $("duration").textContent=rows.filter(item=>item.marked).length.toLocaleString();
    $("duration-label").textContent=t("hil_marked_points");
    $("annotation-count").textContent=rows.reduce((total,item)=>total+item.result_action_count,0).toLocaleString();
    $("annotation-count-label").textContent=t("hil_result_frames");
    $("time-sort").textContent=`${t("sort_time")} ${state.timeSort==='desc'?'↓':'↑'}`;
    setHtml($("episodes"),rows.length?rows.map(item=>{
      const complete=item.status==="complete",label=complete?t("st_complete"):t("st_pending");
      const detail=complete?`${item.result_count} ${t("count_unit")} HIL · ${item.result_action_count.toLocaleString()} ${t("frames_unit")}`:`${t("hil_frame")} ${item.sample_index}`;
      const unmarked=!item.marked?`<span class="flag-dot" title="${t("hil_unmarked")}"></span>`:"";
      const deleteAction=item.results.length===1?`<button class="delete" title="${t("delete")}" onclick="event.stopPropagation();deleteTrajectory(\x27hil\x27,\x27${esc(task.id)}\x27,\x27${esc(item.results[0].id)}\x27)">×</button>`:"";
      return `<div class="episode-item hil-data ${item.id===state.selectedEpisode?'selected':''}" onclick="selectHilCollectionItem(\x27${esc(item.id)}\x27)"><span class="status-cell"><span class="chip ${complete?'available':''}">${label}</span>${unmarked}</span><span class="episode-id" title="${esc(item.run_id)}">${esc(item.episode_id)}</span><span class="asset">${esc(item.asset_id)}</span><span class="meta">${detail}</span>${deleteAction}</div>`;
    }).join(""):`<div class="empty-row">${t("hil_no_collection")}</div>`);
    if(state.selectedEpisode&&!rows.some(item=>item.id===state.selectedEpisode))state.selectedEpisode=null;
    renderHilCollectionPane(rows);
  }
  function renderHilCollectionPane(rows){
    const body=$("pane-body"),segments=$("pane-segments"),actions=$("pane-actions");
    const item=rows.find(row=>row.id===state.selectedEpisode);
    state.reviewPane=null;state.markStart=null;state.annotation=null;
    if(!item){body.innerHTML=`<div class="pane-empty">${t("hil_no_collection")}</div>`;segments.innerHTML="";actions.innerHTML="";return}
    const result=item.results.find(candidate=>candidate.id===state.hilCollectionResult)||null;
    const source=`${item.run_id} / ${item.episode_id} / ${t("hil_frame")} ${item.sample_index}`;
    const resultRows=item.results.map(result=>`<div class="seg-row good-row"><span class="seg-idx">HIL</span><span class="seg-range">${esc(result.id)} · ${result.action_count.toLocaleString()} ${t("frames_unit")}</span><button class="seg-del" title="${t("delete")}" onclick="deleteTrajectory(\x27hil\x27,\x27${esc(state.taskId)}\x27,\x27${esc(result.id)}\x27)">×</button></div>`).join("");
    segments.innerHTML=resultRows||`<div class="seg-empty">${t("st_pending")}</div>`;
    const cameras=result?(result.cameras||[]):item.source_cameras;
    const src=result?`/api/hil/profiles/${encodeURIComponent(state.taskId)}/interventions/${encodeURIComponent(result.id)}/videos/primary`:`/api/hil/profiles/${encodeURIComponent(state.taskId)}/runs/${encodeURIComponent(item.run_id)}/episodes/${encodeURIComponent(item.episode_id)}/videos/primary`;
    if(cameras.length){
      body.innerHTML=`<div class="video-stage" data-src="${src}"><video controls preload="metadata" playsinline src="${src}"></video></div>`;
      const video=body.querySelector("video");video.playbackRate=state.reviewRate;
      video.addEventListener("loadedmetadata",()=>{video.currentTime=result?0:Math.min(video.duration||0,item.sample_index/item.source_fps)});
    }else{
      body.innerHTML=`<div class="pane-empty">${result?t("hil_review_missing"):t("hil_source")}<br>${result?esc(result.id):esc(source)}<br>${t("pane_not_generated")}</div>`;
    }
    const resultSelect=item.results.length>1?`<select class="speed-select intervention-select" aria-label="${t("hil_intervention_video")}" onchange="selectHilCollectionResult(this.value)">${item.results.map(candidate=>`<option value="${esc(candidate.id)}" ${candidate.id===(result?.id||state.hilCollectionResult)?"selected":""}>${t("hil_frame")} ${candidate.sample_index} · ${esc(candidate.id)}</option>`).join("")}</select>`:"";
    const switchButton=item.results.length?`<button onclick="toggleHilCollectionReview()">${result?t("hil_show_rollout"):t("hil_show_intervention")}</button>`:"";
    const label=result?`${t("hil_intervention_video")} · ${result.id}`:`${t("hil_source")} · ${item.episode_id} · ${t("hil_frame")} ${item.sample_index}`;
    actions.innerHTML=`<span class="pane-status" title="${esc(result?.id||source)}">${esc(label)}</span>${switchButton}${resultSelect}<select class="speed-select rate-select" aria-label="播放倍速" onchange="setHilCollectionRate(this.value)">${playbackRateOptions(state.reviewRate)}</select>`;
  }
  function toggleHilCollectionReview(){const item=filteredHilCollectionRows(state.taskId).find(row=>row.id===state.selectedEpisode);if(!item?.results.length)return;state.hilCollectionResult=item.results.some(result=>result.id===state.hilCollectionResult)?null:item.results[0].id;renderHilCollectionPane(filteredHilCollectionRows(state.taskId))}
  function selectHilCollectionResult(resultId){const item=filteredHilCollectionRows(state.taskId).find(row=>row.id===state.selectedEpisode);if(!item?.results.some(result=>result.id===resultId))return;state.hilCollectionResult=resultId;renderHilCollectionPane(filteredHilCollectionRows(state.taskId))}
  function setHilCollectionRate(value){state.reviewRate=Number(value);const video=$("pane-body").querySelector("video");if(video)video.playbackRate=state.reviewRate}
  async function deleteTrajectory(kind,profileId,trajectoryId,sourceId=null){
    const hil=kind==="hil";
    const message=hil?`${t("hil_delete_confirm")} ${trajectoryId}？\n${t("hil_delete_restore")}`:sourceId?`删除 ${episodeTime(sourceId)} 对应的折叠轨迹？\n原展平轨迹会恢复为“待采集”。`:`永久删除 ${episodeTime(trajectoryId)} 的采集数据？\n此操作无法撤销。`;
    if(!confirm(message))return;
    const url=hil?`/api/hil/profiles/${encodeURIComponent(profileId)}/interventions/${encodeURIComponent(trajectoryId)}`:`/api/tasks/${encodeURIComponent(profileId)}/episodes/${encodeURIComponent(trajectoryId)}`;
    try{
      await request(url,{method:"DELETE"});
      if(hil){if(state.hilCollectionResult===trajectoryId)state.hilCollectionResult=null;if(hilState.reviewResult===trajectoryId)hilState.reviewResult=null;toast(t("hil_deleted"));await hilRefresh()}
      else{toast(sourceId?t("fold_deleted"):t("data_deleted"));await refresh()}
    }catch(error){toast(error.message,true)}
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
  function playbackRateOptions(currentRate){return reviewRates.map(rate=>`<option value="${rate}" ${rate===currentRate?"selected":""}>${rate}×</option>`).join("")}
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
    actions.innerHTML=`<select class="speed-select camera-select" aria-label="审核视角" title="审核视角" onchange="setReviewCamera(this.value)">${cameraOptions}</select><select class="speed-select rate-select" aria-label="播放倍速" title="播放倍速" onchange="setPaneRate(this.value)">${playbackRateOptions(state.reviewRate)}</select><button class="${marking?"danger":""}" ${ready?"":"disabled"} onclick="toggleMark()">${marking?t("end_mark"):t("mark_bad")}</button><button ${count===0?"disabled":""} onclick="undoSegment()">${t("undo")}</button><button class="${good?"good-btn on":"good-btn"}" ${count||!ready?"disabled":""} title="${t("good_title")}" onclick="toggleGood()">${good?"✓ ":""}${t("good_btn")}</button>`;
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
    const delArgs=`'task','${esc(task.id)}','${esc(key)}',${task.continuation?`'${esc(episode.id)}'`:'null'}`
    const deleteAction=key&&episode.deletable!==false?`<button class="delete" title="${t("delete")}" onclick="event.stopPropagation();deleteTrajectory(${delArgs})">×</button>`:'';
    return `<div class="episode-item ${key===state.selectedEpisode?'selected':''} " onclick="selectEpisode('${esc(key||'')}')"><span class="status-cell"><span class="chip ${statusClass}">${esc(label)}</span>${flagMark}</span><span class="episode-id">${esc(episodeTime(episode.id))}</span><span class="asset">${esc(episode.asset_id)}</span><span class="meta">${frames} ${t("frames_unit")} · ${duration(episode.duration_seconds)}</span>${deleteAction}</div>`;
  }
  function taskName(id){return state.snapshot?.tasks.find(task=>task.id===id)?.display_name||id}
  function renderAssetPick(){
    const button=$("asset-pick"),hilContext=hilSessionContext();
    button.disabled=false;
    button.textContent=state.sessionMode==="intervention"
      ?(hilContext?`${hilContext.breakpoints.length} ${t("hil_points")} · ${hilContext.asset_count} ${t("asset_prefix")}`:t("hil_worker_hint"))
      :(state.assetId?`${t("asset_prefix")} · ${state.assetId}`:t("pick_asset"));
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
      (profile.runs||[]).forEach(run=>(run.episodes||[]).forEach(episode=>{const c=counts.get(episode.asset_id)||{runs:new Set(),episodes:0};c.runs.add(run.id);c.episodes++;counts.set(episode.asset_id,c)}));
      assets=[...counts.entries()].map(([id,c])=>({id,episode_count:c.episodes,hilRuns:c.runs.size})).sort((l,r)=>r.hilRuns-l.hilRuns||l.id.localeCompare(r.id));
      current=hilState.asset;
      title=`${t("asset_filter")} · ${profile.display_name}`;
    }else{
      task=selectedTask();if(!task)return;
      assets=state.sessionMode==="intervention"
        ?(state.assetPickerMode==="filter"?hilDataAssets(task.id):hilPendingAssets(task.id))
        :sortedAssets(task);
      current=state.assetPickerMode==="filter"?state.dataAsset:state.assetId;
      title=`${state.assetPickerMode==="filter"?t("asset_filter"):t("pick_asset_title")} · ${task.display_name}`;
    }
    const previewTask=state.assetPickerMode==="hil"?hilState.profile:task.id;
    const allCard=state.assetPickerMode==="session"?"":`<button class="asset-card ${current==="all"?"selected":""}" onclick="pickAssetFromModal(\x27all\x27)"><div class="images"><div style="display:grid;place-items:center;color:var(--muted);grid-column:1/-1">${t("all_assets")}</div></div><span class="meta"><span class="id">${t("all_assets")}</span><span class="count">${assets.length}</span></span></button>`;
    $("asset-modal-title").textContent=title;
    setHtml($("asset-cards"),allCard+assets.map(asset=>{
      const base=`/api/tasks/${encodeURIComponent(previewTask)}/assets/${encodeURIComponent(asset.id)}/preview`;
      const images=asset.id==="default"?`<div class="images"><div style="display:grid;place-items:center;color:var(--muted);grid-column:1/-1">${t("default_no_preview")}</div></div>`:`<div class="images"><img loading="lazy" decoding="async" src="${base}/front" alt="${t("front")}"><img loading="lazy" decoding="async" src="${base}/back" alt="${t("back")}"></div>`;
      const count=state.assetPickerMode==="hil"
        ?`${asset.hilRuns} rollout · ${asset.episode_count} ${t("count_unit")}`
        :state.sessionMode==="intervention"
        ?(state.assetPickerMode==="session"?`${asset.breakpoint_count} ${t("pending_unit")}`:`${asset.pending_count} ${t("pending_unit")} · ${asset.episode_count} HIL`)
        :`${task.continuation?`${pendingCount(task,asset.id)} ${t("pending_unit")} · `:""}${asset.episode_count} ${t("count_unit")}`;
      return `<button class="asset-card ${asset.id===current?"selected":""}" onclick="pickAssetFromModal(\x27${esc(asset.id)}\x27)">${images}<span class="meta"><span class="id">${esc(asset.id)}</span><span class="count">${count}</span></span></button>`;
    }).join(""));
    $("asset-modal").classList.add("open");
  }
  function pickAssetFromModal(id){
    closeAssetPicker();
    if(state.assetPickerMode==="filter"){state.dataAsset=id;renderData(selectedTask())}
    else if(state.assetPickerMode==="hil"){hilState.asset=id;if(!hilRuns().some(r=>r.id===hilState.run?.id)){hilState.run=null;hilState.episode=null}hilRender()}
    else{state.assetId=id;render(state.snapshot)}
  }
  function closeAssetPicker(){$("asset-modal").classList.remove('open')}
  function selectLog(type,id){state.logType=type;state.logId=id;render(state.snapshot)}
  async function startSession(event,slotId){
    event.stopPropagation();
    try{
      const hilContext=hilSessionContext();
      const session=state.sessionMode==="intervention"
        ?await request("/api/hil/interventions",{method:"POST",body:JSON.stringify({profile_id:hilContext.profile_id,breakpoints:hilContext.breakpoints,slot_id:slotId})})
        :await request("/api/sessions",{method:"POST",body:JSON.stringify({slot_id:slotId,profile_id:state.taskId,asset_id:state.assetId})});
      state.logType="session";state.logId=session.id;
      if(state.sessionMode==="intervention")hilState.session=session;
      toast(`${session.worker_id||slotId} ${t("session_starting")}`);
      await Promise.all([refresh(),hilRefresh()]);
    }catch(error){toast(error.message,true)}
  }
  async function stopSession(event,id){event.stopPropagation();if(!confirm(t("stop_session")))return;try{await request(`/api/sessions/${encodeURIComponent(id)}/stop`,{method:"POST"});toast(t("stopping_session"));await refresh()}catch(error){toast(error.message,true)}}
  async function switchSessionAsset(event,id){event.stopPropagation();try{await request(`/api/sessions/${encodeURIComponent(id)}/asset`,{method:"POST",body:JSON.stringify({asset_id:state.assetId})});toast(t("switching_asset"));await refresh()}catch(error){toast(error.message,true)}}
  function connectionUrl(slotId){
    const slot=state.snapshot.slots.find(item=>item.id===slotId),target=new URL(`https://${slot.endpoint}`);
    const clientSlot=state.snapshot.slots[0]||slot;
    const clientOrigin=new URL(`https://${clientSlot.endpoint}`);
    return `${clientOrigin.origin}/client/?${new URLSearchParams({serverIP:target.hostname,port:String(slot.proxy_port),proxyUrl:`${clientOrigin.origin}/_clothdojo/${slot.signaling_port}`,...VR_DEVICES[state.vrDevice].params})}`;
  }
  function openConnect(event,slotId){event.stopPropagation();location.href=connectionUrl(slotId)}
  async function updateLog(){
    const {logType:type,logId:id}=state;
    if(!id){state.logCacheId=state.logCacheText=null;$("log-session").textContent="";$("log").innerHTML=`<div class="log-empty">${t("log_select")}</div>`;return}
    const slot=state.snapshot?.slots.find(item=>(type==="session"?item.session:item.export)?.id===id);
    $("log-session").textContent=slot?(slot.worker_id?`${slot.worker_id} · GPU ${slot.gpu_id}`:`GPU ${slot.gpu_id}`):"";
    const url=type==="export"?`/api/exports/${encodeURIComponent(id)}/log`:`/api/sessions/${encodeURIComponent(id)}/log`;
    try{const payload=await request(url),cacheId=`${type}:${id}`,text=payload.log||t("log_waiting");if(state.logCacheId===cacheId&&state.logCacheText===text)return;state.logCacheId=cacheId;state.logCacheText=text;const old=$("log-text"),atBottom=!old||old.scrollHeight-old.scrollTop-old.clientHeight<28;setHtml($("log"),`<pre class="log" id="log-text">${esc(text)}</pre>`);if(atBottom&&$("log-text"))$("log-text").scrollTop=$("log-text").scrollHeight}catch(error){setHtml($("log"),`<div class="log-empty">${esc(error.message)}</div>`)}
  }
  function showQr(event,id,workerId,endpoint){event.stopPropagation();$("qr-title").textContent=`${workerId} · ${VR_DEVICES[state.vrDevice].label} · ${t("qr_title")}`;$("qr-endpoint").textContent=endpoint;$("qr-image").src=`/api/sessions/${encodeURIComponent(id)}/qr?device=${encodeURIComponent(state.vrDevice)}`;$("qr-modal").classList.add("open")}
  function closeQr(){$("qr-modal").classList.remove("open");$("qr-image").removeAttribute("src")}
  let refreshPromise=null;
  function refresh(){
    if(refreshPromise)return refreshPromise;
    refreshPromise=(async()=>{try{const result=await pollRequest("/api/snapshot",state.snapshotEtag);state.snapshotEtag=result.etag;if(result.data)render(result.data);else updateLog()}catch(error){toast(error.message,true)}})().finally(()=>{refreshPromise=null});
    return refreshPromise;
  }
  $("task-select").onchange=event=>{const [taskId,mode]=event.target.value.split("::");state.taskId=taskId;state.sessionMode=mode;state.assetId=null;state.dataAsset="all";state.selectedEpisode=null;state.hilCollectionResult=null;state.activePane="task";if(hilState.data&&hilProfiles().some(profile=>profile.id===taskId)){hilState.profile=taskId;hilState.asset="all";hilState.run=hilState.episode=null;hilState.reviewResult=null}render(state.snapshot);if(hilState.data)hilRender()};
  $("vr-device-select").value=state.vrDevice;
  $("vr-device-select").onchange=event=>{if(VR_DEVICES[event.target.value]){state.vrDevice=event.target.value;localStorage.setItem("clothdojo-vr-device",state.vrDevice)}};
  $("asset-pick").onclick=()=>openAssetPicker("session");
  $("data-asset-pick").onclick=()=>openAssetPicker("filter");
  $("time-sort").onclick=()=>{state.timeSort=state.timeSort==="desc"?"asc":"desc";renderData(selectedTask())};
  const scrub={active:false,dir:0,held:[],gen:0,t0:0,guard:null};
  function scrubCtx(){const useHil=state.activePane==="hil"&&hilState.episode&&hilVideo();const v=useHil?hilVideo():paneVideo();return v?{v:useHil?hilVideo():paneVideo(),fps:useHil?(hilVideoContext()?.fps||50):(paneInfo().fps||50)}:null}
  function scrubStop(){scrub.active=false;scrub.dir=0;scrub.held=[];clearTimeout(scrub.guard);scrub.gen++}
  function scrubStart(dir){if(!scrub.held.includes(dir))scrub.held.push(dir);scrub.dir=dir;if(scrub.active)return;const ctx=scrubCtx();if(!ctx)return;if(!ctx.v.paused)ctx.v.pause();scrub.active=true;scrubStep()}
  function scrubStep(){if(!scrub.active||!scrub.dir)return scrubStop();const ctx=scrubCtx();if(!ctx)return scrubStop();const v=ctx.v,next=Math.max(0,Math.min(v.duration||0,v.currentTime+scrub.dir/ctx.fps));if(Math.abs(next-v.currentTime)<1e-6)return scrubStop();scrub.gen++;const gen=scrub.gen;scrub.t0=performance.now();v.addEventListener("seeked",()=>{if(gen===scrub.gen)scrubNext()},{once:true});clearTimeout(scrub.guard);scrub.guard=setTimeout(()=>{if(gen===scrub.gen)scrubNext()},250);v.currentTime=next}
  function scrubNext(){if(!scrub.active||!scrub.dir)return;const ctx=scrubCtx();if(!ctx)return scrubStop();const wait=Math.max(0,1000/ctx.fps-(performance.now()-scrub.t0));clearTimeout(scrub.guard);scrub.guard=setTimeout(scrubStep,wait)}
  function scrubRelease(dir){scrub.held=scrub.held.filter(d=>d!==dir);if(!scrub.held.length)scrubStop();else scrub.dir=scrub.held[scrub.held.length-1]}
  document.addEventListener("keydown",e=>{const tag=(e.target&&e.target.tagName)||"";if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT"||tag==="VIDEO")return;const useHil=state.activePane==="hil"&&hilState.episode&&hilVideo();const v=useHil?hilVideo():paneVideo();if(!v)return;if(e.code==="ArrowLeft"){e.preventDefault();if(!e.repeat)scrubStart(-1)}else if(e.code==="ArrowRight"){e.preventDefault();if(!e.repeat)scrubStart(1)}else if(e.code==="Space"){e.preventDefault();scrubStop();useHil?hilTogglePlay():togglePanePlay()}});
  document.addEventListener("keyup",e=>{if(e.code==="ArrowLeft")scrubRelease(-1);else if(e.code==="ArrowRight")scrubRelease(1)});
  window.addEventListener("blur",scrubStop);
  const asState={status:"pending",page:1,pageCount:1,searchTimer:null};
  function asStateLabel(asset){return asset.converted?"已转换":""}
  function asActions(asset){
    if(asState.status==="pending")return`<button class="action accept" data-action="accept">接受</button><button class="action reject" data-action="reject">拒绝</button>`;
    if(asState.status==="rejected")return`<button class="action accept" data-action="accept">接受</button>`;
    return`<button class="action reject" data-action="reject">拒绝</button>`;
  }
  function asCard(asset){
    const images=asset.preview?`<img loading="lazy" src="/api/assets/${encodeURIComponent(asset.id)}/preview/front" alt="正面"><img loading="lazy" src="/api/assets/${encodeURIComponent(asset.id)}/preview/back" alt="背面">`:`<div class="no-preview">暂无预览</div>`;
    const label=asStateLabel(asset),trajectory=asState.status==="accepted"?`<span class="state trajectory">${asset.trajectory_count} 条轨迹</span>`:"";
    return`<article class="as-card" data-id="${esc(asset.id)}"><div class="images">${images}</div><div class="meta"><div class="identity"><span class="id">${esc(asset.id)}</span>${trajectory}${label?`<span class="state">${label}</span>`:""}</div><div class="actions">${asActions(asset)}</div></div></article>`;
  }
  async function asLoad(){
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
    }catch(error){toast(error.message,true)}
  }
  $("as-tabs").onclick=event=>{const tab=event.target.closest(".as-tab");if(!tab)return;asState.status=tab.dataset.status;asState.page=1;document.querySelectorAll("#as-tabs .as-tab").forEach(item=>item.classList.toggle("active",item===tab));asLoad()};
  $("as-search").oninput=()=>{clearTimeout(asState.searchTimer);asState.page=1;asState.searchTimer=setTimeout(asLoad,220)};
  $("as-prev").onclick=()=>{if(asState.page>1){asState.page--;asLoad()}};
  $("as-next").onclick=()=>{if(asState.page<asState.pageCount){asState.page++;asLoad()}};
  $("as-jump-form").onsubmit=event=>{event.preventDefault();const requested=Number.parseInt($("as-jump-page").value,10);if(!Number.isFinite(requested))return;asState.page=Math.max(1,Math.min(asState.pageCount,requested));$("as-jump-page").value="";asLoad()};
  $("as-grid").onclick=async event=>{const button=event.target.closest("button[data-action]");if(!button||button.disabled)return;const id=button.closest(".as-card").dataset.id,action=button.dataset.action;try{button.disabled=true;await request(`/api/assets/${id}/decision`,{method:"PUT",body:JSON.stringify({decision:action==="accept"?"accepted":"rejected"})});toast("已保存");await asLoad()}catch(error){button.disabled=false;toast(error.message,true)}};
  asLoad();
  const HIL_POINTS_KEY="clothdojo-hil-points";
  function loadHilPoints(){
    for(const key of [HIL_POINTS_KEY,"clothdojo-hil-points-rescue"]){
      try{
        const value=JSON.parse(localStorage.getItem(key)||"null");
        if(Array.isArray(value)){
          localStorage.setItem(HIL_POINTS_KEY,JSON.stringify(value));
          return value;
        }
      }catch{}
    }
    return [];
  }
  function saveHilPoints(){
    if(hilState.points.length)localStorage.setItem(HIL_POINTS_KEY,JSON.stringify(hilState.points));
    else{localStorage.removeItem(HIL_POINTS_KEY);localStorage.removeItem("clothdojo-hil-points-rescue")}
  }
  const hilState={data:null,profile:null,asset:"all",run:null,episode:null,reviewResult:null,points:loadHilPoints(),session:null,rate:1};
  function hilProfiles(){return hilState.data?.profiles||[]}
  function hilProfile(){return hilProfiles().find(x=>x.id===hilState.profile)}
  function hilRuns(){return (hilProfile()?.runs||[]).map(run=>({...run,episodes:(run.episodes||[]).filter(episode=>hilState.asset==="all"||episode.asset_id===hilState.asset)})).filter(run=>run.episodes.length)}
  function hilInterventionSlots(){return (hilState.data?.slots||[]).filter(x=>x.session?.mode==="intervention")}
  function hilSelectedResult(){return hilState.episode?.interventions?.find(result=>result.id===hilState.reviewResult)||null}
  function hilVideoContext(){return hilSelectedResult()||hilState.episode}
  function hilVideoUrl(view){const result=hilSelectedResult();return result?`/api/hil/profiles/${encodeURIComponent(hilState.profile)}/interventions/${encodeURIComponent(result.id)}/videos/${view}`:`/api/hil/profiles/${encodeURIComponent(hilState.profile)}/runs/${encodeURIComponent(hilState.run.id)}/episodes/${encodeURIComponent(hilState.episode.id)}/videos/${view}`}
  function reviewExport(){return hilState.data?.review_export||null}
  function hilPointKey(point){return `${point.profile_id}\0${point.run_id}\0${point.episode_id}\0${point.sample_index}`}
  function hilServerPoints(profileId){
    const profile=hilProfiles().find(item=>item.id===profileId);if(!profile)return[];
    const points=[];
    (profile.runs||[]).forEach(run=>(run.episodes||[]).forEach(episode=>(episode.breakpoints||[]).forEach(mark=>points.push({profile_id:profile.id,run_id:run.id,episode_id:episode.id,asset_id:episode.asset_id,sample_index:mark.sample_index,created_at:mark.created_at}))));
    return points.sort((left,right)=>String(left.created_at).localeCompare(String(right.created_at)));
  }
  function hilProfilePoints(profileId=hilState.profile){
    const merged=new Map();
    [...hilServerPoints(profileId),...hilState.points.filter(point=>point.profile_id===profileId)].forEach(point=>merged.set(hilPointKey(point),point));
    return [...merged.values()];
  }
  function hilEpisodeRef(point){
    const profile=hilProfiles().find(item=>item.id===point.profile_id),run=profile?.runs?.find(item=>item.id===point.run_id);
    return run?.episodes?.find(item=>item.id===point.episode_id)||null;
  }
  function hilBreakpointUrl(point){return `/api/hil/profiles/${encodeURIComponent(point.profile_id)}/runs/${encodeURIComponent(point.run_id)}/episodes/${encodeURIComponent(point.episode_id)}/breakpoint`}
  async function persistHilPoint(point,marked){
    const result=await request(hilBreakpointUrl(point),{method:"PUT",body:JSON.stringify({sample_index:point.sample_index,marked})});
    const episode=hilEpisodeRef(point);if(episode)episode.breakpoints=result.breakpoints;
    hilState.points=hilState.points.filter(item=>hilPointKey(item)!==hilPointKey(point));
    saveHilPoints();
  }
  let hilPointMigrationAttempted=false;
  async function migrateHilPoints(){
    if(hilPointMigrationAttempted||!hilState.points.length)return;
    hilPointMigrationAttempted=true;
    const count=hilState.points.length;
    for(const point of [...hilState.points]){
      try{await persistHilPoint(point,true)}catch(error){toast(`介入点迁移失败：${error.message}`,true);break}
    }
    if(!hilState.points.length)toast(`${count} ${t("hil_migrated")}`);
  }
  function hilEpisodePoints(ep=hilState.episode){return ep?hilProfilePoints().filter(point=>point.run_id===hilState.run?.id&&point.episode_id===ep.id):[]}
  function hilRender(){
    const withRollouts=new Set((hilProfile()?.runs||[]).flatMap(run=>(run.episodes||[]).map(episode=>episode.asset_id)));
    $("hil-asset-pick").textContent=hilState.asset==="all"?`${t("all_assets")}（${withRollouts.size}）`:hilState.asset;
    const runs=hilRuns();
    $("hil-run-count").textContent=runs.length;
    $("hil-episode-count").textContent=runs.reduce((n,r)=>n+r.episodes.length,0);
    $("hil-points-count").textContent=hilProfilePoints().length;
    $("hil-annotation-count").textContent=runs.reduce((total,run)=>total+run.episodes.filter(episode=>episode.good_sample_index!=null||episode.interventions.length>0).length,0);
    renderHilAssetVisual();
    setHtml($("hil-list"),runs.length?runs.map(run=>`<div class="hil-run">${esc(run.id)}</div>${run.episodes.map(ep=>{
      const sel=hilState.run?.id===run.id&&hilState.episode?.id===ep.id;
      const chip=ep.interventions.length?`<span class="chip active">${ep.interventions.length} ${t("hil_interventions")}</span>`:"";
      const draftCount=hilProfilePoints().filter(point=>point.run_id===run.id&&point.episode_id===ep.id).length;
      const draftChip=draftCount?`<span class="chip draft-chip">${draftCount} ${t("hil_points")}</span>`:"";
      const interventionMark=ep.interventions.length?`<span class="flag-dot" title="${ep.interventions.length} ${t("hil_interventions")}"></span>`:"";
      const completionMark=ep.good_sample_index!=null?`<span class="flag-dot good" title="Good · ${t("hil_frame")} ${ep.good_sample_index}"></span>`:"";
      return `<div class="episode-item hil ${sel?"selected":""}" onclick="hilSelectEpisode(\x27${esc(run.id)}\x27,\x27${esc(ep.id)}\x27)"><span class="status-cell">${chip}${draftChip}${interventionMark}${completionMark}</span><span class="episode-id">${esc(ep.id)}</span><span class="meta">${ep.action_count.toLocaleString()} ${t("frames_unit")} · ${draftCount} ${t("hil_points")}</span></div>`;
    }).join("")}`).join(""):`<div class="empty-row">${t("no_rollouts")}</div>`);
    hilRenderSession();
    updateReviewExportButton();
    hilRenderPane();
    if(state.snapshot)render(state.snapshot);
  }
  function renderHilAssetVisual(){
    const el=$("hil-asset-visual");
    if(!hilState.profile||hilState.asset==="all"){el.innerHTML=`<div class="preview-empty">${t("asset_filter")}</div>`;return}
    const base=`/api/tasks/${encodeURIComponent(hilState.profile)}/assets/${encodeURIComponent(hilState.asset)}/preview`;
    setHtml(el,`<img loading="lazy" decoding="async" src="${base}/front" alt="${t("front")}"><img loading="lazy" decoding="async" src="${base}/back" alt="${t("back")}">`);
  }
  function hilSelectEpisode(runId,episodeId){
    state.activePane="hil";
    hilState.run=hilRuns().find(x=>x.id===runId)||null;
    hilState.episode=hilState.run?.episodes.find(x=>x.id===episodeId)||null;
    hilState.reviewResult=null;
    hilRender();
  }
  function hilRenderPane(){
    const body=$("hil-body"),ep=hilState.episode;
    if(!ep||!hilState.run){body.innerHTML=`<div class="pane-empty">${t("hil_pane_empty")}</div>`;$("hil-segments").innerHTML="";hilRenderActions();return}
    const context=hilVideoContext(),cameras=context?.cameras||[];
    if(!cameras.length){body.innerHTML=`<div class="pane-empty">${hilSelectedResult()?t("hil_review_missing"):t("pane_not_generated")}</div>`;hilRenderActions();return}
    const src=hilVideoUrl("primary");
    const stage=body.querySelector(".video-stage");
    if(!stage||stage.dataset.src!==src){
      body.innerHTML=`<div class="video-stage" data-src="${src}"><video preload="auto" playsinline></video></div><div class="video-bar"><div class="bar-marks"></div><div class="bar-playhead"></div></div>`;
      const video=body.querySelector("video");
      video.src=src;
      video.playbackRate=hilState.rate;
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
  function hilFrameAt(){const v=hilVideo(),context=hilVideoContext();if(!v||!context)return 0;return Math.max(0,Math.min(context.action_count-1,Math.round(v.currentTime*context.fps)))}
  function hilTogglePlay(){const v=hilVideo();if(!v)return;if(v.paused)v.play();else v.pause()}
  function setHilRate(value){hilState.rate=Number(value);const v=hilVideo();if(v)v.playbackRate=hilState.rate}
  function hilSeekFromBar(e){const v=hilVideo();if(!v||!v.duration)return;const r=e.currentTarget.getBoundingClientRect();v.currentTime=Math.max(0,Math.min(v.duration,(e.clientX-r.left)/r.width*v.duration))}
  function hilStep(delta){const v=hilVideo(),context=hilVideoContext();if(!v||!context)return;const fps=context.fps||50;v.currentTime=Math.max(0,Math.min(v.duration||0,v.currentTime+delta/fps))}
  function hilSyncTimeline(){
    const v=hilVideo();if(!v||!hilState.episode)return;
    const body=$("hil-body"),marks=body.querySelector(".bar-marks"),playhead=body.querySelector(".bar-playhead");
    const result=hilSelectedResult(),total=(result||hilState.episode).action_count;
    if(v.duration)playhead.style.left=`${(v.currentTime/v.duration)*100}%`;
    if(result){marks.innerHTML="";return}
    const mine=hilState.episode.interventions.map(x=>`<div class="bad" style="left:${(x.sample_index/total)*100}%;width:0.35%"></div>`).join("");
    const drafts=hilEpisodePoints().map(x=>`<div class="draft" style="left:${(x.sample_index/total)*100}%;width:0.7%"></div>`).join("");
    const completion=hilState.episode.good_sample_index==null?"":`<div class="good" style="left:${(hilState.episode.good_sample_index/total)*100}%;width:0.7%"></div>`;
    marks.innerHTML=mine+drafts+completion;
  }
  function hilRenderActions(){
    const seg=$("hil-segments"),actions=$("hil-actions"),ep=hilState.episode;
    const result=hilSelectedResult(),sourceMode=!result;
    const drafts=hilEpisodePoints(ep);
    const pointRows=drafts.map((p,i)=>`<div class="seg-row"><span class="seg-idx">#${i+1}</span><span class="seg-range">${t("hil_frame")} ${p.sample_index}</span><button class="seg-del" onclick="hilRemovePoint(${i})">×</button></div>`);
    const completionRow=ep?.good_sample_index==null?[]:[`<div class="seg-row good-row"><span class="seg-idx">Good</span><span class="seg-range">${t("hil_frame")} ${ep.good_sample_index}</span></div>`];
    seg.innerHTML=result?`<div class="seg-row good-row"><span class="seg-idx">HIL</span><span class="seg-range">${esc(result.id)} · ${result.action_count.toLocaleString()} ${t("frames_unit")}</span><button class="seg-del" title="${t("delete")}" onclick="deleteTrajectory(\x27hil\x27,\x27${esc(hilState.profile)}\x27,\x27${esc(result.id)}\x27)">×</button></div>`:ep?([...pointRows,...completionRow].join("")||`<div class="seg-empty">${t("hil_empty_hint")}</div>`):"";
    const goodBtn=ep?`<button class="${ep.good_sample_index!=null?"good-btn on":"good-btn"}" title="${t("hil_completion_title")}" onclick="hilToggleCompletion()">${ep.good_sample_index!=null?`✓ Good · ${t("hil_frame")} ${ep.good_sample_index}`:t("good_btn")}</button>`:"";
    const rateSelect=ep?`<select class="speed-select rate-select" aria-label="播放倍速" title="播放倍速" onchange="setHilRate(this.value)">${playbackRateOptions(hilState.rate)}</select>`:"";
    const results=ep?.interventions||[];
    const resultSelect=results.length>1?`<select class="speed-select intervention-select" aria-label="${t("hil_intervention_video")}" onchange="hilSelectResult(this.value)">${results.map(item=>`<option value="${esc(item.id)}" ${item.id===(result?.id||hilState.reviewResult)?"selected":""}>${t("hil_frame")} ${item.sample_index} · ${esc(item.id)}</option>`).join("")}</select>`:"";
    const switchButton=results.length?`<button onclick="hilToggleReview()">${sourceMode?t("hil_show_intervention"):t("hil_show_rollout")}</button>`:"";
    actions.innerHTML=`${switchButton}${resultSelect}${rateSelect}<button class="primary" ${ep&&sourceMode?"":"disabled"} onclick="hilAdd()">${t("hil_add_point")}</button><button ${drafts.length&&sourceMode?"":"disabled"} onclick="hilUndo()">${t("undo")}</button>${sourceMode?goodBtn:""}`;
  }
  function hilToggleReview(){const results=hilState.episode?.interventions||[];if(!results.length)return;hilState.reviewResult=hilSelectedResult()?null:(results.find(item=>item.id===hilState.reviewResult)||results[0]).id;hilRenderPane()}
  function hilSelectResult(resultId){if(!(hilState.episode?.interventions||[]).some(item=>item.id===resultId))return;hilState.reviewResult=resultId;hilRenderPane()}
  function updateReviewExportButton(){const button=$("review-export"),job=reviewExport(),active=job&&["starting","running"].includes(job.status);button.disabled=active;if(active)button.textContent=`${t("hil_exporting_review")} ${job.completed_trajectory_count}/${job.trajectory_count}`;else if(job?.status==="complete")button.textContent=`${t("hil_export_complete")} · ${job.trajectory_count}`;else button.textContent=t("review_export_all");button.title=job?.error||""}
  async function startReviewExport(){try{await request("/api/review-export",{method:"POST"});toast(t("hil_exporting_review"));await hilRefresh()}catch(error){toast(error.message,true)}}
  async function hilAdd(){
    const ep=hilState.episode;if(!ep||hilSelectedResult())return;
    state.activePane="hil";
    const point={profile_id:hilState.profile,run_id:hilState.run.id,episode_id:ep.id,asset_id:ep.asset_id,sample_index:hilFrameAt()};
    if(hilProfilePoints().some(item=>hilPointKey(item)===hilPointKey(point)))return;
    hilState.points.push(point);saveHilPoints();hilRender();
    try{await persistHilPoint(point,true);hilRender()}catch(error){toast(error.message,true)}
  }
  async function hilUndo(){const drafts=hilEpisodePoints();const target=drafts[drafts.length-1];if(target)await hilDeletePoint(target)}
  async function hilDeletePoint(target){
    try{await persistHilPoint(target,false);hilRender()}catch(error){toast(error.message,true)}
  }
  async function hilRemovePoint(i){
    const ep=hilState.episode;if(!ep)return;
    const drafts=hilEpisodePoints(ep);
    const target=drafts[i];if(!target)return;
    await hilDeletePoint(target);
  }
  async function hilToggleCompletion(){
    const ep=hilState.episode;if(!ep||!hilState.run||hilSelectedResult())return;
    const sampleIndex=ep.good_sample_index==null?hilFrameAt():null;
    try{
      const result=await request(`/api/hil/profiles/${encodeURIComponent(hilState.profile)}/runs/${encodeURIComponent(hilState.run.id)}/episodes/${encodeURIComponent(ep.id)}/completion`,{method:"PUT",body:JSON.stringify({sample_index:sampleIndex})});
      ep.good_sample_index=result.good_sample_index;
      toast(sampleIndex==null?t("hil_completion_cleared"):t("hil_completion_done"));
      hilRender();
    }catch(e){toast(e.message,true)}
  }

  function hilRenderSession(){
    const active=hilInterventionSlots().filter(slot=>slot.session?.profile_id===hilState.profile);
    if(active.length){
      const session=active[0].session;
      $("hil-status").textContent=(active.length>1?`${active.length} ${t("hil_active_sessions")} · `:"")+(session.phase||session.status);
    }else{
      $("hil-status").textContent=hilPendingPoints(hilState.profile).length?`${hilPendingPoints(hilState.profile).length} ${t("hil_worker_ready")}`:"—";
    }
  }
  let hilRefreshPromise=null;
  function hilRefresh(){
    if(hilRefreshPromise)return hilRefreshPromise;
    hilRefreshPromise=(async()=>{
      const result=await pollRequest("/api/hil/snapshot",state.hilEtag);state.hilEtag=result.etag;if(!result.data)return;const data=result.data;
      const runId=hilState.run?.id,episodeId=hilState.episode?.id;
      hilState.data=data;
      await migrateHilPoints();
      const active=hilInterventionSlots().sort((a,b)=>(b.session.created_at||"").localeCompare(a.session.created_at||""));
      if(active.length&&!active.some(x=>x.session.id===hilState.session?.id))hilState.session=active[0].session;
      if(!hilProfiles().some(profile=>profile.id===hilState.profile))hilState.profile=hilProfiles().some(profile=>profile.id===state.taskId)?state.taskId:hilProfiles()[0]?.id;
      if(runId){hilState.run=hilRuns().find(x=>x.id===runId)||null;hilState.episode=hilState.run?.episodes.find(x=>x.id===episodeId)||null}
      hilRender();
    })().finally(()=>{hilRefreshPromise=null});
    return hilRefreshPromise;
  }
  $("hil-asset-pick").onclick=()=>openAssetPicker("hil");
  $("review-export").onclick=startReviewExport;
  hilRefresh();setInterval(()=>{if(!document.hidden)hilRefresh()},2000);
  applyTheme();applyI18n();refresh();setInterval(()=>{if(!document.hidden&&state.activePane!=="hil")refresh()},5000);
