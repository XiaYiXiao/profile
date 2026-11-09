"""Parse Nsight SQLite and produce one self-contained Chinese HTML report."""
import collections
import html
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys

from experiment import group_names, load_bench

CATEGORIES = ['NCCL', '其他通信/融合', 'MoE', 'Attention', '其他计算']
COLORS = ['#dc6848', '#b693d8', '#367dcb', '#41a696', '#a7b4c4']


def category(name):
    n = name.lower()
    if 'nccl' in n:
        return 'NCCL'
    if any(s in n for s in ['allreduce', 'all_reduce', 'allgather', 'all_gather', 'reducescatter', 'reduce_scatter']):
        return '其他通信/融合'
    if any(s in n for s in ['moe', 'expert', 'routing', 'swiglu', 'mxfp4', 'bitmatrix', '_reduce_grouped', 'humming', 'marlin']):
        return 'MoE'
    if any(s in n for s in ['attention', 'flash_fwd', 'flash::', 'attn']):
        return 'Attention'
    return '其他计算'


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def seconds(intervals):
    return sum(b-a for a, b in intervals) / 1e9


def rich_export(folder):
    db = folder / 'trace.sqlite'
    with sqlite3.connect(db) as con:
        has_json = con.execute('SELECT COUNT(*) FROM NVTX_EVENTS WHERE jsonText IS NOT NULL').fetchone()[0]
    if not has_json:
        with (folder / 'export.log').open('w') as out:
            subprocess.run([os.environ.get('NSYS_BIN','nsys'), 'export', '--type=sqlite',
                            '--include-json=true', '--dynamic-tables=all', '--force-overwrite=true',
                            '--output='+str(db), str(folder/'trace.nsys-rep')],
                           stdout=out, stderr=subprocess.STDOUT, check=True, timeout=600)
    return db


def extract(folder):
    db = rich_export(folder)
    budget = int(json.loads((folder.parent/'config.json').read_text())['MAX_NUM_BATCHED_TOKENS'])
    con = sqlite3.connect(db)
    if con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end < start').fetchone()[0]:
        raise ValueError(f'Invalid CUPTI kernel intervals in {db}')
    strings = dict(con.execute('SELECT id,value FROM StringIds'))
    kernels = collections.defaultdict(list)
    sums = collections.defaultdict(lambda: [0, 0])
    cat_sums = collections.Counter()
    rank_cats = collections.defaultdict(collections.Counter)
    for a,b,device,pid,name_id in con.execute('SELECT start,end,deviceId,globalPid,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        name = strings[name_id]
        cat = category(name)
        kernels[device].append((a,b,cat,pid))
        sums[name][0] += 1
        sums[name][1] += (b-a)/1e9
        cat_sums[cat] += (b-a)/1e9
        rank_cats[device][cat] += (b-a)/1e9
    lo = min(v[0][0] for v in kernels.values())
    hi = max(max(x[1] for x in v) for v in kernels.values())
    span = (hi-lo)/1e9
    rank_rows, timelines = [], []
    for device, events in sorted(kernels.items()):
        comm = merge([(a,b) for a,b,cat,_ in events if cat in CATEGORIES[:2]])
        compute = merge([(a,b) for a,b,cat,_ in events if cat not in CATEGORIES[:2]])
        all_busy = merge(comm+compute)
        overlap = max(0, seconds(comm)+seconds(compute)-seconds(all_busy))
        rank_rows.append({'device':device,'pid':events[0][3]//16777216 % 16777216,
                          'count':len(events),'busy_s':seconds(all_busy),
                          'comm_s':seconds(comm),'compute_s':seconds(compute),
                          'overlap_s':overlap,'gap_s':span-seconds(all_busy),
                          'category_s':dict(rank_cats[device])})
        # 200 ms bins, interval union per category. These are timeline occupancy,
        # not SM utilisation; overlaps between categories are intentionally retained.
        bin_ns = 200_000_000
        bins = [[0.0]*len(CATEGORIES) for _ in range(math.ceil((hi-lo)/bin_ns))]
        for ci,cat in enumerate(CATEGORIES):
            for a,b in merge([(a,b) for a,b,c,_ in events if c == cat]):
                left, right = int((a-lo)//bin_ns), int((b-lo-1)//bin_ns)
                for j in range(left,right+1):
                    bins[j][ci] += max(0,min(b,lo+(j+1)*bin_ns)-max(a,lo+j*bin_ns))/bin_ns
        timelines.append({'device':device,'bins':[[round(x,4) for x in b] for b in bins]})
    api_sums = collections.defaultdict(lambda: {'calls':0,'bytes':0,'host_s':0.0,'sizes':collections.Counter()})
    runtime_sums = collections.Counter()
    api_seen = set()
    memberships = collections.defaultdict(set)
    pid_devices = {r['pid']:r['device'] for r in rank_rows}
    annotations = collections.defaultdict(list)
    plugin_pids = con.execute("SELECT DISTINCT globalPid FROM DIAGNOSTIC_EVENT WHERE text LIKE 'NCCL plugin init%'").fetchall()
    warnings = con.execute("SELECT DISTINCT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2").fetchall()
    q = '''SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value,''),COALESCE(n.jsonText,js.value)
           FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
           LEFT JOIN StringIds js ON js.id=n.jsonTextId WHERE n.end IS NOT NULL'''
    for a,b,tid,label,payload in con.execute(q):
        pid = (tid or 0) // 16777216 % 16777216
        if not payload:
            if 'token' in label.lower() or 'execute' in label.lower() or 'forward' in label.lower():
                annotations[pid].append({'start_s':(a-lo)/1e9,'duration_s':(b-a)/1e9,'label':label})
            continue
        parts = json.loads(payload)
        if not isinstance(parts,list):
            parts = [parts]
        data = {}
        for p in parts:
            if isinstance(p,dict):
                data.update(p)
        if 'ApiId' not in data or 'Count' not in data:
            continue
        group = str(data.get('Hash', 'unknown'))
        if 'Rank' in data and pid in pid_devices:
            memberships[group].add(pid_devices[pid])
        dtype = strings.get(data.get('Datatype'),'unknown')
        key = (group,label,data.get('NumRanks'),dtype)
        if 'ApiGroupId' in data:
            unique = (pid, data.get('Hash'), data['ApiId'])
            if unique in api_seen:
                continue
            api_seen.add(unique)
            size = {'ncclInt8':1,'ncclUint8':1,'ncclFloat16':2,'ncclBfloat16':2,
                    'ncclFloat32':4,'ncclInt32':4,'ncclUint32':4,'ncclFloat64':8,
                    'ncclInt64':8,'ncclUint64':8}.get(dtype)
            r = api_sums[key]
            r['calls'] += 1
            r['host_s'] += (b-a)/1e9
            if size:
                nbytes = data['Count'] * size
                r['bytes'] += nbytes
                r['sizes'][nbytes] += 1
        elif 'Algorithm' in data and 'NumRanks' in data:
            runtime_sums[(group,label,strings.get(data['Algorithm'],'?'),strings.get(data.get('Protocol'),'?'))] += 1
    nccl = []
    for (group,op,nranks,dtype), r in api_sums.items():
        nccl.append({'group':'GPU '+','.join(map(str,sorted(memberships[group]))),
                     'comm_hash':group,'op':op,'ranks':nranks,'dtype':dtype,
                     'calls':r['calls'],'payload_gib':r['bytes']/2**30,'host_s':r['host_s'],
                     'size_range_mib':[min(r['sizes'],default=0)/2**20,max(r['sizes'],default=0)/2**20]})
    top = [{'name':name,'category':category(name),'count':n,'gpu_s':s,'mean_us':s/n*1e6}
           for name,(n,s) in sorted(sums.items(), key=lambda x:-x[1][1])]
    bench = load_bench(folder)
    step_summary = []
    for pid, ranges in annotations.items():
        steps = []
        for r in ranges:
            m = re.fullmatch(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)',r['label'])
            if m:
                steps.append((int(m[1]),int(m[2]),int(m[3]),int(m[4]),r['duration_s']))
        full = [s[4] for s in steps if s[1] == budget and s[0] == 2]
        step_summary.append({'pid':pid,'device':pid_devices.get(pid),'steps':len(steps),
                             'scheduled_tokens':sum(s[1]+s[3] for s in steps),
                             'full_2request_steps':len(full),
                             'full_2request_median_s':statistics.median(full) if full else None})
    server_log = (folder/'server.log').read_text(errors='replace')
    jit_warnings = [l for l in server_log.splitlines() if 'Triton kernel JIT compilation during inference:' in l]
    result = {'ep':folder.name,'span_s':span,'category_s':dict(cat_sums),'ranks':rank_rows,
              'kernels':top,'nccl':sorted(nccl,key=lambda x:-x['payload_gib']),
              'nccl_algorithms':[{'group':k[0],'op':k[1],'algorithm':k[2],'protocol':k[3],'calls':v}
                                 for k,v in runtime_sums.items()],
              'nccl_plugin_workers':len(plugin_pids),'warnings':[r[0] for r in warnings],
              'annotations':dict(annotations),'steps':step_summary,'jit_warnings':jit_warnings,'timeline':timelines,
              'bench':{k:bench[k] for k in ['num_prompts','completed','failed','duration','total_input_tokens',
                                           'total_output_tokens','mean_ttft_ms','p99_ttft_ms','max_concurrency', 'evaluation_rounds',
                                           'per_round_input_tokens_per_s', 'mean_input_tokens_per_s',
                                           'stdev_input_tokens_per_s']}}
    result['bench']['input_tokens_per_s'] = bench['total_input_tokens']/bench['duration']
    con.close()
    return result


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def bars(data):
    scale = max(sum(d['category_s'].values()) for d in data)
    out = ['<svg viewBox="0 0 1000 180" role="img" aria-label="累计 GPU kernel 时间">']
    for i,d in enumerate(data):
        x,y = 90,20+60*i
        out.append(f'<text x="10" y="{y+26}">{d["ep"].upper()}</text>')
        for cat,col in zip(CATEGORIES,COLORS):
            s = d['category_s'].get(cat,0)
            w = s/scale*780
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="36" fill="{col}"><title>{cat}: {s:.3f} GPU·s</title></rect>')
            x += w
        out.append(f'<text x="{x+8}" y="{y+25}">{sum(d["category_s"].values()):.1f}</text>')
    for i,(cat,col) in enumerate(zip(CATEGORIES,COLORS)):
        out.append(f'<rect x="{i*195+10}" y="150" width="12" height="12" fill="{col}"/><text x="{i*195+28}" y="162" font-size="14">{cat}</text>')
    return ''.join(out)+'</svg>'


def render(root, config, data):
    solo = len(data) == 1
    if not solo:
        a, b = data
        ratio = b['bench']['input_tokens_per_s']/a['bench']['input_tokens_per_s']-1
    model = html.escape(config.get('SERVED_MODEL_NAME', 'MiMo'))
    labels = [d['ep'].upper() for d in data]
    comparison = 'EPLB=0 / EPLB=1（两组均 EP=1）' if config.get('COMPARISON') == 'eplb' else 'EP=0 / EP=1'
    if solo:
        comparison = 'EP=1 单组实验'
    mode = '单组采集' if solo else '并行采集'
    result_text = (f"{labels[0]} 输入吞吐：{data[0]['bench']['input_tokens_per_s']:,.0f} tokens/s。"
                   if solo else f'{labels[1]} 相对 {labels[0]} 的输入吞吐变化：{ratio:+.1%}。')
    experiment_text = '单组实验' if solo else '并行对照'
    devices_text = '，'.join(f"{d['ep'].upper()} 使用 GPU {config['GPU_EP' + d['ep'][-1]]}" for d in data)
    resources_text = '' if solo else '两组共享主机 CPU、内存和存储资源。'
    if 'WARMUP_ROUNDS' in config:
        rounds_text = f"先完整预热 {config['WARMUP_ROUNDS']} 轮，再评估 {config['REPEATS']} 轮；每轮每侧 {config['NUM_PROMPTS']} 个请求，预热和评估负载、seed 一致。预热不计入结果，全程不重启服务。"
        profile_text = '完成两侧全部预热后开启 CUDA profiler，以单个连续窗口覆盖全部评估轮，然后停止。trace 包含客户端启动及轮间间隔；吞吐分母为各轮 benchmark duration 之和，不含这些间隔。启停端点并发调用，但不是精确同步屏障。'
        if solo:
            profile_text = '完成全部预热后开启 CUDA profiler，以单个连续窗口覆盖全部评估轮，然后停止。trace 包含客户端启动及轮间间隔；吞吐分母为各轮 benchmark duration 之和，不含这些间隔。'
    else:
        rounds_text = f"每侧热身 {config['NUM_WARMUPS']} 个、正式 {config['NUM_PROMPTS']} 个请求，各 1 轮。"
        profile_text = '每侧客户端在自身热身后开启 CUDA profiler，完成正式请求后停止；两侧没有精确同步屏障。'
    pieces = [f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{model} {comparison} {mode}分析</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#233247;background:#f4f7fa;margin:0}}main{{max-width:1180px;margin:auto;padding:32px 24px}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}h1{{font-size:30px}}.muted{{color:#63758b}}.lead{{font-size:20px}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e3eaf0;padding:9px 12px;text-align:left;vertical-align:top}}th{{background:#f3f6fa}}td{{max-width:540px;overflow-wrap:anywhere}}svg{{width:100%}}canvas{{width:100%;height:400px}}input{{width:240px}}code{{background:#eef2f7;padding:2px 5px}}summary{{cursor:pointer}}.note{{border-left:4px solid #d49b3d;padding-left:16px}}</style>
<main><h1>{model}：{comparison} {mode}</h1>
<p class="muted">{html.escape(str(root))}<br>{html.escape(config['nsys_version'])}</p>
<section><h2>本轮结果</h2><p class="lead">{result_text}</p>
<p>这是一次带 profiler 的小规模{experiment_text}，不能据此估计稳定 P99，也不能与其他串行运行或未启用 profiler 的实验直接混算。</p>''']
    pieces.append(table(['配置','成功 / 失败','耗时 s','输入 tokens/s','平均 TTFT s'],[
        [d['ep'].upper(),f"{d['bench']['completed']} / {d['bench']['failed']}",
         f"{d['bench']['duration']:.2f}",f"{d['bench']['input_tokens_per_s']:,.0f}",
         f"{d['bench']['mean_ttft_ms']/1000:.3f}"] for d in data]))
    pieces.append(table(['配置', '评估轮', '输入 tokens/s'], [
        [d['ep'].upper(), i, f"{tps:,.2f}"]
        for d in data for i, tps in enumerate(d['bench']['per_round_input_tokens_per_s'], 1)]))
    pieces.append('<p>累计吞吐 = 评估总 tokens / 各轮耗时之和。预热统计单独保存；评估轮共享服务状态，不是独立启动样本。</p>')
    pieces.append(f'''</section><section><h2>采集方式</h2><p>TP=2、DP=2；{devices_text}。{rounds_text} 并发 {config['CONCURRENCY']}，输入 / 输出长度 {config['INPUT_LEN']} / {config['OUTPUT_LEN']} tokens；batch token 预算 {config['MAX_NUM_BATCHED_TOKENS']}，序列数上限 {config['MAX_NUM_SEQS']}。</p>
<p>{profile_text} 全部评估完成后才关闭服务、导出 SQLite、生成分析。</p>
<p>CPU 采样与上下文切换采集关闭；保留 CUDA kernel、NVTX、OSRT 和 NCCL 扩展信息。{resources_text}trace device 是报告内 CUDA 设备编号，请结合各组可见设备及 GPU UUID 核对。</p></section>
<section><h2>GPU kernel 时间分解</h2><p>四卡累计 kernel duration，单位 GPU·s，重叠区间可能重复计入，不等于请求总耗时。分类依据名称；Humming/Marlin/MoE/量化相关名称暂列 MoE 候选，无法确认的计算归到“其他计算”。需要结合具体 kernel 和后端核对，不能直接沿用 GPT-OSS 的内核归因。</p>''')
    pieces.append(bars(data))
    pieces.append(table(['类别'] + [label+' GPU·s' for label in labels]
                        + ([] if solo else [labels[1]+' − '+labels[0]]), [
        [cat] + [f"{d['category_s'].get(cat,0):.3f}" for d in data]
        + ([] if solo else [f"{b['category_s'].get(cat,0)-a['category_s'].get(cat,0):+.3f}"])
        for cat in CATEGORIES]))
    pieces.append('</section><section><h2>每卡活动区间</h2><p>先合并每卡 kernel 区间，再计算通信、计算及其交集。无 kernel 的区间可能包含请求间隔、CPU 工作、数据拷贝等；这不是 SM 利用率，也不能单凭空隙或卡间耗时差证明专家负载不均。</p>')
    pieces.append(table(['配置 / trace device','kernel 数','忙碌 s','通信 s','计算 s','通信∩计算 s','无 kernel s'],[
        [f"{d['ep'].upper()} / {r['device']}",r['count'],
         *[f"{r[k]:.3f}" for k in ['busy_s','comm_s','compute_s','overlap_s','gap_s']]]
        for d in data for r in d['ranks']]))
    pieces.append('<h3>可交互时间线</h3><p>各组以自身首个 kernel 对齐，不能用于比较两侧的绝对启动时间。每格 200 ms，每卡两行；蓝色为所选类别，橙色为通信/融合。</p><label>类别 <select id="category"><option value="compute">计算（不含融合通信）</option><option value="NCCL">NCCL</option><option value="MoE">MoE 候选</option><option value="Attention">Attention</option></select></label> <label>窗口起点 <input id="offset" type="range" min="0" value="0" step="0.2"></label> <span id="timeLabel"></span><canvas id="timeline" width="1100" height="400"></canvas></section>')
    pieces.append('<section><h2>NCCL API 消息</h2><p>按 communicator hash 区分通信组，以参与的 trace device 标注；统计跨 rank 累加。Count × datatype 是 API 逻辑 payload，并非 NVLink 实际流量。AllGather 对应发送大小，ReduceScatter 对应接收大小。窗口内没有对应 API 事件的 CUDA Graph 重放未计入此表，因此不能当作完整通信量或用它计算带宽。</p>')
    pieces.append(table(['配置','组','操作','组大小','类型','rank 调用数','逻辑 payload GiB','每次 MiB 范围'],[
        [d['ep'].upper(),r['group'],r['op'],r['ranks'],r['dtype'],r['calls'],
         f"{r['payload_gib']:.3f}",f"{r['size_range_mib'][0]:.3f}–{r['size_range_mib'][1]:.3f}"]
        for d in data for r in d['nccl']]))
    if solo:
        conclusion = '本轮仅运行 EP1，报告展示单组吞吐、计算及通信统计；没有 EP0 基线，不计算相对吞吐变化。'
    else:
        differences = {c:b['category_s'].get(c,0)-a['category_s'].get(c,0) for c in CATEGORIES}
        largest = max(differences,key=differences.get)
        conclusion = f'本轮吞吐变化为 {ratio:+.1%}。按当前名称分类，kernel 累计时间差最大的是「{largest}」：{differences[largest]:+.3f} GPU·s。这是定位候选开销的入口，不能直接等同于端到端变慢原因。'
    pieces.append(f'</section><section><h2>基于本轮数据可得出的结论</h2><p>{conclusion}</p><p>若 NCCL kernel 变长，需要继续区分传输与等待；若 MoE 候选 kernel 在各卡上耗时不同，需要进一步检查每层、每专家 token 数和内核形状。这里未启用 GPT-OSS 专用的两次 GEMM / ReduceScatter 对齐算法，也不自动推断 MiMo 存在同样的专家失衡问题。</p>')
    if any('KernelCh range begin' in w for d in data for w in d['warnings']):
        pieces.append('<p class="note">NCCL 插件出现通道时间戳校正警告。本页耗时使用 CUPTI kernel 区间，未用异常 channel 事件计算通信时间或带宽；扩展字段只用于识别 API 消息和通信组。</p>')
    pieces.append('</section>')
    for d in data:
        pieces.append('<section><h2>'+d['ep'].upper()+' kernel 明细</h2>')
        pieces.append(table(['kernel（前 18 项）','分类','次数','GPU·s','平均 μs'],[
            [r['name'],r['category'],r['count'],f"{r['gpu_s']:.3f}",f"{r['mean_us']:.1f}"]
            for r in d['kernels'][:18]]))
        pieces.append('<details><summary>采集诊断与推理期 JIT 日志</summary><p>NCCL 插件 worker 数：'+str(d['nccl_plugin_workers'])+'</p><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in d['jit_warnings']+d['warnings'])+'</ul></details></section>')
    pieces.append('<section><h2>使用与复查</h2><p>启动采集：<code>bash profiling_mimo/ep_profile/capture.sh</code>。仅预览：<code>DRY_RUN=1 bash profiling_mimo/ep_profile/capture.sh</code>。重新分析已有数据：<code>python profiling_mimo/ep_profile/analysis/code/analyze.py CAPTURE_DIR</code>。</p><p>原始数据按运行时间戳分开存放，各侧包含 trace.nsys-rep、trace.sqlite、请求结果、metrics 与日志。HTML 可直接离线打开。首次创建这些脚本时没有执行 MiMo 实验；本报告仅在后续实际采集或显式分析已有数据时生成。</p></section></main>')
    payload = json.dumps([{'ep':d['ep'],'timeline':d['timeline'],'span_s':d['span_s']} for d in data]).replace('</','<\\/')


    pieces.append('''<script>const DATA='''+payload+''';const CATS='''+json.dumps(CATEGORIES,ensure_ascii=False)+''';
const cv=document.getElementById('timeline'),ctx=cv.getContext('2d'),off=document.getElementById('offset'),sel=document.getElementById('category');
const span=Math.max(...DATA.map(d=>d.span_s)),windowSec=Math.min(20,span);off.max=Math.max(0,span-windowSec);
function draw(){ctx.clearRect(0,0,1100,400);ctx.font='13px system-ui';const start=+off.value;document.getElementById('timeLabel').textContent=start.toFixed(1)+'–'+(start+windowSec).toFixed(1)+' s';let row=0;
for(const d of DATA)for(const t of d.timeline){let y=30+row*43;ctx.fillStyle='#233247';ctx.fillText(d.ep.toUpperCase()+' / GPU '+t.device,0,y+14);
for(let i=0;i<t.bins.length;i++){let sec=i*.2;if(sec<start||sec>=start+windowSec)continue;let b=t.bins[i],v=sel.value==='compute'?Math.min(1,b[2]+b[3]+b[4]):b[CATS.indexOf(sel.value)],comm=Math.min(1,b[0]+b[1]),x=130+(sec-start)/windowSec*950,w=.2/windowSec*950+.2;
ctx.fillStyle='rgba(54,125,203,'+(.07+.93*v)+')';ctx.fillRect(x,y,w,15);ctx.fillStyle='rgba(220,104,72,'+(.07+.93*comm)+')';ctx.fillRect(x,y+17,w,12);}row++;}
ctx.fillStyle='#63758b';ctx.fillText('蓝：所选计算类别；橙：全部通信/融合。悬停图表查看时间。',130,390);}
off.oninput=draw;sel.onchange=draw;cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1100;cv.title=(+off.value+(x-130)/950*windowSec).toFixed(2)+' s';};draw();</script></html>''')
    return ''.join(pieces)


def main():
    root = Path(sys.argv[1]).resolve()
    config = json.loads((root/'config.json').read_text())
    data = [extract(root/group) for group in group_names(config)]
    analysis = root
    (analysis/'summary.json').write_text(json.dumps({'capture_dir':str(root),'config':config,'data':data},ensure_ascii=False,indent=2))
    (analysis/'index.html').write_text(render(root,config,data))
    print('Report:',analysis/'index.html',flush=True)
    for d in data:
        print(d['ep'],d['bench'],d['category_s'],flush=True)


if __name__ == '__main__':
    main()
