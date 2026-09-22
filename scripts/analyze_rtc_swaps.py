"""Read existing episodes and videos only. No model or environment execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(row):
    actions=[t['action'] for t in row['ticks']]
    grip=[a[6] for a in actions]
    flips=[i for i in range(1,len(grip)) if (grip[i]>0)!=(grip[i-1]>0)]
    requests=[q for q in row['requests'] if q.get('accepted_tick') is not None and not q['initial']]
    return dict(success=row['success'],steps=row['steps'],late_fraction=row['late_fraction'],
        gripper_sign_flip_ticks=flips,positive_gripper_fraction=sum(x>0 for x in grip)/len(grip),
        motion_mean=statistics.mean(math.sqrt(sum(x*x for x in a[:3])) for a in actions),
        median_delay=statistics.median(q['actual_delay'] for q in requests) if requests else None,
        median_replan_interval=statistics.median(b['observed_tick']-a['observed_tick'] for a,b in zip(requests,requests[1:])) if len(requests)>1 else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old',type=Path,default=Path('artifacts/s1_rtc_split_formal_v1'))
    p.add_argument('--new',type=Path,default=Path('artifacts/s1_rtc_weight1_formal_v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/rtc_swap_audit_v1'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.new/'manifest.json').read_text())
    for rel,want in manifest['historical_files'].items():
        if digest(args.old/rel)!=want:raise ValueError('Historical hash mismatch: '+rel)
    original=json.loads((args.old/'manifest.json').read_text());out=[];totals={}
    for path in sorted((args.new/'episodes/formal').rglob('*.json')):
        rtc=json.loads(path.read_text());oldpath=args.old/'episodes'/(rtc['key'].replace('/rtc_','/async_')+'.json')
        baseline=json.loads(oldpath.read_text())
        if rtc['init_state_sha256']!=baseline['init_state_sha256']:raise ValueError('Initial-state mismatch')
        group=(rtc['suite'],rtc['task_id'],rtc['delay_ms'])
        count=totals.setdefault(group,[0,0,0]);count[0]+=baseline['success'];count[1]+=rtc['success'];count[2]+=1
        if rtc['success']==baseline['success']:continue
        task=next(t for t in original['tasks'] if (t['suite'],t['task_id'])==(rtc['suite'],rtc['task_id']))
        key=rtc['key'].replace('/','_');videos={}
        for name,row,root in [('rtc',rtc,args.new),('async',baseline,args.old)]:
            if not row.get('video'):continue
            video=root/row['video']
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)]))
            duration=float(info['format']['duration']);target=args.output/f'{key}_{name}.jpg'
            # Twelve evenly spaced frames, chronological row-major ordering.
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video),
                '-vf',f'fps={12/duration},scale=256:256,tile=4x3','-frames:v','1',str(target)],check=True)
            videos[name]=dict(source=str(video.resolve()),sha256=digest(video),contact_sheet=str(target.resolve()),
                              duration_s=duration,approx_interval_s=duration/12)
        overlap=min(rtc['steps'],baseline['steps'])
        diffs=[max(abs(a-b) for a,b in zip(rtc['ticks'][i]['action'],baseline['ticks'][i]['action'])) for i in range(overlap)]
        hashes_match=all(a['noise_sha256']==b['noise_sha256'] for a,b in zip(rtc['requests'],baseline['requests']))
        out.append(dict(key=rtc['key'],language=task['language'],rtc=metrics(rtc),baseline=metrics(baseline),
            rtc_record_sha256=digest(path),baseline_record_sha256=digest(oldpath),videos=videos,
            first_action_difference_tick=next((i for i,d in enumerate(diffs) if d>1e-5),None),
            first10_max_difference=max(diffs[:10]),request_order_noise_hashes_match=hashes_match))
    (args.output/'analysis.json').write_text(json.dumps(dict(swaps=out,task_counts=[dict(suite=s,task_id=t,delay_ms=d,
        async_success=a,rtc_success=r,total=n) for (s,t,d),(a,r,n) in totals.items()],
        limitation='Actions and sampled video frames only; no object poses, contact logs, or stored RTC correction tensors.'),indent=2))
    print('swap pairs',len(out),'paired videos',sum(len(x['videos'])==2 for x in out))
    for x in out:print(x['key'],x['baseline']['success'],x['rtc']['success'],x['language'],x['first_action_difference_tick'])


if __name__=='__main__':main()
