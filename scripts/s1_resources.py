"""In-process low-frequency resource sampling, independent of assistant polling."""
import json
import resource
import subprocess
import threading
import time
from pathlib import Path


class ResourceLog:
    def __init__(self, output):
        self.path=Path(output)/'resources.jsonl'
        self.stop=threading.Event()
        self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        try:
            with self.path.open('a') as stream:
                while not self.stop.is_set():
                    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
                    utilization,used,total=map(float,result.stdout.strip().splitlines()[0].split(','))
                    row=dict(time=time.time(),gpu_utilization=utilization,gpu_used_mib=used,gpu_total_mib=total,
                        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        container_memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()))
                    stream.write(json.dumps(row)+'\n');stream.flush()
                    if used/total>.85:
                        raise RuntimeError('GPU total used memory exceeds 85 percent')
                    self.stop.wait(5)
        except BaseException as exc:
            self.error=repr(exc)

    def check(self):
        if self.error:
            raise RuntimeError('Resource monitoring failed: '+self.error)

    def close(self):
        self.stop.set();self.thread.join(timeout=12)
        self.check()
