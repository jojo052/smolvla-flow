"""Run all split-process checks, then the independent formal run; stop on any error."""
import argparse
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ('output','checkpoint','student','assets-dir','baseline-manifest','protocol'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args()
    common=[]
    for name,value in vars(args).items():common+=['--'+name.replace('_','-'),value]
    for stage in ('check','run'):
        subprocess.run([sys.executable,'-u','-m','scripts.evaluate_s1_rtc','--split-process',
                        '--stage',stage,*common],check=True)


if __name__=='__main__':main()
