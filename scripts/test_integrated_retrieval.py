"""Inventory and test local integration without printing raw documents."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rag.integrated_retrieve import inventory, retrieve_integrated, public_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inventory','search'])
    p.add_argument('--query', default='시험 때문에 자신감이 떨어지고 스트레스를 느껴요')
    p.add_argument('--mode',choices=['accurate','fast'],default='accurate')
    p.add_argument('--case-source',choices=['none','legacy','private'],default='none')
    p.add_argument('--case-slots',type=int,default=2)
    p.add_argument('--portal-slots',type=int,default=0)
    p.add_argument('--permission-confirmed',action='store_true')
    a=p.parse_args()
    if a.action=='inventory':
        print(json.dumps(inventory(),ensure_ascii=False,indent=2));return
    start=perf_counter()
    rows=retrieve_integrated(a.query,mode=a.mode,case_source=a.case_source,
                             case_slots=a.case_slots,portal_slots=a.portal_slots,
                             permission_confirmed=a.permission_confirmed)
    print(json.dumps({'count':len(rows),'mode':a.mode,'elapsed_seconds':perf_counter()-start,
                      'results':public_metadata(rows)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
