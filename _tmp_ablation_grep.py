import re, sys, os

def show(name, kws, span=2500):
    path = f'_tmp_pdftext/{name}.txt'
    if not os.path.exists(path):
        print(f'--- {name}: NO FILE ---')
        return
    t = open(path).read()
    print(f'\n{"="*70}\n### {name} ###')
    found = False
    for kw in kws:
        idxs = [m.start() for m in re.finditer(kw, t, re.IGNORECASE)]
        if idxs:
            found = True
            i = idxs[0]
            print(f'\n--- [{kw}] at {i} ---')
            print(t[max(0, i-300):i+span])
    if not found:
        print('(no keyword hits)')

if __name__ == '__main__':
    name = sys.argv[1]
    kws = sys.argv[2:]
    show(name, kws)
