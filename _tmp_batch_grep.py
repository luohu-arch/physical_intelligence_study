import re, os

papers = {
    'green-vla': ['ablat', 'R1', 'R2', 'DataQA'],
    'view-invariant-policy': ['ablat', 'crop', 'Plucker', 'cam'],
    'mint': ['Ablation', 'ablat'],
    'fabrica': ['ablat', 'F3', 'feature'],
    'rove': ['ablat', 'visual', 'feature'],
    'octo': ['ablat', 'attention', 'token'],
    'openvla': ['ablat', 'tokenizer', 'pruning'],
    'pi0': ['ablat', 'flow', 'expert'],
    'pi05': ['ablat', 'FAST', 'pre-training'],
    'xr-1': ['ablat', 'UVMC', 'token'],
    'human-as-humanoid': ['ablat', 'humanoid', 'teleop'],
    'imr-llm': ['ablat', 'reward', 'reason'],
    'flashsac': ['ablat', 'SAC', 'critic'],
    'unifp': ['ablat', 'flow', 'unif'],
    'weaver': ['ablat', 'expert', 'world model'],
    'simdist': ['ablat', 'distill', 'teacher'],
    'gr-mg': ['ablat', 'generative', 'mamba'],
    'susie': ['ablat', 'conditioning', 'target'],
    'unipi': ['ablat', 'plan', 'action'],
    'td-mpc2': ['ablat', 'decoder', 'latent'],
    'vjepa': ['ablat', 'predictor', 'target'],
    'paiworld': ['ablat', 'physics', 'world model'],
    'dreamer-v3': ['ablat', 'reinforce', 'critic', 'world model'],
    'mobile-aloha-act': ['ablat', 'ACT', 'chunk', 'temporal'],
    'open-x-embodiment': ['ablat', 'co-training', 'data mix', 'RT-1'],
}

out = []
for name, kws in papers.items():
    path = f'_tmp_pdftext/{name}.txt'
    if not os.path.exists(path):
        out.append(f'\n{"="*70}\n### {name}: NO FILE ###')
        continue
    t = open(path).read()
    out.append(f'\n{"="*70}\n### {name} ###')
    hits = 0
    for kw in kws:
        idxs = [m.start() for m in re.finditer(kw, t, re.IGNORECASE)]
        if idxs:
            hits += 1
            i = idxs[0]
            out.append(f'\n--- [{kw}] x{len(idxs)} at {i} ---')
            out.append(t[max(0, i-200):i+1800])
    if hits == 0:
        out.append('(no keyword hits)')

with open('_tmp_ablation_dump.txt', 'w') as f:
    f.write('\n'.join(out))
print('written', len('\n'.join(out)), 'chars')
