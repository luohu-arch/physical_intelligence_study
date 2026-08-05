import fitz, os

mapping = {
    'green-vla': 'papers/vla-architecture/Green-VLA_2602.00919.pdf',
    'view-invariant-policy': 'papers/vla-architecture/ViewInvariant_2510.02268.pdf',
    'mint': 'papers/vla-architecture/MINT_2602.08602.pdf',
    'fabrica': 'papers/vla-architecture/Fabrica_2506.05168.pdf',
    'rove': 'papers/vla-architecture/ROVE_2606.17011.pdf',
    'octo': 'papers/vla-architecture/Octo_Open_Source_Generalist_Robot_Policy_2405.12213.pdf',
    'openvla': 'papers/vla-architecture/OpenVLA_Open_Source_Vision_Language_Action_Model_2406.09246.pdf',
    'pi0': 'papers/vla-architecture/pi0_2410.24164.pdf',
    'pi05': 'papers/vla-architecture/pi05_2504.16054.pdf',
    'xr-1': 'papers/vla-architecture/XR-1_2511.02776.pdf',
    'human-as-humanoid': 'papers/vla-architecture/Human-as-Humanoid_2606.32009.pdf',
    'imr-llm': 'papers/vla-architecture/IMR-LLM_2603.02669.pdf',
    'flashsac': 'papers/vla-architecture/FlashSAC_2604.04539.pdf',
    'unifp': 'papers/vla-architecture/UniFP_2505.20829.pdf',
    'weaver': 'papers/vla-architecture/WEAVER_2606.13672.pdf',
    'simdist': 'papers/vla-architecture/SimDist_2603.15759.pdf',
    'gr-mg': 'papers/world-model/GR-MG_2408.14368.pdf',
    'susie': 'papers/world-model/SuSIE_2310.10639.pdf',
    'unipi': 'papers/world-model/UniPi_2302.00111.pdf',
    'td-mpc2': 'papers/world-model/TD-MPC2_2310.16828.pdf',
    'vjepa': 'papers/world-model/V-JEPA_2404.08471.pdf',
    'paiworld': 'papers/vla-architecture/PAIWorld_2606.18375.pdf',
    'dreamer-v3': 'papers/world-model/Dreamer_v3_2301.04104.pdf',
    'mobile-aloha-act': 'papers/data-infra/Mobile_ALOHA_ACT_2401.02117.pdf',
    'open-x-embodiment': 'papers/data-infra/Open_X_Embodiment_RT-X_2310.08864.pdf',
}
for name, path in mapping.items():
    if not os.path.exists(path):
        print(f'MISSING: {name} -> {path}')
        continue
    doc = fitz.open(path)
    text = '\n'.join(page.get_text() for page in doc)
    out = f'_tmp_pdftext/{name}.txt'
    os.makedirs('_tmp_pdftext', exist_ok=True)
    with open(out, 'w') as f:
        f.write(text)
    print(f'{name}: {len(text)} chars, {len(doc)} pages')
