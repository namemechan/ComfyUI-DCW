# ComfyUI-DCW

**Differential Correction in Wavelet domain** + **CFG Wavelet Mixing** + **Sliding Mode Control CFG**

> **DCW 논문**: *Elucidating the SNR-t Bias of Diffusion Probabilistic Models*
> Yu et al., arXiv:2604.16044v1 (2026)
> 코드: https://github.com/AMAP-ML/DCW

> **SMC 논문**: *CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance*
> Wang et al., CVPR 2026 / arXiv:2603.03281
> 코드: https://github.com/THU-SI/CFG-Ctrl

---

## 기능 개요

세 가지 독립적인 기능을 하나의 노드에서 제공합니다.

| 기능 | 개입 위치 | 목적 |
|------|-----------|------|
| **DCW** | `sampler_post_cfg_function` (x0_pred 후처리) | SNR-t 편향 보정 |
| **CWM** | `sampler_cfg_function` (CFG 계산 대체) | 주파수 대역별 adaptive CFG |
| **SMC** | CWM 훅 내부 (CWM 이전 단계) | guidance error의 oscillation 억제 및 semantic alignment 개선 |

각각 독립적으로 켜고 끌 수 있으며, 조합해서 사용할 수도 있습니다.

---

## 설치

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-DCW/
        ├── __init__.py
        └── dcw_node.py
```

폴더를 위 경로에 복사하고 ComfyUI를 재시작하세요.
외부 의존성 없음. 순수 PyTorch로만 구현됩니다.

---

## 사용 방법

노드 탐색기에서 **`DCW + CWM + SMC Model Patch`** 를 검색하거나
`model_patches` 카테고리에서 찾으세요.

### 기본 연결

```
[Load Checkpoint]
      ↓
[DCW + CWM + SMC Model Patch]  ← 파라미터 설정
      ↓
[KSampler]
```

### 다른 패치와 함께 사용

```
[Load Checkpoint]
      ↓
[Apply LoRA / FreeU 등]
      ↓
[DCW + CWM + SMC Model Patch]  ← 가능하면 파이프라인 마지막에 연결 권장
      ↓
[KSampler]
```

> **주의**: CWM과 SMC는 `sampler_cfg_function`을 점유합니다.
> 이미 같은 훅을 사용하는 노드가 앞에 연결되어 있으면
> CWM과 SMC는 자동으로 skip되고 콘솔에 경고가 출력됩니다.
> DCW(`sampler_post_cfg_function`)는 항상 안전하게 체이닝됩니다.

> **노드 전체 비활성화**: ComfyUI의 기본 기능인 노드 bypass(우클릭 → Bypass)를 사용하세요.

---

## 파라미터 레퍼런스

### DCW 파라미터 — SNR-t 편향 보정

DCW는 x0_pred를 웨이블릿으로 분해한 뒤 각 주파수 대역을 보정합니다.

```
corrected_f = denoised_f + λ_f(t) · (x_t_f − denoised_f)

λ_l(t) = lambda_l · σ_norm          ← 초기 스텝에서 최대
λ_h(t) = lambda_h · (1 − σ_norm)    ← 후기 스텝에서 최대
```

#### `lambda_l` — 저주파 보정 강도

| 값 범위 | 효과 |
|---------|------|
| `0.0` | DCW 저주파 비활성 |
| `+0.03 ~ +0.05` | 약한 구조 보정, 안전한 시작점 |
| `+0.05 ~ +0.08` | 중간 보정, 논문 권장 범위 |
| `+0.10 이상` | 강한 보정, 색감·구도 변화 가능 |
| `-0.01 ~ -0.05` | 과포화·과채도 억제 |
| `-0.05 ~ -0.5` | 강한 억제, 색감 소실 위험 |

#### `lambda_h` — 고주파 보정 강도

| 값 범위 | 효과 |
|---------|------|
| `0.0` | DCW 고주파 비활성 |
| `+0.005 ~ +0.010` | 약한 디테일 보정 |
| `+0.010 ~ +0.020` | 중간 보정, 논문 권장 범위 |
| `+0.05 이상` | 과도한 샤프닝 위험 |
| `-0.005 ~ -0.01` | 과샤프닝 억제 |
| `-0.01 ~ -0.3` | 강한 억제, 디테일 소실 위험 |

#### `dcw_enabled`

DCW 기능만 독립적으로 on/off.

---

### CWM 파라미터 — CFG Wavelet Mixing

CWM은 CFG guidance error `e = cond − uncond`를 웨이블릿으로 분해한 뒤
주파수 대역별로 다른 CFG 스케일을 적용합니다.

```
w_LL(t) = w · (1 + alpha_l · σ_norm)           ← 초기 스텝에서 LL 부스트
w_HH(t) = w · (1 + alpha_h · (1 − σ_norm))     ← 후기 스텝에서 HH 부스트
w_mid   = √(w_LL × w_HH)                       ← LH, HL에 적용 (기하 평균)

alpha_l = alpha_h = 0 → 표준 CFG와 수학적으로 동일
```

#### `alpha_l` — 저주파 CFG 부스트

| 값 범위 | 효과 |
|---------|------|
| `0.0` | 표준 CFG (CWM 저주파 비활성) |
| `+0.1 ~ +0.2` | 초기 스텝 구조/구도 guidance 약하게 강화 |
| `+0.2 ~ +0.4` | 구도·레이아웃 프롬프트 정렬 강화 |
| `+0.5 이상` | 강한 구도 고정, 과도하면 색감 변이 가능 |
| `-0.1 ~ -0.3` | 구도 guidance 억제, 더 자유로운 레이아웃 |

#### `alpha_h` — 고주파 CFG 부스트

| 값 범위 | 효과 |
|---------|------|
| `0.0` | 표준 CFG (CWM 고주파 비활성) |
| `+0.1 ~ +0.15` | 후기 스텝 디테일/엣지 guidance 약하게 강화 |
| `+0.15 ~ +0.3` | 텍스트·텍스처 프롬프트 정렬 강화 |
| `+0.5 이상` | 과샤프닝·텍스처 노이즈 위험 |
| `-0.1 ~ -0.2` | 엣지 guidance 억제, 부드러운 결과 |

#### `cwm_enabled`

CWM 기능만 독립적으로 on/off. SMC는 CWM이 꺼져 있어도 `smc_preset ≠ Off`이면 독립 작동합니다.

---

### SMC 파라미터 — Sliding Mode Control CFG

SMC는 매 스텝의 guidance error를 보정하여 CFG trajectory의 oscillation을 억제합니다.
CWM이 활성화되어 있으면 SMC가 먼저 error를 보정한 뒤 CWM이 주파수 대역별로 분배합니다.

```
e(t)   = cond − uncond
s(t)   = (e − e_prev) + λ · e_prev        ← sliding surface
‖s‖₂  = L2 norm (배치 샘플별 독립)
Δe     = −k · s / ‖s‖₂                   ← unit_2 switching (논문 Table 4)
e*(t)  = e + Δe                           ← 보정 총 에너지 = 항상 k
e_prev ← e*(t)                            ← 다음 스텝으로 전달
```

#### `smc_preset` — 프리셋 선택

| 값 | 동작 |
|----|------|
| `Off` | SMC 완전 비활성 (기본값) |
| `Auto` | 모델 클래스명으로 자동 감지, 논문 기본값 적용 |
| `SD1.5 / SD2` | λ=5.0, k=0.10 |
| `SDXL` | λ=5.0, k=0.10 |
| `SD3 / SD3.5` | λ=6.0, k=0.10 (논문 grid-search) |
| `Flux` | λ=6.0, k=0.70 (논문 grid-search) |
| `Cosmos / Wan` | λ=6.0, k=0.20 |
| `Custom` | 아래 `smc_lambda` / `smc_k` 슬라이더 사용 |

#### `smc_lambda` — sliding surface 형상 파라미터 λ

`smc_preset = Custom`일 때만 적용됩니다.

| 값 범위 | 효과 |
|---------|------|
| 논문 권장: `2 ~ 8` | 수렴 속도 조절 |
| 극단값 | sliding manifold 왜곡 위험 |

#### `smc_k` — switching gain k

`smc_preset = Custom`일 때만 적용됩니다.

unit_2 방식에서 k는 매 스텝 보정 벡터의 L2 norm을 정확히 k로 고정합니다.
해상도나 채널 수와 무관하게 보정 에너지가 일정하게 유지됩니다.

| 값 범위 | 효과 |
|---------|------|
| 낮은 값 (예: 0.1) | 더 나은 FID / 사실감, 텍스트 정렬 약함 |
| 높은 값 (예: 0.7) | 강한 텍스트 정렬, chattering 위험 |

---

## 모델별 권장 시작값

### DCW

| 모델 | `lambda_l` | `lambda_h` |
|------|-----------|-----------|
| SDXL / SD1.5 / DiT | 0.05 | 0.010 |
| **Flux** | 0.08 – 0.12 | 0.015 – 0.025 |
| **Anima (Cosmos)** | 0.08 – 0.12 | 0.015 – 0.025 |
| EDM | 0.05 | 0.010 |

### CWM

| 모델 | `alpha_l` | `alpha_h` | 비고 |
|------|----------|----------|------|
| SDXL / SD1.5 | 0.10 – 0.20 | 0.10 – 0.15 | |
| **Flux** | 0.20 – 0.40 | 0.15 – 0.25 | Flow 모델, 약 2× |
| **Anima (Cosmos)** | 0.20 – 0.40 | 0.15 – 0.25 | Flow 모델, 약 2× |
| 구도 강화 목적 | 0.20 – 0.40 | 0.0 | alpha_h 비활성 |
| 디테일 강화 목적 | 0.0 | 0.15 – 0.30 | alpha_l 비활성 |

> **Flow 모델 (Flux, Anima/Cosmos) 공통 주의사항**
> σ 스케일이 `[0, 1]`로 제한되어 σ_norm 최대값이 ~0.5에 그칩니다.
> DCW와 CWM 모두 DDPM/EDM 대비 보정 강도가 절반 수준이므로
> lambda/alpha 값을 약 2배로 올려 시작하세요.

### SMC

`Auto` 프리셋이 대부분의 경우 최적값을 자동 선택합니다.
모델이 자동 감지되지 않거나 결과가 만족스럽지 않을 때만 `Custom`을 사용하세요.

---

## 기능 조합 가이드

### DCW만 사용
SNR-t 편향이 주된 문제일 때 (저스텝, 흐릿함, 채도 부족).
`cwm_enabled = False`, `smc_preset = Off`

### CWM만 사용
구도나 디테일의 프롬프트 정렬이 주된 목적일 때.
`dcw_enabled = False`, `smc_preset = Off`

### SMC만 사용 (CWM 없이)
guidance oscillation 억제만 원할 때.
`dcw_enabled = False`, `cwm_enabled = False`, `smc_preset = Auto`
CWM 없이도 SMC는 단독 cfg 훅으로 작동합니다.

### CWM + SMC (권장 조합)
```
SMC: 매 스텝에서 guidance error의 oscillation을 보정
  ↓
CWM: 보정된 error를 주파수 대역별로 분배
```
두 기능은 동일한 cfg 훅 안에서 순서대로 실행됩니다.
SMC가 error 품질을 먼저 개선하고, CWM이 그 위에 주파수 weighting을 적용하므로
단독 사용보다 더 안정적이고 세밀한 제어가 가능합니다.

### DCW + CWM + SMC 동시 사용
```
CWM+SMC: 매 스텝에서 cfg 훅 내부에서 guidance error 보정 및 주파수 weighting
  ↓
DCW: 그 결과로 나온 x0_pred의 SNR 편향을 post-cfg 훅으로 보정
```
세 기능의 개입 위치가 다르므로 충돌 없이 상호보완적으로 작동합니다.
파라미터를 각각 단독으로 먼저 조정한 뒤 합치는 것을 권장합니다.

---

## 튜닝 팁

**단계적 탐색을 권장합니다:**

1. `dcw_enabled = True`, `cwm_enabled = False`, `smc_preset = Off` → DCW 단독 조정
2. `dcw_enabled = False`, `cwm_enabled = True`, `smc_preset = Off` → CWM 단독 조정
3. `dcw_enabled = False`, `cwm_enabled = True`, `smc_preset = Auto` → SMC 추가 효과 확인
4. 모두 활성화 후 `dcw_enabled` / `cwm_enabled` / `smc_preset` 개별 토글로 각 기여분 확인
5. 노드 전체 A/B 비교는 ComfyUI 기본 기능인 노드 bypass를 사용

**스텝 수가 적을수록 효과가 더 큽니다.** 10–20 스텝에서 차이가 가장 명확합니다.

**과도한 보정 징후:**
- 색감 변이 / 채도 과다 → `lambda_l` 또는 `alpha_l` 감소
- 텍스처 노이즈처럼 보이는 디테일 → `lambda_h` 또는 `alpha_h` 감소
- 구도·구조가 원본과 달라짐 → `lambda_l` 또는 `alpha_l` 감소
- 이미지가 과도하게 선명하거나 엣지가 튀는 느낌 → `smc_k` 감소 또는 `smc_preset = Off`

---

## SNR-t 편향이란?

Diffusion 모델은 훈련 시 timestep t와 SNR(신호 대 잡음비)이 1:1로 대응됩니다.

$$\text{SNR}(t) = \bar{\alpha}_t \;/\; (1 - \bar{\alpha}_t)$$

추론 시에는 신경망 예측 오차와 수치 solver 이산화 오차가 누적되어 실제 SNR이 기대값보다 항상 낮아집니다. 모델은 이를 "SNR이 낮은 입력"으로 받아들여 노이즈를 과대추정하고 오차가 누적됩니다.

DCW는 이를 웨이블릿 도메인에서의 differential correction으로 보정합니다.

### 원본 논문 대비 개선: 밴드별 독립 타이밍 + 채널 에너지 가중치

**1. LH/HL 독립 타이밍**

논문 원본과 초기 구현은 LH(수평 엣지), HL(수직 엣지), HH(대각 텍스처)를 동일한 후기 스텝 스케줄로 처리했습니다. 그러나 방향성 엣지(LH/HL)는 HH보다 먼저 형성됩니다.

| 서브밴드 | 내용 | 형성 시기 | 가중치 |
|---|---|---|---|
| LL | 전체 구조·색감 | 초기 | `lambda_l × σ_norm` |
| LH/HL | 방향성 엣지 | 중간 | `(lam_l + lam_h) / 2` |
| HH | 미세 텍스처 | 후기 | `lambda_h × (1-σ_norm)` |

LH/HL의 가중치는 LL 타이밍과 HH 타이밍의 선형 보간으로, 중간 스텝에서 자연스럽게 활성화됩니다.

**2. 채널별 에너지 가중치**

단일 채널 VAE(SD 4ch)와 달리 Cosmos 16ch처럼 다채널 VAE는 각 채널이 서로 다른 의미 정보(포즈, 장신구, 조명 등)를 인코딩합니다. 동일한 lambda를 전 채널에 균일하게 적용하면 미세한 lambda 변화(0.04 → 0.05)가 특정 채널을 의미론적 결정 경계 너머로 밀어 포즈가 바뀌거나 장신구가 생기는 등 이산적 점프 현상이 발생합니다.

채널 에너지 가중치는 이를 완화합니다:

$$w_c = \text{clamp}\!\left(\frac{E[x_{t,c}^2]}{\overline{E[x_{t,c}^2]}},\; 0.25,\; 4.0\right)$$

- 에너지가 높은 채널(의미적으로 활성) → 보정 강하게
- 에너지가 낮은 채널(배경 등) → 보정 약하게
- 초기 스텝에서 HH_x가 노이즈 지배 → 균일 에너지 → 가중치 ≈ 1 (인위적 편향 없음)
- 전체 보정 에너지의 평균은 항상 기존과 동일하게 유지

---

## CFG Wavelet Mixing 수학적 배경

표준 CFG는 `e = cond − uncond`를 단일 스케일 w로 적용합니다.

Diffusion의 coarse-to-fine 특성상:
- **초기 스텝(σ 큼)**: 저주파 성분(구도, 색감)이 결정됨
- **후기 스텝(σ 작음)**: 고주파 성분(엣지, 텍스처)이 결정됨

CWM은 이 특성에 맞게 guidance 스케일을 주파수·시간적으로 분리합니다.
α = 0이면 표준 CFG와 동일하므로 추가 비용 없이 선택적으로 활성화할 수 있습니다.

---

## Sliding Mode Control CFG 수학적 배경

표준 CFG는 매 스텝 동일한 gain w로 error를 피드백하는 비례제어(P-control)입니다.
높은 CFG scale에서 trajectory가 sliding manifold를 중심으로 oscillation하는 불안정성이 발생합니다.

SMC-CFG는 이를 비선형 switching feedback으로 해결합니다:
- **sliding surface** `s_t = (e_t − e_{t-1}) + λ · e_{t-1}` 가 0에 수렴하는 방향으로 강제
- **unit_2 switching** `u_sw = −k · s / ‖s‖₂` 로 방향 정보를 보존하는 정규화 보정
- Lyapunov 안정성 분석에 의해 유한시간 수렴이 이론적으로 보장됨

### 논문 정의: sign(s) ≡ s / ‖s‖₂ (unit_2)

논문 Table 4 / Notation Table은 `sign(s_t) ≡ s_t / ‖s_t‖₂` 로 명시합니다.
이는 element-wise ±1이 아니라 **텐서 전체를 L2 단위벡터로 정규화**하는 연산입니다.

```
s      = (e − e_prev) + λ · e_prev
‖s‖₂  = L2 norm (배치 샘플별 독립 계산)
Δe     = −k · s / ‖s‖₂               보정의 총 에너지 = 항상 k로 고정
e*     = e + Δe
```

| | element-wise sign(s) | unit_2: s / ‖s‖₂ (논문 정의) |
|---|---|---|
| 보정 총 에너지 | k × √(H×W×C) — 해상도에 비례 | 항상 k로 고정 |
| 공간 패턴 | 모든 위치 균등하게 ±k | s의 방향 구조 보존 |
| Lyapunov 안정성 | 비보장 | 이론적으로 보장 |
| Cosmos 16ch 채널 분리 | 에너지 발산 위험 | 보정 에너지 유계 |

---

## 기술적 참고사항

**Haar 웨이블릿 선택 이유**
추가 의존성 없이 PyTorch 텐서 연산만으로 구현 가능하고, 연산이 가장 빠릅니다.

**홀수 해상도 처리**
Haar DWT는 짝수 H, W가 필요합니다. 홀수 해상도 latent는 reflect 패딩 후 보정하고 원본 크기로 크롭합니다.

**fp8 dtype 처리**
fp8 텐서는 산술 연산이 불가합니다. 자동으로 bfloat16으로 업캐스트 후 원본 dtype으로 복원합니다.

**SMC fp32 upcast**
SMC 연산은 항상 float32로 수행되고 원본 dtype으로 복원됩니다. SageAttention 등 fp16 패치와 함께 사용할 때 NaN / 검은 이미지를 방지합니다.

**연산 비용**
DCW 논문 실험 기준 추가 연산 시간 **0.08 – 0.47%** 수준.
CWM과 SMC는 guidance error에 DWT/IDWT 1회 및 L2 norm 연산이 추가되므로 동일 수준입니다.
샘플링 속도에 실질적 영향이 없습니다.

---

## 참고 문헌

```
@article{yu2026dcw,
  title   = {Elucidating the SNR-t Bias of Diffusion Probabilistic Models},
  author  = {Meng Yu and Lei Sun and Jianhao Zeng and Xiangxiang Chu and Kun Zhan},
  journal = {arXiv preprint arXiv:2604.16044},
  year    = {2026}
}

@inproceedings{wang2026cfgctrl,
  title   = {CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance},
  author  = {Hanyang Wang and Yiyang Liu and Jiawei Chi and Fangfu Liu and Ran Xue and Yueqi Duan},
  booktitle = {CVPR},
  year    = {2026},
  eprint  = {2603.03281},
  url     = {https://arxiv.org/abs/2603.03281}
}
```
