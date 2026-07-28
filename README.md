# ComfyUI-DCW

**Differential Correction in Wavelet domain** + **CFG Wavelet Mixing** + **Sliding Mode Control CFG** + **Band-wise Reverse Drift Compensation**

> **DCW 논문**: *Elucidating the SNR-t Bias of Diffusion Probabilistic Models*
> Yu et al., arXiv:2604.16044v1 (2026)
> 코드: https://github.com/AMAP-ML/DCW

> **SMC 논문**: *CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance*
> Wang et al., CVPR 2026 / arXiv:2603.03281
> 코드: https://github.com/THU-SI/CFG-Ctrl

> **RDC**: 본 저장소 자체 확장 기능. 외부 논문 선례 확인 안 됨(전수조사 아님, §"RDC 배경" 참고).
> 원형 개념(전역 단일 텐서 EMA drift 보정)을 DCW의 wavelet band 구조에 맞춰 대역별로 재설계.

---

## 기능 개요

세 가지 독립적인 기능을 하나의 노드에서 제공합니다.

| 기능 | 개입 위치 | 목적 |
|------|-----------|------|
| **DCW** | `sampler_post_cfg_function` (x0_pred 후처리) | SNR-t 편향 보정 |
| **CWM** | `sampler_cfg_function` (CFG 계산 대체) | 주파수 대역별 adaptive CFG |
| **SMC** | CWM 훅 내부 (CWM 이전 단계) | guidance error의 oscillation 억제 및 semantic alignment 개선 |
| **RDC** | DCW 훅 내부 (wavelet 분해 재사용) | 여러 step에 걸친 구도/포즈 누적 drift 억제 |

각각 독립적으로 켜고 끌 수 있으며, 조합해서 사용할 수도 있습니다.

> DCW는 "이번 step이 원래 궤적에서 얼마나 벗어났는지"(순간 보정), RDC는 "여러 step에 걸쳐 누적되는 drift"(궤적 보정)를 다룹니다. 서로 다른 문제라 개입 위치가 같아도(DCW 훅 내부) 역할이 겹치지 않습니다.

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

노드 탐색기에서 **`DCW(+a)`** 를 검색하거나
`model_patches` 카테고리에서 찾으세요.

### 기본 연결

```
[Load Checkpoint]
      ↓
[DCW(+a)]  ← 파라미터 설정
      ↓
[KSampler]
```

### 다른 패치와 함께 사용

```
[Load Checkpoint]
      ↓
[Apply LoRA / FreeU 등]
      ↓
[DCW(+a)]  ← 가능하면 파이프라인 마지막에 연결 권장
      ↓
[KSampler]
```

> **주의**: CWM과 SMC는 `sampler_cfg_function`을 점유합니다.
> 이미 같은 훅을 사용하는 노드가 앞에 연결되어 있으면
> CWM과 SMC는 자동으로 skip되고 콘솔에 경고가 출력됩니다.
> DCW(`sampler_post_cfg_function`)는 항상 안전하게 체이닝됩니다.
> RDC는 별도 훅이 아니라 DCW 훅 내부에서 동작하며, `rdc_tau = 0.0`(기본값)이면
> 완전히 no-op이라 DCW 단독 사용 시 결과에 어떠한 영향도 주지 않습니다.

> **노드 전체 비활성화**: ComfyUI의 기본 기능인 노드 bypass(우클릭 → Bypass)를 사용하세요.

---

## 파라미터 레퍼런스

### DCW 파라미터 — SNR-t 편향 보정

DCW는 x0_pred(모델이 예측한 클린 이미지)를 웨이블릿으로 분해한 뒤 각 주파수 대역의 신호량을 보정합니다. 표준 CFG가 끝난 *이후* 작동하므로 CFG 스케일이나 샘플러 선택에 무관합니다.

```
corrected_f = denoised_f + λ_f(t) · w_ch · (x_t_f − denoised_f)

λ_l(t) = lambda_l · σ_norm        ← 초기 스텝에서 최대, 후기엔 자동 감쇠
λ_m(t) = (λ_l + λ_h) / 2          ← LH/HL(방향 엣지): 중간 스텝에서 활성
λ_h(t) = lambda_h · (1 − σ_norm)  ← 후기 스텝에서 최대, 초기엔 자동 감쇠
w_ch   = 채널별 에너지 가중치      ← 활성 채널에 더 강하게, 조용한 채널엔 약하게
```

**보정 방향의 의미:**
- 양수: x0_pred를 x_t 방향으로 당김 → 신호 성분을 보강, 모델의 과소추정 보정
- 음수: x0_pred를 x_t 반대 방향으로 밈 → 신호 성분을 억제

---

#### `lambda_l` — 저주파 보정 (초기 스텝 활성)

초기 스텝에서 전체 구도, 색감, 피사체 형태 등 저주파 성분을 보정합니다.

**양수일 때 — 구조·색감 신호 보강**

| 범위 | 체감 효과 |
|------|-----------|
| `+0.03 ~ +0.05` | 흐릿했던 전체 구조가 약간 또렷해짐, 색감이 살아남 |
| `+0.05 ~ +0.08` | 논문 권장 범위. 구도가 안정적, 색감 밀도 증가 |
| `+0.10 이상` | 색감·형태 변화 두드러짐. 과하면 구도가 굳어지거나 과채도 발생 |

**음수일 때 — 과잉 구조·채도 억제**

| 범위 | 체감 효과 |
|------|-----------|
| `-0.01 ~ -0.03` | 과채도 또는 딱딱한 구도를 부드럽게 완화 |
| `-0.03 ~ -0.08` | 색감이 차분해지고 구도 고정이 풀림. 모델 자체 표현이 올라옴 |
| `-0.08 이하` | 색감 소실, 구도 붕괴 위험 |

> 💡 Anima/Cosmos에서 lambda 값을 0.04/0.06/0.08 계열 또는 0.05/0.07/0.09 계열 중 하나를 골라 쓰면 비슷한 구도 성향의 결과가 나옵니다. 0.01 차이가 포즈나 장신구 유무 같은 의미론적 경계를 넘을 수 있습니다.

---

#### `lambda_h` — 고주파 보정 (후기 스텝 활성)

후기 스텝에서 엣지, 텍스처, 피부 결, 헤어 스트랜드 등 고주파 성분을 보정합니다.

**양수일 때 — 디테일 신호 보강**

| 범위 | 체감 효과 |
|------|-----------|
| `+0.005 ~ +0.010` | 뭉개져 있던 디테일이 미세하게 살아남 |
| `+0.010 ~ +0.020` | 논문 권장 범위. 텍스처가 선명해지고 엣지가 또렷해짐 |
| `+0.05 이상` | 과도한 샤프닝, AI 특유의 크리스피 질감 발생 위험 |

**음수일 때 — 과샤프닝·인공 텍스처 억제**

| 범위 | 체감 효과 |
|------|-----------|
| `-0.005 ~ -0.010` | AI 특유의 과처리 느낌이 줄어들고 자연스러운 질감이 나옴 |
| `-0.010 ~ -0.050` | 피부·배경 텍스처가 부드러워짐. 필름 느낌에 가까워짐 |
| `-0.050 이하` | 디테일 소실, 전체가 뭉개짐 |

---

#### `dcw_enabled`

DCW만 독립적으로 on/off. A/B 비교용.

---

### CWM 파라미터 — CFG Wavelet Mixing

CWM은 CFG guidance error `e = cond − uncond`를 웨이블릿으로 분해한 뒤 주파수 대역별로 다른 CFG 스케일을 적용합니다. DCW가 x0_pred를 보정하는 것과 달리, CWM은 모델이 프롬프트를 따르는 **강도**를 시간과 주파수 축으로 분리해서 제어합니다.

```
w_LL(t) = w · (1 + alpha_l · σ_norm)          ← 초기 스텝에서 LL 비중 변화
w_HH(t) = w · (1 + alpha_h · (1 − σ_norm))    ← 후기 스텝에서 HH 비중 변화
w_mid   = √(w_LL × w_HH)                      ← LH, HL: 기하 평균 (중간 타이밍)

alpha = 0 → 표준 CFG와 수학적으로 완전히 동일
```

**CFG 스케일 직접 조절과의 차이:**
CFG를 7→8로 올리면 구도와 텍스처가 *동시에* 모든 스텝에서 강해집니다. CWM은 "초반 구도 guidance만 강하게" 또는 "후반 디테일 guidance만 약하게"처럼 *분리* 제어가 가능합니다.

---

#### `alpha_l` — 초반 CFG 제어 (구도·색감·피사체 배치)

초기 디노이징 스텝에서 전체 구도, 색감 배분, 피사체 위치, 조명 방향 등을 결정하는 LL guidance 강도를 조절합니다.

**양수일 때 — 초반 구도 guidance 강화**

| 범위 | 체감 효과 |
|------|-----------|
| `+0.10 ~ +0.20` | 프롬프트의 구도·배치가 더 정확하게 잡힘. 색감이 프롬프트 분위기에 맞게 강해짐 |
| `+0.20 ~ +0.40` | 구도가 강하게 고정됨. 피사체 위치·배경 관계가 프롬프트 기술을 더 충실히 따름 |
| `+0.50 이상` | 구도가 매우 딱딱하게 고정, 포스터 같은 느낌. 색 변이 발생 가능 |

**음수일 때 — 초반 구도 guidance 억제**

| 범위 | 체감 효과 |
|------|-----------|
| `-0.10 ~ -0.20` | 구도가 더 유연해짐. 같은 프롬프트에서 시드별 구도 변화가 커짐 |
| `-0.20 ~ -0.40` | 모델 자체의 미적 판단이 구도에 반영됨. 색감이 중립적이 됨 |
| `-0.40 이하` | 프롬프트 구도 의도에서 많이 벗어날 수 있음 |

> Anima/Cosmos (Flow 모델): σ_norm 최대 ~0.5라 실효값이 절반. 체감을 위해 2배 기준 적용.

---

#### `alpha_h` — 후반 CFG 제어 (디테일·텍스처·엣지)

후기 디노이징 스텝에서 엣지, 피부 질감, 헤어, 장신구, 배경 텍스처 등 미세 디테일을 결정하는 HH guidance 강도를 조절합니다.

**양수일 때 — 후반 디테일 guidance 강화**

| 범위 | 체감 효과 |
|------|-----------|
| `+0.10 ~ +0.15` | 프롬프트에서 언급한 텍스처·소품이 더 명확하게 나타남. 엣지가 샤프해짐 |
| `+0.15 ~ +0.30` | 피부·의류·배경 디테일이 프롬프트를 강하게 따름 |
| `+0.50 이상` | AI 특유의 인공적 샤프닝, 텍스처 노이즈 발생 |

**음수일 때 — 후반 디테일 guidance 억제**

| 범위 | 체감 효과 |
|------|-----------|
| `-0.05 ~ -0.10` | 텍스처가 자연스러워짐. AI 과처리 느낌이 줄고 유기적인 질감이 나옴 |
| `-0.10 ~ -0.20` | 피부·배경이 부드럽고 필름 같은 느낌. 디테일은 모델이 자유롭게 생성 |
| `-0.20 이하` | 디테일이 뭉개지기 시작. 과도하면 배경·텍스처 해상도 손실 |

> Anima/Cosmos에서 alpha_h 양수 값이 크면 16채널 HH 대역에서 캐릭터 분리(1명→복수)가 발생할 수 있습니다. `+0.15` 이하로 시작하거나 음수를 권장합니다.

---

#### `cwm_enabled`

CWM 기능만 독립적으로 on/off. SMC는 CWM이 꺼져 있어도 `smc_preset ≠ Off`이면 독립 작동합니다.

---

### DCW + CWM 방향 조합 가이드

DCW는 x0_pred(모델 출력) 레벨에서, CWM은 CFG guidance(프롬프트 추종력) 레벨에서 작동합니다. 부호를 같은 방향으로 맞추면 효과가 누적되고, 반대로 맞추면 서로 다른 레이어에서 보완적으로 작동합니다.

| 목적 | lambda_l | lambda_h | alpha_l | alpha_h |
|------|----------|----------|---------|---------|
| 과채도·딱딱한 구도 완화 + 자연스러운 텍스처 | 음수 | 양수 | 양수 | 음수 |
| 전체적으로 선명하고 프롬프트 충실 | 양수 | 양수 | 양수 | 양수 |
| 유기적·자연스러운 결과물 | 음수 | 음수 | 음수 | 음수 |
| SNR 보정만, CFG는 건드리지 않음 | 조정 | 조정 | `0.0` | `0.0` |

> 반대 부호 조합(예: lambda_l 음수 + alpha_l 양수)은 "guidance로 구도를 제대로 잡되, x0_pred 레벨에서 과잉 표현을 줄인다"는 역할 분담이 됩니다. 상쇄가 아니라 보완입니다.

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

### RDC 파라미터 — Band-wise Reverse Drift Compensation

RDC는 DCW와 다른 문제를 풉니다. DCW가 "이번 step이 같은 step의 x_t에서 얼마나 벗어났는지"를 매번 순간적으로 비교하는 반면, RDC는 "여러 step에 걸쳐 구도/포즈가 서서히 한쪽으로 새는 현상(trajectory drift)"을 잡습니다. DCW 훅 안에서 이미 끝난 wavelet 분해(LL/LH/HL/HH)를 그대로 재사용하므로 별도의 변환 비용이 없습니다.

```
ema_X_new   = (1 − β) · ema_X_prev + β · X_current      ← 대역별 독립 EMA
drift_X     = X_current − ema_X_new
X_corrected = X_current − alpha_X · drift_X

β = 1 − exp(−Δs / τ)          Δs = 이번 step의 sigma_norm 이동폭
```

LL(구조)과 HH(텍스처)는 성격이 반대입니다. LL은 여러 step에 걸쳐 천천히 일관되게 변해야 정상이라 여기서의 drift가 "얼굴이 계속 바뀐다", "포즈가 흔들린다"의 실체에 가깝습니다. 반대로 HH는 매 step 다시 생성돼도 정상인 성분이라, 여기에 EMA 보정을 걸면 텍스처가 과거 평균 쪽으로 당겨져 디테일이 뭉개질 위험이 있습니다. 그래서 LL은 강하게, HH는 기본값 0(꺼짐)으로 두는 것을 권장합니다. LH/HL은 기존 DCW 코드의 관례를 따라 `(alpha_LL + alpha_HH) / 2`로 자동 계산되어 별도 파라미터가 없습니다.

**별도 on/off 토글이 없는 이유**: `rdc_tau = 0.0`이면 `β`가 항상 0으로 계산되어(`Δs/τ → ∞` → `exp(−∞) = 0`... 정확히는 아래 `rdc_tau` 설명 참고) drift 자체가 0이 되므로 자동으로 no-op이 됩니다. 파라미터 하나(`rdc_tau`)가 강도 조절과 on/off 역할을 겸합니다.

---

#### `rdc_tau` — EMA 기억 구간 (on/off 겸용)

`sigma_norm`(`[0, 1)` 범위, 스케줄러·step 수 무관) 기준으로 EMA가 "얼마나 오래 기억하는지"를 결정합니다. τ만큼 sigma_norm이 진행되면 이전 기억이 약 37%(1/e)로 줄어듭니다.

| 값 | 동작 |
|----|------|
| `0.0` | RDC 완전 비활성 (기본값). DCW 단독 사용과 동일 |
| `0.05 ~ 0.10` | drift에 빠르게 반응, 짧은 기억. 미세한 흔들림까지 잡아내지만 과민할 수 있음 |
| `0.15 ~ 0.20` | 절충 지점. 서서히 진행되는 drift 억제, 정상적인 step-to-step 변화는 유지 |
| `0.30 이상` | 느리게 반응, 긴 기억. 과하면 구도가 초반 상태에 "고착"되어 의도한 변화도 억제됨 |

> step 수나 karras/exponential 등 스케줄러 종류를 바꿔도 같은 τ 값이 항상 동일한 "노이즈 비율 구간"을 기억하도록 설계되어 있어, step 수를 바꿀 때마다 재조정할 필요가 적습니다.

---

#### `rdc_alpha_ll` — 구조 대역(LL) drift 보정 강도

구도, 포즈, 전체 실루엣이 EMA 기준선에서 벗어난 만큼을 되돌리는 강도입니다. `rdc_tau = 0.0`이면 값과 무관하게 효과 없음.

| 범위 | 체감 효과 |
|------|-----------|
| `0.0` | 보정 없음 |
| `0.02 ~ 0.05` | 권장 시작 범위. 구도/포즈의 완만한 표류를 억제하면서 정상적인 변화는 허용 |
| `0.05 ~ 0.10` | 구도 안정성 강하게 유지. 과하면 프롬프트가 의도한 변화(예: 다른 포즈로 유도)까지 억제될 수 있음 |
| `0.10 이상` | 구도가 사실상 "고정"됨. 초반 노이즈에서 결정된 구도에서 벗어나기 어려워짐 |

**음수는 지원하지 않습니다.** DCW의 `lambda_l`/`lambda_h`나 CWM의 `alpha_l`/`alpha_h`는 음수가 "반대 방향의 정상적인 보정"이라는 유효한 의미를 갖지만, RDC의 alpha가 음수가 되면 `current - alpha·drift`의 부호가 뒤집혀 `current + |alpha|·drift`, 즉 자기 자신의 EMA 기준선에서 매 step 더 멀어지는 방향으로 밀어내는 발산성 되먹임 루프가 됩니다. 안정화가 아니라 발산을 조장하므로 UI에서 `min = 0.0`으로 제한해 두었습니다.

---

#### `rdc_alpha_hh` — 텍스처 대역(HH) drift 보정 강도

기본값 0(꺼짐)을 권장합니다. HH는 원래 step마다 새로 생성되는 게 정상이라, 여기에 EMA 보정을 걸면 텍스처가 흐려질 위험이 큽니다.

| 범위 | 체감 효과 |
|------|-----------|
| `0.0` | 권장 기본값. 텍스처/그레인은 매 step 자유롭게 재생성 |
| `0.001 ~ 0.01` | 텍스처 flicker(깜빡임)가 눈에 띌 때만 소량 시도. 과하면 디테일 뭉개짐 |
| `0.01 이상` | 비권장. 블러 발생 위험이 alpha_LL보다 훨씬 큼 |

`rdc_alpha_ll`과 마찬가지로 음수는 지원하지 않습니다(동일한 발산 문제).

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

### RDC

| 모델 | `rdc_tau` | `rdc_alpha_ll` | `rdc_alpha_hh` |
|------|-----------|---------------|---------------|
| SDXL / SD1.5 / DiT | 0.15 | 0.03 | 0.0 |
| **Flux** | 0.15 | 0.03 | 0.0 |
| **Anima (Cosmos)** | 0.15 | 0.03 | 0.0 |
| EDM | 0.15 | 0.03 | 0.0 |

> RDC는 DCW/CWM처럼 모델군별로 검증된 값이 없습니다. 위 표는 개발 문서에 명시된 "실험적으로 검증되지 않은 추정값"을 그대로 옮긴 것으로, 모든 모델에 동일한 시작값을 제시하는 이유도 검증 부족 때문입니다. 실제 사용 시 8-step/20-step 등 자신의 step 수에서 직접 A/B 비교를 권장합니다.

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

### RDC 추가 (구도/포즈가 step을 거치며 표류할 때)
`rdc_tau > 0.0`으로 설정하면 DCW 훅 안에서 RDC가 함께 작동합니다.
CWM/SMC 활성 여부와는 무관하게 독립적으로 켤 수 있지만, **`dcw_enabled = True`가 반드시 필요**합니다.
RDC는 별도 훅이 아니라 DCW 훅(`sampler_post_cfg_function`) 내부에 있으므로,
`dcw_enabled = False`면 훅 자체가 등록되지 않아 RDC도 함께 꺼집니다.
DCW의 보정 항(`lambda_l`, `lambda_h`)만 끄고 RDC만 쓰고 싶다면
`dcw_enabled = True`인 채로 `lambda_l = lambda_h = 0.0`으로 두면 됩니다
(DCW 자체의 보정은 0이 되고, wavelet 분해 결과 위에 RDC만 순수하게 적용됨).

```
[DCW+CWM+SMC 조정 완료] → rdc_tau = 0.15, rdc_alpha_ll = 0.03 부터 시작
  → 여러 시드로 같은 프롬프트 반복 생성
  → 포즈/구도가 시드 간에도 과도하게 비슷해지면(다양성 손실) rdc_alpha_ll 감소
  → 여전히 표류가 보이면 rdc_alpha_ll을 0.05~0.08로 소폭 증가
```

---

## 튜닝 팁

**단계적 탐색을 권장합니다:**

1. `dcw_enabled = True`, `cwm_enabled = False`, `smc_preset = Off` → DCW 단독 조정
2. `dcw_enabled = False`, `cwm_enabled = True`, `smc_preset = Off` → CWM 단독 조정
3. `dcw_enabled = False`, `cwm_enabled = True`, `smc_preset = Auto` → SMC 추가 효과 확인
4. 모두 활성화 후 `dcw_enabled` / `cwm_enabled` / `smc_preset` 개별 토글로 각 기여분 확인
5. 마지막으로 `rdc_tau`를 0.0에서 서서히 올려가며 구도/포즈 안정성 변화만 별도로 확인
6. 노드 전체 A/B 비교는 ComfyUI 기본 기능인 노드 bypass를 사용

**스텝 수가 적을수록 효과가 더 큽니다.** 10–20 스텝에서 차이가 가장 명확합니다.

**과도한 보정 징후:**
- 색감 변이 / 채도 과다 → `lambda_l` 또는 `alpha_l` 감소
- 텍스처 노이즈처럼 보이는 디테일 → `lambda_h` 또는 `alpha_h` 감소
- 구도·구조가 원본과 달라짐 → `lambda_l` 또는 `alpha_l` 감소
- 이미지가 과도하게 선명하거나 엣지가 튀는 느낌 → `smc_k` 감소 또는 `smc_preset = Off`
- 같은 프롬프트인데 시드를 바꿔도 구도/포즈가 지나치게 비슷해짐(다양성 손실) → `rdc_alpha_ll` 감소 또는 `rdc_tau` 감소
- 텍스처가 뭉개지거나 흐릿해짐(RDC 사용 중) → `rdc_alpha_hh`를 0.0으로 되돌림

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

## RDC 배경: 순간 보정과 궤적 보정은 다른 문제다

DCW는 매 step마다 `denoised`(x0_pred)를 **같은 step의 x_t**하고만 비교해서 보정합니다. 이 방식은 "이번 step이 원래 궤적에서 얼마나 벗어났는지"는 잡아내지만, 여러 step에 걸쳐 누적되는 drift(얼굴/포즈/구도가 step을 거치며 서서히 옆으로 새는 현상)는 다루지 않습니다. DCW는 순간(instantaneous) 보정이고, RDC는 궤적(trajectory) 보정이라는 점에서 서로 다른 문제를 풉니다.

### 원형 개념과의 차이

최초 검토된 원형 RDC는 아래와 같이 ε(noise prediction) 전체를 단일 텐서로 다룹니다.

```
d_t        = ε_t − EMA(ε)
EMA        = 0.95·EMA + 0.05·ε
ε_correct  = ε − α·d
```

본 구현은 이를 DCW 구조에 맞춰 확장한 버전입니다.

| 항목 | 원형 RDC | 본 구현 (Band-wise RDC) |
|---|---|---|
| 보정 대상 | ε 전체, 단일 텐서 | `denoised`를 Haar wavelet로 분해한 LL/LH/HL/HH 4개 대역 각각 |
| 적용 위치 | 명시 안 됨 | DCW의 `apply_dcw` 내부, `sampler_post_cfg_function` 훅 |
| EMA 개수 | 1개 | 4개 (대역별 독립 EMA 상태) |
| EMA 시간축 | step 기준 고정 β | `sigma_norm` 기준으로 유도되는 가변 β |
| 강도 파라미터 | α 1개 | `alpha_LL`, `alpha_HH` 2개 (LH/HL은 보간) |

전역 텐서 하나에 동일한 α로 EMA 보정을 거는 원형보다, 대역을 나눠 LL은 강하게·HH는 약하게(또는 0으로) 보정하는 편이 부작용 없이 구도 안정화를 달성하는 데 더 정밀하다고 판단해 확장했습니다.

### 상태(state) 저장

SMC와 동일한 패턴으로 `model_options`에 `_dcw_rdc_state` 키로 대역별 EMA 텐서와 이전 `sigma_norm` 값을 저장합니다. `model_options`는 KSampler 실행마다 새로 생성되는 dict이므로 실행 간 상태가 자동으로 초기화됩니다. 첫 step에는 각 대역의 EMA를 현재값으로만 시딩하고 보정은 적용하지 않습니다(기준선이 없는 상태에서 보정하면 의미가 없기 때문).

### 검증되지 않은 부분 (투명하게 명시)

- `alpha_LL`, `alpha_HH`, `τ`의 기본값은 DCW/SMC 파라미터 스케일에서 유추한 추정값이며, 실험적으로 A/B 검증되지 않았습니다.
- Wavelet-CFG 보정 노드에 band-wise EMA drift compensation을 통합하는 이 방식이 기존 논문에 선례가 있는지는 제한된 검색 범위 내에서 확인되지 않았습니다. 전수조사는 아니므로 "새로운 기법"이라 단정할 수는 없습니다.
- 채널별 에너지 가중치(`_ch_energy_weight`, DCW가 사용 중)를 RDC의 drift 계산에도 재사용할지는 미정입니다. 현재 구현은 재사용하지 않습니다.

---



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
RDC는 DCW가 이미 수행한 wavelet 분해 결과를 그대로 재사용하므로 추가 DWT/IDWT 비용이 없고, 대역별 EMA 갱신(elementwise 연산)만 추가되어 오버헤드가 미미합니다.
샘플링 속도에 실질적 영향이 없습니다.

**RDC 상태 초기화**
`model_options`가 KSampler 실행마다 새로 생성되는 dict이므로, RDC의 대역별 EMA 상태도 매 생성(run)마다 자동으로 초기화됩니다. 배치 크기나 해상도가 이전 step과 달라지면(예: 해상도 변경) 해당 대역의 EMA는 새로 시딩되고 그 step은 보정 없이 넘어갑니다.

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
