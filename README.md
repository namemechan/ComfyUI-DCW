# ComfyUI-DCW

**Differential Correction in Wavelet domain** — SNR-t 편향 보정 플러그인

> 논문: *Elucidating the SNR-t Bias of Diffusion Probabilistic Models*  
> Yu et al., arXiv:2604.16044v1 (2026)  
> 코드: https://github.com/AMAP-ML/DCW

---

## 개요: SNR-t 편향이란?

Diffusion 모델은 **훈련** 시 노이즈 샘플 $x_t$의 SNR(신호 대 잡음비)이 timestep $t$에 의해 완전히 결정됩니다.

$$\text{SNR}(t) = \bar{\alpha}_t \;/\; (1 - \bar{\alpha}_t)$$

그러나 **추론(inference)** 시에는 두 가지 오차의 누적으로 이 결합이 깨집니다.

1. 신경망의 예측 오차  
2. 수치 solver의 이산화 오차  

결과: 역방향 디노이징 샘플 $\hat{x}_t$의 실제 SNR이 해당 timestep이 기대하는 SNR보다 **항상 낮습니다** (더 많은 노이즈 / 더 적은 신호). 모델은 이를 "SNR이 맞지 않는 입력"으로 받아들이고 노이즈 예측을 **과대추정**하여 오차가 누적됩니다. 이것이 SNR-t bias입니다.

---

## 논문의 원본 방법

논문은 각 디노이징 스텝 **완료 후** 예측 샘플 $\hat{x}_{t-1}$을 직접 보정합니다.

$$\hat{x}^{\,f}_{t-1} \;\leftarrow\; \hat{x}^{\,f}_{t-1} \;+\; \lambda^f_t \cdot \bigl(\hat{x}^{\,f}_{t-1} - x^{\,f}_\theta(\hat{x}_t, t)\bigr)$$

- $f \in \{LL,\, LH,\, HL,\, HH\}$ : 웨이블릿 주파수 서브밴드  
- $x^0_\theta(\hat{x}_t, t)$ : 현재 스텝의 x0 재구성 예측  
- 차분 신호 $\hat{x}_{t-1} - x^0_\theta$ 는 이상적인 순방향 샘플 방향을 가리킵니다

동적 가중치 (논문의 Eq. 20, 21):

$$\lambda^l_t = \lambda_l \cdot \sigma_t \qquad \lambda^h_t = (1-\lambda_h) \cdot \sigma_t$$

---

## 이 구현이 논문과 다른 점

### 차이점: 후킹 지점

| | 논문 | 이 구현 |
|---|---|---|
| **개입 시점** | 샘플러가 $x_{t-1}$ 계산 **후** | 샘플러가 $x_{t-1}$ 계산 **전** |
| **보정 대상** | $x_{t-1}$ 직접 | $x^0_\theta$ (x0 예측값) |
| **후킹 방식** | 샘플링 루프 직접 수정 | ComfyUI의 `sampler_post_cfg_function` 훅 |

이 구현에서는 ComfyUI의 `sampler_post_cfg_function` 훅을 사용합니다. 이 훅은 매 디노이징 스텝에서 **CFG 적용 후 x0 예측값(denoised)**을 받아 수정할 수 있게 해줍니다.

### 보정 수식 (이 구현)

$$\text{denoised}^f_{\text{corrected}} = \text{denoised}^f + \lambda^f_t \cdot (x^f_t - \text{denoised}^f)$$

보정 방향이 $(x_t - \text{denoised})$로 바뀌었습니다.

### 왜 그래도 유사한 결과가 나오는가?

**수학적 근거:**

Assumption 5.1 (논문)에 의하면:
$$x^0_\theta(\hat{x}_t, t) = \gamma_t x_0 + \phi_t \epsilon, \qquad 0 < \gamma_t \leq 1$$

$\gamma_t < 1$ 이기 때문에 x0 예측값은 실제 신호를 **과소추정**합니다.

보정 방향 $(x_t - \text{denoised})$를 분해하면:

$$x_t - \text{denoised} \approx (1 - \gamma_t) \cdot x_0 + \text{noise terms}$$

**$(1 - \gamma_t) > 0$** 이므로 이 방향에는 **양의 신호 성분**이 포함됩니다.  
이 보정을 denoised에 더하면:

$$\text{denoised}_{\text{new}} = \text{denoised} + \lambda \cdot (x_t - \text{denoised})$$
$$= (\gamma_t + \lambda(1-\gamma_t)) \cdot x_0 + \text{noise}$$

신호 계수가 $\gamma_t$에서 $\gamma_t + \lambda(1-\gamma_t) > \gamma_t$로 증가합니다. ✓  
즉, **x0 예측의 신호 과소추정을 보정**하는 방향으로 작동합니다.

샘플러는 이 보정된 x0_pred를 사용해 $x_{t-1}$을 계산하므로, $x_{t-1}$ 역시 더 올바른 SNR을 갖게 됩니다.

### 동적 가중치 차이

| | 논문 | 이 구현 |
|---|---|---|
| $\sigma_t$ 정의 | DDPM 후방 분산 $\tilde{\beta}_t$ (작은 값) | k-diffusion sigma (넓은 범위) |
| 정규화 | 없음 | $s = \sigma / (\sigma + 1) \in [0, 1)$ |
| $\lambda^l_t$ | $\lambda_l \cdot \sigma_t$ | $\lambda_l \cdot s$ |
| $\lambda^h_t$ | $(1-\lambda_h) \cdot \sigma_t$ | $\lambda_h \cdot (1 - s)$ |

정규화를 통해 EDM, DDPM, Flow Matching 등 모든 모델에서 sigma 스케일에 무관하게 동작합니다.

**타이밍 특성 (이 구현):**

| 디노이징 단계 | $s$ 값 | $\lambda^l_t$ | $\lambda^h_t$ | 효과 |
|---|---|---|---|---|
| 초기 (sigma 큼) | ≈ 1 | 크다 | ≈ 0 | 저주파 구조 보정 주도 |
| 중기 | 중간 | 중간 | 중간 | 균형 |
| 후기 (sigma 작음) | ≈ 0 | ≈ 0 | 크다 | 고주파 텍스처 보정 주도 |

이는 논문이 의도한 **coarse-to-fine 보정 패턴**과 일치합니다.

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

노드 탐색기에서 **`DCW Model Patch (SNR-t Bias Correction)`** 를 검색하거나  
`model_patches` 카테고리에서 찾으세요.

### 기본 연결

```
[Load Checkpoint]
      ↓
[DCW Model Patch]  ← lambda_l, lambda_h 슬라이더 설정
      ↓
[KSampler]  ← 기존 샘플러 그대로 (euler_a, dpmpp 등)
      ↑
[AYS Scheduler 등]  ← 기존 스케줄러 그대로
```

### 다른 Model Patch와 함께 사용

DCW는 LoRA, FreeU, IPAdapter 등 다른 모델 패치들과 **충돌 없이** 같이 사용할 수 있습니다.

```
[Load Checkpoint]
      ↓
[Apply LoRA / FreeU / etc.]
      ↓
[DCW Model Patch]  ← 가능하면 파이프라인의 마지막에 연결 권장
      ↓
[KSampler]
```

---

## 파라미터 가이드

### `lambda_l` — 저주파 보정 강도

| 값 범위 | 방향 | 효과 |
|---|---|---|
| `0.0` | — | 저주파 보정 비활성화 |
| `+0.03 ~ +0.05` | x_t 방향으로 당김 | 약한 구조 보정, 안전한 시작점 |
| `+0.05 ~ +0.08` | x_t 방향으로 당김 | 중간 보정, 논문 권장 범위 |
| `+0.10 이상` | x_t 방향으로 당김 | 강한 보정, 색감·구도 변화 가능 |
| `-0.01 ~ -0.05` | x_t 반대 방향으로 밈 | 과포화·과채도 억제, 색감 차분해짐 |
| `-0.05 ~ -0.5` | x_t 반대 방향으로 밈 | 강한 억제, 색감 소실·뭉개짐 위험 |

**양수일 때:** x0_pred를 x_t 방향으로 당겨 신호 성분을 보강합니다. SNR-t 편향 보정 본래 목적입니다.  
**음수일 때:** x0_pred를 x_t 반대 방향으로 밀어 신호 성분을 억제합니다. 모델이 과도하게 선명하거나 채도가 높은 결과를 낼 때 억제 효과가 있습니다. 디노이징 초기(sigma 클 때) 작동하므로 **전체 색감·구도**에 영향을 줍니다.

---

### `lambda_h` — 고주파 보정 강도

| 값 범위 | 방향 | 효과 |
|---|---|---|
| `0.0` | — | 고주파 보정 비활성화 |
| `+0.005 ~ +0.010` | x_t 방향으로 당김 | 약한 디테일 보정, 안전한 시작점 |
| `+0.010 ~ +0.020` | x_t 방향으로 당김 | 중간 보정, 논문 권장 범위 |
| `+0.05 이상` | x_t 방향으로 당김 | 강한 보정, 과도한 샤프닝 위험 |
| `-0.005 ~ -0.01` | x_t 반대 방향으로 밈 | 과샤프닝·텍스처 과잉 억제, 디테일이 부드러워짐 |
| `-0.01 ~ -0.3` | x_t 반대 방향으로 밈 | 강한 억제, 디테일 소실·과도한 smoothing 위험 |

**양수일 때:** x0_pred의 고주파 성분을 x_t 방향으로 당겨 엣지·텍스처를 강화합니다.  
**음수일 때:** 고주파 성분을 반대 방향으로 밀어 엣지·텍스처를 억제합니다. 디노이징 후기(sigma 작을 때) 작동하므로 **디테일·텍스처·엣지의 선명도**에 영향을 줍니다.

---

### 음수 lambda의 수학적 배경

보정 수식은 다음과 같습니다.

$$\text{denoised}^f_\text{corrected} = \text{denoised}^f + \lambda^f_t \cdot (x^f_t - \text{denoised}^f)$$

$\lambda > 0$이면 denoised를 x_t 방향으로 당겨 $\gamma_t$를 높입니다 (신호 보강).

$\lambda < 0$이면 반대 방향으로 밀어 $\gamma_t$를 낮춥니다 (신호 억제).

$$\gamma_\text{new} = \gamma_t + \lambda \cdot (1 - \gamma_t)$$

- $\lambda = +0.05$, $\gamma_t = 0.9$ → $\gamma_\text{new} = 0.905$ (신호 1.05× 보강)
- $\lambda = -0.05$, $\gamma_t = 0.9$ → $\gamma_\text{new} = 0.895$ (신호 소폭 억제)

음수 효과는 일반적으로 양수보다 훨씬 약하게 체감됩니다. $\gamma_t$가 이미 1에 가깝기 때문에 억제 여지가 크지 않기 때문입니다. 미세한 조정 도구로 이해하시면 됩니다.

---

### `enabled` — 토글

A/B 비교를 위한 빠른 on/off 스위치입니다.

---

## 모델별 권장 시작값

| 모델 | `lambda_l` | `lambda_h` | 비고 |
|---|---|---|---|
| SDXL | 0.05 | 0.010 | 논문 실험값 기준 |
| SD 1.5 | 0.05 | 0.010 | |
| DiT (PixArt 등) | 0.05 | 0.010 | |
| **Flux** | 0.08 – 0.12 | 0.015 – 0.025 | Flow 모델, sigma 범위 [0,1] → 약 2× |
| **Anima (Cosmos 2B)** | 0.08 – 0.12 | 0.015 – 0.025 | Flow 모델, Flux와 동일 |
| EDM | 0.05 | 0.010 | |

> **Flow-based 모델 (Flux, Anima/Cosmos)에 대하여:**  
> 이 모델들은 sigma 스케일이 `[0, 1]`로 제한되어 정규화 후 최대 $s \approx 0.5$까지만 올라갑니다.  
> DDPM/EDM 모델 대비 보정 강도가 절반 수준이 되므로, 같은 체감 효과를 위해 **lambda 값을 2배 정도로** 올려 시작하세요.

---

## 튜닝 팁

**단계적 탐색을 권장합니다:**

1. `lambda_h = 0.0`으로 고정 후 `lambda_l`만 조정 → 구조/색감이 개선되는 지점 찾기
2. `lambda_l`을 고정 후 `lambda_h` 조정 → 디테일이 개선되는 지점 찾기
3. `enabled` 토글로 A/B 비교 확인

**스텝 수가 적을수록 효과가 더 큽니다.** 논문도 10–20 스텝에서 실험했습니다. 50 스텝 이상에서는 차이가 작아집니다.

**과도한 보정 징후:**
- 색감이 채도 과다 또는 색상 변이
- 디테일 과잉 (텍스처 노이즈처럼 보임)
- 구도나 구조가 원본과 달라짐

→ 해당 lambda 값을 낮추세요.

---

## 기술적 참고사항

### Haar 웨이블릿 선택 이유
논문은 일반적인 DWT를 제안합니다. 이 구현에서는 **Haar 웨이블릿**을 선택했습니다:
- 추가 의존성 없이 PyTorch 텐서 연산만으로 구현 가능
- 연산이 가장 빠름 (이미 무시할 수 있는 오버헤드를 더 줄임)
- 논문 실험의 CIFAR-10 등 소규모 해상도에서도 잘 동작

### 홀수 해상도 처리
Haar DWT는 짝수 H, W가 필요합니다. 홀수 해상도 latent는 자동으로 reflect 패딩 후 보정하고, 결과를 원본 크기로 크롭합니다. 사용자가 신경 쓸 필요 없습니다.

### 연산 비용
논문 실험 기준 추가 연산 시간은 **0.08 – 0.47%** 수준입니다. 샘플링 속도에 실질적 영향이 없습니다.

### 다른 모델 패치와의 호환성
ComfyUI의 `sampler_post_cfg_function`은 리스트로 관리되므로 여러 후처리 함수가 체이닝됩니다. DCW는 원본 모델 옵션을 복사 후 수정하므로 다른 패치를 덮어쓰지 않습니다.

---

## 참고 문헌

```
@article{yu2026dcw,
  title   = {Elucidating the SNR-t Bias of Diffusion Probabilistic Models},
  author  = {Meng Yu and Lei Sun and Jianhao Zeng and Xiangxiang Chu and Kun Zhan},
  journal = {arXiv preprint arXiv:2604.16044},
  year    = {2026}
}
```
