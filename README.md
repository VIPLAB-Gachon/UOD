# RT-JSD: RGB and Thermal Object Detection

현재 M3FD 비교에 사용하는 **Ours-P2 (YOLOv5s + RT-JSD)**의 코드, 설정과
검증에서 선택한 E184 EMA 체크포인트는 [`rt_jsd_p2/`](rt_jsd_p2/README.md)에 있습니다.

이전 **RT-JSD + Score Calibrator(SC) + Varifocal Loss(VFL)**
구현은 [`rt_jsd/`](rt_jsd/README.md)에 있습니다. 실행 방법, 고정 split 목록,
학습 설정과 소스 provenance를 함께 포함합니다. 데이터와 체크포인트는 포함하지 않습니다.

## 이전 RT-JSD + SC/VFL 구조

```text
RGB ──────┐                    ┌─ RGB Neck ─────┐
          ├─ Shared Backbone ──┤                ├─ Shared 25-bin Head ─ Boxes
Thermal ──┘                    └─ Thermal Neck ─┘                        │
                                                                        └─ SC ─ Scores ─ Soft-NMS
```

- Backbone과 Head를 공유하고, 입력 모달리티에 따라 독립 Neck을 선택합니다.
- 네 경계의 거리 분포를 예측하고 기댓값으로 박스를 만듭니다.
- 양쪽 모두 GT 손실로 학습합니다. 학습 시 객체 IoU 조건(Teacher ≥ 0.7,
  차이 ≥ 0.1)과 경계 오차·DFL 조건을 통과한 분포에만 JSD를 적용합니다.
- 검출기 학습 후 동결하고, 별도 SC를 VFL로 학습해 최종 클래스별 점수를 보정합니다.
- 추론은 RGB 또는 Thermal 단독 입력을 지원합니다. RGBT는 두 입력의 후보를
  합친 뒤 Soft-NMS로 평가합니다. GT와 JSD는 추론 시 사용하지 않습니다.

현재 설정은 M3FD 자체 고정 8:1:1 split, 640×640, 검출기 200에포크,
2 GPU에서 전역 배치 16쌍, SC 10에포크입니다. 검출기와 SC를 같은 에포크에
공동 학습하는 방식이나 S/X·경량화 변형은 이 버전에 포함되지 않습니다.

## 시작하기

```bash
cd rt_jsd
```

이후 [`rt_jsd/README.md`](rt_jsd/README.md)의 환경 설치, manifest 준비,
초기 가중치 생성과 학습 명령을 따르세요. 기기별 경로는 템플릿에서 생성하며,
분할 ID는 [`rt_jsd/splits/m3fd_811_seed2026/`](rt_jsd/splits/m3fd_811_seed2026/)에 고정되어 있습니다.

## 초기 reference 코드

루트의 `uod/`, `train.py`, `train_yolov5_uod.py`와
[`README_KO.md`](README_KO.md)는 이전 UOD reference 구현입니다.
현재 RT-JSD 실행에는 `rt_jsd/runtime/`과 `rt_jsd/tools/`를 사용합니다.
vendored YOLOv5의 원본 라이선스는 각 소스 디렉터리에 보존합니다.
