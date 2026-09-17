# 사람 보행 원리와 휴머노이드 적용 가설

## 조사에서 채택한 원리

1. 한쪽 발의 접지에서 같은 발의 다음 접지까지를 한 보행 주기로 본다. 정상 보행은
   대략 지지기 60%, 유각기 40%이며 지지기는 하중 수용·단일 지지·발끝 떼기,
   유각기는 발 전진과 다음 접지 준비 역할을 한다.
   - 근거: [Step-by-step insight into gait analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC13418446/)
   - FDB 적용: 두 다리를 180도 반대 위상으로 구동하고 유각 다리의 무릎만 더 굽힌다.
2. 정상 팔 흔들기는 반대쪽 다리와 함께 움직이며 다리에서 생기는 수직축 각운동량과
   지면 반작용 모멘트를 줄인다. 팔을 고정한 사람 실험에서는 대사 비용이 12% 증가하고
   수직 지면 반작용 모멘트가 63% 증가했다.
   - 근거: [Dynamic arm swinging in human walking](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/)
   - FDB 적용: 왼다리 전진 때 오른팔을 전진시키고 `static_arms` 제거 실험과 비교한다.
3. 발끝 끌림을 줄이려면 유각기의 무릎 굽힘과 발목 보상이 필요하다. 단, 무릎을 많이
   굽히는 것만으로는 지지 다리로 무게중심을 옮기지 못하므로 안정성을 보장하지 않는다.
   - 근거: [Stance and swing phase costs in human walking](https://pmc.ncbi.nlm.nih.gov/articles/PMC2894890/)
   - FDB 적용: `low_clearance` 후보와 정상 유각기 후보의 발 수직 변위와 이동량을 비교한다.
4. 휴머노이드에는 사람 동작 모사만으로 부족하다. 지지면 안의 ZMP/CoP와 미래 발 위치를
   함께 계획해야 동적 안정성을 직접 제어할 수 있다.
   - 근거: Kajita et al., [Biped walking pattern generation by using preview control of ZMP](https://doi.org/10.1109/ROBOT.2003.1241826)
   - FDB 다음 단계: 발 접촉으로 지지 다각형을 계산하고 CoM/ZMP 여유를 상태와 보상에 넣는다.
5. 큰 외란에서는 발목·엉덩이 자세 제어만으로 멈출 수 없고 적절한 위치에 발을 디뎌야 한다.
   Capture Point는 쓰러지지 않고 정지하기 위해 필요한 발 디딤 위치를 정의한다.
   - 근거: Pratt et al., [Capture Point: A Step toward Humanoid Push Recovery](https://www.cs.cmu.edu/~hgeyer/Teaching/R16-899B/Papers/Pratt%26Goswami06Humanoids.pdf)
   - FDB 다음 단계: 현재 낙상 예측 앙상블의 출력에 capture step 후보 생성기를 연결한다.

## 이번 실험의 경계

이번 제어기는 학습된 범용 보행 정책이나 ZMP 제어기가 아니다. 사람 보행에서 가져온
네 가지 관절 협응 가설을 MuJoCo 모델에 적용한 첫 open-loop 주기 운동이다. 이동 방향은
각 모델의 로컬 좌표에 따라 다르므로 절대 `x` 부호가 아니라 수평 이동거리로 비교한다.
성공 기준은 4초 완주, 낙상 없음, 수평 이동거리 10mm 이상, 한쪽 발의 수직 변위 5mm
이상이다. 몸통 pitch와 pitch 속도로 양 발목 목표를 보정하는 폐루프 후보도 별도로
평가한다. 실제 로봇 적용 전에는
접촉 센서, 상태 추정, 지연, 마찰 변화, 힘 제한, 비상 정지와 폐루프 균형 제어가 필요하다.
