# 차선 인지 결과 (판단/시간필터 전의 '그 순간' 기하값).
# lane_detection 노드가 매 프레임 발행. interpret 노드가 구독해 EMA/데드밴드/
# 클램프 등 시간필터·안전처리를 적용한 뒤 LaneInfo 로 재발행한다.
# YOLO 등 다른 인지 소비자는 카메라 원본을 직접 쓰므로 이 메시지에 의존하지 않는다.
std_msgs/Header header

uint32 image_width       # 입력 영상 폭 (px)
uint32 image_height      # 입력 영상 높이 (px)
float32 center_x         # 이미지 중심 x (px)
float32 lane_center_px   # 검출된 차선 중심 x (px); 완전 미검출 시 -1.0

# 정규화 횡방향 오차 [-1.0 ~ 1.0]. 이미지 중심 대비 차선 중심 위치.
# 양수(+) = 차선 중심이 오른쪽 → 차량이 왼쪽으로 치우침. EMA 등 시간필터 없음.
float32 raw_offset

bool  left_detected      # 좌측 차선 검출 여부
bool  right_detected     # 우측 차선 검출 여부
float32 confidence       # 인식 신뢰도 0.0~1.0 (검출 줄 수 / 전체 스캔 줄 수)

# 회전 교차로 FSM 상태 (하류 interpret 가 회전로 구간 감속에 사용).
# 0=LANE_FOLLOW(본선), 1=ENTER(진입), 2=IN_LOOP(회전), 3=EXIT(탈출)
uint8 drive_state
