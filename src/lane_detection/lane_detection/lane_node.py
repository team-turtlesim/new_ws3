"""차선 인지 노드 (순수 인지).

opencv_node 가 발행하는 엣지 영상(/opencv/image/edge)을 구독하여, 관심영역(ROI)
안에서 좌/우 차선을 검출하고 '그 순간'의 기하값(차선 중심, 횡오차, 진행방향
기울기, 신뢰도)을 LaneDetection 으로 발행한다.

이 노드는 인지(perception)만 담당한다: 픽셀에서 선을 뽑고 기하값을 계산할 뿐,
시간 평활(EMA)·데드밴드·클램프·미검출 시 값 유지 같은 '판단'은 하지 않는다.
그 판단은 interpret 노드가 LaneDetection 을 구독해 수행하고 LaneInfo 로 재발행한다.

파이프라인:
    엣지 영상 -> ROI 자르기 -> 행별 차선 픽셀 검출 -> 다항식 피팅/이상치 제거
    -> 단일선 병합 -> 차선폭 학습 -> raw offset/center/confidence 계산
    -> LaneDetection 발행 (+ 선택적 디버그 시각화 영상 발행)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from interface.msg import LaneDetection


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # --- ROS parameters -------------------------------------------------
        self.declare_parameter('edge_topic', '/opencv/image/edge')
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('debug_topic', '/lane_detection/image/debug')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('num_scan_rows', 12)     # ROI 안에서 스캔할 가로줄 개수
        self.declare_parameter('min_detect_rows', 3)    # 차선으로 인정할 최소 검출 줄 수
        # 2026-07-05 실측: 이 트랙 차선폭 ≈ 178px(0.556×320). 기본/범위를 실측에 맞춤.
        self.declare_parameter('default_lane_width_ratio', 0.556)  # 초기 차선폭(이미지폭 대비)
        # 학습된 차선폭(px)을 이미지폭 대비 이 범위로 clamp. 단일차선 추종 시 half 가
        # 과도하게 커져(=반대편으로 overshoot) 반대 차선을 넘는 것을 좌우 대칭으로 방지.
        self.declare_parameter('lane_width_min_ratio', 0.42)
        self.declare_parameter('lane_width_max_ratio', 0.62)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_log', False)  # lane_width/검출상태 진단 로그

        edge_topic = str(self.get_parameter('edge_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.num_scan_rows = max(2, int(self.get_parameter('num_scan_rows').value))
        self.min_detect_rows = max(1, int(self.get_parameter('min_detect_rows').value))
        self.default_lane_width_ratio = float(self.get_parameter('default_lane_width_ratio').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_image = bool(self.get_parameter('debug_image').value)

        # --- vehicle_config.yaml 에서 ROI 읽기 ------------------------------
        self.roi_top, self.roi_left = self.load_roi()
        # config 값을 기본값으로 하되, 실시간 튜닝을 위해 ROS 파라미터로도 노출.
        # detect_lane / publish_debug 는 매 프레임 파라미터를 다시 읽으므로
        # `ros2 param set /lane_detection_node roi_top N` 으로 라이브 조정 가능.
        self.declare_parameter('roi_top', int(self.roi_top))
        self.declare_parameter('roi_left', int(self.roi_left))
        # 라인 피팅 이상치 제거 임계값(px). 이 값보다 선에서 멀면 사물로 보고 버림.
        self.declare_parameter('line_fit_outlier_px', 12.0)
        # 피팅 차수. 근거리 밴드엔 1(직선)이 안정적. 2는 과적합→가짜곡선 요동.
        self.declare_parameter('line_fit_degree', 1)
        # 단일선 판별: 좌x·우x 간격이 (차선폭 * 이 비율)보다 작으면 사실 같은 선
        # 하나가 중심을 가로질러 좌/우로 잘린 것으로 보고 하나의 차선으로 병합.
        self.declare_parameter('single_line_gap_ratio', 0.55)
        # --- 좌/우 분류(클러스터 추적)용 ---
        # cluster_gap_px: 한 행에서 이 간격(px) 이하로 붙은 엣지 픽셀은 한 선으로 묶음.
        # (한 선의 Canny 양쪽 엣지는 붙여서 1개로, 서로 다른 두 차선은 분리)
        self.declare_parameter('cluster_gap_px', 30.0)
        # min_lane_sep_ratio: 근거리 씨앗에서 두 클러스터가 (이 비율*이미지폭) 이상
        # 떨어져야 두 차선으로 인정. 미만이면 단일선(중앙 걸침)으로 취급 → 유령선 방지.
        self.declare_parameter('min_lane_sep_ratio', 0.2)
        # track_tol_px: 인접 스캔행 간 같은 차선으로 매칭할 최대 x 이동(px).
        self.declare_parameter('track_tol_px', 40.0)

        # =================================================================
        # 회전 교차로(중앙 노란 링) 주행 FSM — 상대·기하 기반(sim→실차 이식성)
        # =================================================================
        # 트랙 흐름: 본선(흰) →진입→ 노란 링 →(출구 랜드마크 2번째)→ 탈출→ 본선(빨강쪽)
        # 설계 원칙: 색·해상도에 안 휘둘리게 '절대 픽셀 수' 대신 '비율/개수/시간'을 쓴다.
        #   - 상태 전환: 흰/노란 '픽셀 비율'(yellow/(yellow+white), 해상도 무관)
        #   - 출구 판단: 노란 '선 개수'(위상=색·크기 불변) 를 세서 2번째에 탈출
        #   - odom/IMU 없음 → 개수 카운트가 주 신호, 시간은 안전 백스톱
        self.declare_parameter('yellow_topic', '/opencv/image/yellow')
        # 켜야만 회전로 FSM 동작. 기본 off → 기존 차선주행 무영향(안전).
        self.declare_parameter('roundabout_enabled', False)
        # 탈출/전환 방향: 정방향(빨강 왼쪽)=왼쪽(-1), 역방향(빨강 오른쪽)=오른쪽(+1).
        self.declare_parameter('branch_side', 1)

        # --- 상태 전환: 픽셀 '비율'(해상도·크기 무관, sim→실차 이식) ---
        # 진입: 노란 비율(yellow/(yellow+white))이 이 값 이상이면 링 접근 → ENTER.
        self.declare_parameter('enter_yellow_ratio', 0.40)
        # 링 안착: 흰 비율이 이 값 이하로 떨어지면 본선 벗어나 링 내부 → IN_LOOP.
        self.declare_parameter('onring_white_ratio', 0.15)
        # FSM 종료: 노란 비율이 이 값 이하 = 링 벗어나 본선 복귀 → LANE_FOLLOW.
        self.declare_parameter('exit_yellow_ratio', 0.15)
        # 출구는 '노랑→흰 합류'라 색전환 순간 흰선을 놓쳐 멈추기 쉽다. exit_use_both=True 면
        # EXIT 에서 흰+노랑을 합쳐 따라가(색 바뀌어도 안 놓침), 흰 비율이 exit_white_ratio
        # 이상으로 확보되면(=흰선 잡힘) 그때 LANE_FOLLOW 로 넘긴다.(진입은 발산이라 미적용)
        self.declare_parameter('exit_use_both', True)
        self.declare_parameter('exit_white_ratio', 0.50)
        # 비율을 신뢰할 최소 활성픽셀(ROI 면적 대비). 이보다 적으면 '아무것도 없음'.
        self.declare_parameter('activity_floor_frac', 0.01)

        # --- 출구 랜드마크(노란 선 개수) 카운트 ---
        # 한 스캔행에서 노란 클러스터가 이 개수 이상이면 그 행을 '출구 접합부'로 본다.
        # 평범한 링은 노란선 2개, 출구 접합부는 3개 이상.(개수 = 색·크기 불변)
        self.declare_parameter('junction_min_clusters', 3)
        # 스캔행 중 위 조건 만족 비율이 이 값 이상이면 '지금 출구 랜드마크'로 판정.
        self.declare_parameter('junction_rows_frac', 0.34)
        # 탈출 전 스킵할 출구 개수. 1이면 첫 출구 통과·두 번째에서 탈출(1회전 이상).
        self.declare_parameter('exits_to_skip', 1)

        # --- 시간 백스톱(초) — 임계값이 어긋나 상태에 '갇히는' 것 방지 ---
        # 진입로(ENTER) 통과 시간(초). 이 동안은 순수 중앙잡기(가드 OFF)로 진입로를
        # 지나 오른쪽 쏠림을 없애고, 지나면 원(IN_LOOP)으로 전환한다. 실측으로 튜닝.
        self.declare_parameter('approach_sec', 3.0)
        # IN_LOOP(원) 진입 후 이 시간(≈한 바퀴 도는 시간) 지나면 EXIT 로 전환한다.
        # 실측 랩타임에 맞춰 튜닝(너무 짧으면 덜 돌고 나가고, 너무 길면 더 돈 뒤 나감).
        self.declare_parameter('min_loop_sec', 8.0)
        self.declare_parameter('max_loop_sec', 25.0)      # (미사용, B안/백스톱 대비 보존)
        self.declare_parameter('enter_timeout_sec', 5.0)  # ENTER 갇힘 방지 → 강제 IN_LOOP
        self.declare_parameter('exit_timeout_sec', 6.0)   # EXIT 갇힘 방지 → 강제 LANE_FOLLOW

        # --- 노란 점선 잇기(모폴로지 닫힘) — '중앙잡기용'에만 국소 적용 ---
        # 중앙잡기는 점선 이어야 안정적이지만, 출구 '개수 세기'는 원본이라야 3선이 안
        # 뭉친다. 그래서 닫힘은 중앙잡기 입력에만 걸고, 카운트는 원본 노란으로 센다. 0=끔.
        self.declare_parameter('yellow_close_ksize', 5)

        # --- 진입 전용: 노란 점선 하나 따라가기(닫힌루프) ---
        # 진입 직진 중엔 노란 '점선'만 보이고 반대편 실선은 잘 안 보인다(선 1개).
        # 그때 '두 선 중앙잡기'는 성립 안 하므로, 보이는 그 선을 기준으로 링 안쪽으로
        # entry_target_ratio 만큼 치우친 위치를 목표로 '따라가며' 커밋한다(고정각 아님→
        # 자기보정 됨). 두 선이 다 보이면 자동으로 중앙잡기로 인계. False 면 옛 방식.
        self.declare_parameter('enter_follow_single', True)
        # 보이는 노란선 기준, 링 안쪽(branch_side 방향)으로 얼마나 치우쳐 따라갈지
        # (half-width 대비 비율, 0~1). 클수록 더 확 꺾어 링으로 파고든다. 음수 가능.
        self.declare_parameter('entry_target_ratio', 0.5)
        # 실선 판정: 한 선이 걸린 스캔행 수 / num_scan_rows 가 이 비율 이상이면 연속
        # 실선으로 본다. 점선은 깜빡여 커버리지가 낮으므로 실선 앵커 선택에서 밀린다.
        self.declare_parameter('solid_min_coverage_ratio', 0.5)

        # --- IN_LOOP(원 주행) 전용: 안쪽(중앙섬) 선 앵커 ---
        # 원은 바깥 선이 출구마다 갈라져 열린다 → 바깥을 따라가면 출구로 끌려나감.
        # 안쪽 중앙섬 선은 '닫힌 원'이라 출구에서도 안 끊긴다 → 안쪽에 앵커해 돈다.
        # loop_inner_side: +1=중앙섬이 오른쪽(바깥/출구가 왼쪽), -1=중앙섬이 왼쪽.
        self.declare_parameter('loop_inner_side', 1)
        # 안쪽 선 기준 '바깥쪽'으로 얼마나 치우쳐 돌지(half-width 대비 비율). 링 차로
        # 가운데를 유지하도록 반 링폭 정도. entry_target_ratio 와 별도로 튜닝.
        self.declare_parameter('loop_target_ratio', 0.5)

        # --- 전환부 조향 bias — 목표 방향으로 밀기(다중선 오검출 회피). 음수 가능 ---
        self.declare_parameter('enter_bias', 0.2)
        self.declare_parameter('exit_bias', 0.3)
        # 상태전이/랜드마크 판정 디바운스(연속 프레임 수).
        self.declare_parameter('roundabout_debounce_frames', 2)

        # --- 내부 상태 ------------------------------------------------------
        # 차선폭(px)은 기하 상태라 인지에 둔다. 양쪽 검출 시 EMA로 학습해
        # 한쪽만 보일 때 반대편 차선 위치를 추정하는 데 쓴다. (시간 평활 아님)
        self.lane_width_px = None

        # --- 회전 교차로 FSM 상태 -------------------------------------------
        self.latest_yellow = None        # 최신 노란색 마스크(ndarray, grayscale, 원본)
        self.lane_width_yellow = None    # 노란 링 차선폭(흰색과 분리 → handoff 오염 방지)
        self.rstate = 'LANE_FOLLOW'      # LANE_FOLLOW / ENTER / IN_LOOP / EXIT
        self.trans_counter = 0           # 상태전이 디바운스 카운터(상태 진입 시 0)
        self.state_enter_time = None     # 현재 상태 진입 시각(ENTER/EXIT 타임아웃용)
        self.loop_enter_time = None      # IN_LOOP 진입 시각(min/max_loop 백스톱용)
        self.junction_count = 0          # IN_LOOP 중 만난 출구 랜드마크 수
        self.junction_active = False     # 지금 랜드마크 위에 있나(중복 카운트 방지 래치)
        self.junction_on = 0             # 랜드마크 감지 디바운스(rising)
        self.junction_off = 0            # 랜드마크 해제 디바운스(falling)
        # 디버그 로그용 최신 관측값(회전로 임계값 튜닝에 사용)
        self.dbg_white = 0
        self.dbg_yellow = 0
        self.dbg_yr = 0.0
        self.dbg_wr = 0.0
        self.dbg_jscore = 0.0
        # 진입 실선 앵커의 직전 x (프레임 간 튐 방지 트래킹). LANE_FOLLOW 복귀 시 None.
        self.anchor_x_prev = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            edge_topic,
            self.image_callback,
            image_qos,
        )
        # 노란색 마스크 구독(회전로 FSM 용). 최신 프레임만 들고 있다가 edge 콜백에서
        # 함께 쓴다. 두 스트림은 같은 원본에서 나와 사실상 동기라 근사동기로 충분.
        self.yellow_sub = self.create_subscription(
            CompressedImage,
            str(self.get_parameter('yellow_topic').value),
            self.yellow_callback,
            image_qos,
        )
        self.detection_pub = self.create_publisher(LaneDetection, detection_topic, 10)
        self.debug_pub = None
        # 회전로 튜닝용: detect_on_yellow 가 실제로 쓰는 '닫힘 적용된 노란 마스크'를
        # 눈으로 보게 발행(점선이 이어졌는지 확인 → yellow_close_ksize 튜닝).
        self.yellow_closed_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(CompressedImage, debug_topic, image_qos)
            self.yellow_closed_pub = self.create_publisher(
                CompressedImage, '/lane_detection/image/yellow_closed', image_qos)

        self.get_logger().info(
            'lane_detection node started (perception only):\n'
            f'  edge_topic={edge_topic}\n'
            f'  detection_topic={detection_topic}\n'
            f'  roi_top={self.roi_top}, roi_left={self.roi_left}\n'
            f'  num_scan_rows={self.num_scan_rows}, min_detect_rows={self.min_detect_rows}\n'
            f'  debug_image={self.debug_image}'
        )

    # ------------------------------------------------------------------ config
    def load_roi(self):
        roi_top, roi_left = 0, 0
        if not os.path.exists(self.vehicle_config_file):
            self.get_logger().warning(
                f'vehicle config not found ({self.vehicle_config_file}); ROI defaults 0.'
            )
            return roi_top, roi_left
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(f'Failed to read vehicle config: {exc}')
            return roi_top, roi_left
        roi_top = int(config_data.get('ROI_TOP', 0))
        roi_left = int(config_data.get('ROI_LEFT', 0))
        return roi_top, roi_left

    # ------------------------------------------------------------------ decode
    def decode_edge(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        edge = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            self.get_logger().warning('Failed to decode edge image')
        return edge

    def yellow_callback(self, msg: CompressedImage):
        """노란색 마스크 프레임을 받아 최신본만 보관한다(디코드해서 저장)."""
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        yellow = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if yellow is not None:
            self.latest_yellow = yellow

    # ============================================================= roundabout
    def roi_bounds(self, height, width):
        """현재 파라미터 기준 ROI 상단/좌측 픽셀 경계를 clamp 해서 반환."""
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        roi_left = min(max(int(self.get_parameter('roi_left').value), 0), width - 1)
        return roi_top, roi_left

    def lane_ratios(self, edge, yellow):
        """ROI 안 흰/노란 픽셀의 '비율'을 반환한다(yellow_ratio, white_ratio).
        절대 픽셀 수 대신 비율을 써서 해상도·크기(sim↔실차)에 안 휘둘리게 한다.
        활성 픽셀이 바닥(ROI 면적 대비 activity_floor_frac) 미만이면 둘 다 0."""
        height, width = edge.shape
        roi_top, roi_left = self.roi_bounds(height, width)
        white_n = int(np.count_nonzero(edge[roi_top:, roi_left:]))
        yellow_n = int(np.count_nonzero(yellow[roi_top:, roi_left:]))
        roi_area = max(1, (height - roi_top) * (width - roi_left))
        total = white_n + yellow_n
        floor = float(self.get_parameter('activity_floor_frac').value) * roi_area
        yr = 0.0 if total < floor else yellow_n / float(total)
        wr = 0.0 if total < floor else white_n / float(total)
        # 디버그 스태시(튜닝용)
        self.dbg_white, self.dbg_yellow = white_n, yellow_n
        self.dbg_yr, self.dbg_wr = yr, wr
        return yr, wr

    def is_junction(self, yellow):
        """지금 화면이 회전로 '출구(분기)' junction 인지 판정한다.
        평범한 링은 노란선 2개(=행당 2 클러스터), 진입/출구 접합부는 3개 이상.
        스캔행 중 (junction_min_clusters 이상인 행) 비율이 junction_rows_frac 이상이면
        junction 으로 본다."""
        height, width = yellow.shape
        roi_top, roi_left = self.roi_bounds(height, width)
        scan_ys = np.linspace(roi_top, height - 1, self.num_scan_rows).astype(int)
        cluster_gap = float(self.get_parameter('cluster_gap_px').value)
        min_c = int(self.get_parameter('junction_min_clusters').value)
        rows_seen, rows_hit = 0, 0
        for y in scan_ys:
            clusters = self.row_clusters(yellow[int(y)], roi_left, cluster_gap)
            if clusters:
                rows_seen += 1
                if len(clusters) >= min_c:
                    rows_hit += 1
        score = 0.0 if rows_seen == 0 else rows_hit / float(rows_seen)
        self.dbg_jscore = score  # 디버그: junction 점수(3+클러스터 행 비율)
        return score >= float(self.get_parameter('junction_rows_frac').value)

    def _trans(self, cond):
        """상태전이 조건이 debounce 프레임 연속 참이면 True. 상태당 한 조건만
        검사한다는 전제(공유 카운터). 전이 시 set_state 가 카운터를 리셋한다."""
        need = max(1, int(self.get_parameter('roundabout_debounce_frames').value))
        self.trans_counter = self.trans_counter + 1 if cond else 0
        return self.trans_counter >= need

    def set_state(self, new_state):
        """FSM 상태를 바꾸고 전이 디바운스 카운터·상태진입 시각을 리셋한다."""
        self.rstate = new_state
        self.trans_counter = 0
        self.state_enter_time = self.get_clock().now()

    def elapsed_in_loop(self):
        """IN_LOOP 진입 후 경과 시간(초). 시작 전이면 0."""
        if self.loop_enter_time is None:
            return 0.0
        return (self.get_clock().now() - self.loop_enter_time).nanoseconds * 1e-9

    def elapsed_in_state(self):
        """현재 상태 진입 후 경과 시간(초). ENTER/EXIT 갇힘 방지 타임아웃용."""
        if self.state_enter_time is None:
            return 0.0
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9

    def closed_yellow(self, yellow):
        """중앙잡기용: 노란 점선을 모폴로지 닫힘으로 이어붙인 '사본'. 출구 개수 세기엔
        원본 yellow 를 쓴다(닫힘이 3선을 뭉치는 자기모순 방지)."""
        ksize = int(self.get_parameter('yellow_close_ksize').value)
        if ksize <= 0:
            return yellow
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)

    def detect_on_yellow(self, yellow_closed):
        """노란 마스크로 detect_lane 을 돌리되 차선폭 상태를 흰색과 분리한다(오염 방지).
        학습되는 lane_width_px 를 잠깐 노란용으로 스왑했다가 되돌린다."""
        saved = self.lane_width_px
        self.lane_width_px = self.lane_width_yellow
        result = self.detect_lane(yellow_closed)
        self.lane_width_yellow = self.lane_width_px
        self.lane_width_px = saved
        return result

    def apply_side_bias(self, result, bias):
        """전환부에서 목표 방향(branch_side)으로 offset 을 밀어준다. 여러 노란선이
        섞인 진입·출구에서 순진한 중앙잡기 대신 방향을 강제하기 위함.
        raw_offset 양수 = 차선중심 오른쪽 → 우조향. side=+1(right)이면 +bias."""
        side = 1 if int(self.get_parameter('branch_side').value) >= 0 else -1
        result = dict(result)
        result['raw_offset'] = float(np.clip(result['raw_offset'] + side * bias, -1.0, 1.0))
        # 전환부에선 검출 신뢰를 유지해 하류가 offset 을 쓰게 한다.
        result['left_detected'] = True
        result['right_detected'] = True
        result['confidence'] = max(result['confidence'], 0.3)
        return result

    def follow_dashed_into_ring(self, yc):
        """진입 전용: 노란선 하나(주로 점선)를 기준으로 링 안쪽 목표 위치로 몰아간다.
        진입 직진 중엔 반대편 실선이 잘 안 보여 선이 1개뿐 → '두 선 중앙잡기'가 성립
        안 하므로, 보이는 선을 기준으로 링 안쪽(branch_side)으로 entry_target_ratio 만큼
        치우친 위치를 목표로 따라간다(선 위치를 피드백으로 쓰는 닫힌루프 → 자기보정).
        두 선이 다 보이면 링 안착 근처이므로 중앙잡기(+약한 bias)에 인계한다."""
        result = self.detect_on_yellow(yc)
        side = 1 if int(self.get_parameter('branch_side').value) >= 0 else -1
        ld, rd = result['left_detected'], result['right_detected']
        if ld and rd:
            # 두 선 다 보임 → 중앙잡기 + 약한 링쪽 bias
            return self.apply_side_bias(result, float(self.get_parameter('enter_bias').value))
        # 단일 선: 검출된 선의 x 중앙값을 기준으로 링쪽 target offset 위치를 목표
        pts = result['left_pts'] if ld else (result['right_pts'] if rd else [])
        if not pts:
            return result   # 아무 선도 없음: 그대로(offset 0)
        width = result['image_width']
        center_x = result['center_x']
        line_x = float(np.median([x for _, x in pts]))
        target = float(self.get_parameter('entry_target_ratio').value) * (width / 2.0)
        lane_center = line_x + side * target       # 링 쪽으로 target 만큼 치우쳐 따라감
        result = dict(result)
        result['lane_center'] = lane_center
        result['raw_offset'] = float(
            np.clip((lane_center - center_x) / (width / 2.0), -1.0, 1.0))
        result['left_detected'] = result['right_detected'] = True
        result['confidence'] = max(result['confidence'], 0.3)
        return result

    def _make_result(self, width, height, center_x, raw_offset, lane_center, pts):
        """앵커 추종 결과를 detect_lane 과 동일한 dict 형식으로 만든다.
        좌/우 모두 검출로 표기(하류 interpret 가 단독선 축소 없이 offset 을 그대로
        쓰게)하고, 선이 잡히면 confidence 에 바닥(0.5)을 줘 회전로 중 lane-lost
        오정지(interpret 페일세이프)를 막는다."""
        return {
            'raw_offset': float(np.clip(raw_offset, -1.0, 1.0)),
            'left_detected': bool(pts),
            'right_detected': bool(pts),
            'confidence': 0.5 if pts else 0.0,
            'lane_center': lane_center,
            'center_x': center_x,
            'image_width': int(width),
            'image_height': int(height),
            'left_pts': pts,
            'right_pts': [],
            'left_poly': None,
            'right_poly': None,
        }

    def detect_yellow_lines(self, mask):
        """노란 마스크에서 세로선(차선)들을 '개수 제한 없이' 각각 분리한다.
        scan_lanes 는 좌/우 딱 2선을 가정하지만, 진입로엔 점선+실선+링선이 겹쳐
        3개 이상 나온다. 여기선 모든 선을 뽑고 각 선의 '행 커버리지'(연속성)를 재
        실선(높은 커버리지)과 점선(낮은 커버리지)을 구분할 수 있게 한다.
        반환: [{'median_x','bottom_x','coverage','pts'}...] (근거리→원거리 추적)."""
        height, width = mask.shape
        roi_top, roi_left = self.roi_bounds(height, width)
        scan_ys = sorted(
            {int(v) for v in np.linspace(roi_top, height - 1, self.num_scan_rows)},
            reverse=True,
        )
        cluster_gap = float(self.get_parameter('cluster_gap_px').value)
        track_tol = float(self.get_parameter('track_tol_px').value)
        lines = []  # 각 선: {'ref': 직전행 x, 'pts': [(y,x)...]}
        for y in scan_ys:  # 근거리(하단)부터 위로 추적
            clusters = self.row_clusters(mask[y], roi_left, cluster_gap)
            used = set()
            # 1) 기존 선들을 가장 가까운 미사용 클러스터에 tol 내에서 매칭
            for ln in lines:
                best_d, best_j = track_tol + 1.0, None
                for j, c in enumerate(clusters):
                    if j in used:
                        continue
                    d = abs(c[0] - ln['ref'])
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j is not None and best_d <= track_tol:
                    ln['ref'] = clusters[best_j][0]
                    ln['pts'].append((y, int(round(clusters[best_j][0]))))
                    used.add(best_j)
            # 2) 매칭 안 된 클러스터는 새 선으로 시작
            for j, c in enumerate(clusters):
                if j not in used:
                    lines.append({'ref': c[0], 'pts': [(y, int(round(c[0])))]})
        out = []
        for ln in lines:
            pts = ln['pts']
            xs = [x for _, x in pts]
            bottom_x = max(pts, key=lambda p: p[0])[1]  # 가장 아래(근거리) x
            out.append({
                'median_x': float(np.median(xs)),
                'bottom_x': float(bottom_x),
                'coverage': len(pts),
                'pts': pts,
            })
        return out

    def follow_solid_into_ring(self, yellow):
        """진입로/링에서 '가장 연속적인(=실선) 노란선 하나'에 앵커해 따라간다.
        다중선(점선+실선+링선)일 때 좌/우 짝짓기가 매 프레임 튀어 이탈하던 문제를
        없앤다:
          - 각 선의 행 커버리지로 실선을 고르고(점선은 깜빡여 커버리지 낮음),
          - 직전 앵커 x 에 가장 가까운 선으로 유지(프레임 간 튐 방지),
          - 그 선에서 '도로 중앙 쪽'으로 entry_target_ratio 만큼(반 차선폭) 치우쳐
            따라간다(닫힌루프 → 자기보정). 선이 굽으면 따라 굽어 자연히 링으로 진입.
        color/해상도 대신 '연속성+위치'로만 판단해 sim↔실차 이식성을 유지한다.
        원본(닫힘 미적용) yellow 를 쓴다 — 닫으면 점선이 메워져 실선과 구분이 안 됨."""
        height, width = yellow.shape
        center_x = width / 2.0
        lines = self.detect_yellow_lines(yellow)
        if not lines:
            return self._make_result(width, height, center_x, 0.0, None, [])
        solid_min = (
            float(self.get_parameter('solid_min_coverage_ratio').value) * self.num_scan_rows
        )
        solid = [ln for ln in lines if ln['coverage'] >= solid_min]
        pool = solid if solid else lines
        # 프레임 간 튐 방지: 직전 앵커에 가장 가까운 선을 유지. 직전이 없거나 너무
        # 멀면(다른 선으로 튐) 커버리지 최대(=가장 실선다운) 선으로 재획득.
        if self.anchor_x_prev is not None:
            anchor = min(pool, key=lambda ln: abs(ln['median_x'] - self.anchor_x_prev))
            if abs(anchor['median_x'] - self.anchor_x_prev) > float(
                    self.get_parameter('track_tol_px').value):
                anchor = max(pool, key=lambda ln: ln['coverage'])
        else:
            anchor = max(pool, key=lambda ln: ln['coverage'])
        self.anchor_x_prev = anchor['median_x']

        line_x = anchor['bottom_x']  # 근거리 x 기준(조향 반응성)
        target = float(self.get_parameter('entry_target_ratio').value) * (width / 2.0)
        # 앵커한 선을 branch_side 쪽으로 target 만큼 치우쳐 따라간다(라이브로 방향 조정
        # 가능: +1=오른쪽, -1=왼쪽). 진입로 방향에 맞춰 branch_side 로 부호를 맞춘다.
        side = 1.0 if int(self.get_parameter('branch_side').value) >= 0 else -1.0
        lane_center = line_x + side * target
        raw_offset = (lane_center - center_x) / (width / 2.0)
        return self._make_result(width, height, center_x, raw_offset, lane_center, anchor['pts'])

    def follow_ring(self, yellow, anchor_side=None):
        """한쪽 극단의 노란 실선 하나에 앵커해 따라간다(가장자리에서 도로 안쪽으로
        반 링폭 오프셋).
          - IN_LOOP(원 주행): anchor_side=안쪽(loop_inner_side) → 안쪽 중앙섬 선은
            닫힌 원이라 출구서 안 갈라짐 → 안 끌려나가고 계속 돎.
          - EXIT(탈출): anchor_side=바깥(-loop_inner_side) → 바깥 선은 출구서 갈라져
            나가므로 그 선을 따라 밖으로 빠져나간다.
        anchor_side 미지정 시 loop_inner_side(안쪽) 사용. loop_inner_side +1=중앙섬이
        오른쪽(바깥/출구 왼쪽). 라이브로 뒤집어 트랙 회전방향에 맞춘다."""
        yc = self.closed_yellow(yellow)   # 링 선은 실선이라 닫힘으로 안정화
        height, width = yc.shape
        center_x = width / 2.0
        lines = self.detect_yellow_lines(yc)
        if not lines:
            return self._make_result(width, height, center_x, 0.0, None, [])
        solid_min = (
            float(self.get_parameter('solid_min_coverage_ratio').value) * self.num_scan_rows
        )
        solid = [ln for ln in lines if ln['coverage'] >= solid_min]
        pool = solid if solid else lines
        if anchor_side is None:
            anchor_side = 1 if int(self.get_parameter('loop_inner_side').value) >= 0 else -1

        def extreme(cands):
            # anchor_side 쪽으로 가장 치우친 선(+1=가장 오른쪽, -1=가장 왼쪽).
            return (max(cands, key=lambda ln: ln['bottom_x']) if anchor_side > 0
                    else min(cands, key=lambda ln: ln['bottom_x']))

        # 튐 방지: 직전 앵커에 가장 가까운 선 우선, 너무 멀면 anchor_side 극단으로 재획득.
        if self.anchor_x_prev is not None:
            anchor = min(pool, key=lambda ln: abs(ln['bottom_x'] - self.anchor_x_prev))
            if abs(anchor['bottom_x'] - self.anchor_x_prev) > float(
                    self.get_parameter('track_tol_px').value):
                anchor = extreme(pool)
        else:
            anchor = extreme(pool)
        self.anchor_x_prev = anchor['bottom_x']

        line_x = anchor['bottom_x']
        target = float(self.get_parameter('loop_target_ratio').value) * (width / 2.0)
        # 앵커한 가장자리 선에서 '도로 안쪽'(anchor_side 반대)으로 오프셋 → 차로 유지.
        lane_center = line_x - anchor_side * target
        raw_offset = float(np.clip((lane_center - center_x) / (width / 2.0), -1.0, 1.0))
        return self._make_result(width, height, center_x, raw_offset, lane_center, anchor['pts'])

    def guard_inner(self, result, yellow):
        """중앙잡기(detect_lane) 결과를 '안쪽 선 기준 링차로 중앙'보다 바깥(출구쪽)으로
        못 가게 한쪽 방향으로만 클램프한다. 출구에서 바깥 선이 갈라져 중앙잡기가 출구로
        끌려나가는 걸 막되, 링 직선부에선 중앙잡기와 한계가 거의 같아 영향이 없다.
        loop_inner_side 로 안쪽(중앙섬) 방향을 지정한다(+1=오른쪽)."""
        lc = result.get('lane_center')
        if lc is None:
            return result
        yc = self.closed_yellow(yellow)
        lines = self.detect_yellow_lines(yc)
        if not lines:
            return result
        solid_min = (
            float(self.get_parameter('solid_min_coverage_ratio').value) * self.num_scan_rows
        )
        solid = [ln for ln in lines if ln['coverage'] >= solid_min] or lines
        inner_side = 1 if int(self.get_parameter('loop_inner_side').value) >= 0 else -1
        inner = (max(solid, key=lambda ln: ln['bottom_x']) if inner_side > 0
                 else min(solid, key=lambda ln: ln['bottom_x']))
        width = result['image_width']
        target = float(self.get_parameter('loop_target_ratio').value) * (width / 2.0)
        nominal = inner['bottom_x'] - inner_side * target   # 안쪽 기준 링차로 중앙
        # lane_center 가 nominal 보다 '바깥(-inner_side 방향)'으로 가면 nominal 로 막음.
        clamped = max(lc, nominal) if inner_side > 0 else min(lc, nominal)
        if clamped != lc:
            result = dict(result)
            result['lane_center'] = clamped
            center_x = result['center_x']
            result['raw_offset'] = float(
                np.clip((clamped - center_x) / (width / 2.0), -1.0, 1.0))
        return result

    def follow_yellow(self, yellow):
        """노란 구역 주행: 상황에 따라 '중앙잡기'와 '단일 실선 앵커'를 자동 전환한다.
          - 내가 있는 차로를 이루는 '중앙을 사이에 둔 두 실선'이 충분히 떨어져 보이면
            → 그 사이 중앙 주행(원래 라인 가운데 잡기). 진입로 직진/링에서 이 경우라
              한쪽에 안 붙고 안정적이다.
          - 그런 두 선이 없으면(흰→노랑 전환부·다중선·단일선) → 가장 연속적인 실선
            하나에 앵커해 링 쪽으로 커밋(follow_solid_into_ring).
        닫힘 마스크로 점선 flicker 를 메워 중앙잡기를 안정화한다. 오버레이는 중앙잡기면
        빨강+파랑(두 선), 앵커면 빨강 한 줄로 보여 현재 모드를 눈으로 알 수 있다."""
        yc = self.closed_yellow(yellow)
        height, width = yc.shape
        center_x = width / 2.0
        solid_min = (
            float(self.get_parameter('solid_min_coverage_ratio').value) * self.num_scan_rows
        )
        min_sep = float(self.get_parameter('min_lane_sep_ratio').value) * width
        lines = self.detect_yellow_lines(yc)
        solid = [ln for ln in lines if ln['coverage'] >= solid_min]
        # 중앙(center_x)을 사이에 둔, 가장 가까운 좌/우 실선 = 내가 있는 차로의 두 선
        lefts = [ln for ln in solid if ln['bottom_x'] < center_x]
        rights = [ln for ln in solid if ln['bottom_x'] >= center_x]
        if lefts and rights:
            left = max(lefts, key=lambda ln: ln['bottom_x'])
            right = min(rights, key=lambda ln: ln['bottom_x'])
            if right['bottom_x'] - left['bottom_x'] >= min_sep:
                lane_center = 0.5 * (left['bottom_x'] + right['bottom_x'])
                raw_offset = float(
                    np.clip((lane_center - center_x) / (width / 2.0), -1.0, 1.0))
                self.anchor_x_prev = lane_center  # 앵커 트래킹 연속성 유지
                return {
                    'raw_offset': raw_offset,
                    'left_detected': True,
                    'right_detected': True,
                    'confidence': 0.6,
                    'lane_center': lane_center,
                    'center_x': center_x,
                    'image_width': int(width),
                    'image_height': int(height),
                    'left_pts': left['pts'],
                    'right_pts': right['pts'],
                    'left_poly': None,
                    'right_poly': None,
                }
        # 중앙을 사이에 둔 두 실선이 없음 → 단일 실선 앵커(전환부 커밋)
        return self.follow_solid_into_ring(yellow)

    def update_junction_count(self, yellow):
        """IN_LOOP 중 출구(junction)를 rising-edge 로 카운트한다. junction 위에
        올라서는 '그 순간' 한 번만 +1(래치+디바운스)."""
        need = max(1, int(self.get_parameter('roundabout_debounce_frames').value))
        if self.is_junction(yellow):
            self.junction_on += 1
            self.junction_off = 0
        else:
            self.junction_off += 1
            self.junction_on = 0
        if not self.junction_active and self.junction_on >= need:
            self.junction_active = True
            self.junction_count += 1
        elif self.junction_active and self.junction_off >= need:
            self.junction_active = False

    def enter_loop(self):
        """ENTER→IN_LOOP 전이. junction_active=True 로 두어 진입부에 아직 남아있는
        junction 을 출구로 오카운트하지 않게 한다(빠져나온 뒤부터 카운트)."""
        self.set_state('IN_LOOP')
        self.loop_enter_time = self.get_clock().now()
        self.junction_count = 0
        self.junction_active = True
        self.junction_on = 0
        self.junction_off = 0
        self.anchor_x_prev = None   # 원(안쪽선) 앵커 새로 획득

    def reset_roundabout(self):
        """FSM 을 LANE_FOLLOW 로 되돌리고 회전 상태를 초기화."""
        self.set_state('LANE_FOLLOW')
        self.loop_enter_time = None
        self.junction_count = 0
        self.junction_active = False
        self.anchor_x_prev = None

    def roundabout_step(self, edge, yellow):
        """회전 교차로 FSM. 상태별로 흰/노란 마스크를 detect_lane 에 태워 처리한다.

        LANE_FOLLOW : 흰 차선 주행. 노란 비율 급증 → ENTER.
        ENTER       : 가장 연속적인 노란 실선에 앵커해 링으로 파고듦(follow_solid_into_ring).
                      흰 비율 낮아짐(or 타임아웃) → IN_LOOP.
        IN_LOOP     : 같은 실선 앵커로 링 주행 + 출구 랜드마크 카운트(원본 yellow).
                      (카운트 > exits_to_skip AND 시간≥min) 또는 시간≥max → EXIT.
        EXIT        : 출구쪽 bias + 노란 램프 추종. 노란 비율 낮아짐(or 타임아웃) → LANE_FOLLOW.
        진입/회전은 실선 앵커(다중선 튐 제거), 출구 카운트는 원본 yellow 로 한다."""
        yr, wr = self.lane_ratios(edge, yellow)      # 비율(해상도 무관)
        yc = self.closed_yellow(yellow)              # 중앙잡기용(닫힘). 카운트는 원본.
        enter_yr = float(self.get_parameter('enter_yellow_ratio').value)
        onring_wr = float(self.get_parameter('onring_white_ratio').value)
        exit_yr = float(self.get_parameter('exit_yellow_ratio').value)
        exits_to_skip = int(self.get_parameter('exits_to_skip').value)
        min_loop = float(self.get_parameter('min_loop_sec').value)
        max_loop = float(self.get_parameter('max_loop_sec').value)
        enter_to = float(self.get_parameter('enter_timeout_sec').value)
        exit_to = float(self.get_parameter('exit_timeout_sec').value)

        # ---- LANE_FOLLOW: 흰 차선 정상 주행, 노란 비율 급증 시 진입 ----
        if self.rstate == 'LANE_FOLLOW':
            if self._trans(yr >= enter_yr):
                self.set_state('ENTER')
            return self.detect_lane(edge)

        # ---- ENTER(진입로): 순수 중앙잡기(가드 OFF). approach_sec 지나면 원으로 ----
        # 진입로엔 점선이 있어 오른선 앵커를 켜면 오른쪽으로 쏠린다. 그래서 여기선
        # 흰색과 동일한 중앙잡기로 두 선 사이 가운데를 지나고, approach_sec 뒤 원 진입.
        if self.rstate == 'ENTER':
            if self.elapsed_in_state() >= float(self.get_parameter('approach_sec').value):
                self.enter_loop()
            return self.detect_on_yellow(yc)

        # ---- IN_LOOP(원): 안쪽(오른쪽)선만 앵커해 돎. min_loop 지나면 탈출 ----
        # 안쪽 중앙섬 선은 닫힌 원이라 출구서 안 갈라짐 → 안 끌려나가고 계속 돎.
        if self.rstate == 'IN_LOOP':
            if self.elapsed_in_loop() >= min_loop:   # ≈한 바퀴 시간 → 탈출
                self.set_state('EXIT')
                self.anchor_x_prev = None             # EXIT 에서 바깥선 재획득
            return self.follow_ring(yellow)           # anchor_side=None → 안쪽선

        # ---- EXIT: 바깥선만 앵커 → 출구서 갈라진 바깥선을 따라 밖으로 나감 ----
        # 흰선 확보(wr↑) 또는 노랑 소멸(yr↓) 또는 타임아웃이면 본선(LANE_FOLLOW) 복귀.
        exit_wr = float(self.get_parameter('exit_white_ratio').value)
        if (self._trans(wr >= exit_wr or yr <= exit_yr)
                or self.elapsed_in_state() >= exit_to):
            self.reset_roundabout()
            return self.detect_lane(edge)
        inner_side = 1 if int(self.get_parameter('loop_inner_side').value) >= 0 else -1
        return self.follow_ring(yellow, anchor_side=-inner_side)   # 바깥선 따라 탈출
    def detect_lane(self, edge):
        """ROI 안에서 행별로 좌/우 차선 x좌표를 찾아 '그 순간'의 차선 중심과
        offset/confidence 를 계산한다. 시간 평활은 하지 않는다."""
        height, width = edge.shape
        center_x = width / 2.0
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        roi_left = min(max(int(self.get_parameter('roi_left').value), 0), width - 1)

        if self.lane_width_px is None:
            self.lane_width_px = self.default_lane_width_ratio * width

        scan_ys = np.linspace(roi_top, height - 1, self.num_scan_rows).astype(int)

        # 좌/우 차선 점 검출: 화면 중심으로 자르지 않고, 행별 엣지 픽셀을 '선(클러스터)'
        # 으로 묶은 뒤 근거리(하단)에서 차선 수를 확정하고 위로 추적한다. 중앙 근처에
        # 걸친 한 개의 선이 좌/우 두 개로 쪼개지는 오분류(유령 반대선)를 막는다.
        left_raw, right_raw = self.scan_lanes(edge, scan_ys, roi_left, center_x, width)

        # 검출점에 다항식(곡선)을 피팅해 선에서 벗어난 엉뚱한 사물 점(이상치)을
        # 걸러내고, 차선을 정교한 곡선으로 표현한다.
        left_pts, left_poly = self.fit_and_filter(left_raw)
        right_pts, right_poly = self.fit_and_filter(right_raw)

        left_detected = len(left_pts) >= self.min_detect_rows
        right_detected = len(right_pts) >= self.min_detect_rows
        left_x = float(np.median([x for _, x in left_pts])) if left_detected else None
        right_x = float(np.median([x for _, x in right_pts])) if right_detected else None

        # --- 단일선 판별 (곡선에서 한 선이 중심을 가로질러 좌/우로 잘리는 문제) ---
        # 좌x·우x 간격이 실제 차선폭보다 훨씬 작으면 둘은 같은 선. 하나의 차선으로
        # 병합하고, 근거리(맨 아래) 위치가 중심의 어느 쪽인지로 좌/우를 판정한다.
        if left_detected and right_detected:
            ref_width = self.lane_width_px if self.lane_width_px else float(width)
            gap_ratio = float(self.get_parameter('single_line_gap_ratio').value)
            if (right_x - left_x) < gap_ratio * ref_width:
                all_pts = left_pts + right_pts
                _, x_near = max(all_pts, key=lambda p: p[0])  # 가장 아래(근거리) 점
                line_x = float(np.median([x for _, x in all_pts]))
                line_poly = left_poly if left_poly is not None else right_poly
                if x_near < center_x:      # 근거리에서 중심 왼쪽 -> 좌차선
                    left_detected, right_detected = True, False
                    left_x, right_x = line_x, None
                    left_pts, right_pts = all_pts, []
                    left_poly, right_poly = line_poly, None
                else:                       # 근거리에서 중심 오른쪽 -> 우차선
                    left_detected, right_detected = False, True
                    left_x, right_x = None, line_x
                    left_pts, right_pts = [], all_pts
                    left_poly, right_poly = None, line_poly

        # 필터된 점으로 per-row 차선중심 재구성 (단일선 병합 반영)
        left_map = {y: x for y, x in left_pts}
        right_map = {y: x for y, x in right_pts}
        center_pts = []
        for y in scan_ys:
            y = int(y)
            lx = left_map.get(y)
            rx = right_map.get(y)
            if lx is not None and rx is not None:
                center_pts.append((y, (lx + rx) / 2.0))
            elif lx is not None:
                center_pts.append((y, lx + self.lane_width_px / 2.0))
            elif rx is not None:
                center_pts.append((y, rx - self.lane_width_px / 2.0))

        # 양쪽 검출 시 차선폭 학습(EMA) — 기하 상태 추정(시간 평활 아님)
        if left_detected and right_detected and right_x > left_x:
            self.lane_width_px = 0.8 * self.lane_width_px + 0.2 * (right_x - left_x)

        # 차선폭을 안전 범위로 clamp -> 단일차선 half overshoot(반대선 침범) 방지.
        w_min = float(self.get_parameter('lane_width_min_ratio').value) * width
        w_max = float(self.get_parameter('lane_width_max_ratio').value) * width
        if w_max > w_min:
            self.lane_width_px = float(np.clip(self.lane_width_px, w_min, w_max))

        half = self.lane_width_px / 2.0
        if left_detected and right_detected:
            lane_center = (left_x + right_x) / 2.0
        elif left_detected:
            lane_center = left_x + half
        elif right_detected:
            lane_center = right_x - half
        else:
            lane_center = None

        detected_rows = len(center_pts)
        confidence = detected_rows / float(self.num_scan_rows)

        # raw offset: 그 순간의 정규화 횡오차. 미검출 시엔 0(=값 유지는 interpret 담당).
        if lane_center is not None:
            raw_offset = (lane_center - center_x) / (width / 2.0)
            raw_offset = float(np.clip(raw_offset, -1.0, 1.0))
        else:
            raw_offset = 0.0
            confidence = 0.0  # 완전 미검출: 신뢰도 0

        return {
            'raw_offset': raw_offset,
            'left_detected': left_detected,
            'right_detected': right_detected,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'lane_center': lane_center,
            'center_x': center_x,
            'image_width': int(width),
            'image_height': int(height),
            'left_pts': left_pts,
            'right_pts': right_pts,
            'left_poly': left_poly,
            'right_poly': right_poly,
        }

    def row_clusters(self, edge_row, roi_left, cluster_gap):
        """한 행의 엣지 픽셀을 x 간격 기준으로 묶어 클러스터 목록을 만든다.
        각 클러스터 = (mean_x, min_x, max_x). x(mean) 오름차순 정렬."""
        xs = np.where(edge_row[roi_left:] > 0)[0]
        if xs.size == 0:
            return []
        xs = np.sort(xs + roi_left)
        if xs.size == 1:
            x = int(xs[0])
            return [(float(x), x, x)]
        splits = np.where(np.diff(xs) > cluster_gap)[0]
        groups = np.split(xs, splits + 1)
        clusters = [(float(g.mean()), int(g.min()), int(g.max())) for g in groups]
        clusters.sort(key=lambda c: c[0])
        return clusters

    def scan_lanes(self, edge, scan_ys, roi_left, center_x, width):
        """행별 클러스터를 근거리(하단)→원거리(상단)로 추적해 좌/우 차선 점열을 만든다.

        목표: (1) 두 선이 있으면 둘 다 잡아 '두 선 사이 중앙'을 유지(정상 동작),
              (2) 한 선만 있으면(중앙에 걸쳐도) 유령 반대선을 만들지 않는다.

        - 매 행: 먼저 기존 좌/우 차선을 가장 가까운 클러스터에 track_tol 내에서
          매칭·갱신. 그다음 아직 없는 차선을 '충분히 떨어진(≥min_lane_sep)'
          미사용 클러스터에서 새로 시작한다 → 두 번째 선이 위쪽에서 늦게 나타나도
          받아들이되(정상 두 선 복원), 단일선은 행마다 클러스터가 하나뿐이라
          먼 미사용 클러스터가 없어 유령선이 생기지 않는다.
        - 좌차선 점은 안쪽 엣지(=오른쪽=max_x), 우차선은 안쪽(=왼쪽=min_x)을 기록해
          기존 캘리브레이션(차선폭/half) 관례를 유지한다."""
        cluster_gap = float(self.get_parameter('cluster_gap_px').value)
        track_tol = float(self.get_parameter('track_tol_px').value)
        min_lane_sep = float(self.get_parameter('min_lane_sep_ratio').value) * width

        left_raw, right_raw = [], []
        left_ref, right_ref = None, None  # 각 차선의 직전 행 mean x (추적 기준)

        def nearest_unused(ref, means, used):
            cand = [(abs(means[k] - ref), k) for k in range(len(means)) if k not in used]
            return min(cand)[1] if cand else None

        for y in sorted((int(v) for v in scan_ys), reverse=True):  # 근거리부터
            clusters = self.row_clusters(edge[y], roi_left, cluster_gap)
            if not clusters:
                continue
            means = [c[0] for c in clusters]
            used = set()

            # 1) 기존 차선 추적: 가장 가까운 미사용 클러스터를 tol 내에서 매칭
            if left_ref is not None:
                j = nearest_unused(left_ref, means, used)
                if j is not None and abs(means[j] - left_ref) <= track_tol:
                    left_ref = means[j]
                    left_raw.append((y, clusters[j][2]))  # 좌 안쪽 엣지 = max_x
                    used.add(j)
            if right_ref is not None:
                j = nearest_unused(right_ref, means, used)
                if j is not None and abs(means[j] - right_ref) <= track_tol:
                    right_ref = means[j]
                    right_raw.append((y, clusters[j][1]))  # 우 안쪽 엣지 = min_x
                    used.add(j)

            remaining = [k for k in range(len(clusters)) if k not in used]

            # 2) 아직 없는 차선을 '충분히 떨어진' 미사용 클러스터에서 시작
            if left_ref is None and right_ref is None:
                if len(remaining) >= 2 and \
                        (means[remaining[-1]] - means[remaining[0]]) >= min_lane_sep:
                    # 두 선 동시 씨앗 (최좌=좌, 최우=우)
                    a, b = remaining[0], remaining[-1]
                    left_ref, right_ref = means[a], means[b]
                    left_raw.append((y, clusters[a][2]))
                    right_raw.append((y, clusters[b][1]))
                elif remaining:
                    # 단일선(또는 붙은 덩어리): 한 덩어리로 보고 화면 중심 기준 한쪽만
                    all_min = min(clusters[k][1] for k in remaining)
                    all_max = max(clusters[k][2] for k in remaining)
                    m = 0.5 * (all_min + all_max)
                    if m < center_x:
                        left_ref = m
                        left_raw.append((y, all_max))
                    else:
                        right_ref = m
                        right_raw.append((y, all_min))
            elif left_ref is None:
                # 우차선만 있음 → 우차선보다 min_lane_sep 이상 왼쪽인 클러스터로 좌차선 시작
                cands = [k for k in remaining if right_ref - means[k] >= min_lane_sep]
                if cands:
                    k = min(cands, key=lambda k: means[k])
                    left_ref = means[k]
                    left_raw.append((y, clusters[k][2]))
            elif right_ref is None:
                # 좌차선만 있음 → 좌차선보다 min_lane_sep 이상 오른쪽인 클러스터로 우차선 시작
                cands = [k for k in remaining if means[k] - left_ref >= min_lane_sep]
                if cands:
                    k = max(cands, key=lambda k: means[k])
                    right_ref = means[k]
                    right_raw.append((y, clusters[k][1]))

        left_raw.sort()
        right_raw.sort()
        return left_raw, right_raw

    def fit_degree(self, n_points):
        """요청 차수를 파라미터에서 읽되, 점 수로 상한을 둔다(차수 = 점수-1 이하).
        근거리 밴드엔 기본 1차(직선)가 안정적."""
        req = int(self.get_parameter('line_fit_degree').value)
        return max(1, min(req, n_points - 1))

    def fit_and_filter(self, pts):
        """검출점들에 다항식 x=f(y)를 피팅(차선은 수직에 가까움)해 이상치를
        제거하고 (필터된 점 리스트, np.poly1d 또는 None)을 반환한다.
        점이 적으면 그대로 반환. 차수는 line_fit_degree 파라미터(기본 1차)."""
        if len(pts) < 3:
            return list(pts), None
        ys = np.array([p[0] for p in pts], dtype=np.float64)
        xs = np.array([p[1] for p in pts], dtype=np.float64)
        try:
            poly = np.poly1d(np.polyfit(ys, xs, self.fit_degree(len(pts))))
        except Exception:
            return list(pts), None

        resid = np.abs(xs - poly(ys))
        thresh = max(
            float(self.get_parameter('line_fit_outlier_px').value),
            2.5 * float(np.std(resid)),
        )
        keep = resid <= thresh
        if keep.all() or keep.sum() < 2:
            return list(pts), poly

        # 이상치 제거 후 1회 재피팅으로 선을 더 정교화
        ys2, xs2 = ys[keep], xs[keep]
        try:
            degree2 = self.fit_degree(int(ys2.size))
            poly = np.poly1d(np.polyfit(ys2, xs2, degree2))
        except Exception:
            pass
        filtered = [(int(y), int(x)) for y, x in zip(ys2, xs2)]
        return filtered, poly

    # ------------------------------------------------------------------ callbk
    def image_callback(self, msg: CompressedImage):
        edge = self.decode_edge(msg)
        if edge is None:
            return

        # 회전 교차로 FSM: 켜져 있고 노란 마스크가 준비됐으면 FSM 으로 처리.
        # 꺼져 있으면(기본) 기존 흰 차선 주행 그대로 → 무영향.
        yellow = self.latest_yellow
        if (bool(self.get_parameter('roundabout_enabled').value)
                and yellow is not None and yellow.shape == edge.shape):
            result = self.roundabout_step(edge, yellow)
        else:
            result = self.detect_lane(edge)

        if bool(self.get_parameter('debug_log').value):
            lc = result['lane_center']
            self.get_logger().info(
                f"rstate={self.rstate} jcnt={self.junction_count} "
                f"jactive={int(self.junction_active)} jscore={self.dbg_jscore:.2f} "
                f"yr={self.dbg_yr:.2f} wr={self.dbg_wr:.2f} "
                f"white={self.dbg_white} yellow={self.dbg_yellow} "
                f"lane_width_px={self.lane_width_px:.0f} "
                f"L={int(result['left_detected'])} R={int(result['right_detected'])} "
                f"lane_center={('%.0f' % lc) if lc is not None else 'None'} "
                f"raw_offset={result['raw_offset']:+.3f} conf={result['confidence']:.2f}",
                throttle_duration_sec=0.5,
            )

        detection = LaneDetection()
        detection.header.stamp = msg.header.stamp
        detection.header.frame_id = 'lane_detection'
        detection.image_width = result['image_width']
        detection.image_height = result['image_height']
        detection.center_x = float(result['center_x'])
        detection.lane_center_px = (
            float(result['lane_center']) if result['lane_center'] is not None else -1.0
        )
        detection.raw_offset = result['raw_offset']
        detection.left_detected = result['left_detected']
        detection.right_detected = result['right_detected']
        detection.confidence = result['confidence']
        # 회전로 FSM 상태를 하류로 전달(interpret 가 회전로 구간 감속에 사용).
        detection.drive_state = {
            'LANE_FOLLOW': 0, 'ENTER': 1, 'IN_LOOP': 2, 'EXIT': 3,
        }.get(self.rstate, 0)
        self.detection_pub.publish(detection)

        if self.debug_pub is not None:
            # 회전로 상태(ENTER/IN_LOOP/EXIT)에선 노란선을 따라가므로 디버그 배경을
            # 흰+노랑 합친 마스크로 그린다 → 빨간 앵커가 '어느 노란선'을 물었는지 보임.
            # (기존엔 흰 마스크 위에만 그려 노란선이 안 보였음)
            debug_base = edge
            if (self.rstate in ('ENTER', 'IN_LOOP', 'EXIT') and yellow is not None
                    and yellow.shape == edge.shape):
                debug_base = cv2.bitwise_or(edge, self.closed_yellow(yellow))
            self.publish_debug(debug_base, result, msg)

        # 회전로 튜닝: 닫힘 적용된 노란 마스크를 발행(점선 이어짐 확인용).
        if (self.yellow_closed_pub is not None and yellow is not None
                and yellow.shape == edge.shape):
            yc = self.closed_yellow(yellow)
            ok, enc = cv2.imencode(
                '.jpg', yc, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                out = CompressedImage()
                out.header.stamp = msg.header.stamp
                out.header.frame_id = 'yellow_closed'
                out.format = 'jpeg'
                out.data = enc.tobytes()
                self.yellow_closed_pub.publish(out)

    # ------------------------------------------------------------------- debug
    def publish_debug(self, edge, result, source_msg: CompressedImage):
        canvas = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        height, width = edge.shape
        center_x = int(result['center_x'])

        # ROI 상단 경계선(노랑)
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        cv2.line(canvas, (0, roi_top), (width, roi_top), (0, 255, 255), 1)
        # 이미지 중심선(흰색)
        cv2.line(canvas, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        # 좌/우 차선 검출점(좌=빨강, 우=파랑) — 참고용 작은 점
        for y, x in result['left_pts']:
            cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)
        for y, x in result['right_pts']:
            cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)
        # 피팅된 차선 곡선(좌=빨강, 우=파랑) — 정교한 실선
        for poly, color in ((result.get('left_poly'), (0, 0, 255)),
                            (result.get('right_poly'), (255, 0, 0))):
            if poly is None:
                continue
            ys = np.arange(roi_top, height)
            xs = np.clip(poly(ys), 0, width - 1).astype(np.int32)
            pts_line = np.stack([xs, ys.astype(np.int32)], axis=1)
            cv2.polylines(canvas, [pts_line], False, color, 2)
        # 차선 중심선(초록)
        if result['lane_center'] is not None:
            lc = int(result['lane_center'])
            cv2.line(canvas, (lc, roi_top), (lc, height), (0, 255, 0), 2)

        ok, encoded = cv2.imencode(
            '.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'lane_detection_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
