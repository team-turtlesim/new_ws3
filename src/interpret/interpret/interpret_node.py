"""차선 판단+제어결정 노드 (interpret).

lane_detection 이 발행하는 '그 순간'의 인지값(LaneDetection, /lane/detection)을
구독하여, 한 콜백 안에서 두 가지를 연속으로 수행한다:

  1) 판단(시간필터/안전): offset EMA 저역통과. 결과를 LaneInfo(/lane_info)로도
     발행(디버그/rosbag 용; 현재 런타임 구독자는 없음).
  2) 제어결정(PID): offset 횡오차 PID 를 돌려 조향/스로틀을 계산하고
     Control(/control)로 발행. 하드웨어는 안 만진다(그건 control_node).

왜 인지(lane_detection)와 이 노드를 나누고, 예전의 lane_follow(제어)를 여기로
합쳤나:
  - 인지 = "지금 프레임에 선이 어디 있나"(기하). 그것만 lane_detection 이 한다.
  - 판단+제어결정 = "그걸 얼마나 믿고, 얼마나 꺾을까". 시간적 맥락이 필요한 이
    둘을 한 노드에 모아 한 콜백에서 처리한다.
  - 이렇게 하면 '프레임 도착 → 즉시 조향명령'이 되어(이벤트구동) 예전 lane_follow
    의 고정 20Hz 타이머가 만들던 최대 ~50ms 위상지연과 /lane_info 홉이 사라진다.

제어 방식(offset 전용):
  - 2026-07-06: heading(진행방향 기울기)은 이 셋업에서 주행 중 신뢰불가로 판명
    (종횡비 640x480->320x160 세로 1.5배 압축으로 기울기 증폭 + 점선 중앙선/비대칭
    검출 + ROI 외삽으로 직진에서도 heading 이 ±0.5 스파이크). 그래서 heading 기반
    곡선제어(선행조향/curve_bias)를 전부 걷어내고 offset 만으로 제어한다.
  - 직진: 순수 offset=0 중앙추종(PID). 곡선: offset 이 커지면(바깥 밀림) kp 를
    올리고(반응형) 감속해 라인을 유지. 곡선 감지도 |offset| 기반.

안전(페일세이프):
  - 이벤트구동이라 프레임이 끊기면 발행이 멈춘다. "명령이 끊기면 중립+정지"는
    control_node 의 stale 워치독이 담당한다.
  - 차선 신뢰도가 낮으면(lane lost) 스로틀을 0 으로 램프다운한다.
  - cruise_throttle 기본 0.0: 첫 기동엔 조향만 계산하고 차는 안 움직인다.
"""

import os
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from interface.msg import Control, LaneDetection, LaneInfo


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def smoothstep(x, lo, hi):
    """lo~hi 구간을 0~1 로 S자 보간(양끝 기울기 0 -> 경계에서 게인이 부드럽게).
    x<=lo -> 0, x>=hi -> 1. 하드 분기 대신 연속 블렌딩용."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


class InterpretNode(Node):
    def __init__(self):
        super().__init__('interpret_node')

        # --- Topics / IO ----------------------------------------------------
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('lane_topic', '/lane_info')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 판단: offset 시간필터 -----------------------------------------
        # offset 저역통과 필터 계수(0~1, 클수록 민감/덜 평활).
        self.declare_parameter('ema_alpha', 0.4)
        # 단독선(한쪽만 검출)일 때 offset 축소/클램프 실험용 훅.
        # 2026-07-04 트랙 테스트: 축소(scale=0.5)하면 오히려 위치보정이 죽어 직진을
        # 못했다 → 기본값 1.0(비활성)으로 되돌림. 두 선 중앙잡기가 정상 동작이므로
        # 단독선 offset도 그대로 신뢰한다. 훅은 남겨두되 함부로 낮추지 말 것.
        self.declare_parameter('single_line_offset_scale', 1.0)
        self.declare_parameter('single_line_offset_limit', 1.0)

        # --- 제어결정: 횡오차 PID (offset 전용) ----------------------------
        # logical steer: 양수 = 물리적으로 우회전(LaneInfo 규약: +offset 이면 차선
        # 중심이 이미지 오른쪽 -> 차가 왼쪽으로 치우침 -> 우회전). steer_sign 이
        # 이를 서보 퍼센트 극성으로 매핑(-1.0 은 이전 차량 인계값, 트랙에서 검증).
        # 2026-07-06 주행 측정으로 확정한 게인. 이벤트구동 전환 후 남은 직진
        # 뱀주행(리밋사이클, ~0.5Hz)을 실측 튜닝으로 잡음:
        #   원본(kp0.45/kd0.12/ki0.2) offset std 0.110 -> 최종 std 0.060(-45%).
        self.declare_parameter('kp_offset', 0.25)   # 0.45->0.25 루프게인 축소(리밋사이클)
        self.declare_parameter('kd_offset', 0.16)   # 0.12->0.16 감쇠 강화(offset 깨끗해 여유)
        # 적분: 원래 정상상태 offset 제거용이나, 저속 위치루프에선 위상지연을 키워
        # 뱀주행을 되살린다(실측: ki=0.2/0.05 모두 std~0.11, ki=0 은 0.06). 그래서
        # 기본 0 으로 비활성. 정상상태 편향은 steer_trim 이 담당(측정 잔차 ±0.03).
        # 긴 직선에서 드리프트가 문제되면 i_limit 을 크게 낮춰(≤0.1) 소량만 재도입.
        self.declare_parameter('ki_offset', 0.0)
        self.declare_parameter('i_limit', 0.3)       # clamp on |ki*integral|
        self.declare_parameter('steer_limit', 0.7)   # max |logical steer|
        self.declare_parameter('steer_sign', -1.0)
        # steer_trim(직진/중립 조향값)은 vehicle_config 에서 읽는다.
        # --- 평활(직진 twitch 억제) ---
        # d_offset: 검출이 잠깐 끊겼다 복귀할 때 offset 이 튀어 미분이 폭발 -> 클램프.
        # steer EMA 로 최종 명령을 저역통과.
        self.declare_parameter('d_offset_limit', 2.0)     # clamp |d(offset)/dt|
        self.declare_parameter('steer_smooth_alpha', 0.30)  # 1.0 = no smoothing
        # --- 게인 스케줄링(직진<->곡선 연속 블렌딩, |offset| 기준) ---
        # w = smoothstep(|offset|, lo, hi): 0=중앙(직진) ~ 1=크게 벌어짐(곡선/이탈).
        # w 로 kp 를 직진값<->곡선값 사이에서 연속 보간. 하드 분기의 경계 튐을 피하려
        # S자 블렌딩. offset 이 커지면(곡선에서 바깥 밀림) kp 를 올려 강하게 복구.
        # 직진 뱀주행(|offset|~0.1)은 lo(0.3) 아래라 안 건드림 -> 직진 튜닝 유지.
        # heading 기반 곡선감지는 신뢰불가로 제거, offset(항상 유효)만으로 감지.
        self.declare_parameter('kp_offset_curve', 0.45)  # 곡선 kp(offset 교정 강화)
        self.declare_parameter('sched_offset_lo', 0.3)   # 이 |offset| 부터 복구게인 시작
        self.declare_parameter('sched_offset_hi', 0.6)   # 이 |offset| 에서 완전 곡선게인

        # --- 스로틀 ---------------------------------------------------------
        self.declare_parameter('cruise_throttle', 0.0)  # 0 => 조향부터 검증
        self.declare_parameter('max_throttle', 0.30)
        self.declare_parameter('throttle_slew_per_sec', 0.6)  # ramp rate
        # 곡선 감속: 조향 스케줄과 같은 w 로 throttle 을 줄인다("코너에서 브레이크").
        # 2026-07-06: kp0.45 로 강하게 걸어도 급곡선은 속도가 빠르면 못 버티고
        # 가장자리를 스침 -> 감속해야 라인 유지.
        # throttle = cruise × lerp(1.0, curve_throttle_scale, w). w=1 에서 이 비율로.
        self.declare_parameter('curve_throttle_scale', 0.9)
        # 회전로 감속: lane_detection 의 drive_state(0=본선, 1~3=회전로 진입/회전/탈출)
        # 가 회전로 구간이면 스로틀에 이 배율을 곱해 강제 저속화한다. 진입로 다중선을
        # 안정적으로 인지·추종하려면 저속이라야 한다(고속에선 반응 프레임 부족으로
        # 이탈). 본선 주행(drive_state=0)엔 영향 없음(배율 1.0).
        self.declare_parameter('roundabout_throttle_scale', 0.5)
        # 회전(IN_LOOP) 스로틀 배율. 노란 진입로 직진 구간에서 코너 감속 뒤 다시
        # 속도를 내기 위해 기본 1.0(감속 해제). 링 진입이 빨라 불안정하면 낮춘다.
        self.declare_parameter('roundabout_loop_throttle_scale', 1.0)

        # --- 페일세이프 -----------------------------------------------------
        self.declare_parameter('min_confidence', 0.2)   # 미만 -> lost 취급
        self.declare_parameter('debug_log', False)

        detection_topic = str(self.get_parameter('detection_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('ema_alpha must be in range (0, 1]')

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.steer_trim = self.load_steer_trim()

        # --- 내부 상태: 판단(시간필터) ------------------------------------
        self.offset_filtered = 0.0     # 필터링된 lane_offset (미검출 시 마지막 값 유지)

        # --- 내부 상태: 제어(PID/평활/슬루) -------------------------------
        self.prev_offset_for_d = 0.0
        self.prev_time = None           # 미분/슬루 dt 계산용 (콜백 간 시간)
        self.throttle_cmd = 0.0
        self.steer_cmd_filtered = self.steer_trim  # EMA-smoothed 조향 출력
        self.was_low_conf = False       # 직전 프레임 신뢰도 미달?
        self.integral = 0.0             # offset 오차 적분(I 항)

        self.subscription = self.create_subscription(
            LaneDetection,
            detection_topic,
            self.detection_callback,
            10,
        )
        self.lane_pub = self.create_publisher(LaneInfo, lane_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        self.get_logger().info(
            'interpret node started (judgment + control law, offset-only):\n'
            f'  detection_topic={detection_topic}\n'
            f'  lane_topic={lane_topic}\n'
            f'  control_topic={control_topic}\n'
            f'  steer_trim={self.steer_trim} steer_sign='
            f'{float(self.get_parameter("steer_sign").value)}\n'
            f'  ema_alpha={self.ema_alpha}\n'
            f'  kp={float(self.get_parameter("kp_offset").value)}'
            f'(curve {float(self.get_parameter("kp_offset_curve").value)}) '
            f'kd={float(self.get_parameter("kd_offset").value)} '
            f'ki={float(self.get_parameter("ki_offset").value)}\n'
            f'  cruise_throttle={float(self.get_parameter("cruise_throttle").value)} '
            f'(0 => 조향만; 검증 후 param 으로 올릴 것)'
        )

    # ------------------------------------------------------------------ callbk
    def detection_callback(self, msg: LaneDetection):
        """프레임 도착 즉시: 판단(offset 시간필터) -> LaneInfo 발행 -> PID -> Control."""
        detected = bool(msg.left_detected or msg.right_detected)
        # 단독선 = 좌/우 중 정확히 한쪽만 검출 (XOR)
        single_line = bool(msg.left_detected) != bool(msg.right_detected)

        offset = self.filter_offset(float(msg.raw_offset), detected, single_line)
        confidence = float(msg.confidence)
        # 회전로 감속용 FSM 상태. 구버전(필드 없는) msg 에도 안전하게 0 fallback.
        drive_state = int(getattr(msg, 'drive_state', 0))

        # 1) 판단 결과를 LaneInfo 로도 발행(디버그/rosbag; 런타임 구독자는 없음).
        lane_info = LaneInfo()
        lane_info.header.stamp = msg.header.stamp
        lane_info.header.frame_id = 'interpret'
        lane_info.lane_offset = offset
        lane_info.left_detected = bool(msg.left_detected)
        lane_info.right_detected = bool(msg.right_detected)
        lane_info.confidence = confidence
        self.lane_pub.publish(lane_info)

        # 2) 제어결정(PID) -> Control 발행.
        self.run_control(offset, confidence, drive_state)

    # ------------------------------------------------------------------ filters
    def filter_offset(self, raw_offset, detected, single_line):
        """검출 시 EMA 저역통과, 미검출 시 마지막 필터값 유지.
        단독선일 때는 offset 을 축소·클램프해 신뢰도를 낮춘다(과대/요동 억제)."""
        if single_line:
            scale = float(self.get_parameter('single_line_offset_scale').value)
            limit = float(self.get_parameter('single_line_offset_limit').value)
            raw_offset = max(-limit, min(limit, raw_offset * scale))
        if detected:
            self.offset_filtered = (
                self.ema_alpha * raw_offset
                + (1.0 - self.ema_alpha) * self.offset_filtered
            )
        # 미검출: self.offset_filtered 를 직전 값 그대로 유지
        return float(max(-1.0, min(1.0, self.offset_filtered)))

    # ------------------------------------------------------------------ control
    def run_control(self, offset, confidence, drive_state=0):
        """판단된 offset/confidence 로 offset PID 를 돌려 조향/스로틀을 계산하고
        Control 을 발행한다. 프레임 도착마다 호출(이벤트구동).
        dt 는 콜백 간 실제 경과시간(프레임 간격)을 쓴다."""
        now = self.get_clock().now()
        if self.prev_time is None:
            dt = 1.0 / 30.0   # 첫 프레임 가정치(카메라 ~30fps)
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 30.0
        self.prev_time = now

        kp_straight = float(self.get_parameter('kp_offset').value)
        kd = float(self.get_parameter('kd_offset').value)
        ki = float(self.get_parameter('ki_offset').value)
        i_limit = float(self.get_parameter('i_limit').value)
        kp_curve = float(self.get_parameter('kp_offset_curve').value)
        sched_off_lo = float(self.get_parameter('sched_offset_lo').value)
        sched_off_hi = float(self.get_parameter('sched_offset_hi').value)
        steer_limit = float(self.get_parameter('steer_limit').value)
        steer_sign = float(self.get_parameter('steer_sign').value)
        d_limit = float(self.get_parameter('d_offset_limit').value)
        alpha = clip(float(self.get_parameter('steer_smooth_alpha').value), 0.05, 1.0)
        min_conf = float(self.get_parameter('min_confidence').value)

        low_conf = confidence < min_conf

        # --- 게인 스케줄링: |offset| 로 직진<->곡선 블렌딩(반응형). ---
        # offset 이 커지면(=곡선/이탈로 바깥 밀림) kp 부스트 + 감속으로 반응 복구.
        # 직진(|offset|~0.1)은 lo 아래라 kp=직진값 유지(뱀주행 튜닝 보존).
        w = smoothstep(abs(offset), sched_off_lo, sched_off_hi)
        kp = lerp(kp_straight, kp_curve, w)

        error = offset  # 목표 = 차선중앙(offset=0)

        # --- 미분(클램프); 검출 복귀 프레임엔 리셋해 슬램 방지. ---
        if self.was_low_conf and not low_conf:
            # 방금 차선 재획득: offset 이 정지 HOLD 값에서 튀어 이 프레임의 raw
            # 미분은 garbage -> 억제.
            self.prev_offset_for_d = offset
        d_offset = clip((offset - self.prev_offset_for_d) / dt, -d_limit, d_limit)
        self.prev_offset_for_d = offset
        self.was_low_conf = low_conf

        # --- 적분(anti-windup): 추종 중 & 전진 중일 때만 누적; 아니면 리셋. ---
        if low_conf or self.throttle_cmd <= 0.0:
            self.integral = 0.0
        elif ki > 0.0:
            self.integral += error * dt
            self.integral = clip(self.integral, -i_limit / ki, i_limit / ki)
        i_term = clip(ki * self.integral, -i_limit, i_limit)

        # --- PID 후 EMA 저역통과로 부드러운 출력. ---
        logical = kp * error + i_term + kd * d_offset
        logical = clip(logical, -steer_limit, steer_limit)
        steering_raw = clip(self.steer_trim + steer_sign * logical, -1.0, 1.0)
        self.steer_cmd_filtered = (
            alpha * steering_raw + (1.0 - alpha) * self.steer_cmd_filtered
        )
        steering = clip(self.steer_cmd_filtered, -1.0, 1.0)

        # --- 스로틀: lane lost 아니면 cruise; 곡선(w↑) 감속; 슬루 제한. ---
        cruise = float(self.get_parameter('cruise_throttle').value)
        max_throttle = float(self.get_parameter('max_throttle').value)
        slew = float(self.get_parameter('throttle_slew_per_sec').value)
        curve_thr_scale = float(self.get_parameter('curve_throttle_scale').value)

        throttle_scale = lerp(1.0, curve_thr_scale, w)
        # 회전로 속도 프로파일(FSM drive_state):
        #   ENTER(1)/EXIT(3) = 코너·전환이라 감속(roundabout_throttle_scale),
        #   IN_LOOP(2) = 노란 직진 접근이라 속도 회복(roundabout_loop_throttle_scale),
        #   본선(0) = 무영향(1.0).
        if drive_state == 2:        # IN_LOOP: 직진 접근 → 다시 속도
            rb_scale = float(self.get_parameter('roundabout_loop_throttle_scale').value)
        elif drive_state != 0:      # ENTER/EXIT: 코너·전환 → 감속
            rb_scale = float(self.get_parameter('roundabout_throttle_scale').value)
        else:
            rb_scale = 1.0
        target_throttle = (0.0 if low_conf
                           else clip(cruise * throttle_scale * rb_scale, 0.0, max_throttle))

        step = slew * dt
        if target_throttle > self.throttle_cmd:
            self.throttle_cmd = min(self.throttle_cmd + step, target_throttle)
        else:
            self.throttle_cmd = max(self.throttle_cmd - step, target_throttle)

        self.publish_control(steering, self.throttle_cmd)

        if bool(self.get_parameter('debug_log').value):
            self.get_logger().info(
                f'off={offset:+.3f} i={i_term:+.3f} d={d_offset:+.3f} '
                f'conf={confidence:.2f} w={w:.2f} kp={kp:.2f} '
                f'-> steer={steering:+.3f} thr={self.throttle_cmd:.3f} rb={drive_state}'
            )

    def publish_control(self, steering, throttle):
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        self.control_pub.publish(msg)

    # ------------------------------------------------------------------ config
    def load_steer_trim(self):
        if not os.path.exists(self.vehicle_config_file):
            return 0.0
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return 0.0
        return float(config_data.get('STEER_TRIM', 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = InterpretNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
