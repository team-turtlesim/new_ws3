import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class OpenCvNode(Node):
    """카메라 압축영상을 받아 차선용 엣지영상을 발행하는 전처리 노드.

    카메라는 YOLO 를 위해 640x480 으로 발행하지만, 차선 파이프라인의 ROI/px 튜닝값은
    저해상도(LANE_PROC, 기본 320x160) 기준이다. 그래서 여기서 먼저 그 크기로
    다운스케일한 뒤 Canny 를 돌린다 -> 차선 튜닝 보존 + Canny/인코딩 부하도 저해상도 유지.

    grayscale/blur 는 디버그용이라 기본적으로 발행하지 않는다(프레임당 JPEG 인코딩
    2회 절약). 필요하면 publish_debug_streams:=true 로 켠다.
    """

    def __init__(self):
        super().__init__('opencv_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_log', True)
        # gray/blur 발행 여부(디버그용). 기본 off 로 CPU 절약.
        self.declare_parameter('publish_debug_streams', False)

        # 차선 추출 방식: 'color'(흰/노랑 HSV 색 임계값, 기본) 또는 'edge'(Canny).
        # 2026-07-05 트랙 테스트에서 color 가 차선을 더 안정적으로 잡아 기본값으로 채택.
        # 실시간 토글: ros2 param set /opencv_node detect_mode edge
        self.declare_parameter('detect_mode', 'color')
        # 흰색(HSV): 채도 낮고 명도 높음. 조명 밝으면 white_v_min 을 낮춘다.
        # 2026-07-05 튜닝: 200/40 은 너무 엄격해 흰 차선이 깜빡여서 180/55 로 완화.
        self.declare_parameter('white_s_max', 55)
        self.declare_parameter('white_v_min', 180)
        # 노란색(HSV): H 18~38 부근. OpenCV 는 H 가 0~179 임에 주의.
        self.declare_parameter('yellow_h_min', 18)
        self.declare_parameter('yellow_h_max', 38)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_v_min', 80)

        # --- 회전 교차로(중앙 노란 링) 지원 -------------------------------
        # 트랙은 색으로 길을 구분한다: 흰색=본선 차선, 노란색=회전로.
        # 기존엔 white|yellow 를 OR 로 합쳐 하나로 냈지만, 회전로 진입/주행/탈출을
        # 판단하려면 흰/노랑을 분리해야 한다.
        #   edge_color: /opencv/image/edge 에 실을 마스크.
        #     'white'(본선 차선만·기본) | 'yellow'(회전로만) | 'both'(기존처럼 합침)
        #   평소 구간엔 노란색이 없어 white 와 both 가 같으므로 기본 white 로 둬도
        #   기존 차선주행은 안 바뀐다(회전로 근처에서만 달라짐).
        self.declare_parameter('edge_color', 'white')
        # /opencv/image/yellow 로 노란색 전용 마스크를 따로 발행(lane_detection 이
        # 회전로 FSM 에 사용). edge 모드(Canny)에선 색 정보가 없어 발행 안 함.
        # 원본(점선 안 이어붙인) 마스크를 발행한다 — 점선 잇기(모폴로지 닫힘)는
        # lane_detection 에서 '중앙잡기 입력'에만 국소 적용하고, 출구 '선 개수 세기'
        # 는 원본이라야 선이 안 뭉쳐 정확하다(닫힘↔카운트 자기모순 해소).
        self.declare_parameter('publish_yellow', True)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.publish_debug_streams = bool(self.get_parameter('publish_debug_streams').value)
        self.publish_yellow = bool(self.get_parameter('publish_yellow').value)

        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')

        # 차선 처리 해상도. config 값을 기본으로 하되 param 으로도 노출/오버라이드.
        proc_w, proc_h = self.load_lane_proc_size()
        self.declare_parameter('lane_proc_width', proc_w)
        self.declare_parameter('lane_proc_height', proc_h)
        self.lane_proc_width = int(self.get_parameter('lane_proc_width').value)
        self.lane_proc_height = int(self.get_parameter('lane_proc_height').value)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            subscribe_topic,
            self.image_callback,
            image_qos,
        )

        # 차선검출이 실제로 쓰는 건 edge 뿐. 항상 발행.
        self.edge_pub = self.create_publisher(
            CompressedImage,
            '/opencv/image/edge',
            image_qos,
        )
        # 노란색 전용 마스크(회전로 FSM 용) -> 켜졌을 때만 퍼블리셔 생성.
        self.yellow_pub = None
        if self.publish_yellow:
            self.yellow_pub = self.create_publisher(
                CompressedImage, '/opencv/image/yellow', image_qos,
            )

        # gray/blur 는 디버그용 -> 켜졌을 때만 퍼블리셔 생성.
        self.gray_pub = None
        self.blur_pub = None
        if self.publish_debug_streams:
            self.gray_pub = self.create_publisher(
                CompressedImage, '/opencv/image/grayscale', image_qos,
            )
            self.blur_pub = self.create_publisher(
                CompressedImage, '/opencv/image/blur', image_qos,
            )

        self.get_logger().info(
            'OpenCV node started:\n'
            f'  subscribe_topic={subscribe_topic}\n'
            f'  detect_mode={str(self.get_parameter("detect_mode").value)}\n'
            f'  lane_proc={self.lane_proc_width}x{self.lane_proc_height}\n'
            f'  publish_debug_streams={self.publish_debug_streams}\n'
            f'  jpeg_quality={self.jpeg_quality}'
        )

    def load_lane_proc_size(self):
        default_size = (320, 160)
        if not os.path.exists(self.vehicle_config_file):
            return default_size
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(f'Failed to read vehicle config: {exc}')
            return default_size
        width = int(config_data.get('LANE_PROC_WIDTH', default_size[0]))
        height = int(config_data.get('LANE_PROC_HEIGHT', default_size[1]))
        return width, height

    def white_yellow_masks(self, bgr):
        """흰색 차선 마스크와 노란색(회전로) 마스크를 '분리해서' 반환한다.

        Canny 는 모든 경계(그림자·트랙끝·다른 차)를 다 잡아 노이즈가 많지만,
        색 임계값은 차선 색만 골라내 더 안정적일 수 있다(대신 조명 변화에 민감).
        임계값은 파라미터로 노출 -> venue 조명에서 라이브 튜닝 가능.

        흰색=본선 차선, 노란색=중앙 회전로. 회전로에서 진입/주행/탈출을 판단하려면
        두 색을 합치지 말고 각각 돌려줘야 한다. 노란 마스크는 '원본'을 돌려준다
        (점선 잇기는 lane_detection 이 중앙잡기용으로만 국소 적용).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gp = self.get_parameter

        # 흰색: H 전체, S 낮음, V 높음
        white = cv2.inRange(
            hsv,
            np.array([0, 0, int(gp('white_v_min').value)], dtype=np.uint8),
            np.array([179, int(gp('white_s_max').value), 255], dtype=np.uint8),
        )
        # 노란색: H 특정 대역, S/V 충분
        yellow = cv2.inRange(
            hsv,
            np.array([
                int(gp('yellow_h_min').value),
                int(gp('yellow_s_min').value),
                int(gp('yellow_v_min').value),
            ], dtype=np.uint8),
            np.array([int(gp('yellow_h_max').value), 255, 255], dtype=np.uint8),
        )
        # 점잡음 제거 (원본 노란 마스크 그대로 발행)
        white = cv2.medianBlur(white, 5)
        yellow = cv2.medianBlur(yellow, 5)
        return white, yellow

    def color_lane_mask(self, bgr):
        """흰색+노란색을 합친 이진 마스크(하위호환용). 신규 코드는
        white_yellow_masks 로 분리해 쓰는 것을 권장."""
        white, yellow = self.white_yellow_masks(bgr)
        return cv2.bitwise_or(white, yellow)

    def to_compressed_msg(self, image, source_msg: CompressedImage, frame_id: str):
        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning(f'Failed to encode image for frame_id={frame_id}')
            return None

        out_msg = CompressedImage()
        out_msg.header.stamp = source_msg.header.stamp
        out_msg.header.frame_id = frame_id
        out_msg.format = 'jpeg'
        out_msg.data = encoded.tobytes()
        return out_msg

    def image_callback(self, msg: CompressedImage):
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        np_arr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        if np_arr is None:
            self.get_logger().warning('Failed to decode compressed image')
            return

        # 차선 처리 해상도로 다운스케일(카메라가 640x480 이어도 여기서 320x160 으로).
        # 이후 Canny/인코딩이 전부 저해상도에서 돌아 차선 튜닝값(ROI/px)이 그대로 유효.
        if self.lane_proc_width > 0 and self.lane_proc_height > 0 and (
            np_arr.shape[1] != self.lane_proc_width
            or np_arr.shape[0] != self.lane_proc_height
        ):
            np_arr = cv2.resize(
                np_arr,
                (self.lane_proc_width, self.lane_proc_height),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 차선 이진영상 생성: color 모드면 흰/노랑 분리, 아니면 Canny 엣지.
        # color 모드에선 edge_color 로 고른 마스크만 /edge 에 싣고, 노란색 전용
        # 마스크는 /opencv/image/yellow 로 따로 보낸다(회전로 FSM 용).
        mode = str(self.get_parameter('detect_mode').value)
        yellow_mask = None
        if mode == 'color':
            white_mask, yellow_mask = self.white_yellow_masks(np_arr)
            edge_color = str(self.get_parameter('edge_color').value)
            if edge_color == 'both':
                edge = cv2.bitwise_or(white_mask, yellow_mask)
            elif edge_color == 'yellow':
                edge = yellow_mask
            else:  # 'white'
                edge = white_mask
        else:
            edge = cv2.Canny(blur, 50, 150)

        edge_msg = self.to_compressed_msg(edge, msg, frame_id='opencv_edge')
        if edge_msg is None:
            return
        self.edge_pub.publish(edge_msg)

        # 노란색 전용 마스크 발행(color 모드에서만 존재).
        if self.yellow_pub is not None and yellow_mask is not None:
            yellow_msg = self.to_compressed_msg(yellow_mask, msg, frame_id='opencv_yellow')
            if yellow_msg is not None:
                self.yellow_pub.publish(yellow_msg)

        # gray/blur 는 디버그용 -> 켜졌을 때만 인코딩/발행(평소 CPU 절약).
        if self.publish_debug_streams:
            gray_msg = self.to_compressed_msg(gray, msg, frame_id='opencv_grayscale')
            blur_msg = self.to_compressed_msg(blur, msg, frame_id='opencv_blur')
            if gray_msg is not None:
                self.gray_pub.publish(gray_msg)
            if blur_msg is not None:
                self.blur_pub.publish(blur_msg)

        if self.debug_log:
            self.get_logger().info('Published edge frame')


def main(args=None):
    rclpy.init(args=args)
    node = OpenCvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
