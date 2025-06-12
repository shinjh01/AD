# 자율주행 시스템

## 1. 시스템 아키텍처 개요
```
Raspberry Pi 5와 ROS2 Framework을 중심으로 구성된 임베디드 시스템으로 실시간 이미지 처리와 모터 제어를 통해 자율주행을 구현합니다. 

ROS2 Docker 컨테이너에서 실행되는 자율주행 노드가 카메라 데이터를 처리하고, ESP32를 통해 메카넘 휠을 제어합니다. GPIO를 통한 버튼제어와 LED 점등은 사용자 인터페이스로 활용됩니다.
```

## 2. 계층 구조

### 하드웨어 계층 (Hardware Layer)
- **Raspberry Pi 5**: 메인 컴퓨팅 장치
- **ESP32**: 모터 제어용 마이크로컨트롤러
- **Depth /RGB Camera**: 자율 주행 인식용 카메라 모듈
- **MentorPi Mecanum**: 전방향 이동이 가능한 메카넘 휠
- **버튼/LED 커스텀 빵판**: 사용자 인터페이스 및 상태 표시

### 운영체제 계층 (OS Layer)
- **Ubuntu**: Raspberry Pi 5에 설치된 기본 운영체제
- **ROS2 Docker Container**: 로봇 운영체제 환경

### 애플리케이션 계층 (Application Layer)
- **Self Driving Node**: 자율주행 메인 로직
- **YOLOv5**: 객체 인식 (신호등, 표지판, 횡단보도)
- **OpenCV Lane Detect**: 이미지 처리 및 차선 인식
- **GPIO Zero**: GPIO 인터페이스 라이브러리


## 3. 구성 요소 상호작용
1. **Raspberry Pi와 ESP32**는 UART 또는 I2C로 연결되어 명령을 전달합니다.
2. **Depth / RGB Camera**는 Raspberry Pi에 연결되어 영상 데이터를 제공합니다.
3. **ROS2 환경**에서 각 노드 간 Pub/Sub 구조를 통해 통신합니다.
4. **Self Driving Node**가 카메라 노드를 구독하여 이미지를 받고 분석 후 결과에 따라 ESP32를 통해 메카넘 휠과 LED를 제어합니다.
6. **GPIO Zero** 라이브러리를 통해 시작/종료를 제어하고 **LED**로 상태를 표시합니다.


## 4. 시스템 아키텍처

<div align="center">
    <img src="HLD.jpg" alt="HLD" />
</div>
<br/>



```mermaid
graph TD
    %% 하드웨어 계층
    subgraph "Hardware Layer"
        RP[Raspberry Pi 5]
        ESP[ESP32]
        CAM[Depth / RGB Camera]
        WHEEL[MentorPi Mecanum Wheels]
        GPIO_HW[GPIO Buttons/LEDs]
        
        RP --- ESP
        RP --- CAM
        ESP --- WHEEL
        RP --- GPIO_HW
    end
    
    %% 운영체제 계층
    subgraph "OS Layer"
        UBUNTU[Ubuntu OS]
        DOCKER[Docker]
        ROS2[ROS2 Container]
        
        UBUNTU --- RP
        DOCKER --- UBUNTU
        ROS2 --- DOCKER
    end
    
    %% 애플리케이션 계층
    subgraph "Application Layer"
        SD[Self Driving Node]
        YOLO[YOLOv5]
        CV[OpenCV]
        GPIO_SW[GPIO Zero]
        
        SD --- ROS2
        YOLO --- SD
        CV --- SD
        GPIO_SW --- SD
        
        %% 동작 흐름
        CAM -.-> SD
        SD -.-> ESP
        GPIO_HW -.-> GPIO_SW
        GPIO_SW -.-> GPIO_HW
    end
    
    %% 스타일 정의
    classDef hardware fill:#ffcdd2,stroke:#c62828
    classDef os fill:#bbdefb,stroke:#0d47a1
    classDef app fill:#c8e6c9,stroke:#2e7d32
    
    class RP,ESP,CAM,WHEEL,GPIO_HW hardware
    class UBUNTU,DOCKER,ROS2 os
    class SD,YOLO,CV,GPIO_SW app
```

## 플로우

```mermaid
graph TD
    %% 액터 정의
    User[사용자] --> Start[Start_Self-Driving]
    User --> Stop[Stop_Self-Driving]
    Start --> ObjectDetection[Object_Detection]
    Start --> LineDetection[Line_Detection]
    Start --> LEDControlStart[Control_LEDs_Start]
    ObjectDetection --> LEDControl[Control_LEDs]
    ObjectDetection --> MecanumControl[Control_Mecanum_Wheels]
    LineDetection --> LEDControl
    LineDetection --> MecanumControl
    Stop --> LEDControlStop[Control_LEDs_Stop]
    Stop --> MecanumStop[Stop_Mecanum_Wheels]
```

##  시퀀스 다이어그램

```mermaid
sequenceDiagram
    participant User as 사용자
    participant GPIO as GPIO 버튼/LED
    participant SelfDrivingNode as Self Driving Node
    participant Camera as Depth Camera
    participant YOLOv5 as YOLOv5 Node
    participant Mecanum as Mecanum Wheels

    User->>GPIO: 시작 버튼 누름
    GPIO->>SelfDrivingNode: 시작 신호 전달
    SelfDrivingNode->>Camera: 카메라 데이터 요청
    Camera-->>SelfDrivingNode: 카메라 데이터 전송
    SelfDrivingNode->>YOLOv5: 객체 감지 요청
    YOLOv5-->>SelfDrivingNode: 감지 결과 반환
    alt 횡단보도 감지
        SelfDrivingNode->>Mecanum: 정지 명령
        SelfDrivingNode->>GPIO: 붉은색 LED 점등
    else 곡선 차선 감지
        SelfDrivingNode->>Mecanum: 우회전 명령
        SelfDrivingNode->>GPIO: 노란색 LED 점등
    else 신호등 감지
        alt 신호등이 빨간색
            SelfDrivingNode->>Mecanum: 정지 명령
            SelfDrivingNode->>GPIO: 붉은색 LED 점등
        else 신호등이 녹색
            SelfDrivingNode->>Mecanum: 진행 명령
            SelfDrivingNode->>GPIO: 파란색 LED 점등
        end
    else 주차 표지판 감지
        SelfDrivingNode->>Mecanum: 주차 명령
        SelfDrivingNode->>GPIO: 모든 LED 끔
    end
    User->>GPIO: 종료 버튼 누름
    GPIO->>SelfDrivingNode: 종료 신호 전달
    SelfDrivingNode->>Mecanum: 정지 명령
    SelfDrivingNode->>GPIO: 모든 LED 끔
```