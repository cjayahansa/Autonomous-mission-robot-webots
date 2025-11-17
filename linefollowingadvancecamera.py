from controller import Robot
import numpy as np
import cv2

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')



left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0)
right_motor.setVelocity(0)

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()  

camera = robot.getDevice('camera')
camera.enable(timestep)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

arm_motor_1 = robot.getDevice('arm_motor_1')
arm_motor_1.setPosition(float('inf'))
arm_motor_1.setVelocity(0)

arm_l_motor_1 = robot.getDevice('arm_l_motor_1')
arm_l_motor_1.setPosition(float('inf'))
arm_l_motor_1.setVelocity(0)


arm_head_L = robot.getDevice('arm_head_L')
arm_head_L.setPosition(float('inf'))
arm_head_L.setVelocity(0)

arm_head_R = robot.getDevice('arm_head_R')
arm_head_R.setPosition(float('inf'))
arm_head_R.setVelocity(0)

def wait(robot, timestep, seconds):
    steps = int(seconds * 1000 / timestep)
    for _ in range(steps):
        robot.step(timestep)




# IR SENSORS
ir_sensors = []
for i in range(5):
    s = robot.getDevice(f'ps{i}')
    s.enable(timestep)
    ir_sensors.append(s)

# Threshold
threshold = 210.0

# PID
kp = 0.75
kd = 0.3
last_error = 0.0
base_speed = 3.0

# Weights for 5 sensors
weights = [-2, -1, 0, 1, 2]

# STATES
STATE_FOLLOW = 0
STATE_TURN_LEFT = 1
STATE_TURN_RIGHT = 2
STATE_U_TURN = 3
STATE_OBJECT_AHEAD = 4
STATE_MISSION = 5

state = STATE_FOLLOW


# ------------------------ 90° TURN DETECTOR -----------------------
def detect_90_bend(binary):
    # 90° LEFT bend patterns
    left_patterns = [
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1]
    ]

    # 90° RIGHT bend patterns
    right_patterns = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ]

    if binary in left_patterns:
        return "LEFT"

    if binary in right_patterns:
        return "RIGHT"
    return None

def detact_object_ahead(front):
    if front < 0.05:
        return True
    return False

# ------------------------ RED COLOR DETECTION -----------------------
def red_detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    red_pixels = cv2.countNonZero(mask)
   
    if red_pixels > 500 :
        return True
    return False


# ----------------------- MAIN LOOP ---------------------------------
while robot.step(timestep) != -1:

    lidar_data = lidar.getRangeImage()
    # print("LIDAR Ranges:", lidar_data[0:10])  # Print first 5 LIDAR ranges for debugging
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    frame = frame[:, :, :3]

    n = len(lidar_data)

    lidar_data = lidar.getRangeImage()
    n = len(lidar_data)

    # FRONT sector
    front = sum(lidar_data[n//2 - 5 : n//2 + 5]) / 10

    # RIGHT sector (index 0 → 100)
    right = sum(lidar_data[0 : 10]) / 10

    # LEFT sector  (last 100 samples)
    left = sum(lidar_data[n-10 : n]) / 10

    # print("front:", front)
    # print("left:", left)
    # print("right:", right)

    # Read IR sensors
    values = [s.getValue() for s in ir_sensors]
    binary = [1 if v > threshold else 0 for v in values]

    # Detect turns
    turn = detect_90_bend(binary)

    # PID error using binary
    error = sum(w * b for w, b in zip(weights, binary))

    # Derivative
    derivative = error - last_error
    last_error = error

    # PID output
    correction = (kp * error) + (kd * derivative)

    # ---------------- STATE MACHINE ----------------


    
            

    
    # FOLLOWING LINE
    if state == STATE_FOLLOW:
        print("STATE_FOLLOW")

        if detact_object_ahead(front):
            print("Obstacle detected ahead! Stopping.")
            state = STATE_OBJECT_AHEAD

        if turn == "LEFT":
            print("Detected 90° LEFT — Turning...")
            state = STATE_TURN_LEFT
            continue

        if turn == "RIGHT":
            print("Detected 90° RIGHT — Turning...")
            state = STATE_TURN_RIGHT
            continue

        if sum(binary) == 0:  # No sensors see line → U turn
            print("U-Turn Detected")
            state = STATE_U_TURN
            continue

            # Normal PID Line Following
        left_speed = base_speed - correction
        right_speed = base_speed + correction

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)


        # TURN LEFT (90°)
    elif state == STATE_TURN_LEFT:
        left_motor.setVelocity(-1.5)
        right_motor.setVelocity(6.0)

            # Center sensor sees line → done turn
        if binary[2] == 1:
            print("Finished LEFT turn — Resume")
            state = STATE_FOLLOW


        # TURN RIGHT (90°)
    elif state == STATE_TURN_RIGHT:
        left_motor.setVelocity(6.0)
        right_motor.setVelocity(-1.5)

        if binary[2] == 1:
            print("Finished RIGHT turn — Resume")
            state = STATE_FOLLOW


        # U-TURN
    elif state == STATE_U_TURN:
        left_motor.setVelocity(-1.5)
        right_motor.setVelocity(4.0)

        if binary[2] == 1:
            print("Finished U-TURN — Resume")
            state = STATE_FOLLOW
            

    elif state == STATE_OBJECT_AHEAD:
    
        print(right)
        if red_detect(frame):
            print("Red Object detected ahead! Stopping.")
            state = STATE_MISSION


        elif right < 0.065:
            print("Path clear on the left— Turning left")
            left_motor.setVelocity(4.0)
            right_motor.setVelocity(1.0)
            
            if binary[2] == 1:
                print(error)
                print("object following finish")
                left_motor.setVelocity(3.0)
                right_motor.setVelocity(1.0)
                state = STATE_FOLLOW

        elif right < 0.073:
            print("Path clear on the RIGHT — Turning RIGHT")
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(3.0)
            
            if binary[2] == 1:
                print(error)
                print("object following finish")
                left_motor.setVelocity(6.0)
                right_motor.setVelocity(1.0)
                state = STATE_FOLLOW

        elif right > 0.8:
            print("Path clear on the LEFT — Turning LEFT")
            left_motor.setVelocity(6.0)
            right_motor.setVelocity(-1.0)

    elif state == STATE_MISSION:
        left_motor.setVelocity(1.5)
        right_motor.setVelocity(1.5)

        if front < 0.02:
            print(front)
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            wait(robot, timestep, 1.0)
            arm_l_motor_1.setPosition(-0.06)
            arm_l_motor_1.setVelocity(0.01)
            
            wait(robot, timestep, 5.0)
            arm_head_L.setPosition(0.02)
            arm_head_L.setVelocity(0.01)
            
            
            arm_head_R.setPosition(-0.02)
            arm_head_R.setVelocity(0.01)
            
            wait(robot, timestep, 3.0)

                
            arm_motor_1.setPosition(1.57)
            arm_motor_1.setVelocity(0.5)
            
            arm_l_motor_1.setPosition(-0.03)
            arm_l_motor_1.setVelocity(0.01)
            
            wait(robot, timestep, 5.0)
            
            arm_l_motor_1.setPosition(-0.06)
            arm_l_motor_1.setVelocity(0.01)
            
            wait(robot, timestep, 4.0)
            
            arm_head_L.setPosition(0.0)
            arm_head_L.setVelocity(0.01)
            
            
            arm_head_R.setPosition(0.0)
            arm_head_R.setVelocity(0.01)
            
            wait(robot, timestep, 2.0)
            arm_l_motor_1.setPosition(0.0)
            arm_l_motor_1.setVelocity(0.01)
            
            wait(robot, timestep, 4.0)
        
            arm_motor_1.setPosition(0.0)
            arm_motor_1.setVelocity(0.5)
            
                
            wait(robot, timestep, 5.0)
            print("Mission Accomplished! Robot Stopped.")
            wait(robot, timestep, 1.0)
            
            state = STATE_FOLLOW

            
        
        
        
            
            
           
            
    