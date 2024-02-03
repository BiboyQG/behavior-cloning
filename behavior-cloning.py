import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import RPi.GPIO as GPIO
import keyboard

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
motor1 = 13
motor2 = 12
GPIO.setup(motor1, GPIO.OUT)
GPIO.setup(motor2, GPIO.OUT)
motor1Servo = GPIO.PWM(motor1, 50)
motor1Servo.start(8)
motor2Servo = GPIO.PWM(motor2, 50)
motor2Servo.start(8)
motor1Servo.ChangeDutyCycle(7.5)
motor2Servo.ChangeDutyCycle(7.5)


def servoControl(value):
    motor1Servo.ChangeDutyCycle(7.5 + value)
    motor2Servo.ChangeDutyCycle(7.5 - value)


class Agent:
    def __init__(self):
        self.userSteering = 0
        self.aiMode = False
        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (7, 7),
                    input_shape=(240, 320, 3),
                    strides=(2, 2),
                    activation="relu",
                    padding="same",
                ),
                MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding="valid"),
                Conv2D(64, (4, 4), activation="relu", strides=(1, 1), padding="same"),
                MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding="valid"),
                Conv2D(128, (4, 4), strides=(1, 1), activation="relu", padding="same"),
                MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding="valid"),
                Flatten(),
                Dense(384, activation="relu"),
                Dense(64, activation="relu", name="layer1"),
                Dense(8, activation="relu", name="layer2"),
                Dense(1, activation="linear", name="layer3"),
            ]
        )
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.05))
        self.cap = cv2.VideoCapture(
            0
        ) 
        self.cap.set(
            3, 320
        )
        self.cap.set(4, 240)

    def act(self, state):
        state = np.reshape(state, (1, 240, 320, 3))
        action = self.model.predict(state)[0][0]
        action = (action * 2) - 1
        servoControl(action)
        return action

    def learn(self, state, action):
        state = np.reshape(state, (1, 240, 320, 3))
        history = self.model.fit(state, [action], batch_size=1, epochs=1, verbose=0)
        print("LOSS: ", history.history.get("loss")[0])

    def getState(self):
        ret, frame = self.cap.read()
        pic = np.array(frame)
        processedImg = np.reshape(pic, (240, 320, 3)) / 255
        return processedImg

    def observeAction(self):
        return (self.userSteering + 1) / 2


agent = Agent()

def handle_key_input():
    if keyboard.is_pressed('left'):  # If the left arrow key is pressed
        print("Steering left")
        agent.userSteering = -1  # Update steering to left
        servoControl(agent.userSteering)
    elif keyboard.is_pressed('right'):  # If the right arrow key is pressed
        print("Steering right")
        agent.userSteering = 1  # Update steering to right
        servoControl(agent.userSteering)
    elif keyboard.is_pressed('up'):  # Toggle AI mode with the up arrow key
        agent.aiMode = not agent.aiMode
        print(f"AI Mode {'Enabled' if agent.aiMode else 'Disabled'}")

counter = 0
while True:
    handle_key_input()
    if agent.aiMode == False: #AI's Learning mode
        start = time.time()
        state = agent.getState()
        action = agent.observeAction()
        counter += 1
        if counter % 1 == 0: #To ensure generalization, we don't let AI learn every iteration
            start = time.time()
            agent.learn(state, action)
            agent.memory = []
        if counter % 50 == 0: #Frequency to save its weights
           agent.model.save_weights("selfdrive.h5")
        print("framerate: ", 1/(time.time() - start))
    else: 
        while agent.aiMode == True: #Autonomous loop
            start = time.time() 
            state = agent.getState()
            action = agent.act(state)
            print("action", action)
            print("framerate: ", 1/(time.time() - start))
