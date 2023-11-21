from pymodi.modi import *

CAR_SPEED = 50
CRANE_SPEED = 30
PUSH_SPEED = 50

class Control(MODI):
    def __init__(self, modi_version: int = 1, conn_type: str = "", verbose: bool = False, port=None, network_uuid: str = "", virtual_modules=None):
        super().__init__(modi_version, conn_type, verbose,
                         port, network_uuid, virtual_modules)
        self.CAR_MOTOR_FRONT = self.motors[0]
        self.CRANE_MOTER = self.motors[1]
        self.PUSH_MOTOR = self.motors[2]

    def car_go_straight(self):
        self._car_move(CAR_SPEED, CAR_SPEED)
    
    def car_go_back(self):
        self._car_move(-CAR_SPEED, -CAR_SPEED)

    def car_stop(self):
        self._car_move(0, 0)

    def crane_pull_up(self):
        self._crane_move(CRANE_SPEED);
    
    def crane_stop(self):
        self._crane_move(0)

    def push(self, t=2.5):
        self._push_move(PUSH_SPEED)
        sleep(t)
        self._push_move(0)

    def _car_move(self, left_speed, right_speed):
        self.CAR_MOTOR_FRONT.speed = left_speed, right_speed

    def _crane_move(self, speed):
        self.CRANE_MOTER.speed = speed, speed

    def _push_move(self, speed):
        self.PUSH_MOTOR.speed = speed, 0