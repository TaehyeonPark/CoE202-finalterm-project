from pymodi.modi import *
from time import sleep


class Control(MODI):
    def __init__(self, modi_version: int = 1, conn_type: str = "", verbose: bool = False, port=None, network_uuid: str = "", virtual_modules=None):
        super().__init__(modi_version, conn_type, verbose,
                         port, network_uuid, virtual_modules)
        self.ROOT_MOTOR = self.motors[0]

    def start(self):
        self.ROOT_MOTOR.speed = 0, 40
        sleep(4)
        self.ROOT_MOTOR.speed = 0, -40
        sleep(4)
        self.ROOT_MOTOR.speed = 0, 0


modi = Control()
modi.start()
