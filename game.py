from control import Control
from enum import Enum
from time import sleep


class Stage(Enum):
    WAIT = 0
    HORIZONTAL = 1
    VERTICAL = 2
    PUSHING = 3


class Game:
    def __init__(self):
        self.stage = Stage.WAIT
        self.control = Control()

    def game_start(self):
        if (self.stage != Stage.WAIT):
            return
        self.stage = Stage.HORIZONTAL
        self.control.car_go_straight()

    def start_crane(self):
        if (self.stage != Stage.HORIZONTAL):
            return
        self.control.car_stop()
        self.stage = Stage.VERTICAL
        sleep(1)
        self.control.crane_pull_up()

    def start_push(self):
        if (self.stage != Stage.VERTICAL):
            return
        self.control.crane_stop()
        self.stage = Stage.PUSHING
        sleep(1)
        self.control.push()
        self.stage = Stage.WAIT
