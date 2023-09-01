import traci
import time


DEFULT_SUMO_BINARY_PATH = "/opt/homebrew/opt/sumo/share/sumo/bin/sumo"
# DEFULT_SUMO_BINARY_PATH = "/opt/homebrew/bin/sumo-gui"  # gui version, need to press "play" button

MAX_SIMULATION_TIME = 2 * 60   # 2 minutes

class SUMOClient(object):
    def __init__(self, config_path: str, sumo_binary_path: str = DEFULT_SUMO_BINARY_PATH):
        self.sumo_binary_path = sumo_binary_path
        self.config_path = config_path

    def run(self):
        start_time = time.time()
        sumo_cmd = [self.sumo_binary_path, "-c", self.config_path]
        traci.start(sumo_cmd)
        traci.simulationStep()
        # Run a simulation until all vehicles have arrived
        while traci.simulation.getMinExpectedNumber() > 0:
            if time.time() > start_time + MAX_SIMULATION_TIME:
                break
            traci.simulationStep()
        traci.close()
