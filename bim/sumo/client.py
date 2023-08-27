import traci
import time


DEFULT_SUMO_BINARY_PATH = "/opt/homebrew/opt/sumo/share/sumo/bin/sumo"
# DEFULT_SUMO_BINARY_PATH = "/opt/homebrew/bin/sumo-gui"  # TODO: or just sumo, not gui?


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
        max_simulation_time = 30 * 60  # 30 minutes
        while traci.simulation.getMinExpectedNumber() > 0:
            if time.time() > start_time + max_simulation_time:
                break
            traci.simulationStep()
        traci.close()
