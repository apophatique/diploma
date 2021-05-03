from kernel import RecognitionSystem
from work_states import SystemWorkStates
from student_database import Database
from schedule.schedule import ScheduleController
from IO import IO

config = {
    'recognition_threshold': 0.1,
    'save_frame': True,
    'save_results': True,
    'frame_output_path': '../output/frame.jpg',
    'results_output_path': '../output/test1_detect_res.txt'
}

class Controller:
    def __init__(self):
        self.system_work_state = SystemWorkStates.webcam
        self.recognition_system = RecognitionSystem()
        self.database = Database()
        self.database_path = '../input/database/popular/'

    def run(self):
        database = self.database.get_database(self.database_path)
        results, frame, boxes = self.recognition_system.run(self.system_work_state, database)

        IO.write_csv_results(results)
