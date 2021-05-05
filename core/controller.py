from kernel import RecognitionSystem
from work_states import SystemWorkStates
from student_database import Database
from IO import IO

config = {
    'recognition_threshold': 0.1,
    'save_frame': True,
    'save_results': True,
    'frame_output_path': 'output/frame.jpg',
    'results_output_path': 'output/results.csv',
    'faces_detected': 'output/faces_detected/'
}


class Controller:
    def __init__(self):
        self.recognition_system = RecognitionSystem()
        self.database = Database()
        self.database_path = 'input/database/'

    def run(self, system_work_state):
        database = self.database.get_database(self.database_path)
        results, frame = self.recognition_system.run(system_work_state, database)

        IO.write_csv_results(results, config['results_output_path'])

        if system_work_state is SystemWorkStates.image:
            IO.save_image(frame, config['frame_output_path'])
            IO.save_detected_faces(results, frame, config['faces_detected'])
