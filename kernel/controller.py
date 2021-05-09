from config import config
from work_states import SystemWorkStates
from IO import IO
from recognition_system import RecognitionSystem
from database import Database
from schedule.schedule_controller import ScheduleController


# Ядро/контроллер всей системы
class Controller:
    def __init__(self):
        self.recognition_system = RecognitionSystem()
        self.schedule_controller = ScheduleController()
        self.database = Database(config['faces_folder_path'])

    def run(self, system_work_state, auditorium=None, date=None):
        # groups_in_auditorium = self.schedule_controller.get_groups_in_auditorium(auditorium, date)
        # database = self.database.get_data_by_network(groups_in_auditorium)
        database = self.database.get_database_from_folder()
        results, frame = self.recognition_system.run(system_work_state, database)

        if system_work_state is SystemWorkStates.image.value:
            IO.save_image(frame, config['frame_output_path'])
            IO.save_detected_faces(results, frame, config['faces_detected'])
        if system_work_state is SystemWorkStates.webcam.value:
            pass
        if system_work_state is SystemWorkStates.video.value:
            pass
        IO.write_csv_results(results, config['results_output_path'])
