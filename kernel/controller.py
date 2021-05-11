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

    def run(self, system_work_state, auditorium, date, time):
        groups_in_auditorium = self.schedule_controller.get_groups_in_auditorium(auditorium, date, time)
        data_students = self.database.get_students_data(groups_in_auditorium)
        results, frame = self.recognition_system.run(system_work_state, data_students)

        IO.save_work_results(results, frame, system_work_state, config)
